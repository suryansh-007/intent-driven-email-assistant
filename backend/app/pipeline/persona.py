"""Persona store (Table 2, row 6) + evaluation log.

Learns each user's preferred tone, sign-off and reply length from
accept / edit / reject feedback ([6][11][14]), and logs every pipeline run and
its outcome so the accept-or-minor-edit rate and per-stage latency (Section
9.2) can be computed directly from real usage.

Users are identified only by a salted SHA-256 hash of their address, computed
in the add-on, and stored replies are PII-redacted (constraint C2).
"""
import difflib
import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import settings
from .preprocess import Redactor

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    user_id TEXT PRIMARY KEY,
    tone TEXT DEFAULT 'professional',
    signoff TEXT DEFAULT '',
    display_name TEXT DEFAULT '',
    avg_words REAL DEFAULT 0,
    n_accept INTEGER DEFAULT 0,
    n_edit INTEGER DEFAULT 0,
    n_reject INTEGER DEFAULT 0,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT, intent TEXT, text TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS sender_rules (
    user_id TEXT, sender_key TEXT, category TEXT, updated_at REAL,
    PRIMARY KEY (user_id, sender_key)
);
CREATE TABLE IF NOT EXISTS triage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT, source TEXT, category TEXT, email_type TEXT, needs_reply TEXT,
    method TEXT, corrected_to TEXT, sender_key TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS events (
    request_id TEXT PRIMARY KEY,
    user_id TEXT, intent TEXT, intent_confidence REAL, is_open_intent INTEGER,
    phishing_risk REAL, n_turns INTEGER,
    summary_method TEXT, faithfulness REAL, generator_method TEXT,
    latency_ms TEXT, created_at REAL,
    action TEXT, edit_ratio REAL, rating INTEGER, feedback_at REAL
);
"""


@contextmanager
def _db():
    with _lock:
        conn = sqlite3.connect(settings.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(_SCHEMA)
            for table, col in (("triage_log", "topic"), ("triage_log", "corrected_flag")):
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")   # upgrade older databases
                except sqlite3.OperationalError:
                    pass
            yield conn
            conn.commit()
        finally:
            conn.close()


@dataclass
class Persona:
    user_id: str
    tone: str = "professional"
    signoff: str = ""
    display_name: str = ""
    avg_words: float = 0.0
    n_accept: int = 0
    n_edit: int = 0
    n_reject: int = 0
    examples: List[str] = field(default_factory=list)

    def as_prompt(self) -> str:
        lines = [f"Preferred tone: {self.tone}."]
        if self.avg_words:
            lines.append(f"Their replies are usually about {int(self.avg_words)} words long.")
        if self.signoff:
            lines.append(f"They sign off with: \"{self.signoff}\".")
        if self.examples:
            lines.append("Replies they previously approved (match this style, not the content):")
            lines += [f"---\n{e}" for e in self.examples]
        return "\n".join(lines)


def get_persona(user_id: str, intent: Optional[str] = None) -> Persona:
    with _db() as db:
        row = db.execute("SELECT * FROM personas WHERE user_id=?", (user_id,)).fetchone()
        p = Persona(user_id=user_id)
        if row:
            for k in ("tone", "signoff", "display_name", "avg_words", "n_accept", "n_edit", "n_reject"):
                setattr(p, k, row[k])
        rows = db.execute(
            "SELECT text FROM examples WHERE user_id=? ORDER BY (intent=?) DESC, created_at DESC LIMIT 2",
            (user_id, intent or ""),
        ).fetchall()
        p.examples = [r["text"] for r in rows]
    return p


def update_settings(user_id: str, tone: Optional[str] = None, signoff: Optional[str] = None,
                    display_name: Optional[str] = None) -> Persona:
    with _db() as db:
        db.execute("INSERT OR IGNORE INTO personas(user_id, updated_at) VALUES(?, ?)", (user_id, time.time()))
        for col, val in (("tone", tone), ("signoff", signoff), ("display_name", display_name)):
            if val is not None:
                db.execute(f"UPDATE personas SET {col}=?, updated_at=? WHERE user_id=?", (val, time.time(), user_id))
    return get_persona(user_id)


def log_run(request_id: str, user_id: str, record: Dict) -> None:
    with _db() as db:
        db.execute(
            """INSERT OR REPLACE INTO events(request_id, user_id, intent, intent_confidence, is_open_intent,
               phishing_risk, n_turns, summary_method, faithfulness, generator_method, latency_ms, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (request_id, user_id, record["intent"], record["intent_confidence"], int(record["is_open_intent"]),
             record["phishing_risk"], record["n_turns"], record["summary_method"], record["faithfulness"],
             record["generator_method"], json.dumps(record["latency_ms"]), time.time()),
        )


def _detect_signoff(text: str) -> str:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for i in range(len(lines) - 1, max(-1, len(lines) - 4), -1):
        if re.match(r"^(best|regards|thanks|thank you|cheers|warm regards|kind regards|sincerely|"
                    r"best regards|many thanks|yours)\b[\w ,]*,?$", lines[i], re.IGNORECASE):
            return lines[i].rstrip(",") + ","
    return ""


def _redact_example(text: str, names: List[str]) -> str:
    text = Redactor().redact(text, names)
    # also mask whoever is greeted on the first line ("Hi Rohan," -> "Hi [NAME],")
    return re.sub(r"^(\s*(?:hi|hello|hey|dear)\s+)([^,\n]{1,40})(,)", r"\1[NAME]\3", text, count=1,
                  flags=re.IGNORECASE)


def record_feedback(user_id: str, request_id: str, action: str, draft: str, final_text: str = "",
                    intent: str = "", rating: Optional[int] = None, names: List[str] = ()) -> Dict:
    """action: 'accept' | 'edit' | 'reject'. Returns the edit ratio and updated counts."""
    action = action.lower()
    if action not in ("accept", "edit", "reject"):
        raise ValueError("action must be accept, edit or reject")
    final_text = final_text or (draft if action == "accept" else "")
    edit_ratio = 0.0
    if action in ("accept", "edit") and draft:
        edit_ratio = round(1 - difflib.SequenceMatcher(None, draft, final_text).ratio(), 3)
        if action == "edit" and edit_ratio < 0.02:
            action = "accept"  # opened the editor but changed nothing

    with _db() as db:
        db.execute("INSERT OR IGNORE INTO personas(user_id, updated_at) VALUES(?, ?)", (user_id, time.time()))
        db.execute(f"UPDATE personas SET n_{action}=n_{action}+1, updated_at=? WHERE user_id=?",
                   (time.time(), user_id))
        if action in ("accept", "edit") and final_text:
            words = len(final_text.split())
            row = db.execute("SELECT avg_words, n_accept + n_edit AS n FROM personas WHERE user_id=?",
                             (user_id,)).fetchone()
            n = max(1, row["n"])
            new_avg = words if not row["avg_words"] else row["avg_words"] + (words - row["avg_words"]) / n
            db.execute("UPDATE personas SET avg_words=? WHERE user_id=?", (round(new_avg, 1), user_id))
            signoff = _detect_signoff(final_text)
            if signoff:
                db.execute("UPDATE personas SET signoff=? WHERE user_id=?", (signoff, user_id))
            db.execute("INSERT INTO examples(user_id, intent, text, created_at) VALUES(?,?,?,?)",
                       (user_id, intent, _redact_example(final_text, list(names))[:1200], time.time()))
            db.execute("""DELETE FROM examples WHERE user_id=? AND id NOT IN
                          (SELECT id FROM examples WHERE user_id=? ORDER BY created_at DESC LIMIT 20)""",
                       (user_id, user_id))
        db.execute("UPDATE events SET action=?, edit_ratio=?, rating=?, feedback_at=? WHERE request_id=?",
                   (action, edit_ratio, rating, time.time(), request_id))
        row = db.execute("SELECT n_accept, n_edit, n_reject FROM personas WHERE user_id=?", (user_id,)).fetchone()
    return {"action": action, "edit_ratio": edit_ratio, **dict(row)}


def stats(user_id: Optional[str] = None) -> Dict:
    """Aggregates for the evaluation section: acceptance, edit size, latency, per-intent breakdown."""
    where, args = ("WHERE user_id=?", (user_id,)) if user_id else ("", ())
    with _db() as db:
        rows = [dict(r) for r in db.execute(f"SELECT * FROM events {where}", args).fetchall()]
    judged = [r for r in rows if r["action"]]
    n = len(judged)

    def rate(pred):
        return round(sum(1 for r in judged if pred(r)) / n, 3) if n else None

    lat = [json.loads(r["latency_ms"]) for r in rows if r["latency_ms"]]
    stages = sorted({k for l in lat for k in l})
    by_intent: Dict[str, Dict] = {}
    for r in judged:
        key = "open" if r["is_open_intent"] else r["intent"]
        d = by_intent.setdefault(key, {"n": 0, "accept": 0, "edit": 0, "reject": 0})
        d["n"] += 1
        d[r["action"]] += 1
    return {
        "runs": len(rows),
        "with_feedback": n,
        "accept_rate": rate(lambda r: r["action"] == "accept"),
        "accept_or_minor_edit_rate": rate(lambda r: r["action"] == "accept" or
                                          (r["action"] == "edit" and (r["edit_ratio"] or 0) <= 0.2)),
        "reject_rate": rate(lambda r: r["action"] == "reject"),
        "mean_edit_ratio": round(sum(r["edit_ratio"] or 0 for r in judged) / n, 3) if n else None,
        "mean_faithfulness": round(sum(r["faithfulness"] or 0 for r in rows) / len(rows), 3) if rows else None,
        "mean_latency_ms": {s: round(sum(l.get(s, 0) for l in lat) / len(lat), 1) for s in stages} if lat else {},
        "by_intent": by_intent,
        "triage": triage_stats(user_id),
    }


# ------------------------------------------------------------------ triage rules & log

def sender_keys(sender: str) -> List[str]:
    """Most specific first: full address, then domain."""
    addr = (re.findall(r"<([^>]+)>", sender or "") or [sender or ""])[0].strip().lower()
    keys = [addr] if "@" in addr else []
    if "@" in addr:
        keys.append("@" + addr.split("@", 1)[1])
    return keys


def get_sender_rule(user_id: str, sender: str) -> Optional[str]:
    keys = sender_keys(sender)
    if not keys:
        return None
    with _db() as db:
        for k in keys:
            row = db.execute("SELECT category FROM sender_rules WHERE user_id=? AND sender_key=?",
                             (user_id, k)).fetchone()
            if row:
                return row["category"]
    return None


def set_sender_rule(user_id: str, sender: str, category: str, whole_domain: bool = False) -> str:
    keys = sender_keys(sender)
    if not keys:
        raise ValueError("Could not read the sender address")
    key = keys[-1] if whole_domain and len(keys) > 1 else keys[0]
    with _db() as db:
        db.execute("INSERT OR REPLACE INTO sender_rules(user_id, sender_key, category, updated_at) VALUES(?,?,?,?)",
                   (user_id, key, category, time.time()))
    return key


def log_triage(user_id: str, source: str, rows: List[Dict]) -> None:
    with _db() as db:
        db.executemany(
            "INSERT INTO triage_log(user_id, source, topic, category, email_type, needs_reply, method, sender_key,"
            " created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            [(user_id, source, r.get("topic", ""), r["category"], r["email_type"], r["needs_reply"], r["method"],
              r.get("sender_key", ""), time.time()) for r in rows])


def log_correction(user_id: str, sender_key: str, predicted: str, corrected: str, flag: str = "") -> None:
    with _db() as db:
        db.execute("INSERT INTO triage_log(user_id, source, topic, category, email_type, needs_reply, method,"
                   " corrected_to, corrected_flag, sender_key, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (user_id, "correction", predicted, "", "", "", "", corrected, flag, sender_key, time.time()))


def triage_stats(user_id: Optional[str] = None) -> Dict:
    where, args = ("WHERE user_id=?", (user_id,)) if user_id else ("", ())
    with _db() as db:
        rows = [dict(r) for r in db.execute(f"SELECT * FROM triage_log {where}", args).fetchall()]
    judged = [r for r in rows if r["source"] != "correction"]
    corrections = [r for r in rows if r["source"] == "correction"]
    topic_fixes = [r for r in corrections if r["corrected_to"] and r["corrected_to"] != r["topic"]]
    by_topic: Dict[str, int] = {}
    by_imp: Dict[str, int] = {}
    for r in judged:
        by_topic[r.get("topic") or "?"] = by_topic.get(r.get("topic") or "?", 0) + 1
        by_imp[r["category"]] = by_imp.get(r["category"], 0) + 1
    return {
        "emails_triaged": len(judged),
        "by_topic": by_topic,
        "by_importance": by_imp,
        "corrections": len(corrections),
        "topic_corrections": len(topic_fixes),
        # every topic correction marks one wrong prediction, so this is an upper bound on topic accuracy
        "topic_agreement_rate": round(1 - len(topic_fixes) / len(judged), 3) if judged else None,
        "confusions": [{"predicted": r["topic"], "corrected_to": r["corrected_to"],
                        "can_delete": r.get("corrected_flag") or None} for r in corrections[-50:]],
    }
