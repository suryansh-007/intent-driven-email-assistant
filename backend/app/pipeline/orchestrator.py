"""Runs the full categorize → summarize → respond pipeline for one thread."""
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from typing import Dict, List, Optional

from ..config import settings
from . import intent as intent_mod
from . import persona as persona_mod
from . import retrieval
from . import triage as triage_mod
from .generate import FusedContext, generate, TONES
from .preprocess import Redactor, clean_body, first_name, participant_names
from .summarize import Turn, summarize, extractive_summary


# request_id -> participant names, kept briefly so feedback (which arrives
# later) can redact the user's saved reply with the same names (C2).
_RECENT_NAMES: "OrderedDict[str, List[str]]" = OrderedDict()


def names_for(request_id: str) -> List[str]:
    return _RECENT_NAMES.get(request_id, [])


class Timer:
    def __init__(self):
        self.ms: Dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._start = self._t0

    def lap(self, stage: str):
        now = time.perf_counter()
        self.ms[stage] = round((now - self._t0) * 1000, 1)
        self._t0 = now

    def total(self):
        self.ms["total"] = round((time.perf_counter() - self._start) * 1000, 1)


def restore_obj(red: Redactor, obj):
    """Puts real names/addresses back into every string of a nested structure."""
    if isinstance(obj, str):
        return red.restore(obj)
    if isinstance(obj, list):
        return [restore_obj(red, x) for x in obj]
    if isinstance(obj, dict):
        return {k: restore_obj(red, v) for k, v in obj.items()}
    return obj


def llm_sender(sender: str, red: Redactor, names: List[str]) -> str:
    """'Rohan Mehta <rohan@college.edu>' -> '[PERSON_1] <…@college.edu>': keeps the domain
    (useful for triage) but not the personal part of the address."""
    _, local, domain, display = triage_mod.sender_parts(sender)
    shown = red.redact(display, names) if (display and settings.redact_pii) else display
    return f"{shown} <…@{domain}>" if domain else (shown or "unknown")


def apply_sender_rule(user_id: str, sender: str, t: "triage_mod.TriageResult") -> None:
    """Rule format: "<topic>", "<topic>|delete" (always disposable) or "<topic>|keep" (never)."""
    rule = persona_mod.get_sender_rule(user_id, sender)
    if not rule:
        return
    topic, _, flag = rule.partition("|")
    if topic in triage_mod.CATEGORIES and topic not in triage_mod.TOPICS:   # rule saved by an older version
        topic, flag = ("spam" if topic == "spam" else t.topic), ("delete" if topic in ("trivial", "spam") else "keep")
    if topic in triage_mod.TOPICS:
        t.topic = topic
    if topic == "spam" or flag == "delete":
        t.category = "spam" if topic == "spam" else "trivial"
        t.can_delete, t.needs_reply = True, "no"
    elif flag == "keep":
        t.can_delete = False
        if t.category in ("trivial", "spam"):
            t.category = "fyi"
    t.category_reason = "Your rule for this sender"
    t.method += "+rule"


def run(user_id: str, subject: str, messages: List[Dict], tone: Optional[str] = None,
        guidance: str = "", force_draft: bool = False, max_turns: int = 10,
        has_attachment: bool = False) -> Dict:
    """messages: oldest-first list of {"from", "date", "body"} dicts."""
    timer = Timer()
    request_id = uuid.uuid4().hex[:12]
    messages = messages[-max_turns:]

    # 1. preprocessing + PII redaction --------------------------------------------
    red = Redactor()
    names = participant_names([m.get("from", "") for m in messages])
    _RECENT_NAMES[request_id] = names
    while len(_RECENT_NAMES) > 2000:
        _RECENT_NAMES.popitem(last=False)
    turns: List[Turn] = []
    raw_latest = ""
    for m in messages:
        body = clean_body(m.get("body", ""))[:6000]
        if not body:
            continue
        raw_latest = body
        text = red.redact(body, names) if settings.redact_pii else body
        sender = first_name(m.get("from", "")) or "Sender"
        sender = red.redact(sender, names) if settings.redact_pii else sender
        turns.append(Turn(sender=sender, date=m.get("date", ""), text=text))
    if not turns:
        turns = [Turn(sender="Sender", date="", text="(empty message)")]
    subj = red.redact(subject or "", names) if settings.redact_pii else (subject or "")
    latest = turns[-1]
    timer.lap("preprocess")

    # 2. intent + triage ---------------------------------------------------------------
    latest_from = messages[-1].get("from", "") if messages else ""
    # When both use the LLM, a single "understand" call also yields the intent label.
    shared_llm = settings.intent_backend == "llm" and triage_mod.use_llm()
    it = intent_mod.classify(subj, latest.text, latest_from, backend="keyword" if shared_llm else None)
    timer.lap("intent")

    tri = triage_mod.heuristic_triage(subject or "", raw_latest, latest_from, it.label,
                                      it.is_phishing_suspect, has_attachment)
    if triage_mod.use_llm():
        item = {"id": "0", "from": llm_sender(latest_from, red, names), "subject": subj,
                "body": "\n\n".join(f"[{t.sender}]: {t.text}" for t in turns[-3:]) if len(turns) > 1 else latest.text,
                "has_attachment": has_attachment}
        out = triage_mod.triage_many([item], intent_mod.INTENT_LABELS, {"0": tri}, {"0": it.is_phishing_suspect})
        tri = out["0"]
        if shared_llm and tri.intent:
            if tri.intent in intent_mod.SEED_INTENTS:
                it.label, it.is_open, it.action_object = tri.intent, False, None
            it.confidence = round(tri.intent_confidence or it.confidence, 3)
            it.scores = {it.label: it.confidence}
            it.method = tri.method
        if tri.phishing_risk is not None:
            it.phishing_risk = round(max(it.phishing_risk, tri.phishing_risk), 2)
    apply_sender_rule(user_id, latest_from, tri)
    if tri.category == "spam" and it.phishing_risk < 0.5 and tri.method.startswith("llm"):
        it.phishing_risk = 0.5           # the model saw clear spam/phishing signs
    timer.lap("triage")

    # 3. summary (thread) ------------------------------------------------------------
    if len(turns) > 1 or len(latest.text.split()) > 120:
        sm = summarize(turns)
    else:
        sm = None
    timer.lap("summarize")

    # 4. context fusion: persona + KB --------------------------------------------------
    persona = persona_mod.get_persona(user_id, it.label)
    use_tone = tone or persona.tone or "professional"
    kb = retrieval.retrieve(f"{subj} {latest.text}")
    seed = intent_mod.SEED_INTENTS.get(it.label)
    intent_desc = seed["desc"] if seed else (
        f"Open intent discovered from the text (action–object: {it.action_object})" if it.action_object
        else "No seed intent fits well.")
    ctx = FusedContext(
        intent=it.label, intent_description=intent_desc,
        summary=sm.summary if sm else extractive_summary(turns, 2),
        latest_message=latest.text, latest_sender_name=latest.sender, subject=subj,
        persona=persona, tone=use_tone, guidance=red.redact(guidance, names) if guidance else "", kb=kb,
    )
    timer.lap("fusion")

    # 5. generation (skipped for phishing / newsletters unless forced) ----------------
    skipped = None
    if (it.is_phishing_suspect or tri.category == "spam") and not force_draft:
        skipped = "phishing_suspect"
    elif (tri.needs_reply == "no" or it.label in intent_mod.NO_REPLY_INTENTS) and not force_draft:
        skipped = "no_reply_needed"
    draft = None if skipped else generate(ctx)
    timer.lap("generate")
    timer.total()

    # 6. restore real names/addresses for display ---------------------------------------
    result = {
        "request_id": request_id,
        "intent": {
            "label": it.label,
            "confidence": it.confidence,
            "is_open": it.is_open,
            "action_object": it.action_object,
            "method": it.method,
            "top": sorted(it.scores.items(), key=lambda kv: kv[1], reverse=True)[:3],
        },
        "phishing": {"risk": it.phishing_risk, "suspect": it.is_phishing_suspect, "reasons": it.phishing_reasons},
        "triage": restore_obj(red, tri.to_dict()),
        "sender_key": (persona_mod.sender_keys(latest_from) or [""])[0],
        "summary": None if sm is None else {
            "text": red.restore(sm.summary), "faithfulness": sm.faithfulness,
            "method": sm.method, "fell_back": sm.fell_back,
        },
        "draft": None if draft is None else {"text": red.restore(draft.text), "method": draft.method},
        "draft_skipped": skipped,
        "tone": use_tone,
        "tones": list(TONES),
        "kb_sources": sorted({src for src, _ in kb}),
        "n_turns": len(turns),
        "redactions": len(red.mapping),
        "latency_ms": timer.ms,
    }
    persona_mod.log_triage(user_id, "open", [{"topic": tri.topic, "category": tri.category, "email_type": tri.email_type,
                                              "needs_reply": tri.needs_reply, "method": tri.method,
                                              "sender_key": result["sender_key"]}])
    persona_mod.log_run(request_id, user_id, {
        "intent": it.label, "intent_confidence": it.confidence, "is_open_intent": it.is_open,
        "phishing_risk": it.phishing_risk, "n_turns": len(turns),
        "summary_method": sm.method if sm else "none", "faithfulness": sm.faithfulness if sm else None,
        "generator_method": draft.method if draft else f"skipped:{skipped}", "latency_ms": timer.ms,
    })
    return result


def triage_bulk(user_id: str, emails: List[Dict]) -> List[Dict]:
    """Bulk inbox sorting. emails: [{"id", "from", "subject", "body", "has_attachment"}]."""
    timer = Timer()
    heur, phish, items, reds = {}, {}, [], {}
    for e in emails:
        eid = str(e["id"])
        body = clean_body(e.get("body", ""))[:4000]
        subject = e.get("subject", "") or ""
        sender = e.get("from", "")
        it = intent_mod.classify(subject, body, sender, backend="keyword")
        heur[eid] = triage_mod.heuristic_triage(subject, body, sender, it.label, it.is_phishing_suspect,
                                                bool(e.get("has_attachment")))
        phish[eid] = it.is_phishing_suspect
        red = Redactor()
        names = participant_names([sender])
        reds[eid] = red
        items.append({"id": eid, "from": llm_sender(sender, red, names),
                      "subject": red.redact(subject, names) if settings.redact_pii else subject,
                      "body": red.redact(body, names) if settings.redact_pii else body,
                      "has_attachment": bool(e.get("has_attachment"))})
    if settings.bulk_triage_backend == "llm":
        results = triage_mod.triage_many(items, intent_mod.INTENT_LABELS, heur, phish)
    else:
        results = heur                       # quota saver: rules only for bulk sorting
    out, log_rows = [], []
    for e in emails:
        eid = str(e["id"])
        t = results[eid]
        apply_sender_rule(user_id, e.get("from", ""), t)
        key = (persona_mod.sender_keys(e.get("from", "")) or [""])[0]
        d = restore_obj(reds[eid], t.to_dict())
        d["id"], d["sender_key"] = eid, key
        out.append(d)
        log_rows.append({"topic": t.topic, "category": t.category, "email_type": t.email_type, "needs_reply": t.needs_reply,
                         "method": t.method, "sender_key": key})
    persona_mod.log_triage(user_id, "bulk", log_rows)
    timer.total()
    return out
