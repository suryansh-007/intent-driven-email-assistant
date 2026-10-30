"""Email triage: what is this email, what does it want, does it need a reply,
and how important is it?

Output per email (TriageResult):
  topic          personal | work_academic | finance | shopping_orders | advertisements |
                 social_apps | newsletters_blogs | accounts_security | spam
                 -> one Gmail label per email, e.g. "AI/Finance"
  category       importance: very_important | important | fyi | trivial | spam
                 -> very_important adds the "AI/Priority" label; trivial/spam add "AI/Can delete"
  email_type     finer-grained type used for wording (order_shipping, meeting, marketing …)
  gist           one sentence: what the email is about
  sender_wants   what the sender wants from the user ("Nothing; it's informational")
  needs_reply    yes | optional | no, with reply_reason
  action_items   things the user should do
  key_details    [{"label", "value"}]: order numbers, amounts, dates, codes …
  deadline       date or null
  can_delete     safe to delete without reading

Two backends, merged:
  * heuristic: sender/header patterns + cue words + regex extraction; always runs,
    needs no API, and acts as the baseline
  * llm: one call understands up to BATCH emails at once (used by /analyze and
    the bulk /triage endpoint); its output overrides the heuristic, except
    that strong phishing signals always win
Per-user sender rules (learned from "Wrong label?" corrections) override both.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..config import settings
from .. import llm

CATEGORIES = {   # importance
    "very_important": {"name": "Priority", "desc": "needs the user's personal attention or action soon"},
    "important": {"name": "Important", "desc": "worth reading and keeping, but no immediate action"},
    "fyi": {"name": "For information", "desc": "informational; fine to skim"},
    "trivial": {"name": "Low priority", "desc": "safe to delete without reading"},
    "spam": {"name": "Spam", "desc": "unsolicited junk, scam or phishing"},
}

TOPICS = {
    "personal": {"name": "Personal", "label": "AI/Personal",
                 "desc": "written by a real person to the user about non-work matters (family, friends, personal plans)"},
    "work_academic": {"name": "Work & Academic", "label": "AI/Work and Academic",
                      "desc": "work, teaching, research, courses, students, colleagues, meetings, institutional notices"},
    "finance": {"name": "Finance", "label": "AI/Finance",
                "desc": "banks, statements, transactions, bills, invoices, insurance, investments, tax"},
    "shopping_orders": {"name": "Shopping & Orders", "label": "AI/Shopping and Orders",
                        "desc": "order confirmations, shipping and delivery updates, receipts, returns"},
    "advertisements": {"name": "Advertisements", "label": "AI/Advertisements",
                       "desc": "promotions, sales, offers, marketing campaigns"},
    "social_apps": {"name": "Social & Apps", "label": "AI/Social and Apps",
                    "desc": "social network, game and app notifications, reminders, streaks, likes, mentions"},
    "newsletters_blogs": {"name": "Newsletters & Blogs", "label": "AI/Newsletters and Blogs",
                          "desc": "newsletters, digests, blog posts, articles, webinars, subscriptions to content"},
    "accounts_security": {"name": "Accounts & Security", "label": "AI/Accounts and Security",
                          "desc": "sign-ins, password resets, verification codes, security alerts, account changes"},
    "spam": {"name": "Spam", "label": "AI/Spam", "desc": "unsolicited junk, scams and phishing"},
}
PRIORITY_LABEL = "AI/Priority"
DELETE_LABEL = "AI/Can delete"

EMAIL_TYPES = {
    "conversation": "Personal or work conversation",
    "work_academic": "Work / academic",
    "meeting": "Meeting or calendar",
    "account_security": "Account & security",
    "finance": "Bank & finance",
    "order_shipping": "Order & delivery",
    "marketing": "Promotion / marketing",
    "social_notification": "App or social notification",
    "newsletter": "Newsletter",
    "other": "Other",
}

_TYPE_TO_TOPIC = {
    "conversation": "personal", "work_academic": "work_academic", "meeting": "work_academic",
    "account_security": "accounts_security", "finance": "finance", "order_shipping": "shopping_orders",
    "marketing": "advertisements", "social_notification": "social_apps", "newsletter": "newsletters_blogs",
}
_WORK_DOMAIN = re.compile(r"(\.edu|\.ac\.[a-z]{2}|\.edu\.[a-z]{2}|\.gov|\.gov\.[a-z]{2}|\.nic\.in|\.res\.in)$", re.I)


def topic_for(email_type: str, automated: bool, domain: str) -> str:
    if email_type == "other":
        return "newsletters_blogs" if automated else "personal"
    t = _TYPE_TO_TOPIC.get(email_type, "personal")
    if t == "personal" and _WORK_DOMAIN.search(domain or ""):
        return "work_academic"
    return t


@dataclass
class TriageResult:
    topic: str = "personal"
    category: str = "fyi"
    email_type: str = "other"
    gist: str = ""
    sender_wants: str = ""
    needs_reply: str = "no"
    reply_reason: str = ""
    action_items: List[str] = field(default_factory=list)
    key_details: List[Dict[str, str]] = field(default_factory=list)
    deadline: Optional[str] = None
    can_delete: bool = False
    category_reason: str = ""
    automated: bool = False
    method: str = "heuristic"
    intent: Optional[str] = None          # filled by the LLM path (reused as the intent label)
    intent_confidence: Optional[float] = None
    phishing_risk: Optional[float] = None

    def gmail_labels(self) -> List[str]:
        labels = [TOPICS.get(self.topic, TOPICS["personal"])["label"]]
        if self.category == "very_important":
            labels.append(PRIORITY_LABEL)
        if self.can_delete:
            labels.append(DELETE_LABEL)
        return labels

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update(topic_name=TOPICS.get(self.topic, TOPICS["personal"])["name"],
                 importance=self.category,
                 importance_name=CATEGORIES.get(self.category, CATEGORIES["fyi"])["name"],
                 priority=self.category == "very_important",
                 gmail_labels=self.gmail_labels(),
                 email_type_name=EMAIL_TYPES.get(self.email_type, self.email_type))
        return d


# ------------------------------------------------------------------ heuristics

_AUTOMATED_LOCAL = re.compile(
    r"^(no-?reply|do-?not-?reply|donotreply|notifications?|alerts?|updates?|order-?update|info|news(letter)?|"
    r"hello|hi|marketing|mailer(-daemon)?|digest|auto-?confirm|shipment-?tracking|account-?update|security|"
    r"support|team|contact|care|service|bounce.*|offers?|deals?|promo(tions)?|ship-?confirm|cbssbi\.cas|"
    r"[a-z]*\.?(alerts?|noreply))$", re.I)
_AUTOMATED_DOMAIN = re.compile(r"^(alerts?|mail|email|e|em|news|notify|notifications?|info|marketing|updates?)\.", re.I)
_AUTOMATED_BODY = re.compile(
    r"do not reply|don't reply|please do not respond|computer[- ]generated|automated (message|email)|"
    r"unsubscribe|manage (your )?(notification|email) (settings|preferences)|you are receiving this|"
    r"view (it )?in (your )?browser|email preferences|this is a system generated", re.I)

_TYPE_CUES = [
    ("account_security", r"password|sign[- ]?in|log[- ]?in attempt|verification code|\botp\b|one[- ]time (pass|code)|"
                         r"2-step|two-factor|security alert|new device|verify your (email|account)|reset your"),
    ("finance", r"statement|transaction|debited|credited|payment (received|due|successful)|invoice|\bbill\b|\bemi\b|"
                r"account balance|\bupi\b|credit card|net ?banking|refund (processed|initiated)|premium due"),
    ("order_shipping", r"\border\b|shipped|delivered|out for delivery|dispatched|tracking (id|number)|package|"
                       r"shipment|arriving|your item"),
    ("meeting", r"invitation:|calendar|meeting|\binvite\b|zoom\.us|meet\.google|rsvp|webex|teams meeting"),
    ("marketing", r"% off|\bsale\b|\boffer\b|\bdeal(s)?\b|discount|coupon|shop now|buy now|limited time|"
                  r"exclusive|save up to|free shipping|cashback|best price"),
    ("social_notification", r"streak|liked your|commented|mentioned you|friend request|new follower|connection request|"
                            r"play a game|puzzle|your weekly|tagged you|new message from|reacted|endorse"),
    ("newsletter", r"newsletter|digest|this week in|edition|issue #|read more|webinar"),
    ("work_academic", r"assignment|project|deadline|\bexam\b|submission|review|report|thesis|lecture|syllabus|"
                      r"marks|grade|internship|interview|proposal|manuscript|journal|conference"),
]
_REQUEST = re.compile(r"\?|\b(please|kindly|could you|can you|would you|let me know|need you to|request you|"
                      r"awaiting your|confirm|approve|rsvp)\b", re.I)


def sender_parts(sender: str):
    addr = (re.findall(r"<([^>]+)>", sender or "") or [sender or ""])[0].strip().lower()
    local, _, domain = addr.partition("@")
    display = re.sub(r"<[^>]*>", "", sender or "").strip().strip('"')
    return addr, local, domain, display


def is_automated(sender: str, body: str) -> bool:
    _, local, domain, _ = sender_parts(sender)
    return bool(_AUTOMATED_LOCAL.match(local) or _AUTOMATED_DOMAIN.match(domain) or _AUTOMATED_BODY.search(body))


def detect_type(subject: str, body: str, automated: bool) -> str:
    text = f"{subject}\n{body[:3000]}".lower()
    subj = subject.lower()
    scores = {}
    for t, pat in _TYPE_CUES:
        n_body = len(re.findall(pat, text))
        n_subj = len(re.findall(pat, subj))
        if n_body:
            scores[t] = n_body + 2 * n_subj
    if automated and "unsubscribe" in text and not scores.get("finance") and not scores.get("account_security"):
        scores["marketing"] = scores.get("marketing", 0) + 2
    if not scores:
        return "conversation" if not automated else "other"
    best = max(scores, key=scores.get)
    if not automated and best in ("marketing", "social_notification", "newsletter", "order_shipping"):
        # a real person mentioning "order" or "review" is still a conversation
        return "work_academic" if scores.get("work_academic") else "conversation"
    return best


# key detail extraction ---------------------------------------------------------

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DATE = re.compile(rf"\b(?:(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s+)?(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?|"
                   rf"{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})\b", re.I)
_AMOUNT = re.compile(r"(?:₹|rs\.?|inr|\$|usd|€|eur)\s?[\d,]+(?:\.\d{1,2})?", re.I)
_ORDER = re.compile(r"\border\s*(?:no\.?|number|#|id)?[:\s#]*([A-Z0-9][A-Z0-9-]{5,})", re.I)
_CODE = re.compile(r"\b(?:otp|code|pin)\b[^\d]{0,25}(\d{4,8})\b", re.I)
_DEADLINE = re.compile(rf"\b(?:by|before|due(?: on| by)?|no later than|deadline[: ]+)\s+((?:today|tomorrow|tonight|eod|end of (?:day|week)|"
                       rf"(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*|midnight|{_DATE.pattern}))", re.I)


def extract_key_details(subject: str, body: str) -> (List[Dict[str, str]], Optional[str]):
    text = f"{subject}\n{body[:4000]}"
    details, seen = [], set()

    def add(label, value):
        value = value.strip(" .,;")
        if value and (label, value.lower()) not in seen and len(details) < 6:
            seen.add((label, value.lower()))
            details.append({"label": label, "value": value})

    for m in _ORDER.finditer(text):
        if re.search(r"\d", m.group(1)):
            add("Order number", m.group(1))
    for m in _AMOUNT.finditer(text):
        add("Amount", m.group(0))
    for m in _CODE.finditer(text):
        add("Code", m.group(1))
    for m in list(_DATE.finditer(text))[:2]:
        add("Date", m.group(0))
    dl = _DEADLINE.search(text)
    deadline = dl.group(1) if dl else None
    return details, deadline


def _first_informative_sentence(body: str, skip_greeting=True) -> str:
    for line in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = line.strip()
        if len(s) < 25 or re.match(r"^(hi|hello|hey|dear|greetings)\b", s, re.I) and len(s) < 60:
            continue
        if re.search(r"https?://|unsubscribe|view (it )?in (your )?browser|\[image", s, re.I):
            continue
        return s[:220]
    return ""


def _request_sentence(body: str) -> str:
    for s in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = s.strip()
        if 15 < len(s) < 300 and _REQUEST.search(s) and not re.search(r"unsubscribe|do not reply", s, re.I):
            return s
    return ""


_WANTS_BY_TYPE = {
    "marketing": "Wants you to buy or engage with an offer. No action needed.",
    "social_notification": "Wants you to come back to the app. No action needed.",
    "newsletter": "Shares news or articles. Read only if interested.",
    "order_shipping": "Updates you on an order or delivery.",
    "account_security": "Asks you to act on your account (e.g. verify or reset). Do this only if you started the request.",
    "finance": "Informs you about account activity or a statement.",
    "meeting": "Invites you to, or updates you about, a meeting.",
}


def heuristic_triage(subject: str, body: str, sender: str, intent_label: str = "",
                     phishing_suspect: bool = False, has_attachment: bool = False) -> TriageResult:
    automated = is_automated(sender, body)
    etype = detect_type(subject, body, automated)
    details, deadline = extract_key_details(subject, body)
    if has_attachment:
        details.append({"label": "Attachment", "value": "Yes"})
    low = f"{subject}\n{body}".lower()
    _, _, domain, display = sender_parts(sender)
    who = display or domain or "The sender"

    # gist
    first = _first_informative_sentence(body)
    type_name = EMAIL_TYPES[etype]
    gist = f"{type_name} from {who}: {subject.strip() or first}"[:240]

    # what they want
    req = _request_sentence(body) if not automated else ""
    if req:
        wants = req[:220]
    elif etype in _WANTS_BY_TYPE:
        wants = _WANTS_BY_TYPE[etype]
    elif automated:
        wants = "Nothing; it's an automated, informational message."
    else:
        wants = "Shares information; no specific request found."
    if etype == "order_shipping" and "delivered" in low:
        wants = "Confirms that your order was delivered. No action needed."
    if etype == "finance" and re.search(r"statement", low):
        wants = "Sends your account statement" + (" (attached)" if has_attachment else "") + \
                ("; the file is password-protected" if re.search(r"protected by a password|password[- ]protected", low) else "") + \
                ". Keep it for your records."
    if etype == "account_security" and re.search(r"didn't request|did not request|if this was(n't| not) you", low):
        wants += " If you didn't request it, ignore it or secure your account."

    # reply decision
    if automated or phishing_suspect:
        needs, reason = "no", ("It comes from an automated / no-reply sender." if automated
                               else "It looks like phishing; don't reply.")
        if re.search(r"do not reply|don't reply|please do not respond", low):
            reason = "The email itself says not to reply."
    elif intent_label in ("request_action", "question", "meeting_scheduling", "approval_request",
                          "complaint", "follow_up") or req:
        needs, reason = "yes", "A person is asking you something or waiting for your response."
    elif intent_label in ("social_thanks", "information_fyi"):
        needs, reason = "optional", "A short acknowledgement would be polite, but isn't required."
    else:
        needs, reason = "optional", "It's from a person, but no clear question or request was found."

    # category
    if phishing_suspect:
        cat, why = "spam", "Shows phishing signals."
    elif etype in ("marketing", "social_notification"):
        cat, why = "trivial", f"{type_name}; safe to delete."  # noqa
    elif etype == "order_shipping":
        if re.search(r"\bdelivered\b", low):
            cat, why = "trivial", "Delivery confirmation; nothing left to do."
        else:
            cat, why = "fyi", "Order update; keep until it arrives."
    elif etype == "newsletter":
        cat, why = "trivial" if automated else "fyi", "Newsletter content."
    elif etype == "account_security":
        if re.search(r"\b(otp|verification code|one[- ]time)\b", low):
            cat, why = "trivial", "One-time code; useless once it expires."
        else:
            cat, why = "important", "Concerns access to one of your accounts."
    elif etype == "finance":
        if re.search(r"statement|invoice|\bbill\b|due|premium", low):
            cat, why = "important", "Financial document you may need later."
        else:
            cat, why = "fyi", "Routine account activity alert."
    elif not automated and needs == "yes":
        cat, why = "very_important", "A person needs a response from you."
    elif etype == "meeting":
        cat, why = ("very_important" if not automated else "important"), "Meeting or calendar item."
    elif not automated:
        cat, why = "important", "Written to you by a person."
    else:
        cat, why = "fyi", "Automated informational email."
    if deadline and cat in ("fyi", "important") and not automated:
        cat, why = "very_important", f"Has a deadline ({deadline})."

    actions = []
    if needs == "yes":
        actions.append("Reply to " + (display.split()[0] if display else "the sender"))
    if deadline:
        actions.append(f"Note the deadline: {deadline}")
    if etype == "account_security" and cat != "trivial":
        actions.append("Use the link only if you started this request; otherwise ignore it")
    if cat in ("trivial", "spam"):
        actions = ["Delete it"] if cat == "trivial" else ["Don't click links; delete or report as phishing"]

    topic = "spam" if phishing_suspect else topic_for(etype, automated, domain)
    if topic == "personal" and etype == "conversation" and re.search(_TYPE_CUES[-1][1], low):
        topic = "work_academic"
    return TriageResult(topic=topic, category=cat, email_type=etype, gist=gist, sender_wants=wants, needs_reply=needs,
                        reply_reason=reason, action_items=actions, key_details=details, deadline=deadline,
                        can_delete=cat in ("trivial", "spam"), category_reason=why, automated=automated)


# ------------------------------------------------------------------ LLM understanding

def _llm_system(intent_labels: List[str]) -> str:
    cats = "\n".join(f"- {k}: {v['desc']}" for k, v in CATEGORIES.items())
    topics = "\n".join(f"- {k}: {v['desc']}" for k, v in TOPICS.items())
    types = ", ".join(EMAIL_TYPES)
    return (
        "You are an email triage assistant. Read each email the way a sharp personal assistant would, "
        "then decide what it is about, what the sender wants, whether the user must reply, and how important it is.\n"
        f"Topic (exactly one):\n{topics}\n"
        f"Importance (field 'category'):\n{cats}\n"
        "Guidance: promotions, app/game/social notifications, delivery confirmations and expired one-time codes "
        "are trivial. Bank statements, bills, invoices, official or legal notices and receipts are important. "
        "A real person asking the user something, deadlines, interviews and approvals are very_important. "
        "Unexpected password-reset or account-verification emails with links may be phishing: say so in "
        "sender_wants, and use spam only when the signs are clear.\n"
        "needs_reply: 'no' for automated or no-reply senders, or when the email says not to reply; "
        "'yes' when a person asks a question or needs a response; otherwise 'optional'.\n"
        f"email_type is one of: {types}.\n"
        f"intent is one of: {', '.join(intent_labels)}, other.\n"
        "Tokens like [PERSON_1], [EMAIL_2] or [PHONE_1] are redacted personal data: copy them unchanged. "
        "Never invent facts, numbers or dates.")


_ITEM_SCHEMA = (
    '{"id": <same id>, "topic": <topic>, "category": <importance>, "category_reason": <short phrase>, '
    '"email_type": <type>, '
    '"gist": <one sentence: what the email is about, with specifics>, '
    '"sender_wants": <one sentence: what the sender wants the user to do, or "Nothing; informational">, '
    '"needs_reply": "yes"|"optional"|"no", "reply_reason": <short phrase>, '
    '"action_items": [<short imperative strings, max 3>], '
    '"key_details": [{"label": <e.g. Order number, Amount, Date, Deadline, Code>, "value": <exact text>}], '
    '"deadline": <text or null>, "intent": <intent>, "intent_confidence": <0-1>, "phishing_risk": <0-1>}')


def llm_triage_batch(items: List[Dict], intent_labels: List[str]) -> Dict[str, Dict]:
    """items: [{"id", "from", "subject", "body"}] (already redacted). Returns id -> dict."""
    blocks = []
    for it in items:
        blocks.append(f'### id={it["id"]}\nFrom: {it["from"]}\nSubject: {it["subject"]}\n'
                      f'Attachment: {"yes" if it.get("has_attachment") else "no"}\n\n{it["body"][:2500]}')
    user = ("Triage these emails.\n\n" + "\n\n".join(blocks) +
            f'\n\nReturn {{"results": [one object per email, in order, each shaped like {_ITEM_SCHEMA}]}}')
    out = llm.complete_json(_llm_system(intent_labels), user, max_tokens=350 * len(items) + 200)
    results = out.get("results", out if isinstance(out, list) else [])
    return {str(r.get("id")): r for r in results if isinstance(r, dict)}


def merge(h: TriageResult, l: Dict, phishing_suspect: bool) -> TriageResult:
    """LLM fields override the heuristic baseline, with validation."""
    r = TriageResult(**asdict(h))
    if l.get("topic") in TOPICS:
        r.topic = l["topic"]
    if l.get("category") in CATEGORIES:
        r.category = l["category"]
    if l.get("email_type") in EMAIL_TYPES:
        r.email_type = l["email_type"]
    for k in ("gist", "sender_wants", "reply_reason", "category_reason"):
        if isinstance(l.get(k), str) and l[k].strip():
            setattr(r, k, l[k].strip()[:300])
    if l.get("needs_reply") in ("yes", "optional", "no"):
        r.needs_reply = l["needs_reply"]
    if isinstance(l.get("action_items"), list):
        r.action_items = [str(a)[:160] for a in l["action_items"][:3]]
    if isinstance(l.get("key_details"), list) and l["key_details"]:
        kd = [{"label": str(d.get("label", ""))[:40], "value": str(d.get("value", ""))[:120]}
              for d in l["key_details"] if isinstance(d, dict) and d.get("value")]
        r.key_details = kd[:6] or r.key_details
    if l.get("deadline"):
        r.deadline = str(l["deadline"])[:80]
    r.intent = l.get("intent")
    try:
        r.intent_confidence = float(l.get("intent_confidence")) if l.get("intent_confidence") is not None else None
        r.phishing_risk = float(l.get("phishing_risk")) if l.get("phishing_risk") is not None else None
    except (TypeError, ValueError):
        pass
    if phishing_suspect:                        # strong heuristic signals always win
        r.category, r.needs_reply = "spam", "no"
    if r.category == "spam":
        r.topic = "spam"
    elif r.topic == "spam":
        r.category = "spam"
    if r.automated and r.needs_reply == "yes" and r.category != "very_important":
        r.needs_reply = "optional"
    r.can_delete = r.category in ("trivial", "spam")
    r.method = f"llm:{settings.llm_provider}"
    return r


def use_llm() -> bool:
    return settings.triage_backend == "llm" and llm.available()


def triage_many(items: List[Dict], intent_labels: List[str], heuristics: Dict[str, TriageResult],
                phishing: Dict[str, bool]) -> Dict[str, TriageResult]:
    """Runs LLM triage in parallel batches; any failed batch keeps its heuristic results."""
    results = dict(heuristics)
    if not use_llm() or not items:
        return results
    size = max(1, settings.triage_batch)
    chunks = [items[i:i + size] for i in range(0, len(items), size)]

    def run(chunk):
        try:
            return llm_triage_batch(chunk, intent_labels), None
        except Exception as exc:
            return {}, str(exc)[:160]

    with ThreadPoolExecutor(max_workers=max(1, min(settings.llm_concurrency, len(chunks)))) as pool:
        for out, err in pool.map(run, chunks):
            for eid, l in out.items():
                if eid in results:
                    results[eid] = merge(results[eid], l, phishing.get(eid, False))
            if err:
                for eid, r in results.items():
                    if r.method == "heuristic":
                        r.method = f"heuristic (llm failed: {err})"
    return results
