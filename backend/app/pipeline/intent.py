"""Intent classification module (Table 2, row 2).

Closed-set seed taxonomy + open-set extension:
  1. a classifier scores the email against the seed intents
     (zero-shot NLI [4], LLM prompting [5], or a keyword baseline);
  2. if the best score is below OPEN_INTENT_THRESHOLD, Action–Object
     open-intent discovery (OPINE-style [10]) produces a new label such as
     "open:renew_subscription" instead of forcing a wrong seed label (C3).
  3. a separate phishing check runs on every email; a phishing-suspect email
     is never sent to the reply generator.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from ..config import settings
from .. import llm
from .preprocess import split_sentences

# Seed taxonomy: label -> (natural-language hypothesis for zero-shot, keyword cues)
SEED_INTENTS: Dict[str, Dict] = {
    "request_action": {
        "desc": "The sender asks the recipient to do a task or send something.",
        "cues": ["please", "could you", "can you", "kindly", "would you", "need you to", "send me",
                 "share the", "update the", "review", "by eod", "asap", "action required"],
    },
    "question": {
        "desc": "The sender asks a question and wants information.",
        "cues": ["?", "wondering", "do you know", "what is", "when is", "how do", "any idea", "clarify"],
    },
    "meeting_scheduling": {
        "desc": "The sender wants to schedule, move or confirm a meeting or call.",
        "cues": ["meeting", "call", "schedule", "reschedule", "calendar", "slot", "available",
                 "availability", "zoom", "teams", "google meet", "catch up", "sync"],
    },
    "approval_request": {
        "desc": "The sender asks the recipient to approve, sign off or authorize something.",
        "cues": ["approve", "approval", "sign off", "sign-off", "authorize", "go-ahead", "permission", "consent"],
    },
    "follow_up": {
        "desc": "The sender is following up or reminding about an earlier message.",
        "cues": ["following up", "follow up", "follow-up", "reminder", "gentle reminder", "circling back",
                 "any update", "haven't heard", "checking in"],
    },
    "complaint": {
        "desc": "The sender is unhappy and reports a problem or complaint.",
        "cues": ["disappointed", "unacceptable", "complaint", "not working", "issue with", "problem with",
                 "refund", "frustrated", "still waiting", "poor"],
    },
    "information_fyi": {
        "desc": "The sender is sharing information or an update; no reply is required.",
        "cues": ["fyi", "for your information", "sharing", "attached", "please find", "update:", "announce",
                 "heads up", "note that"],
    },
    "social_thanks": {
        "desc": "The sender is thanking, congratulating or greeting the recipient.",
        "cues": ["thank you", "thanks", "congratulations", "congrats", "appreciate", "happy birthday",
                 "well done", "great job"],
    },
    "newsletter_promotion": {
        "desc": "The email is a newsletter, marketing message or automated notification.",
        "cues": ["unsubscribe", "newsletter", "% off", "sale", "offer", "promo", "webinar",
                 "view in browser", "no-reply", "noreply"],
    },
}

INTENT_LABELS = list(SEED_INTENTS)

# Intents for which a reply draft is usually not wanted.
NO_REPLY_INTENTS = {"newsletter_promotion"}

_PHISHING_CUES = [
    "verify your account", "account will be suspended", "confirm your password", "update your payment",
    "unusual sign-in", "click here to unlock", "won a", "lottery", "gift card", "wire transfer",
    "urgent action required", "kyc", "your account has been locked", "reset your password immediately",
    "claim your prize", "bitcoin",
]


@dataclass
class IntentResult:
    label: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    is_open: bool = False
    action_object: Optional[str] = None
    phishing_risk: float = 0.0
    phishing_reasons: List[str] = field(default_factory=list)
    method: str = ""

    @property
    def is_phishing_suspect(self) -> bool:
        return self.phishing_risk >= 0.5


# --- classifiers ----------------------------------------------------------------

def _keyword_scores(text: str) -> Dict[str, float]:
    low = text.lower()
    raw = {}
    for label, spec in SEED_INTENTS.items():
        hits = 0
        for c in spec["cues"]:
            if c[0].isalnum():
                hits += len(re.findall(rf"\b{re.escape(c)}\b", low))
            else:
                hits += low.count(c)
        raw[label] = hits
    total = sum(raw.values())
    if total == 0:
        return {k: 0.0 for k in raw}
    # Softened proportions so a single cue never claims full confidence.
    return {k: round(v / (total + 1.5), 3) for k, v in raw.items()}


@lru_cache(maxsize=2)
def _zeroshot_pipe(model_name: str):
    from transformers import pipeline  # imported lazily: heavy
    return pipeline("zero-shot-classification", model=model_name)


def _zeroshot_scores(text: str) -> Dict[str, float]:
    pipe = _zeroshot_pipe(settings.intent_model)
    hyps = {spec["desc"]: label for label, spec in SEED_INTENTS.items()}
    out = pipe(text[:2000], candidate_labels=list(hyps), hypothesis_template="{}", multi_label=False)
    return {hyps[l]: round(float(s), 3) for l, s in zip(out["labels"], out["scores"])}


def _llm_classify(text: str) -> Dict:
    taxonomy = "\n".join(f"- {k}: {v['desc']}" for k, v in SEED_INTENTS.items())
    system = ("You classify the intent of an email for an email assistant. "
              "Choose the single best label from the taxonomy, or 'other' if none fits.")
    user = (f"Taxonomy:\n{taxonomy}\n\nEmail:\n\"\"\"\n{text[:4000]}\n\"\"\"\n\n"
            'Return {"label": <label or "other">, "confidence": <0-1>, '
            '"action": <main verb the sender wants, lowercase>, "object": <its object noun, lowercase>, '
            '"phishing_risk": <0-1>}')
    return llm.complete_json(system, user)


# --- open-intent discovery (Action–Object) -------------------------------------

_REQUEST_LEAD = re.compile(
    r"\b(?:please|kindly|could you|can you|would you|need (?:you )?to|want to|would like to|i'd like to|"
    r"we need to|let's|let us)\s+(?:please\s+)?([a-z]+)\s+(?:(?:the|a|an|my|our|your|this|that|these|those|some)\s+)?"
    r"([a-z]+(?:\s[a-z]+)?)", re.IGNORECASE)
_STOP_OBJ = {"me", "you", "us", "it", "them", "know", "to", "on", "in", "for", "with", "at", "and", "or",
             "whether", "if", "that", "we", "i", "what", "how", "when", "where", "which", "who", "why", "is", "are"}


def discover_action_object(text: str) -> Optional[str]:
    """Light OPINE-style Action–Object extraction. Uses spaCy when installed."""
    try:
        import spacy  # optional
        nlp = _spacy()
        doc = nlp(text[:3000])
        for sent in doc.sents:
            for tok in sent:
                if tok.pos_ == "VERB" and tok.dep_ in ("ROOT", "xcomp", "ccomp"):
                    objs = [c for c in tok.children if c.dep_ in ("dobj", "obj")]
                    if objs:
                        obj = objs[0]
                        comp = [c.text.lower() for c in obj.children if c.dep_ == "compound"]
                        return f"{tok.lemma_.lower()}_{'_'.join(comp + [obj.lemma_.lower()])}"
    except Exception:
        pass
    for sent in split_sentences(text):
        m = _REQUEST_LEAD.search(sent)
        if m:
            verb = m.group(1).lower()
            words = m.group(2).lower().split()
            if words and words[0] in _STOP_OBJ:       # "confirm whether we …" is not an object
                continue
            obj_words = [w for w in words if w not in _STOP_OBJ]
            if obj_words and verb not in _STOP_OBJ:
                return f"{verb}_{'_'.join(obj_words)}"
    return None


@lru_cache(maxsize=1)
def _spacy():
    import spacy
    return spacy.load("en_core_web_sm")


# --- phishing screen ------------------------------------------------------------

def phishing_check(subject: str, body: str, sender: str) -> (float, List[str]):
    low = f"{subject}\n{body}".lower()
    reasons = [c for c in _PHISHING_CUES if c in low]
    links = re.findall(r"https?://([^/\s]+)", body, re.IGNORECASE)
    if any(re.match(r"^\d{1,3}(\.\d{1,3}){3}", h) for h in links):
        reasons.append("link to a raw IP address")
    if re.search(r"bit\.ly|tinyurl|t\.co/|rb\.gy", body, re.IGNORECASE):
        reasons.append("shortened link")
    sender_dom = (re.findall(r"@([\w.-]+)", sender or "") or [""])[0].lower()
    brands = ["paypal", "amazon", "sbi", "hdfc", "icici", "google", "microsoft", "apple", "netflix"]
    for b in brands:
        if b in low and sender_dom and b not in sender_dom and any(k in low for k in ("account", "password", "verify")):
            reasons.append(f"mentions {b} but sender domain is {sender_dom}")
            break
    risk = min(1.0, 0.3 * len(reasons))
    return round(risk, 2), reasons


# --- entry point ----------------------------------------------------------------

def classify(subject: str, body: str, sender: str = "", backend: Optional[str] = None) -> IntentResult:
    text = f"Subject: {subject}\n\n{body}".strip()
    backend = backend or settings.intent_backend
    llm_out = None
    if backend == "llm" and llm.available():
        try:
            llm_out = _llm_classify(text)
            label = llm_out.get("label", "other")
            conf = float(llm_out.get("confidence", 0.0))
            scores = {label: conf}
            method = f"llm:{settings.llm_provider}"
        except Exception:
            llm_out, backend = None, "keyword"
    if llm_out is None:
        if backend == "zeroshot":
            scores, method = _zeroshot_scores(text), f"zeroshot:{settings.intent_model}"
        else:
            scores, method = _keyword_scores(text), "keyword"
        label = max(scores, key=scores.get)
        conf = scores[label]

    result = IntentResult(label=label, confidence=round(conf, 3), scores=scores, method=method)

    # keyword scores are cue proportions, not probabilities, so they get a lower bar
    threshold = 0.15 if method == "keyword" else settings.open_intent_threshold
    if label not in SEED_INTENTS or conf < threshold:
        ao = None
        if llm_out and llm_out.get("action") and llm_out.get("object"):
            ao = f"{llm_out['action']}_{llm_out['object']}".replace(" ", "_").lower()
        ao = ao or discover_action_object(body)
        result.action_object = ao
        if ao:
            result.label, result.is_open = f"open:{ao}", True
        else:
            result.label, result.is_open = "other", label not in SEED_INTENTS

    risk, reasons = phishing_check(subject, body, sender)
    if llm_out and isinstance(llm_out.get("phishing_risk"), (int, float)):
        risk = max(risk, float(llm_out["phishing_risk"]))
    result.phishing_risk, result.phishing_reasons = round(risk, 2), reasons
    return result
