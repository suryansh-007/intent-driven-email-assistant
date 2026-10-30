"""FastAPI service the Gmail add-on calls.

Run locally:   uvicorn app.main:app --reload --port 8000   (from backend/)
Docs:          http://localhost:8000/docs
"""
import hmac
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import orchestrator, persona, retrieval
from .pipeline.generate import TONES
from .pipeline.triage import CATEGORIES, DELETE_LABEL, EMAIL_TYPES, PRIORITY_LABEL, TOPICS

app = FastAPI(title="Intent-Driven Email Assistant", version="0.1.0")


def require_key(x_api_key: str = Header(default="")):
    if not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class Message(BaseModel):
    from_: str = Field("", alias="from")
    date: str = ""
    body: str

    model_config = {"populate_by_name": True}


class AnalyzeRequest(BaseModel):
    user_id: str = Field(..., description="Salted hash of the user's address, computed client-side")
    subject: str = ""
    messages: List[Message] = Field(..., min_length=1, description="Oldest first")
    tone: Optional[str] = None
    guidance: str = Field("", max_length=500, description="Optional user instructions for the reply")
    force_draft: bool = False
    has_attachment: bool = False


class TriageEmail(BaseModel):
    id: str
    from_: str = Field("", alias="from")
    subject: str = ""
    date: str = ""
    body: str = ""
    has_attachment: bool = False

    model_config = {"populate_by_name": True}


class TriageRequest(BaseModel):
    user_id: str
    emails: List[TriageEmail] = Field(..., min_length=1, max_length=40)


class CorrectionRequest(BaseModel):
    user_id: str
    sender: str = Field(..., description="The From header of the email")
    predicted: str = Field("", description="Topic the assistant had chosen")
    topic: str
    can_delete: Optional[bool] = Field(None, description="true = always disposable, false = never, null = let the AI decide")
    whole_domain: bool = False


class FeedbackRequest(BaseModel):
    user_id: str
    request_id: str
    action: str = Field(..., pattern="^(accept|edit|reject)$")
    draft: str = ""
    final_text: str = ""
    intent: str = ""
    rating: Optional[int] = Field(None, ge=1, le=5)


class SettingsRequest(BaseModel):
    user_id: str
    tone: Optional[str] = None
    signoff: Optional[str] = Field(None, max_length=80)
    display_name: Optional[str] = Field(None, max_length=80)


@app.get("/")
def root():
    return {"service": "Intent-Driven Email Assistant", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backends": {
            "intent": settings.intent_backend, "summarizer": settings.summarizer_backend,
            "faithfulness": settings.faithfulness_backend, "generator": settings.generator_backend,
            "llm_provider": settings.llm_provider if settings.llm_api_key else None,
        },
        "redact_pii": settings.redact_pii,
    }


@app.get("/llm-check")
def llm_check():
    """Open in a browser to see whether the LLM answers right now (no API key needed; sends no email data)."""
    import time as _t
    from . import llm
    if not llm.available():
        return {"ok": False, "reason": "No LLM_API_KEY in backend/.env, so offline mode is used"}
    t0 = _t.time()
    try:
        text = llm.complete("Reply with exactly: OK", "Health check", max_tokens=20, temperature=0)
        return {"ok": True, "provider": settings.llm_provider, "model": settings.llm_model,
                "reply": text.strip()[:40], "seconds": round(_t.time() - t0, 2)}
    except Exception as exc:
        return {"ok": False, "provider": settings.llm_provider, "model": settings.llm_model,
                "reason": str(exc)[:500], "seconds": round(_t.time() - t0, 2)}


@app.get("/llm-models")
def llm_models():
    """Ollama: lists the model names your API key can use (put one in LLM_MODEL)."""
    from . import llm
    try:
        return {"provider": settings.llm_provider, "current": settings.llm_model,
                "fallbacks": settings.llm_fallback_models, "available": llm.list_models()}
    except Exception as exc:
        return {"provider": settings.llm_provider, "error": str(exc)[:300]}


@app.post("/analyze", dependencies=[Depends(require_key)])
def analyze(req: AnalyzeRequest):
    if req.tone and req.tone not in TONES:
        raise HTTPException(400, f"tone must be one of {list(TONES)}")
    msgs = [{"from": m.from_, "date": m.date, "body": m.body} for m in req.messages]
    return orchestrator.run(req.user_id, req.subject, msgs, tone=req.tone, guidance=req.guidance,
                            force_draft=req.force_draft, has_attachment=req.has_attachment)


@app.post("/triage", dependencies=[Depends(require_key)])
def triage(req: TriageRequest):
    """Bulk inbox sorting: category, type, needs-reply and a one-line gist for each email."""
    emails = [{"id": e.id, "from": e.from_, "subject": e.subject, "body": e.body,
               "has_attachment": e.has_attachment} for e in req.emails]
    return {"results": orchestrator.triage_bulk(req.user_id, emails)}


@app.post("/triage/correct", dependencies=[Depends(require_key)])
def triage_correct(req: CorrectionRequest):
    """User says the label was wrong: remember it for this sender (or domain) and log it for evaluation."""
    if req.topic not in TOPICS:
        raise HTTPException(400, f"topic must be one of {list(TOPICS)}")
    flag = "" if req.can_delete is None else ("delete" if req.can_delete else "keep")
    rule = req.topic + (f"|{flag}" if flag else "")
    try:
        key = persona.set_sender_rule(req.user_id, req.sender, rule, req.whole_domain)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    persona.log_correction(req.user_id, key, req.predicted, req.topic, flag)
    labels = [TOPICS[req.topic]["label"]] + ([DELETE_LABEL] if req.can_delete or req.topic == "spam" else [])
    return {"sender_key": key, "topic": req.topic, "can_delete": req.can_delete, "gmail_labels": labels}


@app.get("/categories")
def categories():
    return {"topics": TOPICS, "importance": CATEGORIES, "priority_label": PRIORITY_LABEL,
            "delete_label": DELETE_LABEL, "email_types": EMAIL_TYPES}


@app.post("/feedback", dependencies=[Depends(require_key)])
def feedback(req: FeedbackRequest):
    return persona.record_feedback(req.user_id, req.request_id, req.action, req.draft,
                                   req.final_text, req.intent, req.rating,
                                   names=orchestrator.names_for(req.request_id))


@app.get("/settings/{user_id}", dependencies=[Depends(require_key)])
def get_settings(user_id: str):
    p = persona.get_persona(user_id)
    return {"tone": p.tone, "signoff": p.signoff, "display_name": p.display_name,
            "counts": {"accept": p.n_accept, "edit": p.n_edit, "reject": p.n_reject}, "tones": list(TONES)}


@app.post("/settings", dependencies=[Depends(require_key)])
def save_settings(req: SettingsRequest):
    if req.tone and req.tone not in TONES:
        raise HTTPException(400, f"tone must be one of {list(TONES)}")
    p = persona.update_settings(req.user_id, req.tone, req.signoff, req.display_name)
    return {"tone": p.tone, "signoff": p.signoff, "display_name": p.display_name}


@app.get("/stats", dependencies=[Depends(require_key)])
def get_stats(user_id: Optional[str] = None):
    """Evaluation numbers for Section 9.2: accept/edit/reject rates, faithfulness, latency."""
    return persona.stats(user_id)


@app.post("/kb/reload", dependencies=[Depends(require_key)])
def kb_reload():
    return {"passages": retrieval.reload()}
