"""Run from backend/:  pytest -q
Uses the lightweight fallback backends, so no GPU, model download or API key is needed."""
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["BACKEND_API_KEY"] = "test-key"
# Force offline backends. Explicit values win over backend/.env (which only fills unset variables).
os.environ.update({"INTENT_BACKEND": "keyword", "SUMMARIZER_BACKEND": "extractive",
                   "FAITHFULNESS_BACKEND": "overlap", "GENERATOR_BACKEND": "template",
                   "LLM_API_KEY": "", "ABLATION": "full", "REDACT_PII": "true",
                   "TRIAGE_BACKEND": "llm"})

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.pipeline.preprocess import Redactor, clean_body  # noqa: E402
from app.pipeline import intent  # noqa: E402

client = TestClient(app)
H = {"X-API-Key": "test-key"}

MEETING_THREAD = {
    "user_id": "u_test",
    "subject": "DA2 project review",
    "messages": [
        {"from": "Rohan Mehta <rohan.mehta@college.edu>", "date": "Mon 10:02",
         "body": "Dear Professor, our team has finished the intent classifier and the summarizer. "
                 "We would like to show you the end-to-end demo before the DA2 deadline."},
        {"from": "Prof. M. Wagh <mwagh@college.edu>", "date": "Mon 11:30",
         "body": "Good progress. Please make sure the faithfulness check is working.\n\n"
                 "On Mon, Rohan Mehta wrote:\n> Dear Professor, our team..."},
        {"from": "Rohan Mehta <rohan.mehta@college.edu>", "date": "Tue 09:15",
         "body": "Yes, it is working now. Could we schedule a meeting on Thursday afternoon? "
                 "Are you available at 3 pm? You can reach me on +91 98765 43210.\n\nThanks,\nRohan"},
    ],
}


def test_auth_required():
    assert client.post("/analyze", json=MEETING_THREAD).status_code == 401


def test_redaction_roundtrip():
    r = Redactor()
    text = "Mail priya.s@x.com or call +91 98765 43210, PAN ABCDE1234F. Priya will join."
    red = r.redact(text, ["Priya"])
    assert "priya.s@x.com" not in red and "98765" not in red and "ABCDE1234F" not in red
    assert "Priya" not in red
    assert r.restore(red) == text


def test_clean_body_strips_quotes_and_signature():
    body = "Sounds good.\n\nSent from my iPhone\n\nOn Mon, A wrote:\n> old text"
    assert clean_body(body) == "Sounds good."


def test_meeting_thread_end_to_end():
    r = client.post("/analyze", json=MEETING_THREAD, headers=H)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["intent"]["label"] == "meeting_scheduling"
    assert d["summary"] is not None and d["summary"]["faithfulness"] >= 0.5
    assert d["draft"] and "Rohan" in d["draft"]["text"]       # name restored after redaction
    assert "[PERSON" not in d["draft"]["text"]
    assert d["redactions"] >= 2                                # name + phone number
    assert set(d["latency_ms"]) >= {"preprocess", "intent", "summarize", "generate", "total"}


def test_phishing_is_flagged_and_not_drafted():
    req = {"user_id": "u_test", "subject": "Urgent action required: account locked",
           "messages": [{"from": "SBI Security <alert@sbi-secure-verify.ru>",
                         "body": "Your account has been locked. Verify your account within 24 hours "
                                 "or your account will be suspended: http://192.168.4.7/sbi/login"}]}
    d = client.post("/analyze", json=req, headers=H).json()
    assert d["phishing"]["suspect"] is True
    assert d["draft"] is None and d["draft_skipped"] == "phishing_suspect"


def test_open_intent_discovery():
    res = intent.classify("Subscription", "Hey, we need to renew the software licence before it lapses.")
    assert res.is_open and res.label.startswith("open:renew")


def test_feedback_updates_persona_and_stats():
    d = client.post("/analyze", json=MEETING_THREAD, headers=H).json()
    final = d["draft"]["text"].replace("[Day/time]", "Thursday 3 pm") + "\n\nWarm regards,\nM. Wagh"
    fb = client.post("/feedback", headers=H, json={
        "user_id": "u_test", "request_id": d["request_id"], "action": "edit",
        "draft": d["draft"]["text"], "final_text": final, "intent": d["intent"]["label"]}).json()
    assert fb["action"] == "edit" and fb["n_edit"] == 1
    s = client.get("/settings/u_test", headers=H).json()
    assert s["signoff"] == "Warm regards,"
    st = client.get("/stats", headers=H).json()
    assert st["with_feedback"] == 1 and st["runs"] >= 2


def test_settings_and_tone():
    client.post("/settings", headers=H, json={"user_id": "u2", "tone": "formal", "display_name": "Dr. Wagh"})
    d = client.post("/analyze", headers=H, json={**MEETING_THREAD, "user_id": "u2"}).json()
    assert d["tone"] == "formal" and d["draft"]["text"].startswith("Dear") and "Dr. Wagh" in d["draft"]["text"]


def test_kb_retrieval_grounds_extension_request():
    req = {"user_id": "u3", "subject": "Assignment extension",
           "messages": [{"from": "Ananya <ananya@college.edu>",
                         "body": "Could I please get an extension on the assignment deadline? I was unwell."}]}
    d = client.post("/analyze", headers=H, json=req).json()
    assert "sample_faq.md" in d["kb_sources"]


def test_llm_generator_path_with_stub(monkeypatch):
    """The LLM sees only redacted text, and placeholders are restored in the output."""
    from app import llm
    from app.config import settings
    seen = {}

    def fake_complete(system, user, **kw):
        seen["prompt"] = user
        return "Hi [PERSON_1],\n\nThursday at 3 pm works. I'll call you on [PHONE_1].\n\nBest,\nM. Wagh"

    monkeypatch.setattr(settings, "generator_backend", "llm")
    monkeypatch.setattr(settings, "llm_api_key", "x")
    monkeypatch.setattr(llm, "complete", fake_complete)
    d = client.post("/analyze", json=MEETING_THREAD, headers=H).json()
    assert "98765" not in seen["prompt"] and "Rohan" not in seen["prompt"]   # C2: no raw PII sent out
    assert "Hi Rohan" in d["draft"]["text"] and "+91 98765 43210" in d["draft"]["text"]
    assert d["draft"]["method"].startswith("llm")


def test_llm_failure_falls_back_to_template(monkeypatch):
    from app import llm
    from app.config import settings

    def boom(*a, **kw):
        raise llm.LLMError("quota exceeded")

    monkeypatch.setattr(settings, "generator_backend", "llm")
    monkeypatch.setattr(settings, "llm_api_key", "x")
    monkeypatch.setattr(llm, "complete", boom)
    d = client.post("/analyze", json=MEETING_THREAD, headers=H).json()
    assert d["draft"]["method"].startswith("template")


def test_ablation_removes_conditioning(monkeypatch):
    from app import llm
    from app.config import settings
    prompts = []
    monkeypatch.setattr(settings, "generator_backend", "llm")
    monkeypatch.setattr(settings, "llm_api_key", "x")
    monkeypatch.setattr(llm, "complete", lambda s, u, **kw: prompts.append(u) or "Hi,\n\nOk.\n\nBest")
    for mode in ("full", "none"):
        monkeypatch.setattr(settings, "ablation", mode)
        client.post("/analyze", json=MEETING_THREAD, headers=H)
    gen = [p for p in prompts if "User style profile" in p]     # reply-generation prompts only
    assert "Detected intent" in gen[0] and "Thread summary" in gen[0]
    assert "Detected intent" not in gen[1] and "Thread summary" not in gen[1]


def test_html_newsletter_is_converted_to_text():
    html = ('<!DOCTYPE html><html><head><style>.a{color:red}</style></head><body>'
            '<span style="display:none">preheader</span><table><tr><td><p>Big sale on courses, 50% off this week.</p>'
            '<p>Register for our webinar.</p><a href="#">Unsubscribe</a></td></tr></table></body></html>')
    d = client.post("/analyze", headers=H, json={"user_id": "u4", "subject": "Autumn offer",
                    "messages": [{"from": "Promo <news@shop.com>", "body": html}]}).json()
    assert d["intent"]["label"] == "newsletter_promotion"
    assert d["draft_skipped"] == "no_reply_needed"
    assert "<" not in (d["summary"] or {}).get("text", "")
