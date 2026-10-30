"""Gemini retry / fallback behaviour with a fake HTTP layer."""
import pytest
import tests.test_pipeline  # noqa: F401  (env setup)
from app import llm
from app.config import settings


class Resp:
    def __init__(self, code, text="ok"):
        self.status_code, self._text = code, text
        self.text = text

    def json(self):
        if self.status_code == 200:
            return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}
        return {"error": {"message": f"status {self.status_code}"}}


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "llm_api_key", "k")
    monkeypatch.setattr(settings, "llm_model", "lite")
    monkeypatch.setattr(settings, "llm_fallback_models", ["flash", "preview"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm._COOLDOWN.clear()
    calls = []

    def install(script):
        def post(url, **kw):
            model = url.split("/models/")[1].split(":")[0]
            calls.append(model)
            return script(model, len(calls))
        monkeypatch.setattr(llm.requests, "post", post)
        return calls
    return install


def test_503_retries_same_model_then_succeeds(gemini):
    calls = gemini(lambda m, n: Resp(503) if n == 1 else Resp(200, "hello"))
    assert llm.complete("s", "u") == "hello" and calls == ["lite", "lite"]


def test_429_falls_back_to_next_model_and_cools_down(gemini):
    calls = gemini(lambda m, n: Resp(429) if m == "lite" else Resp(200, f"from {m}"))
    assert llm.complete("s", "u") == "from flash"
    assert llm.complete("s", "u") == "from flash"          # lite skipped while cooling down
    assert calls == ["lite", "flash", "flash"]


def test_bad_key_is_not_retried(gemini):
    calls = gemini(lambda m, n: Resp(400))
    with pytest.raises(llm.LLMError):
        llm.complete("s", "u")
    assert calls == ["lite"]


def test_all_overloaded_raises_with_reasons(gemini):
    calls = gemini(lambda m, n: Resp(503))
    with pytest.raises(llm.LLMError, match="unavailable"):
        llm.complete("s", "u")
    assert calls == ["lite", "lite", "flash", "flash", "preview", "preview"]


def test_ollama_openai_compatible(monkeypatch):
    sent = {}

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": '<think>hmm</think>\n{"results": []}'}}]}

    def post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, body=json)
        return R()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_model", "qwen2.5:7b")
    monkeypatch.setattr(llm.requests, "post", post)
    assert llm.available()
    assert llm.complete_json("s", "u") == {"results": []}
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in sent["headers"] and sent["body"]["response_format"] == {"type": "json_object"}


@pytest.fixture
def ollama(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_api_key", "olk")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "llm_model", "gpt-oss:20b")
    monkeypatch.setattr(settings, "llm_fallback_models", ["gemma4:31b"])
    llm._COOLDOWN.clear()
    sent = []

    class R:
        def __init__(self, code, payload):
            self.status_code, self._p, self.text = code, payload, str(payload)

        def json(self):
            return self._p

    def install(script):
        def post(url, headers=None, json=None, timeout=None):
            sent.append({"url": url, "headers": headers, "body": json})
            code, payload = script(json["model"], json)
            return R(code, payload)
        monkeypatch.setattr(llm.requests, "post", post)
        return sent
    return install


def test_ollama_cloud_request_shape(ollama):
    sent = ollama(lambda m, b: (200, {"message": {"content": '{"results": []}', "thinking": "…"}}))
    assert llm.complete_json("sys", "user") == {"results": []}
    req = sent[0]
    assert req["url"] == "https://ollama.com/api/chat"
    assert req["headers"]["authorization"] == "Bearer olk"
    assert req["body"]["format"] == "json" and req["body"]["think"] == "low" and req["body"]["stream"] is False


def test_ollama_falls_back_when_model_not_in_plan(ollama):
    sent = ollama(lambda m, b: (403, {"error": "this model requires a subscription"}) if m == "gpt-oss:20b"
                  else (200, {"message": {"content": "hi from gemma"}}))
    assert llm.complete("s", "u") == "hi from gemma"
    assert [r["body"]["model"] for r in sent] == ["gpt-oss:20b", "gemma4:31b"]
    assert sent[1]["body"]["think"] is False


def test_ollama_retries_without_think_param(ollama):
    def script(m, b):
        if "think" in b:
            return 400, {"error": "model does not support thinking"}
        return 200, {"message": {"content": "ok"}}
    sent = ollama(script)
    assert llm.complete("s", "u") == "ok" and "think" not in sent[-1]["body"]


def test_ollama_bad_key_is_reported(ollama):
    ollama(lambda m, b: (401, {"error": "unauthorized"}))
    with pytest.raises(llm.LLMError, match="401"):
        llm.complete("s", "u")
