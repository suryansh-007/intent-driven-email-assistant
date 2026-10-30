"""Minimal, dependency-free client for the LLM providers the pipeline can use.

Only `requests` is needed. One function, `complete(system, user)`, returns the
model's text so every module can swap providers by configuration alone.
"""
import json
import re
import threading
import time
import requests

from .config import settings


class LLMError(RuntimeError):
    pass


def available() -> bool:
    if settings.llm_api_key:
        return True
    # keyless only for self-hosted servers (local Ollama, LM Studio …)
    return settings.llm_provider in ("openai", "ollama") and bool(settings.llm_base_url)


_slots = None


def _slot():
    """Caps simultaneous LLM requests across all users and threads (LLM_CONCURRENCY)."""
    global _slots
    if _slots is None:
        _slots = threading.BoundedSemaphore(max(1, settings.llm_concurrency))
    return _slots


def complete(system: str, user: str, max_tokens: int = 600, temperature: float = 0.4,
             json_mode: bool = False) -> str:
    provider = settings.llm_provider.lower()
    try:
        if provider == "ollama":
            return _ollama(system, user, max_tokens, temperature, json_mode)
        if provider == "gemini":
            return _gemini(system, user, max_tokens, temperature, json_mode)
        if provider == "anthropic":
            return _anthropic(system, user, max_tokens, temperature)
        if provider == "openai":
            return _openai(system, user, max_tokens, temperature, json_mode)
    except requests.RequestException as exc:
        raise LLMError(f"{provider} request failed: {exc}") from exc
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")


def complete_json(system: str, user: str, max_tokens: int = 400) -> dict:
    """Ask for a JSON object and parse it, tolerating ```json fences."""
    text = complete(system + "\nRespond with a single JSON object and nothing else.", user,
                    max_tokens=max_tokens, temperature=0.0, json_mode=True).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"No JSON object in model output: {text[:200]}")
    return json.loads(text[start:end + 1])


# model -> time until which it is skipped (after a 429 quota error)
_COOLDOWN = {}
_RETRYABLE = (429, 500, 503, 504)


def _gemini(system, user, max_tokens, temperature, json_mode=False):
    """Tries LLM_MODEL, then each of LLM_FALLBACK_MODELS. A 503/500 ("high demand") gets one quick
    retry; a 429 (quota) puts that model on a 60 s cool-down and moves straight to the next one.
    Everything stays inside a ~20 s budget so the Gmail card (30 s limit) still gets an answer."""
    models = [settings.llm_model] + [m for m in settings.llm_fallback_models if m != settings.llm_model]
    deadline = time.time() + 20
    errors = []
    for model in models:
        if _COOLDOWN.get(model, 0) > time.time():
            errors.append(f"{model}: cooling down after quota error")
            continue
        for attempt in range(2):
            if time.time() > deadline:
                raise LLMError("Gemini: gave up after 20 s; " + "; ".join(errors))
            try:
                return _gemini_once(model, system, user, max_tokens, temperature, json_mode)
            except LLMError as exc:
                code = getattr(exc, "status", None)
                errors.append(f"{model}: {str(exc)[:120]}")
                if code == 429:
                    _COOLDOWN[model] = time.time() + 60
                    break                      # next model
                if code in _RETRYABLE and attempt == 0:
                    time.sleep(1.5)
                    continue                   # retry same model once
                if code in _RETRYABLE:
                    break                      # next model
                raise                          # bad key, bad request …: no point trying others
    raise LLMError("Gemini unavailable: " + " | ".join(errors))


def _gemini_once(model, system, user, max_tokens, temperature, json_mode):
    base = settings.llm_base_url or "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base}/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        # newer Gemini models spend part of this budget on internal "thinking", so leave headroom
        "generationConfig": {"maxOutputTokens": max(max_tokens * 4, 4096), "temperature": temperature},
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    r = requests.post(url, params={"key": settings.llm_api_key}, json=body, timeout=settings.llm_timeout)
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text
        err = LLMError(f"Gemini {r.status_code}: {msg[:200]}")
        err.status = r.status_code
        raise err
    data = r.json()
    try:
        cand = data["candidates"][0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        if not text:
            raise LLMError(f"Gemini returned no text (finishReason={cand.get('finishReason')})")
        return text
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected Gemini response: {str(data)[:300]}") from exc


def _anthropic(system, user, max_tokens, temperature):
    base = settings.llm_base_url or "https://api.anthropic.com"
    r = requests.post(
        f"{base}/v1/messages",
        headers={"x-api-key": settings.llm_api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": settings.llm_model, "max_tokens": max_tokens, "temperature": temperature,
              "system": system, "messages": [{"role": "user", "content": user}]},
        timeout=settings.llm_timeout,
    )
    if r.status_code != 200:
        raise LLMError(f"Anthropic {r.status_code}: {r.text[:300]}")
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")


def _openai(system, user, max_tokens, temperature, json_mode=False):
    """OpenAI and any compatible server: Ollama (http://localhost:11434/v1), Groq, LM Studio, vLLM …"""
    import re
    base = (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"content-type": "application/json"}
    if settings.llm_api_key:
        headers["authorization"] = f"Bearer {settings.llm_api_key}"
    body = {"model": settings.llm_model, "max_tokens": max(max_tokens, 1024), "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{base}/chat/completions", headers=headers, json=body, timeout=settings.llm_timeout)
    if r.status_code != 200:
        raise LLMError(f"OpenAI-compatible {r.status_code}: {r.text[:300]}")
    text = r.json()["choices"][0]["message"].get("content") or ""
    # reasoning models (qwen3, deepseek-r1 …) may prepend their hidden thinking
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        raise LLMError("Model returned an empty answer")
    return text


# ------------------------------------------------------------------ Ollama (cloud or local)

OLLAMA_CLOUD = "https://ollama.com"
_FALLBACK_CODES = (403, 404, 429, 500, 502, 503, 504)   # not in plan / unknown / quota / busy


def _ollama_base() -> str:
    return (settings.llm_base_url or OLLAMA_CLOUD).rstrip("/").removesuffix("/v1").removesuffix("/api")


def _ollama_headers() -> dict:
    h = {"content-type": "application/json"}
    if settings.llm_api_key:
        h["authorization"] = f"Bearer {settings.llm_api_key}"
    return h


def _ollama(system, user, max_tokens, temperature, json_mode=False):
    """Ollama native /api/chat. Tries LLM_MODEL then LLM_FALLBACK_MODELS; a model that is busy,
    rate-limited or not included in your plan is skipped for 60 s. Waits for a free request slot
    (the free cloud plan allows one request at a time) within a ~22 s budget."""
    models = [settings.llm_model] + [m for m in settings.llm_fallback_models if m != settings.llm_model]
    deadline = time.time() + 22
    errors = []
    for model in models:
        if _COOLDOWN.get(model, 0) > time.time():
            errors.append(f"{model}: cooling down")
            continue
        remaining = deadline - time.time()
        if remaining <= 0 or not _slot().acquire(timeout=remaining):
            raise LLMError("Ollama: busy with other requests; " + "; ".join(errors or ["no free slot"]))
        try:
            return _ollama_once(model, system, user, max_tokens, temperature, json_mode)
        except LLMError as exc:
            errors.append(f"{model}: {str(exc)[:140]}")
            if getattr(exc, "status", None) in _FALLBACK_CODES:
                _COOLDOWN[model] = time.time() + 60
                continue
            raise
        finally:
            _slot().release()
    raise LLMError("Ollama unavailable: " + " | ".join(errors))


def _ollama_once(model, system, user, max_tokens, temperature, json_mode, think=True):
    body = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": temperature, "num_predict": max(max_tokens * 4, 2048)},
    }
    if json_mode:
        body["format"] = "json"
    if think:
        # gpt-oss takes an effort level; other thinking models take true/false
        body["think"] = settings.llm_think if model.startswith("gpt-oss") else False
    r = requests.post(f"{_ollama_base()}/api/chat", headers=_ollama_headers(), json=body,
                      timeout=settings.llm_timeout)
    if r.status_code == 400 and think and "think" in r.text.lower():
        return _ollama_once(model, system, user, max_tokens, temperature, json_mode, think=False)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", r.text)
        except Exception:
            msg = r.text
        err = LLMError(f"Ollama {r.status_code}: {str(msg)[:200]}")
        err.status = r.status_code
        raise err
    text = (r.json().get("message") or {}).get("content", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        raise LLMError("Ollama returned an empty answer (try a larger LLM_TIMEOUT or another model)")
    return text


def list_models() -> list:
    """Models your key can use (Ollama) — handy for picking LLM_MODEL."""
    if settings.llm_provider != "ollama":
        return []
    r = requests.get(f"{_ollama_base()}/api/tags", headers=_ollama_headers(), timeout=15)
    if r.status_code != 200:
        raise LLMError(f"Ollama {r.status_code}: {r.text[:200]}")
    return sorted(m.get("name") or m.get("model") for m in r.json().get("models", []))
