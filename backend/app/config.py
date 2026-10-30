"""Runtime configuration, read from environment variables (or a .env file).

Every heavy component has a lightweight fallback so the whole pipeline runs
on a laptop with no GPU and no API key — useful for development and for the
add-on demo. Switch components to the research-grade models by changing the
*_BACKEND variables (see .env.example).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


_DEFAULT_MODEL = {"gemini": "gemini-3.1-flash-lite", "ollama": "gpt-oss:20b", "anthropic": "claude-haiku-5-5",
                  "openai": "gpt-4o-mini"}
_DEFAULT_FALLBACKS = {"gemini": "gemini-flash-latest,gemini-3-flash-preview",
                      "ollama": "gemma4:31b,nemotron-3-nano:30b"}


@dataclass
class Settings:
    # Shared secret the Gmail add-on sends in the X-API-Key header.
    api_key: str = field(default_factory=lambda: _env("BACKEND_API_KEY", "change-me"))

    # intent: "zeroshot" (HF NLI model) | "llm" | "keyword"
    intent_backend: str = field(default_factory=lambda: _env("INTENT_BACKEND", "keyword"))
    intent_model: str = field(default_factory=lambda: _env("INTENT_MODEL", "facebook/bart-large-mnli"))
    # below this confidence the label is treated as unknown and open-intent discovery kicks in
    open_intent_threshold: float = field(default_factory=lambda: float(_env("OPEN_INTENT_THRESHOLD", "0.35")))

    # summarizer: "bart" (HF seq2seq) | "llm" | "extractive"
    summarizer_backend: str = field(default_factory=lambda: _env("SUMMARIZER_BACKEND", "extractive"))
    summarizer_model: str = field(default_factory=lambda: _env("SUMMARIZER_MODEL", "philschmid/bart-large-cnn-samsum"))

    # faithfulness: "nli" (HF NLI model) | "overlap"
    faithfulness_backend: str = field(default_factory=lambda: _env("FAITHFULNESS_BACKEND", "overlap"))
    faithfulness_model: str = field(default_factory=lambda: _env("FAITHFULNESS_MODEL", "facebook/bart-large-mnli"))
    faithfulness_threshold: float = field(default_factory=lambda: float(_env("FAITHFULNESS_THRESHOLD", "0.5")))

    # triage (category / needs-reply / gist): "llm" (falls back to heuristic) | "heuristic"
    triage_backend: str = field(default_factory=lambda: _env("TRIAGE_BACKEND", "llm"))

    # generator: "llm" | "template"
    generator_backend: str = field(default_factory=lambda: _env("GENERATOR_BACKEND", "template"))

    # LLM provider used wherever a backend is set to "llm":
    #   "ollama"    Ollama Cloud (https://ollama.com, API key) or a local Ollama (LLM_BASE_URL)
    #   "gemini" | "anthropic"
    #   "openai"    OpenAI or any OpenAI-compatible server (Groq, LM Studio, vLLM …)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", _DEFAULT_MODEL.get(
        _env("LLM_PROVIDER", "gemini").lower(), "gemini-3.1-flash-lite")))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    # Gemini / Ollama: tried in order when the main model is busy, out of quota, or not in your plan
    llm_fallback_models: list = field(default_factory=lambda: [
        m.strip() for m in _env("LLM_FALLBACK_MODELS", _DEFAULT_FALLBACKS.get(
            _env("LLM_PROVIDER", "gemini").lower(), "")).split(",") if m.strip()])
    # simultaneous LLM requests across all users (Ollama Cloud free plan allows 1)
    llm_concurrency: int = field(default_factory=lambda: int(_env(
        "LLM_CONCURRENCY", "1" if _env("LLM_PROVIDER", "gemini").lower() == "ollama" else "3")))
    # reasoning effort for thinking models such as gpt-oss: low keeps replies fast
    llm_think: str = field(default_factory=lambda: _env("LLM_THINK", "low"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_timeout: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT", "20")))

    # bulk "Sort my inbox": llm | heuristic (heuristic saves quota on big inboxes)
    bulk_triage_backend: str = field(default_factory=lambda: _env("BULK_TRIAGE_BACKEND", "llm"))
    triage_batch: int = field(default_factory=lambda: int(_env("TRIAGE_BATCH", "10")))

    # Conditioning ablation for the LLM generator: full | intent_only | summary_only | none
    ablation: str = field(default_factory=lambda: _env("ABLATION", "full"))

    # PII redaction before any text leaves the backend (constraint C2)
    redact_pii: bool = field(default_factory=lambda: _env("REDACT_PII", "true").lower() == "true")

    # SQLite file for the persona store and the evaluation log
    db_path: str = field(default_factory=lambda: _env("DB_PATH", str(Path(__file__).resolve().parent.parent / "assistant.db")))


settings = Settings()
