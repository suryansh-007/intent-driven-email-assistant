# Intent-Driven Email Assistant

Intent-driven email categorization, abstractive summarization and conditional
response generation, delivered as a **Gmail add-on** backed by a **Python
(FastAPI) pipeline**.

Open an email in Gmail and the side panel shows:
- its **topic**: Personal, Work & Academic, Finance, Shopping & Orders, Advertisements, Social & Apps,
  Newsletters & Blogs, Accounts & Security or Spam, plus **Priority** (needs attention soon) and
  **Can delete** (safe to remove without reading) flags, with the reason
- **what the email is** and **what the sender wants from you**, plus key details (order numbers, amounts, dates), a deadline and to-dos
- **whether it needs a reply**: automated, no-reply and "do not reply" emails get no draft
- the **intent** (seed taxonomy plus open-set Action–Object discovery) and a **phishing warning** when needed
- a **thread summary** with its **faithfulness score** (multi-message threads)
- a **draft reply** conditioned on intent + summary + your learned style, which you can edit, re-tone or steer with an instruction, then insert as a Gmail reply

With no email open, **Sort my inbox** gives every conversation one topic label (`AI/Personal`, `AI/Work and Academic`, `AI/Finance`, `AI/Shopping and Orders`, `AI/Advertisements`, `AI/Social and Apps`, `AI/Newsletters and Blogs`, `AI/Accounts and Security`, `AI/Spam`), plus
`AI/Priority` or `AI/Can delete` where they apply. **Clean up** moves a whole group (Can delete,
Advertisements, Social and Apps, Spam) to Trash after a preview. "Wrong label?" corrections
become per-sender rules, and are logged as triage-accuracy data.

Every accept, edit or reject is logged, which gives the human-evaluation numbers
for Section 9.2 directly from real use.

## Repository structure

```text
intent-driven-email-assistant/
├── backend/                     Python pipeline + REST API
│   ├── app/
│   │   ├── main.py              FastAPI endpoints (/analyze, /feedback, /settings, /stats)
│   │   ├── config.py            all settings via environment variables
│   │   ├── llm.py               Gemini / Anthropic / OpenAI-compatible client
│   │   └── pipeline/
│   │       ├── preprocess.py    cleaning, quote/signature stripping, PII redaction (C2)
│   │       ├── intent.py        intent classifier + open-intent discovery + phishing screen
│   │       ├── triage.py        category, email type, needs-reply, gist, key details (heuristic + batched LLM)
│   │       ├── summarize.py     thread summarizer + faithfulness check + extractive fallback
│   │       ├── retrieval.py     knowledge-base retrieval (RAG)
│   │       ├── generate.py      context fusion + conditional reply generation + ablations
│   │       ├── persona.py       persona store + evaluation log
│   │       └── orchestrator.py  runs the stages and times each one
│   ├── knowledge_base/          FAQ / policy files used to ground replies
│   ├── tests/                   pytest suite (runs offline)
│   ├── requirements.txt         core service
│   ├── requirements-ml.txt      optional local HF models
│   ├── .env.example
│   └── Dockerfile
├── gmail-addon/                 Google Apps Script add-on (see its README for setup)
│   ├── appsscript.json
│   ├── Api.gs                   backend calls
│   ├── Sort.gs                  inbox sorting, labels, bulk delete, hourly auto-sort
│   └── Code.gs                  email card, reply drafting, settings
├── src/load_dataset.py          dataset loader (offline experiments)
├── data/
└── notebooks/
```

## How the code maps to the proposed architecture (Table 2)

| Module (report) | Code | Backends (set in `.env`) |
|---|---|---|
| Preprocessing + PII redaction | `preprocess.py` | regex + participant names; placeholders are restored after generation |
| Email triage (category, needs-reply, gist) | `triage.py` | `heuristic` baseline · `llm` (8 emails per call, merged with heuristic; phishing signals and user sender rules always win) |
| Intent classification | `intent.py` | `keyword` baseline · `zeroshot` (BART-MNLI) · `llm`; open intent below the confidence threshold [4][10] |
| Thread summarization + faithfulness | `summarize.py` | `extractive` · `bart` (dialogue-tuned BART) · `llm`; faithfulness `overlap` or `nli`; falls back to extractive if the check fails [3][8] |
| Context fusion | `generate.py: FusedContext` | intent + summary + persona + KB + user instruction |
| Response generation | `generate.py` | `template` (retrieval-style baseline [1][2]) · `llm` |
| Persona store | `persona.py` | SQLite; learns tone, sign-off and length from accept/edit feedback [6][11][14] |
| Orchestration (RAG) | `retrieval.py`, `orchestrator.py` | TF-IDF over `knowledge_base/` [12] |
| Human review | Gmail add-on | accept / edit / reject capture, tone and instruction steering [15] |

## Quick start (backend)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add an LLM key, or leave the LLM backends unset for offline mode
uvicorn app.main:app --reload --port 8000
```

- Interactive API docs: <http://localhost:8000/docs>
- Tests: `pytest -q`. They run offline, with no model downloads or API keys.

**Offline mode** (no key needed): set `INTENT_BACKEND=keyword`, `SUMMARIZER_BACKEND=extractive`
and `GENERATOR_BACKEND=template`. This is useful for development and works as the baseline row of the ablation table.

**LLM mode** (best quality): the default is **Ollama Cloud**. Create a key at
<https://ollama.com/settings/keys>, then set `LLM_PROVIDER=ollama` and `LLM_API_KEY=…` in `.env`.
Model `gpt-oss:20b`, with `gemma4:31b` and `nemotron-3-nano:30b` as fallbacks. Check it with
`/llm-check`; `/llm-models` lists the models your key can use. Local Ollama and Gemini also work
(see `.env.example`).

**Local research models**: `pip install -r requirements-ml.txt`, then set
`INTENT_BACKEND=zeroshot`, `SUMMARIZER_BACKEND=bart` and `FAITHFULNESS_BACKEND=nli`. To use
your own EmailSum fine-tune, point `SUMMARIZER_MODEL` at its checkpoint. On CPU these models
can be slow (several seconds per stage), so check `latency_ms` against Gmail's 30-second card limit.

Then set up the add-on: see [`gmail-addon/README.md`](gmail-addon/README.md).

## Evaluation hooks (Section 9)

- `POST /triage` labels up to 40 emails per call; `POST /triage/correct` records a user correction.
- `GET /stats` → `triage` gives emails triaged per category, the number of user corrections, an
  agreement rate (1 − corrections/triaged) and the confusion pairs (predicted → corrected).
- `GET /stats` also returns runs, accept rate, **accept-or-minor-edit rate** (edit ratio ≤ 0.2),
  reject rate, mean edit ratio, mean faithfulness, **mean latency per stage** and a per-intent breakdown.
  Add `?user_id=…` for a single participant.
- `ABLATION=full | intent_only | summary_only | none` controls what conditions the LLM
  generator: the full-vs-ablated comparison in Section 9.3.
- `GENERATOR_BACKEND=template` is the fixed-candidate baseline (Smart-Reply style).
- Each `/analyze` response includes `request_id`, the method used by every stage and the per-stage latency.

## Privacy (constraint C2)

- The add-on can read only the email that is open (`gmail.addons.current.message.readonly`).
- Users are identified by a salted SHA-256 hash of their address.
- Emails, phone numbers, Aadhaar/PAN/card numbers, IFSC codes, URLs and participant names are
  replaced with placeholders **before** any text reaches a model or external API. Real values
  are restored only in the draft shown to the user.
- Replies saved as style examples are redacted the same way.
- Email bodies are not stored. The SQLite file holds only persona settings, redacted
  example replies and evaluation metrics.

## Datasets

CSpace and Avocado (intent), EmailSum (summarization) and Amazon Product-QA (persona) are
used for the offline module benchmarks. Note: **Avocado is distributed by the LDC under
licence (LDC2015T03)**. If your institution has no LDC membership, the public Enron email
corpus can replace it for intent experiments.
