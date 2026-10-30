"""Thread summarization module (Table 2, row 3) + faithfulness check.

Summarizer backends:
  - "bart":       HF seq2seq model (default philschmid/bart-large-cnn-samsum,
                  a BART model fine-tuned on dialogue summarization; swap in
                  your own EmailSum fine-tune via SUMMARIZER_MODEL)
  - "llm":        prompted LLM, asked to keep who-asked-whom-for-what (the
                  sender-intent / role failure reported by EmailSum [3])
  - "extractive": centroid-style sentence ranking; no model download needed

Faithfulness backends ([3][8] flag ROUGE as unreliable, so we score the
summary against the source instead of a reference):
  - "nli":     each summary sentence must be entailed by some source chunk
  - "overlap": share of summary content words found in the source
If the abstractive summary scores below FAITHFULNESS_THRESHOLD, the module
falls back to the extractive summary, which is faithful by construction.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from ..config import settings
from .. import llm
from .preprocess import split_sentences

_STOP = set("""a an the and or but if of to in on at for from by with about as is are was were be been being
it its this that these those i you he she we they me him her us them my your our their do does did have has had
will would can could should may might shall not no yes so than then there here what which who whom when where why how
please thanks thank hi hello dear regards best just also very really let know get got""".split())


@dataclass
class Turn:
    sender: str
    date: str
    text: str


@dataclass
class SummaryResult:
    summary: str
    faithfulness: float
    method: str
    fell_back: bool = False


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9\[\]_]+", text.lower()) if w not in _STOP and len(w) > 2]


def _format_thread(turns: List[Turn]) -> str:
    return "\n\n".join(f"[{i + 1}] {t.sender} ({t.date}):\n{t.text}" for i, t in enumerate(turns))


# --- extractive -----------------------------------------------------------------

def extractive_summary(turns: List[Turn], max_sentences: int = 3) -> str:
    sents: List[Tuple[int, str, str]] = []  # (turn index, sender, sentence)
    for i, t in enumerate(turns):
        for s in split_sentences(t.text):
            if len(_content_words(s)) >= 3:
                sents.append((i, t.sender, s))
    if not sents:
        return " ".join(t.text for t in turns)[:300]
    df = Counter(w for _, _, s in sents for w in set(_content_words(s)))
    n = len(sents)

    def score(idx: int) -> float:
        i, _, s = sents[idx]
        words = _content_words(s)
        tfidf = sum(math.log(1 + n / df[w]) for w in words) / (len(words) + 1)
        recency = 1 + 0.5 * i / max(1, len(turns) - 1)   # latest turns matter more for replying
        ask = 1.3 if "?" in s or re.search(r"\b(please|could you|can you|need)\b", s, re.I) else 1.0
        return tfidf * recency * ask

    best = sorted(range(n), key=score, reverse=True)[:max_sentences]
    out = []
    for idx in sorted(best):
        i, sender, s = sents[idx]
        out.append(f"{sender}: {s}" if len(turns) > 1 else s)
    return " ".join(out)


# --- abstractive ----------------------------------------------------------------

@lru_cache(maxsize=1)
def _bart_pipe(model_name: str):
    from transformers import pipeline
    return pipeline("summarization", model=model_name)


def _bart_summary(turns: List[Turn]) -> str:
    pipe = _bart_pipe(settings.summarizer_model)
    dialogue = "\n".join(f"{t.sender}: {t.text}" for t in turns)
    out = pipe(dialogue[:3500], max_length=120, min_length=25, do_sample=False, truncation=True)
    return out[0]["summary_text"].strip()


def _llm_summary(turns: List[Turn]) -> str:
    system = ("You summarize email threads for the person who must reply next. Be faithful: state only "
              "facts present in the thread. Keep placeholders like [EMAIL_1] or [PERSON_2] exactly as written.")
    user = (f"Thread (oldest first):\n{_format_thread(turns)}\n\n"
            "Write 2-4 sentences: who wants what from whom, what has been agreed or decided, "
            "and what is still open (deadlines, questions awaiting an answer).")
    return llm.complete(system, user, max_tokens=250, temperature=0.2).strip()


# --- faithfulness -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _nli(model_name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    ent_idx = next(i for i, l in model.config.id2label.items() if l.lower().startswith("entail"))
    return tok, model, ent_idx


def _nli_faithfulness(summary: str, source: str) -> float:
    import torch
    tok, model, ent_idx = _nli(settings.faithfulness_model)
    src_sents = split_sentences(source)
    chunks = [" ".join(src_sents[i:i + 6]) for i in range(0, len(src_sents), 4)] or [source]
    scores = []
    for hyp in split_sentences(summary):
        best = 0.0
        for premise in chunks:
            enc = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                probs = model(**enc).logits.softmax(-1)[0]
            best = max(best, float(probs[ent_idx]))
        scores.append(best)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _overlap_faithfulness(summary: str, source: str) -> float:
    src = set(_content_words(source))
    words = [w for w in _content_words(summary)]
    if not words:
        return 0.0
    return round(sum(w in src for w in words) / len(words), 3)


def faithfulness(summary: str, source: str) -> float:
    if settings.faithfulness_backend == "nli":
        try:
            return _nli_faithfulness(summary, source)
        except Exception:
            pass
    return _overlap_faithfulness(summary, source)


# --- entry point ----------------------------------------------------------------

def summarize(turns: List[Turn]) -> SummaryResult:
    source = "\n".join(t.text for t in turns)
    backend = settings.summarizer_backend
    abstractive, method = None, "extractive"
    try:
        if backend == "bart":
            abstractive, method = _bart_summary(turns), f"bart:{settings.summarizer_model}"
        elif backend == "llm" and llm.available():
            abstractive, method = _llm_summary(turns), f"llm:{settings.llm_provider}"
    except Exception:
        abstractive = None

    if abstractive:
        score = faithfulness(abstractive, source)
        if score >= settings.faithfulness_threshold:
            return SummaryResult(abstractive, score, method)
        fallback = extractive_summary(turns)
        return SummaryResult(fallback, faithfulness(fallback, source), "extractive", fell_back=True)

    ext = extractive_summary(turns)
    return SummaryResult(ext, faithfulness(ext, source), "extractive")
