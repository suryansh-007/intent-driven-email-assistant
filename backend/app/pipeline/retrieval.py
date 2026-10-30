"""Knowledge-base retrieval for the orchestration step (Table 2, row 7).

Drop .md / .txt files (policies, FAQs, office hours, course rules …) into
backend/knowledge_base/. Files are split into paragraphs and the top matches
for the email are passed to the generator as grounding context, in the spirit
of the RAG policy-QA agent in [12]. Plain TF-IDF keeps it dependency-free;
swap in an embedding index later if the KB grows.
"""
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

KB_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base"

_STOP = set("a an the and or of to in on for is are be it this that with as at by from you your we our i".split())


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2]


@lru_cache(maxsize=1)
def _index() -> Tuple[List[Tuple[str, str]], Counter]:
    passages = []
    if KB_DIR.exists():
        for f in sorted(KB_DIR.glob("*")):
            if f.suffix.lower() in (".md", ".txt") and f.name.lower() != "readme.md":
                for para in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")):
                    if len(_tokens(para)) >= 4:
                        passages.append((f.name, para.strip()))
    df = Counter(w for _, p in passages for w in set(_tokens(p)))
    return passages, df


def reload() -> int:
    _index.cache_clear()
    return len(_index()[0])


def retrieve(query: str, k: int = 2, min_score: float = 1.0) -> List[Tuple[str, str]]:
    passages, df = _index()
    if not passages:
        return []
    q = Counter(_tokens(query))
    n = len(passages)
    scored = []
    for src, p in passages:
        toks = Counter(_tokens(p))
        s = sum(q[w] * toks[w] * math.log(1 + n / df[w]) for w in q if w in toks)
        s /= math.sqrt(sum(toks.values()) + 1)
        if s >= min_score:
            scored.append((s, src, p))
    scored.sort(reverse=True)
    return [(src, p) for _, src, p in scored[:k]]
