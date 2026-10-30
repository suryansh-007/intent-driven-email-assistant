"""Preprocessing module (Table 2, row 1).

- cleans each message body (quoted history, signatures, disclaimers, whitespace)
- segments a thread into ordered turns
- redacts PII with reversible placeholders so that no raw PII reaches any
  model or external API (constraint C2); the placeholders are restored in the
  final draft so the user still sees real names/numbers.
"""
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List

# --- cleaning -----------------------------------------------------------------

_QUOTE_HEADER = re.compile(
    r"^(On .{5,200}wrote:|-{2,}\s*Original Message\s*-{2,}|From:\s.+\nSent:\s.+|_{5,})",
    re.IGNORECASE | re.MULTILINE,
)
_SIGNATURE = re.compile(r"^(--\s*$|Sent from my \w+|Get Outlook for)", re.IGNORECASE | re.MULTILINE)
_DISCLAIMER = re.compile(r"(This e-?mail (and any attachments )?(is|are) confidential.*)", re.IGNORECASE | re.DOTALL)


class _HTMLText(HTMLParser):
    """HTML → readable text: drops <head>/<style>/<script> and hidden preheaders, keeps line breaks."""
    _SKIP = {"head", "style", "script", "title", "noscript", "xml", "svg"}
    _BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "td"}
    _VOID = {"br", "img", "meta", "link", "hr", "input", "col", "area", "base", "wbr", "source"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._stack: List[bool] = []   # True = this element hides its content

    def handle_starttag(self, tag, attrs):
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        hidden = tag in self._SKIP or "display:none" in style
        if tag in self._BLOCK:
            self.parts.append("\n")
        if tag not in self._VOID:
            self._stack.append(hidden)

    def handle_endtag(self, tag):
        if tag not in self._VOID and self._stack:
            self._stack.pop()
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not any(self._stack):
            self.parts.append(data)


def looks_like_html(text: str) -> bool:
    head = text[:2000].lower()
    return "<html" in head or "<!doctype" in head or len(re.findall(r"</?(?:div|table|td|p|span|br)\b", head)) >= 5


def html_to_text(html: str) -> str:
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)   # incl. Outlook [if mso] blocks
    parser = _HTMLText()
    try:
        parser.feed(html)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("\xa0", " ").replace("‌", "").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


def clean_body(body: str) -> str:
    text = body.replace("\r\n", "\n")
    if looks_like_html(text):
        text = html_to_text(text)
    m = _QUOTE_HEADER.search(text)
    if m:
        text = text[: m.start()]
    text = "\n".join(line for line in text.split("\n") if not line.lstrip().startswith(">"))
    m = _SIGNATURE.search(text)
    if m:
        text = text[: m.start()]
    text = _DISCLAIMER.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- PII redaction ------------------------------------------------------------

_PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("URL", re.compile(r"\bhttps?://\S+", re.IGNORECASE)),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{4,5}(?!\w)")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
]


@dataclass
class Redactor:
    """Replaces PII with stable placeholders like [EMAIL_1] and can undo it."""
    mapping: Dict[str, str] = field(default_factory=dict)  # placeholder -> original
    _reverse: Dict[str, str] = field(default_factory=dict)  # original -> placeholder
    _counts: Dict[str, int] = field(default_factory=dict)

    def _placeholder(self, kind: str, value: str) -> str:
        if value in self._reverse:
            return self._reverse[value]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        ph = f"[{kind}_{self._counts[kind]}]"
        self.mapping[ph] = value
        self._reverse[value] = ph
        return ph

    def redact(self, text: str, names: List[str] = ()) -> str:
        for kind, pattern in _PII_PATTERNS:
            text = pattern.sub(lambda m, k=kind: self._placeholder(k, m.group(0)), text)
        # Names of thread participants (from headers) are the most common PII.
        for name in sorted({n for n in names if n and len(n) > 2}, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(name)}\b", lambda m: self._placeholder("PERSON", m.group(0)), text)
        return text

    def restore(self, text: str) -> str:
        for ph, original in self.mapping.items():
            text = text.replace(ph, original)
        return text


_HONORIFICS = {"prof", "professor", "dr", "mr", "mrs", "ms", "miss", "sir", "madam", "shri", "smt", "team"}


def participant_names(raw_from_fields: List[str]) -> List[str]:
    """'Priya Sharma <priya@x.com>' -> ['Priya Sharma', 'Priya', 'Sharma']"""
    names = []
    for f in raw_from_fields:
        display = re.sub(r"<[^>]*>", "", f or "").strip().strip('"')
        if not display or "@" in display:
            continue
        names.append(display)
        names.extend(p.strip(".,") for p in display.split()
                     if len(p.strip(".,")) > 2 and p.strip(".,").lower() not in _HONORIFICS)
    return names


def first_name(raw_from: str) -> str:
    display = re.sub(r"<[^>]*>", "", raw_from or "").strip().strip('"')
    if not display or "@" in display:
        local = re.sub(r".*<|>.*", "", raw_from or "").split("@")[0]
        return re.split(r"[._\d]", local)[0].capitalize() if local else ""
    parts = display.split()
    if parts[0].strip(".,").lower() in _HONORIFICS and len(parts) > 1:
        return f"{parts[0]} {parts[-1]}"          # "Prof. M. Wagh" -> "Prof. Wagh"
    return parts[0]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]
