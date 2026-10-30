"""Conditional response generation (Table 2, row 5) with context fusion (row 4).

The fused context = intent label + faithfulness-checked summary + latest
message + persona profile + optional retrieved KB passages + optional user
guidance ("say yes but propose Thursday", in the spirit of ResQ [15]).

Backends:
  - "llm":      generative decoding conditioned on the fused context (C5:
                the prompt forbids facts that are not in the context and asks
                for [square-bracket] blanks instead)
  - "template": intent-specific templates; a deterministic baseline that also
                plays the role of the Smart-Reply-style retrieval baseline
                [1][2] in the ablation study
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..config import settings
from .. import llm
from .persona import Persona

TONES = {
    "professional": "polite, clear and professional",
    "friendly": "warm and friendly but still work-appropriate",
    "brief": "very short and direct (2-3 sentences), no pleasantries beyond a greeting",
    "formal": "formal and respectful, suitable for senior colleagues or official correspondence",
}


@dataclass
class FusedContext:
    intent: str
    intent_description: str
    summary: str
    latest_message: str
    latest_sender_name: str
    subject: str
    persona: Persona
    tone: str
    guidance: str = ""
    kb: List[Tuple[str, str]] = ()

    def as_prompt(self) -> str:
        # ABLATION (Section 9.3): full | intent_only | summary_only | none
        mode = settings.ablation
        parts = [f"Subject: {self.subject}"]
        if mode in ("full", "intent_only"):
            parts.append(f"Detected intent of the latest email: {self.intent} — {self.intent_description}")
        if mode in ("full", "summary_only"):
            parts.append(f"Thread summary: {self.summary}")
        parts += [
            f"Latest email (from {self.latest_sender_name or 'the sender'}):\n\"\"\"\n{self.latest_message[:3000]}\n\"\"\"",
            f"User style profile:\n{self.persona.as_prompt()}",
        ]
        if self.kb:
            parts.append("Relevant knowledge-base passages (use only if relevant):\n" +
                         "\n".join(f"[{src}] {p}" for src, p in self.kb))
        if self.guidance:
            parts.append(f"The user's instructions for this reply (follow them): {self.guidance}")
        return "\n\n".join(parts)


@dataclass
class DraftResult:
    text: str
    method: str


def _llm_draft(ctx: FusedContext) -> str:
    tone = TONES.get(ctx.tone, ctx.tone)
    name = ctx.persona.display_name or "[your name]"
    system = (
        "You draft email replies on behalf of the user. Write only the reply body (greeting to sign-off), "
        "no subject line and no commentary. Ground every statement in the provided context. Never invent "
        "dates, numbers, commitments or facts: where the user must supply something, insert a short "
        "placeholder in square brackets, e.g. [confirm date]. Keep tokens like [PERSON_1] or [EMAIL_2] "
        "exactly as written; they are redacted names and addresses."
    )
    user = (f"{ctx.as_prompt()}\n\nWrite a reply that is {tone}. Address the intent directly. "
            f"Sign as {name}" + (f", using the sign-off \"{ctx.persona.signoff}\"" if ctx.persona.signoff else "") + ".")
    return llm.complete(system, user, max_tokens=500, temperature=0.5).strip()


_TEMPLATES = {
    "request_action": "Thanks for your email. I'll take care of this and get back to you by [date]. "
                      "{ask_back}",
    "question": "Thanks for reaching out. To answer your question: [your answer]. "
                "Let me know if anything else would help.",
    "meeting_scheduling": "Thanks for setting this up. [Day/time] works for me. "
                          "If that doesn't suit you, I'm also free [alternative slot].",
    "approval_request": "Thanks for sending this over. I've reviewed it and [approve / have a few comments]. "
                        "[Comments, if any].",
    "follow_up": "Thanks for following up, and apologies for the delay. [Status update]. "
                 "I'll have [next step] to you by [date].",
    "complaint": "Thank you for letting me know, and I'm sorry for the trouble. I'm looking into "
                 "[the issue] now and will update you by [date] with next steps.",
    "information_fyi": "Thanks for the update — noted.",
    "social_thanks": "Thank you so much for your kind words, I really appreciate it.",
    "newsletter_promotion": "",
    "other": "Thanks for your email. [Your reply].",
}


def _template_draft(ctx: FusedContext) -> str:
    key = ctx.intent if ctx.intent in _TEMPLATES else "other"
    body = _TEMPLATES[key].format(ask_back="Please let me know if there's anything specific you need from my side.")
    if ctx.tone == "brief":
        body = body.split(". ")[0].rstrip(".") + "."
    if ctx.guidance:
        body += f"\n\n[Add: {ctx.guidance}]"
    greet = {"friendly": "Hi", "brief": "Hi", "formal": "Dear"}.get(ctx.tone, "Hello")
    who = ctx.latest_sender_name or ""
    signoff = ctx.persona.signoff or ("Regards," if ctx.tone == "formal" else "Best,")
    name = ctx.persona.display_name or "[your name]"
    return f"{greet} {who},\n\n{body}\n\n{signoff}\n{name}".replace(" ,", ",")


def generate(ctx: FusedContext) -> DraftResult:
    if settings.generator_backend == "llm" and llm.available():
        try:
            return DraftResult(_llm_draft(ctx), f"llm:{settings.llm_provider}:{settings.llm_model}")
        except Exception as exc:  # degrade gracefully: the add-on must always get a draft
            return DraftResult(_template_draft(ctx), f"template (llm failed: {str(exc)[:80]})")
    return DraftResult(_template_draft(ctx), "template")
