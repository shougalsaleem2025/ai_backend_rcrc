"""Sanad's one final response surface for ALL assistant routes.

This module is deliberately deterministic. Never let a free-form model rewrite
verified employee, attendance, leave, policy or contact facts just to sound warm.

Routing, references, safety and factual grounding remain orchestrator-owned.
"""
from __future__ import annotations

import re

_AR_OPENERS = re.compile(r"\A\s*(?:(?:أكيد|تمام|حاضر|أبشر|أبشري)\s*[😊👌🌷🙂]?[ \t]*(?:\n[ \t]*)?){2,}")
_EN_OPENERS = re.compile(r"\A\s*(?:(?:Sure|Of course|Certainly|Absolutely)\s*[😊👌🌷🙂]?[ \t]*(?:\n[ \t]*)?){2,}", re.I)

# Only remove meaningless mechanical intro immediately ahead of a complete
# factual sentence. Never rewrite the factual lines themselves.
_AR_DIRECT = re.compile(r"\A\s*(?:أكيد|تمام|حاضر)\s*[😊👌🌷🙂]?[ \t]*\n(?=(?:مديرك|مدير |رصيد|عندك|ما عندك|موظفو|الإدارات|هذي|هذه|البريد|الإيميل|الجوال|التحويلة))")
_EN_DIRECT = re.compile(r"\A\s*(?:Sure|Of course|Certainly)\s*[😊👌🌷🙂]?[ \t]*\n(?=(?:Your |The |There |I found|Here are|You have|You don't))", re.I)


def polish_response(answer: str, *, language: str = "ar") -> str:
    """Clean tone consistently with zero modification to enterprise facts.

    Run once at the response boundary, regardless of the Endpoint / prompt path.
    Preserves lists, URLs, email addresses, numbers and error text.
    """
    text = str(answer or "").strip()
    if not text:
        return text
    text = text.replace("\r\n", "\n")
    # Preserve Markdown headings/links and document text. Rendering Markdown is
    # a frontend concern; deleting all '**' corrupts document drafts.
    if language == "en":
        text = _EN_OPENERS.sub("", text)
        text = _EN_DIRECT.sub("", text)
    else:
        text = _AR_OPENERS.sub("", text)
        text = _AR_DIRECT.sub("", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()
