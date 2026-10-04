"""core/general_guard.py

Guard rails for the GENERAL (no-tools) conversation mode.

The general model has NO access to enterprise data. Anything that looks like an
employee/system fact in its output is fabricated by definition and must never
reach the user.
"""

from __future__ import annotations

import re
from typing import Any


NEEDS_DATA_TOKEN = "NEEDS_ENTERPRISE_DATA"


GENERAL_ONLY_PROMPT = f"""
You are Sanad (سَنَد), the internal assistant of the Royal Commission for Riyadh City,
running in GENERAL MODE.

IN THIS MODE YOU HAVE NO ACCESS TO ANY INTERNAL RCRC DATA.

You MAY help with:
- greetings and ordinary conversation
- writing, drafting, rewriting, shortening, making text formal
- translation
- explaining general (non-RCRC) concepts, brainstorming, general advice

HARD RULE:
If answering would require ANY fact about a specific person (employee, manager,
colleague, the user themself), such as their name, job title, department, email,
mobile, extension, manager, team, leave balance, attendance, meetings,
Tarasol transactions — OR any RCRC-specific policy, procedure, news,
announcement, benefit or offer — reply with exactly this single token and nothing else:
{NEEDS_DATA_TOKEN}

Never invent names, numbers, emails, phone numbers, extensions, dates, balances,
or internal facts. Never guess who someone is.

VERIFIED_FACTS (when present) were fetched from RCRC systems by the backend and are
correct. Use them in writing tasks: address people by these names and sign with
the user's name instead of placeholders like [اسم المدير] or [اسمك].

Writing requests: write the requested text directly using only what the user,
GENERAL_HISTORY and VERIFIED_FACTS provided. Do not add fill-in-the-blank placeholders unless
the user asks for a template. If one essential fact is missing, ask one short question.

Reply in CURRENT_LANGUAGE only (ar = Arabic, en = English). Keep answers concise and natural.
""".strip()


# RCRC content domains (things that must come from APIs, never from the model).
CONTENT_DOMAIN_RE = re.compile(
    r"(?:سياس|اجراء|إجراء|لائح|خبر|اخبار|أخبار|اعلان|إعلان|تعميم|تعاميم|مزايا|عروض|"
    r"policy|policies|procedure|news|announcement|circular|benefit|offers?|mazaya)",
    re.I,
)

EXPLAIN_VERB_RE = re.compile(
    r"(?:اشرح|وضح|وضّح|لخص|اختصر|explain|summari[sz]e|shorten)",
    re.I,
)


def is_content_explain_request(text: str) -> bool:
    """'اشرح لي سياسة ...' / 'لخص الأخبار' must go to APIs, not the general model."""
    value = str(text or "")
    return bool(EXPLAIN_VERB_RE.search(value) and CONTENT_DOMAIN_RE.search(value))


_AR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
})
_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def _n(value: Any) -> str:
    text = str(value or "").casefold()
    text = _DIACRITICS_RE.sub("", text).replace("ـ", "").translate(_AR_MAP)
    text = re.sub(r"[^0-9a-z\u0600-\u06ff@.+]+", " ", text)
    return " ".join(text.split())


_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # email
    re.compile(r"[\w.\-]+@[\w\-]+(?:\.[\w\-]+)+"),
    # Saudi mobile
    re.compile(r"(?<!\d)(?:\+?966|0)?5\d{8}(?!\d)"),
    # extension with a number
    re.compile(r"(?:تحويل\S*|extension|ext\.?)\s*[:：]?\s*\d{3,5}", re.I),
    # "your manager is ..."
    re.compile(r"(?:مدير(?:ك|ه|ها)(?:\s+المباشر)?\s+(?:هو|هي)|your\s+(?:direct\s+)?manager\s+is)", re.I),
    # leave balance statements
    re.compile(r"(?:رصيدك|رصيد\s+[اإ]جازاتك|your\s+(?:leave|vacation)\s+balance\s+is)", re.I),
    # counts of meetings / transactions
    re.compile(r"(?:عندك\s+\d+\s+(?:اجتماع|معامل)|you\s+have\s+\d+\s+(?:meetings?|transactions?))", re.I),
)


def looks_fabricated(answer: str, state: dict[str, Any] | None, message: str = "", allowed: str = "") -> bool:
    """True when a general-mode answer contains enterprise-looking facts that the
    user did not type themselves, or when the model asked for enterprise data."""
    text = str(answer or "")
    if not text.strip():
        return False
    if NEEDS_DATA_TOKEN in text:
        return True

    # Facts verified by the backend (VERIFIED_FACTS) count as user-supplied.
    msg_n = _n(f"{message} {allowed}")

    for pattern in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            found = _n(match.group(0))
            if found and found in msg_n:
                continue  # the user supplied it (e.g. pasted an email to rewrite)
            return True

    # Known employee names from this conversation should never appear in a
    # general answer unless the user typed them.
    state = state if isinstance(state, dict) else {}
    rows: list[Any] = list(state.get("entity_history") or [])
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    rows.extend([focus.get("current"), focus.get("previous"), state.get("authenticated_user")])
    answer_n = _n(text)
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("name_ar", "name_en", "name"):
            name = _n(row.get(key))
            if name and len(name.split()) >= 2 and name in answer_n and name not in msg_n:
                return True
    return False


def sanitize_general_history(history: list[Any], state: dict[str, Any] | None) -> list[dict[str, str]]:
    """Drop past assistant turns that contain enterprise-looking facts so an old
    hallucination is never fed back to the model."""
    clean: list[dict[str, str]] = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if role == "assistant" and looks_fabricated(content, state, ""):
            continue
        clean.append({"role": role, "content": content})
    return clean


def enterprise_clarification(language: str) -> str:
    if language == "ar":
        return (
            "هذا السؤال يحتاج بيانات من أنظمة الهيئة، وما قدرت أحدد المطلوب بدقة. "
            "وضّح لي اسم الموظف أو الخدمة اللي تقصدها، مثلًا: «إيميل عماد الهاجري»، "
            "«رصيد إجازاتي»، «آخر الأخبار»."
        )
    return (
        "This question needs RCRC system data, and I couldn't identify exactly what you need. "
        "Please mention the employee or service, e.g. \"Emad's email\", \"my leave balance\", "
        "or \"latest news\"."
    )