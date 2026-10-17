"""Year clarification for Sanad's date-sensitive requests.

This module does not call APIs or save conversation state. The orchestrator
persists the original question and passes it to ``date_year_turn`` after the
user answers with a year.
"""
from __future__ import annotations

import re

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_MONTHS = (
    "يناير", "فبراير", "مارس", "ابريل", "أبريل", "مايو", "يونيو",
    "يوليو", "اغسطس", "أغسطس", "سبتمبر", "اكتوبر", "أكتوبر",
    "نوفمبر", "ديسمبر", "january", "jan", "february", "feb",
    "march", "mar", "april", "apr", "may", "june", "jun",
    "july", "jul", "august", "aug", "september", "sept", "sep",
    "october", "oct", "november", "nov", "december", "dec",
)
_MONTH = "(?:" + "|".join(re.escape(m) for m in sorted(_MONTHS, key=len, reverse=True)) + ")"
_DAY = r"\d{1,2}"
# A day with a named month, e.g. '20 أكتوبر', 'October 20'.
_EXPLICIT_DAY_MONTH = re.compile(
    rf"(?<!\w)(?:{_DAY}\s+{_MONTH}|{_MONTH}\s+{_DAY})(?!\w)", re.I
)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_YEAR_ONLY = re.compile(r"^\s*((?:19|20)\d{2})\s*$")
# Two numeric dates with their years, e.g. 20/10/2026 to 24/10/2026.
_EXPLICIT_DATED_NUMERIC = re.compile(r"(?<!\d)(?:\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)\d{2}|(?:19|20)\d{2}[/.-]\d{1,2}[/.-]\d{1,2})(?!\d)")


def dated_message_needs_year(message: str) -> bool:
    """True only for explicit day+month dates lacking a four-digit year.

    Relative expressions such as 'الأسبوع الماضي' are not intercepted here.
    """
    value = str(message or "").translate(_DIGITS)
    if _YEAR.search(value) or _EXPLICIT_DATED_NUMERIC.search(value):
        return False
    return bool(_EXPLICIT_DAY_MONTH.search(value))


def date_year_turn(message: str, original_question: str) -> tuple[str, bool]:
    """Apply a standalone year reply to the saved original question.

    Returns (completed_question, consumed). The caller must keep the pending
    state until consumed or superseded by a new explicit request.
    """
    response = str(message or "").translate(_DIGITS).strip()
    match = _YEAR_ONLY.fullmatch(response)
    original = str(original_question or "").strip()
    if not match or not dated_message_needs_year(original):
        return str(message or ""), False
    year = int(match.group(1))
    if not 1900 <= year <= 2099:
        return str(message or ""), False
    return f"{original} {year}", True


__all__ = ["dated_message_needs_year", "date_year_turn"]
