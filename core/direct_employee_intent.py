"""Explicit person-lookup grammar for Sanad (Arabic/English).

Only claims a named employee when a name is actually supplied. Contextual
pronouns and numbered-list selections remain the orchestrator's responsibility.
"""
from __future__ import annotations
import re

FILLER = r"(?:(?:طيب|طب|ممكن|لو سمحت|بالله|please|can you)\s+)*"
START = r"(?:(?:ابي|أبي|ابغى|أبغى|اريد|أريد|اعطني|أعطني|عطني|ورني|اظهر|أظهر|هات|show me|give me|what is|what's|find|search for)\s+)*"
FIELD = {
    "email": r"(?:ايميل|إيميل|الايميل|الإيميل|البريد(?:\s+الالكتروني|\s+الإلكتروني)?|email|e-mail)",
    "mobile": r"(?:رقم\s+(?:الجوال|جوال|جواله|جوالها|الهاتف)|جوال|جواله|جوالها|موبايل|mobile(?:\s+number)?|phone(?:\s+number)?)",
    "extension": r"(?:التحويلة|تحويلة|التحويله|extension|ext)",
}
NAME = r"[\w\u0600-\u06ff][\w\u0600-\u06ff ._'-]*"
SERVICE = re.compile(r"\b(?:إجازة|اجازة|الإجازات|اجازاتي|الحضور|بصمة|بصمات|دوام|اجتماعات|إعلانات|اعلانات|سياسات|مزايا|leave|attendance|meetings|announcements|policy)\b", re.I)
SELF = re.compile(r"^(?:انا|أنا|لي|حقي|خاصتي|نفسي|بياناتي|معلوماتي|ايميلي|إيميلي|بريدي|جوالى|جوالي|مديري|manager|my|mine|me)$", re.I)
ORDINAL = re.compile(r"^(?:رقم\s*)?(?:\d+|الاول|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|first|second|third)$", re.I)

def _valid_name(value: str) -> bool:
    name = value.strip(" ؟?!.،, ")
    if not name or len(name) < 2 or SELF.fullmatch(name) or ORDINAL.fullmatch(name) or SERVICE.search(name):
        return False
    if re.fullmatch(r"(?:هو|هي|له|لها|إيميله|ايميله|ايميلها|جواله|جوالها|his|her|their)", name, re.I):
        return False
    return True

def direct_employee_lookup(text: str):
    """Return (mode, explicit target, fields), or None if not safely identifiable."""
    value = " ".join(str(text or "").strip().split()).strip(" ؟?!،,.")
    if not value:
        return None
    email = re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", value)
    if email:
        return ("email_lookup", value, ["name", "job_title", "department", "email"])
    prefix = FILLER + START
    fields = []
    for field, pattern in FIELD.items():
        if re.search(pattern, value, re.I):
            fields.append(field)
    if fields:
        # Contact field(s) followed by the explicit employee name, optionally a relational preposition.
        patterns = [
            prefix + r"(?:" + "|".join(FIELD.values()) + r")(?:\s*(?:و|and)\s*(?:" + "|".join(FIELD.values()) + r"))*\s+(?:(?:حق|لـ|ل|عن|of|for)\s+)?(" + NAME + r")",
            prefix + r"(?:" + "|".join(FIELD.values()) + r")(?:\s*(?:و|and)\s*(?:" + "|".join(FIELD.values()) + r"))*\s+(?:الموظف|الموظفة)\s+(" + NAME + r")",
        ]
        for pattern in patterns:
            matched = re.fullmatch(pattern, value, re.I)
            if matched and _valid_name(matched.group(1)):
                return ("person", matched.group(1).strip(), fields)
    for pattern in (
        prefix + r"(?:بيانات|معلومات|تفاصيل|ملف|بروفايل|profile|details|information)\s+(?:(?:عن|حق|لـ|ل|of|for)\s+)?(" + NAME + r")",
        FILLER + r"(?:ابحث|أبحث|دور|دوّر|دورلي|دور\s+لي|search(?:\s+for)?|find)\s*(?:لي\s+)?(?:عن|على|for)?\s+(" + NAME + r")",
    ):
        matched = re.fullmatch(pattern, value, re.I)
        if matched and _valid_name(matched.group(1)):
            return ("person", matched.group(1).strip(), ["name", "job_title", "department", "email", "mobile", "extension"])
    return None

# Multi-field requests: remove *all* requested field expressions before
# extracting the person name. Leave the existing single-field grammar intact.
_legacy_direct_employee_lookup = direct_employee_lookup
_CONTACT_WORDS = re.compile(
    r"\bو?(?:البريد\s+(?:الإلكتروني|الالكتروني)|(?:ال)?[اإ]يميل|"
    r"رقم\s+(?:ال)?جوال(?:ه|ها)?|رقم\s+الهاتف|"
    r"(?:ال)?جوال(?:ه|ها)?|(?:ال)?موبايل|(?:ال)?تحويل(?:ة|ه)|"
    r"(?:ال)?مسمى\s+(?:الوظيفي)?|(?:ال)?منصب|"
    r"email|e-mail|mobile(?:\s+number)?|phone(?:\s+number)?|"
    r"extension|job\s+title)", re.I,
)
_FIELD_GROUPS = {
    "email": re.compile(FIELD["email"], re.I),
    "mobile": re.compile(FIELD["mobile"], re.I),
    "extension": re.compile(FIELD["extension"], re.I),
    "job_title": re.compile(r"(?:المسمى\s+الوظيفي|مسمى|منصب|job\s+title)", re.I),
}
_INTRO = re.compile(
    r"^(?:(?:طيب|طب|ممكن|لو\s+سمحت|بالله)\s+)*"
    r"(?:(?:ابي|أبي|ابغى|أبغى|اريد|أريد|عطني|اعطني|أعطني|"
    r"ورني|هات|وش|ايش|ما|show\s+me|give\s+me|what(?:'s|\s+is))\s+)*",
    re.I,
)

def direct_employee_lookup(text: str):
    """Parse standalone person lookups, including multiple requested fields."""
    value = " ".join(str(text or "").strip().split()).strip(" ؟?!،,.")
    if not value:
        return None
    # Possessive pronouns refer to the current focus. They must NEVER be parsed
    # as an explicitly named employee (in particular a name of "ه").
    if re.search(
        r"(?:[اإ]يميل(?:ه|ها)|بريد(?:ه|ها)|جوال(?:ه|ها)|تحويلت(?:ه|ها)|"
        r"رقم\s+جوال(?:ه|ها)|(?:his|her|their)\s+(?:email|mobile|phone|extension))",
        value, re.I,
    ):
        return None
    # Explicit entire email address is handled by the old proven path.
    if re.fullmatch(r"[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}", value):
        return _legacy_direct_employee_lookup(value)
    fields = [field for field, matcher in _FIELD_GROUPS.items() if matcher.search(value)]
    if len(fields) < 2 or SERVICE.search(value):
        return _legacy_direct_employee_lookup(value)
    candidate = _INTRO.sub("", value)
    # Remove field words in one pass so "ورقم جوال" never becomes a name.
    candidate = _CONTACT_WORDS.sub(" ", candidate)
    candidate = re.sub(r"\b(?:and|of|for|the|employee|person)\b", " ", candidate, flags=re.I)
    candidate = re.sub(r"(?:^|\s)(?:و|حق|عن|الموظف|الموظفة|لـ)(?=\s|$)", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ؟?!،,. ")
    if candidate.startswith("و") and len(candidate) > 1:
        candidate = candidate[1:].strip()
    if candidate and _valid_name(candidate):
        return ("person", candidate, fields)
    return _legacy_direct_employee_lookup(value)
