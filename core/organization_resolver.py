"""Safe normalization of organizational aliases before verified directory matching.

Not an authoritative organizational chart. Never manufacture unit IDs or membership.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass

ALIAS_TO_QUERY = {
    "it": "التقنية والرقمنة", "ict": "التقنية والرقمنة",
    "information technology": "التقنية والرقمنة",
    "digital technology": "التقنية والرقمنة",
    "hr": "الموارد البشرية", "human resources": "الموارد البشرية",
    "ia": "المراجعة الداخلية", "internal audit": "المراجعة الداخلية",
    "cyber": "الأمن السيبراني", "cybersecurity": "الأمن السيبراني",
    "cyber security": "الأمن السيبراني",
    "pmo": "المشاريع", "project management office": "المشاريع",
    "finance": "المالية", "legal": "الشؤون القانونية",
    "procurement": "المشتريات",
    "enterprise solutions": "الحلول المؤسسية",
}

@dataclass(frozen=True)
class UnitQuery:
    original: str
    canonical_query: str
    alias: str = ""


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text)
    text = text.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه"}))
    return " ".join(text.split())


def find_unit_alias(message: str) -> UnitQuery | None:
    """Detect alias as a whole unit. Do not match 'it' inside other words."""
    raw = str(message or "")
    normalized = _norm(raw)
    for alias in sorted(ALIAS_TO_QUERY, key=len, reverse=True):
        if re.search(r"(?<![a-z])(?:بال|لل|ال)?\s*" + re.escape(alias) + r"(?![a-z])", normalized, re.I):
            return UnitQuery(original=raw, canonical_query=ALIAS_TO_QUERY[alias], alias=alias)
    return None


def require_verified_unit(candidates: list[dict], query: str) -> list[dict]:
    """Fail closed on unverified department filter: never show entire directory.

    Needs rows carrying actual, trustworthy department/unit columns. Parent/child
    hierarchy must be handled by a verified organization table, not guessed here.
    """
    q = _norm(query)
    matched=[]
    for row in candidates:
        if not isinstance(row,dict):continue
        for key in ("general_department","general_department_ar","generalDepartment",
                    "department","department_ar","departmentName"):
            value=_norm(row.get(key, ""))
            if value and (q == value or q in value):
                matched.append(row)
                break
    return matched
