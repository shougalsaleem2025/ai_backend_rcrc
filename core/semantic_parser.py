from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from core.llm_client import chat
from core.semantic_models import ParsedRequest, TargetRef

SEMANTIC_PARSER_BUILD = "rcrc-v163-context-api-routing-fixes-2026-10-03"



ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")


# Common English department names/acronyms -> verified Arabic directory query fragments.
# The directory matcher also checks English department fields returned by the API,
# so this table is mainly for short acronyms and common business names.
_DEPARTMENT_QUERY_ALIASES: tuple[tuple[str, str], ...] = (
    (r"\b(?:it|i\s*\.?t\.?|ict)\b", "التقنية والرقمنة"),
    (r"\b(?:information\s+technology|technology\s+and\s+(?:digitization|digitalization)|digital\s+technology)\b", "التقنية والرقمنة"),
    (r"\b(?:hr|h\s*\.?r\.?|human\s+resources?|people\s+and\s+culture)\b", "الموارد البشرية"),
    (r"\b(?:finance|financial(?:\s+affairs)?)\b", "المالية"),
    (r"\b(?:legal|legal\s+affairs)\b", "الشؤون القانونية"),
    (r"\b(?:procurement|purchasing)\b", "المشتريات"),
    (r"\b(?:strategy|strategy\s+and\s+polic(?:y|ies))\b", "الاستراتيجية والسياسات"),
    (r"\b(?:pmo|p\s*\.?m\s*\.?o\.?|project\s+management\s+office|project\s+management)\b", "المشاريع"),
    (r"\b(?:cybersecurity|cyber\s+security)\b", "الأمن السيبراني"),
)


def _extract_department_alias(message: str) -> str:
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return ""
    normalized = _normalize_for_semantic_guards(raw)
    for pattern, canonical_query in _DEPARTMENT_QUERY_ALIASES:
        if re.search(pattern, normalized, flags=re.I):
            return canonical_query
    return ""


_AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def _normalize_for_semantic_guards(text: str) -> str:
    """Minimal normalization for closed grammatical guards only."""
    value = str(text or "").strip().lower()
    value = _AR_DIACRITICS_RE.sub("", value)
    value = value.translate(
        str.maketrans({
            "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",
            "ؤ": "و", "ئ": "ي",
        })
    )
    # Common Arabic typing/transposition variants.
    value = (
        value
        .replace("الادراه", "الاداره")
        .replace("الادراة", "الاداره")
        .replace("ادراه", "اداره")
        .replace("ادراة", "اداره")
        .replace("الادرارات", "الادارات")
        .replace("ادرارات", "ادارات")
        .replace("الادراات", "الادارات")
        .replace("ادراات", "ادارات")
    )
    value = re.sub(r"\s+", " ", value)
    return value



def _has_verified_focus(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    current = focus.get("current") if isinstance(focus.get("current"), dict) else {}
    return bool(current.get("username") or current.get("userid"))


def _name_is_literal_in_message(name: str, message: str) -> bool:
    """True only when the employee name was actually typed in CURRENT_MESSAGE."""
    target_tokens = _norm_span(name).split()
    message_tokens = _norm_span(message).split()
    if not target_tokens:
        return False
    width = len(target_tokens)
    return any(
        message_tokens[i:i + width] == target_tokens
        for i in range(0, max(0, len(message_tokens) - width + 1))
    )


def _is_anaphoric_reference(message: str) -> bool:
    """Detect deictic/pronominal wording, not business intents.

    This is deliberately grammar-level. It does not decide which API/intent to use.
    It only tells us that the user is referring to a previously established person.
    """
    raw = str(message or "").strip().lower()
    norm = _normalize_for_semantic_guards(raw)

    # English pronouns / possessives.
    if re.search(r"\b(?:his|her|him|their|theirs|that person|the same person|last one)\b", norm):
        return True

    # Arabic standalone prepositional/possessive pronouns.
    if re.search(r"\b(?:له|لها|عنه|عنها|منه|منها|عليه|عليها|حقه|حقها|نفسه|نفسها)\b", norm):
        return True

    # Explicit manager possessives such as مديره / مديرها / مديرته / مديرتها.
    if re.search(r"\bمدير(?:ه|ها|ته|تها)\b", norm):
        return True

    # Arabic attached possessive/object pronouns such as:
    # ايميله، تحويلته، اجازاته، معلوماته، مديره، جوالها.
    tokens = re.findall(r"[\u0600-\u06ff]+", raw)
    excluded = {
        "الله", "هذه", "هذا", "الهيئه", "الهيئة", "جهه", "جهة",
        "خدمه", "خدمة", "قائمه", "قائمة",
    }
    for token in tokens:
        nt = _normalize_for_semantic_guards(token)
        if nt in {_normalize_for_semantic_guards(x) for x in excluded}:
            continue
        if len(nt) >= 4 and (nt.endswith("ها") or nt.endswith("ه")):
            return True
    return False


def _focus_target() -> TargetRef:
    return TargetRef(kind="focus", name="")


def _focus_matches_person(state: dict[str, Any] | None, person: str) -> bool:
    """Return True when an explicitly typed short name matches verified focus.

    This lets a follow-up such as "طيب بكرة اسامه فاضي؟" reuse the verified
    Osama from the immediately preceding turn instead of starting an ambiguous
    directory search. A different explicit name (for example شاهه العقيل) still
    wins and is resolved as a new employee.
    """
    if not isinstance(state, dict):
        return False
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    current = focus.get("current") if isinstance(focus.get("current"), dict) else {}
    focus_name = str(
        current.get("name_ar") or current.get("name") or current.get("name_en") or ""
    ).strip()
    p = _normalize_for_semantic_guards(person)
    f = _normalize_for_semantic_guards(focus_name)
    if not p or not f:
        return False
    tokens = [t for t in p.split() if len(t) >= 2]
    return bool(tokens) and all(t in f.split() for t in tokens)


def _repair_untrusted_employee_reference(
    request: ParsedRequest,
    message: str,
    state: dict[str, Any] | None,
) -> ParsedRequest:
    """Never trust an employee name invented by the LLM.

    A target of kind=employee is valid only when its name is literally present
    in CURRENT_MESSAGE. If the utterance is pronominal/anaphoric, resolve it
    through verified conversation focus instead.
    """
    req = request.model_copy(deep=True)
    anaphoric = _is_anaphoric_reference(message)

    repaired: list[TargetRef] = []
    for target in req.targets:
        if target.kind != "employee":
            repaired.append(target)
            continue

        if _name_is_literal_in_message(target.name, message):
            repaired.append(target)
            continue

        # Model supplied a name the user did not type. Never search that name.
        # For anaphoric wording, use structured verified focus.
        if anaphoric:
            repaired.append(_focus_target())
        else:
            # Keep no fabricated person. Validation will ask who the user means.
            repaired.append(TargetRef(kind="none", name=""))

    req.targets = repaired

    # If the model called an anaphoric field request an employee search,
    # normalize it back to employee_details. Search is for finding people;
    # email/mobile/extension/title/profile fields are details about the focus.
    detail_fields = {
        "email", "mobile", "extension", "job_title", "job_title_en",
        "department", "department_en", "sector", "sector_en",
        "username", "employee_type", "all",
    }
    if anaphoric and req.intent == "employee_search" and set(req.fields or []) & detail_fields:
        req.intent = "employee_details"
        req.operation = "details"

    return req



def _looks_like_organizational_role(value: str) -> bool:
    """Distinguish an organisational unit/title from a person's name.

    This is a narrow ontology guard, not general keyword routing. It protects
    requests such as "مين مدير إدارة الثقافة المؤسسية؟" from being treated as
    "who is the manager of an employee named إدارة الثقافة المؤسسية".
    """
    norm = _normalize_for_semantic_guards(value)
    ar_org_terms = ("اداره", "قسم", "قطاع", "وحده", "مكتب", "مركز", "برنامج", "مشروع")
    en_org_terms = ("department", "division", "section", "unit", "office", "center", "centre", "program", "programme")
    tokens = set(norm.split())
    return (
        any(term in tokens or ("ال" + term) in tokens for term in ar_org_terms)
        or any(term in tokens for term in en_org_terms)
    )


def _first_person_manager_reference(message: str) -> bool:
    msg = _normalize_for_semantic_guards(message)
    return bool(
        re.search(r"\bمديري\b", msg)
        or re.search(r"\bالمدير\s+(?:المباشر\s+)?لي\b", msg)
        or re.search(r"\bmy\s+(?:direct\s+)?manager\b", msg)
    )


def _relation_has_row(state: dict[str, Any] | None, key: str) -> bool:
    if not isinstance(state, dict):
        return False
    relation = state.get("relation") if isinstance(state.get("relation"), dict) else {}
    row = relation.get(key) if isinstance(relation.get(key), dict) else {}
    return bool(row.get("username") or row.get("userid"))


def _explicit_manager_object_reference(message: str) -> bool:
    """Explicit manager-of-topic wording, not a plain pronoun."""
    msg = _normalize_for_semantic_guards(message)
    return bool(
        re.search(r"\bمدير(?:ه|ها|هم|ته|تها)\b", msg)
        or re.search(r"\bmanager(?:'s)?\b", msg)
    )


def _repair_relation_reference(request: ParsedRequest, message: str, state: dict[str, Any] | None) -> ParsedRequest:
    req = request.model_copy(deep=True)
    if (
        req.intent in {
            "employee_details", "leave_balance", "meetings", "meeting_list",
            "attendance_today", "attendance_history", "working_hours",
            "tarasol_count", "tarasol_details", "reporting_employees",
        }
        and _relation_has_row(state, "object")
        and _explicit_manager_object_reference(message)
    ):
        req.targets = [TargetRef(kind="relation_object", name="")]
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
    return req


def _repair_role_and_list_context(
    request: ParsedRequest,
    message: str,
    state: dict[str, Any] | None,
) -> ParsedRequest:
    """Repair two structural ambiguities using verified dialogue state.

    1) Organisational role/title lookups must use employee_role_search.
    2) A person-scoped employee list means reporting employees. Elliptical
       follow-ups can use verified focus, but a true global directory list stays
       employee_list with no person target.
    """
    req = request.model_copy(deep=True)
    has_focus = _has_verified_focus(state)
    raw = " ".join(str(message or "").strip().split())

    # Direct organisational role question, e.g.:
    # "مين هو مدير عام الإدارة العامة للتقنية والرقمنة؟"
    direct_role_match = re.search(
        r"(?:^|\s)(?:مين|من)(?:\s+هو|\s+هي)?\s+((?:ال)?مدير(?:\s+عام)?\s+.+?)[؟?!.]*$",
        raw,
        flags=re.I,
    )
    if direct_role_match:
        role_query = str(direct_role_match.group(1) or "").strip()
        role_norm = _normalize_for_semantic_guards(role_query)
        if re.search(r"\b(?:ال)?(?:اداره|قسم|قطاع|وحده|مكتب|مركز)\b", role_norm):
            req.intent = "employee_role_search"
            req.operation = "search"
            req.targets = []
            req.role_query = role_query
            if not req.fields:
                req.fields = ["name", "job_title", "department"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
            return req

    # Role holder vs manager-of-person distinction.
    if req.intent == "manager" and len(req.targets) == 1:
        target = req.targets[0]
        if target.kind == "employee" and _looks_like_organizational_role(target.name):
            role = " ".join(str(req.role_query or "").split())
            if not role:
                msg = _normalize_for_semantic_guards(message)
                manager_word = "مدير" if "مدير" in msg else "manager"
                role = f"{manager_word} {target.name}".strip()
            req.intent = "employee_role_search"
            req.operation = "search"
            req.targets = []
            if not req.fields:
                req.fields = ["name", "job_title", "department"]
            req.role_query = role
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.98)

    # Any list scoped to a person is semantically a direct-report/team list,
    # not the global employee directory.
    if req.intent == "employee_list" and req.targets:
        target = req.targets[0]
        if target.kind in {"employee", "focus", "previous_focus", "manager_of_self"}:
            # A generic/elliptical follow-up should continue from current focus;
            # only explicit first-person manager wording keeps manager_of_self.
            if target.kind == "manager_of_self" and has_focus and not _first_person_manager_reference(message):
                req.targets = [TargetRef(kind="focus", name="")]
            req.intent = "reporting_employees"
            req.operation = "list"
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.98)

    # Reporting/team follow-ups should continue from verified focus unless the
    # CURRENT message explicitly refers to the authenticated user's manager.
    if req.intent == "reporting_employees":
        if req.targets:
            target = req.targets[0]
            if (
                target.kind == "manager_of_self"
                and has_focus
                and not _first_person_manager_reference(message)
            ):
                req.targets = [TargetRef(kind="focus", name="")]
        elif has_focus and _is_anaphoric_reference(message):
            req.targets = [TargetRef(kind="focus", name="")]

    return req

def _repair_general_department_followup(
    request: ParsedRequest,
    message: str,
    state: dict[str, Any] | None,
) -> ParsedRequest:
    req = request.model_copy(deep=True)
    if not isinstance(state, dict):
        return req
    ctx = state.get("directory_context") if isinstance(state.get("directory_context"), dict) else {}
    if not ctx.get("general_department_key"):
        return req
    msg = _normalize_for_semantic_guards(message).strip(" ؟?!.")
    closed_followups = {
        "مين هم", "من هم", "طيب مين هم", "طيب من هم",
        "مين الموظفين", "من الموظفين", "who are they", "which employees",
    }
    if msg in closed_followups:
        req.intent = "general_department_members"
        req.operation = "list"
        req.targets = []
        req.department_query = ""
        req.fields = []
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    remembered_department = str(ctx.get("general_department") or "").strip()
    if remembered_department and re.fullmatch(
        r"(?:مين|من)(?:\s+هو|\s+هي)?\s+(?:ال)?مدير\s+العام\s+فيها",
        msg,
    ):
        req.intent = "employee_role_search"
        req.operation = "search"
        req.targets = []
        req.role_query = f"مدير عام {remembered_department}"
        req.department_query = remembered_department
        req.fields = ["name", "job_title", "department"]
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
    return req


def _apply_semantic_invariants(request: ParsedRequest, message: str, state: dict[str, Any] | None = None) -> ParsedRequest:
    """Apply small grammar-level invariants after LLM parsing.

    Open-ended intent understanding remains model-based. These guards only cover
    closed deictic grammar where the referent is mathematically/linguistically
    determined by the current utterance, e.g. "مديري" == manager of self.
    """
    msg = _normalize_for_semantic_guards(message)
    req = request.model_copy(deep=True)

    # Keep explicit employee-name searches grounded in the user's literal text.
    # The LLM must not transliterate/translate names (e.g. عدنان العقيلي -> adenan alaqeel),
    # because SearchEmployee expects the directory spelling entered by the user.
    explicit_employee_search = re.fullmatch(
        r"(?:(?:طيب|تمام|اوكي|أوكي|ممكن|لو\s+سمحت)\s+)?"
        r"(?:(?:ابي|أبي|ابغى|أبغى)\s+)?"
        r"(?:ابحث|أبحث|دور|دوّر|دورلي|دور\s+لي|ابحث\s+لي|أبحث\s+لي|find|search(?:\s+for)?)"
        r"\s+(?:لي\s+)?(?:عن\s+)?(.+?)[؟?!.،,]*",
        str(message or "").strip(),
        flags=re.I,
    )
    if explicit_employee_search:
        literal_name = " ".join(str(explicit_employee_search.group(1) or "").split()).strip()
        if literal_name and not _looks_like_organizational_role(literal_name):
            req.intent = "employee_search"
            req.operation = "search"
            req.targets = []
            req.query = literal_name
            if not req.fields:
                req.fields = ["name", "job_title", "department"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)

    # "Who is my manager?" must be anchored to the authenticated user, never to
    # conversational focus. "Who is my manager's manager?" is the same subject
    # with relation_depth=2.
    if req.intent == "manager" and set(req.fields or []) <= {"name"}:
        second_level_ar = bool(re.search(r"\bمدير\s+مديري\b", msg))
        second_level_en = bool(re.search(r"\bmanager\s+of\s+my\s+(?:direct\s+)?manager\b|\bmy\s+(?:direct\s+)?manager'?s\s+manager\b", msg))
        first_person_ar = bool(
            re.search(r"\bمديري\b", msg)
            or re.search(r"\bالمدير\s+(?:المباشر\s+)?لي\b", msg)
        )
        first_person_en = bool(re.search(r"\bmy\s+(?:direct\s+)?manager\b", msg))

        if second_level_ar or second_level_en:
            req.targets = [TargetRef(kind="self", name="")]
            req.relation_depth = 2
            if "name" not in req.fields:
                req.fields = ["name"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
        elif first_person_ar or first_person_en:
            req.targets = [TargetRef(kind="self", name="")]
            req.relation_depth = 1
            if "name" not in req.fields:
                req.fields = ["name"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)

    req = _repair_general_department_followup(req, message, state)
    req = _repair_role_and_list_context(req, message, state)
    req = _repair_relation_reference(req, message, state)
    req = _repair_untrusted_employee_reference(req, message, state)

    # Calendar/attendance APIs are currently token-owner self-service only.
    # Keep the semantic subject intact (so the orchestrator can explain the
    # limitation for another employee) and never silently rewrite a third-person
    # calendar request into an executable self request.
    if req.intent == "employee_availability":
        req.fields = ["start_time", "end_time"]
        if not req.time_query:
            req.time_query = _extract_time_query(message)
        if _has_explicit_period_cue(message):
            if req.period == "none":
                req.period, req.start_date, req.end_date = _availability_period(message)
            req.clarification_reason = ""
        else:
            req.period = "none"
            req.start_date = ""
            req.end_date = ""
            req.clarification_reason = "missing_date_range"

    if req.intent in {"meetings", "meeting_list", "attendance_today", "attendance_history", "working_hours"}:
        if not _has_explicit_period_cue(message):
            req.period = "none"
            req.start_date = ""
            req.end_date = ""
            req.clarification_reason = "missing_date_range"

    if req.intent == "leave_balance" and req.operation == "calculate" and req.amount is None:
        req.amount = _extract_leave_amount(message)

    return req
WORD_RE = re.compile(r"[A-Za-z\u0600-\u06ff]+")


def _norm_span(value: str) -> str:
    text = str(value or "").casefold()
    for source, target in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        text = text.replace(source, target)
    return " ".join(WORD_RE.findall(text))


def _ground_employee_names(request: ParsedRequest, message: str) -> ParsedRequest:
    """Ground model-extracted employee names back to literal spans in the message.

    The LLM may occasionally drop one character while copying a name. We never
    invent a replacement name: we only substitute a highly-similar contiguous
    span that literally exists in CURRENT_MESSAGE.
    """
    message_tokens = WORD_RE.findall(str(message or ""))
    message_norm = _norm_span(message)
    for target in request.targets:
        if target.kind != "employee" or not target.name.strip():
            continue
        target_norm = _norm_span(target.name)
        target_norm_tokens = target_norm.split()
        message_norm_tokens = message_norm.split()
        exact_span = any(
            message_norm_tokens[i:i + len(target_norm_tokens)] == target_norm_tokens
            for i in range(0, max(0, len(message_norm_tokens) - len(target_norm_tokens) + 1))
        ) if target_norm_tokens else False
        if exact_span:
            continue
        target_tokens = WORD_RE.findall(target.name)
        if not target_tokens or not message_tokens:
            continue
        best_span = ""
        best_ratio = 0.0
        desired = len(target_tokens)
        for width in range(max(1, desired - 1), min(len(message_tokens), desired + 1) + 1):
            for i in range(0, len(message_tokens) - width + 1):
                span = " ".join(message_tokens[i:i + width])
                ratio = SequenceMatcher(None, target_norm, _norm_span(span)).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = span
        if best_span and best_ratio >= 0.86:
            target.name = best_span
    return request


SEMANTIC_PARSER_PROMPT = r"""
You are the semantic parser for RCRC AI Assistant.

Return exactly one valid JSON object that matches ParsedRequest.
Do not answer the user. Do not include markdown or commentary.

Your job is to infer meaning, target, and domain from natural language and
conversation context. Do not depend on exact wording.

SUPPORTED LANGUAGE
Understand:
- Saudi/Gulf Arabic
- Modern Standard Arabic
- English
- mixed Arabic/English
- common typos
- shorthand
- incomplete follow-ups

CORE INVARIANTS
1. Same meaning => same intent, even with different wording.
2. Never invent employee names, usernames, IDs, departments, roles, or facts.
3. Never convert a third-person reference to self.
4. First-person wording means authenticated self.
5. Third-person wording means the most recent verified focus when context supports it.
6. If target is ambiguous, preserve the domain and return a clarification reason.
7. Job titles are roles, not employee names.
8. Department-membership requests are not reporting-hierarchy requests.
9. Fresh factual values are never inferred from dialogue history.
10. Authorization is not your job. Preserve the requested target and let backend
    authorization allow or deny access.

TARGET RESOLUTION
First-person/self examples:
- انا / عندي / لي / حقي
- اجازاتي / حضوري / اجتماعاتي / مديري
- my / me / mine / my manager
=> target self or manager_of_self as appropriate

Third-person/focus examples:
- هو / هي / هذا / هذه
- حضوره / حضورها
- اجازاته / إجازاته / اجازتها / إجازتها
- ايميله / إيميلها
- جواله / جوالها
- تحويلته / تحويلتها
- مديره / مديرها
- منصبه / منصبها
- his / her / he / she
=> target focus / previous_focus when context contains a verified employee

Never map:
"رصيد اجازاته" -> self
"حضوره اليوم" -> self
"وش ايميله؟" -> self
when a verified focus exists.

ROLE SEARCH
Use employee_role_search when the user asks who holds a job title or provides a
job title without a person's name.

Equivalent role phrasings:
- مين كبير تقني الحلول الرقمية
- كبير تقني الحلول الرقمية
- مين ماسك منصب كبير تقني الحلول الرقمية
- من يشغل منصب كبير تقني الحلول الرقمية
=> employee_role_search
role_query = "كبير تقني الحلول الرقمية"

- مين مدير ادارة حلول الاعمال
- من مدير إدارة حلول الأعمال
- مين ماسك إدارة حلول الأعمال
=> employee_role_search
role_query = "مدير إدارة حلول الأعمال"

Role-title indicators include:
مدير عام, مدير إدارة, مدير قسم, رئيس قسم, مستشار, كبير تقني,
general manager, director, head of, chief, manager.

Do NOT send a title such as "كبير تقني الحلول الرقمية" to employee_search.

EMPLOYEE SEARCH
Use employee_search when the user is looking for a person by actual name.
Extract only the person name. Remove filler wording.
Do not use pronouns or possessive words as a name.

EMPLOYEE DETAILS
Examples:
"وش ايميل سعود؟"
"عطني البريد حق سعود"
=> employee_details, target employee("سعود"), fields=["email"]

With verified focus:
"وش ايميله؟"
"طيب البريد؟"
=> employee_details, target focus, fields=["email"]

MANAGER
"مين مديري" => manager, target self
"مين مديره؟" with focus => manager, target focus

REPORTING EMPLOYEES
"الموظفين اللي تحتي" => reporting_employees, target self
"الموظفين اللي تحته" with focus => reporting_employees, target focus

DEPARTMENT MEMBERS
Equivalent:
"موظفين HR"
"قائمة بموظفين الhr"
"ابي موظفين الموارد البشرية"
=> general_department_members

Equivalent:
"موظفين IT"
"قائمة بموظفين الit"
"ابي موظفين التقنية والرقمنة"
=> general_department_members

Aliases:
HR => الموارد البشرية
IT / ICT => التقنية والرقمنة
PMO => المشاريع

Do not convert department-membership requests to reporting_employees because a
manager happens to be the current focus.

MEETINGS
Equivalent self meeting requests:
- اجتماعاتي
- جدول اجتماعاتي
- قائمة اجتماعاتي
- مواعيدي
- وش عندي ميتنقات
- هل عندي اجتماع اليوم
- my meetings
- my schedule today
=> meetings / meeting_list depending existing schema conventions

ATTENDANCE
"حضوري اليوم" => attendance_today, self
"حضوره اليوم" with focus => attendance_today, focus
"مديري متى حضر" => attendance_today, manager_of_self
Never switch a third-person attendance request to self.

LEAVE
"رصيد اجازاتي" => leave_balance, self
"رصيد اجازاته" with focus => leave_balance, focus
"إجازاتي السابقة" => leave_history, self
For leave history requiring a period:
- if explicit relative period exists, preserve it
- if no period is given, set clarification_reason="missing_date_range"
Do not invent a date range.

CONTENT DOMAINS
Keep these separate:
- news / news_details
- announcements / announcement_details
- policy_search / policy_details
- mazaya / mazaya_details
- tarasol_count / tarasol_details
- hr_letter_types / hr_letter_recipients

Do not classify:
"ابرز المزايا الي تقدمها الهيئة"
as general if it refers to the Mazaya/benefits service.

GENERAL INFORMATION
Organization-level explanatory questions such as achievements, projects, goals,
vision, mission, or general writing/explanation may be general.
Do not treat phrases like "ابرز انجازات الهيئة" as employee names.

CONVERSATION-FIRST UNDERSTANDING
You receive RECENT_DIALOGUE in DIALOGUE_CONTEXT.
Use it to understand ellipsis, corrections, pronouns, short answers, numbers,
and follow-up intent exactly as a human conversation partner would.

Examples:
- Previous assistant asked meeting duration; user says "2" => this is an answer
  to the pending duration question, not a new unrelated request.
- Previous topic is a focused employee; user says "طيب ايميله" => email of focus.
- User says "لا قصدي مديري" => correct the target to manager_of_self while keeping
  the relevant domain/field from the immediately preceding request when possible.
- User says "ساعه", "ساعتين", "نص ساعة", "45 دقيقة" after a duration question =>
  interpret as duration context, not general chat.
- User says "الأول", "الثاني", "هذا", "حقه", "مديره" after a verified list/focus =>
  resolve through structural verified context.

IMPORTANT:
- RECENT_DIALOGUE is only for conversational meaning and continuity.
- Do NOT treat names, IDs, balances, attendance, emails, or other factual values
  appearing in dialogue text as verified enterprise facts.
- Verified focus/list/relation state remains the authority for entity resolution.
- A new explicit request overrides old context.
- Do not force a scripted slot flow when the user's current wording already gives
  enough information.

CONFIDENCE
Use high confidence only when the intent and target are well supported.
If uncertain:
- keep the most likely domain
- do not fabricate a target
- do not default to self
- use clarification_reason to request the missing distinction

OUTPUT
Return only one JSON object matching ParsedRequest.
"""

SEMANTIC_REVIEW_PROMPT = r"""
You are the second-pass semantic adjudicator for an internal RCRC assistant.
Re-read CURRENT_MESSAGE independently and correct FIRST_DRAFT if it misclassified the meaning. Do not infer a person from conversation history; only decide whether the message means self, manager_of_self, an explicit employee name, or an anaphoric focus reference.
The most important distinction is between ordinary `general` conversation and supported internal RCRC intents.
Short colloquial requests about the signed-in user, their manager, employees, leave, meetings, Tarasol, benefits, news, policies, announcements, attendance, HR letters, or reporting relationships are NOT general conversation.

Return exactly one full JSON object matching the same schema as FIRST_DRAFT. No prose.
If the request is clear, use confidence 0.90-1.00 and empty clarification_reason.
Only use low confidence / clarification_reason when there is a real unresolved ambiguity that changes what action should be taken.
Never invent usernames, IDs, emails, phone numbers, or enterprise facts.
If the message corrects the previous topic (for example "لا مو معاملات ..."), the new explicit request wins completely; do not inherit the previous intent.
"""


def detect_language(message: str, fallback: str = "ar") -> str:
    """Choose the response language from the current turn, with explicit cues."""
    text = str(message or "").strip()
    folded = text.casefold()
    if any(cue in folded for cue in ("بالانجليزي", "بالإنجليزي", "بالانقليزي", "in english", "english please", "speak english")):
        return "en"
    if any(cue in folded for cue in ("بالعربي", "بالعربية", "in arabic", "arabic please", "speak arabic")):
        return "ar"
    ar = len(ARABIC_RE.findall(text))
    en = len(LATIN_RE.findall(text))
    if ar == 0 and en == 0:
        return fallback if fallback in {"ar", "en"} else "ar"
    if ar and not en:
        return "ar"
    if en and not ar:
        return "en"
    # Mixed Arabic/English: use the language the user starts the utterance in.
    # This keeps Arabic workplace phrasing Arabic even when it contains IT/HR,
    # an English meeting term, or a Latin proper noun.
    first_alpha = re.search(r"[A-Za-z\u0600-\u06ff]", text)
    if first_alpha:
        ch = first_alpha.group(0)
        return "ar" if ARABIC_RE.match(ch) else "en"
    return fallback if fallback in {"ar", "en"} else "ar"


def _extract_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None



def _extract_relaxed_semantic(value: Any) -> dict[str, Any] | None:
    """Recover a small, predictable pseudo-JSON format from the local model.

    Qwen occasionally answers the NLU prompt with compact notation such as::

        employee_details, employee(name="Ali"), fields=[email, mobile]

    instead of JSON.  Throwing that whole turn away caused supported questions
    to fall into the generic "I did not understand" response.  This function
    only converts structure that is already explicit in the model output; it
    does not invent employee facts or resolve identities.  Normal grounding and
    authorization still run afterwards.
    """
    raw = str(value or "").strip()
    if not raw or "{" in raw:
        return None

    first = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)", raw)
    if not first:
        return None

    intent = first.group(1).strip()
    intent = _INTENT_ALIASES.get(intent, intent)
    allowed_intents = {
        "general", "web_search", "summarize_previous", "profile",
        "employee_details", "employee_search", "employee_role_search",
        "employee_count", "general_department_counts",
        "general_department_members", "employee_list", "manager",
        "leave_balance", "leave_types", "leave_history", "vacations_list",
        "leave_overlap", "meetings", "meeting_list", "employee_availability",
        "attendance_today", "attendance_history", "working_hours",
        "reporting_employees", "team_attendance", "tarasol_count",
        "tarasol_details", "employee_image", "hr_letter_recipients",
        "hr_letter_types", "policy_search", "policy_details", "news",
        "news_details", "announcements", "announcement_details", "mazaya",
        "mazaya_details",
    }
    if intent not in allowed_intents:
        return None

    operation_by_intent = {
        "general": "chat", "web_search": "search", "summarize_previous": "summarize",
        "profile": "details", "employee_details": "details", "employee_search": "search",
        "employee_role_search": "search", "employee_count": "calculate",
        "general_department_counts": "calculate", "general_department_members": "list",
        "employee_list": "list", "manager": "details", "leave_balance": "lookup",
        "leave_types": "list", "leave_history": "list", "vacations_list": "list",
        "leave_overlap": "list", "meetings": "list", "meeting_list": "list",
        "employee_availability": "lookup", "attendance_today": "lookup",
        "attendance_history": "lookup", "working_hours": "lookup",
        "reporting_employees": "list", "team_attendance": "lookup",
        "tarasol_count": "lookup", "tarasol_details": "list", "employee_image": "lookup",
        "hr_letter_recipients": "list", "hr_letter_types": "list",
        "policy_search": "search", "policy_details": "details", "news": "list",
        "news_details": "details", "announcements": "list",
        "announcement_details": "details", "mazaya": "list", "mazaya_details": "details",
    }

    data: dict[str, Any] = {
        "intent": intent,
        "operation": operation_by_intent.get(intent, "lookup"),
        "targets": [],
        "fields": [],
        "query": "",
        "role_query": "",
        "department_query": "",
        "period": "none",
        "start_date": "",
        "end_date": "",
        "ordinal": 0,
        "relation_depth": 1,
        "amount": None,
        "time_query": "",
        "confidence": 0.80,
        "clarification_reason": "",
    }

    employee_match = re.search(r"employee\s*\(\s*name\s*=\s*['\"]([^'\"]+)['\"]\s*\)", raw, re.I)
    if employee_match:
        data["targets"] = [{"kind": "employee", "name": employee_match.group(1).strip()}]
    else:
        target_match = re.search(r"\btarget\s*=\s*([A-Za-z_]+)", raw, re.I)
        if target_match:
            kind = _TARGET_KIND_ALIASES.get(target_match.group(1).strip(), target_match.group(1).strip())
            if kind in {"self", "manager_of_self", "focus", "previous_focus", "relation_subject", "relation_object", "none"}:
                data["targets"] = [] if kind == "none" else [{"kind": kind, "name": ""}]
        else:
            # Common compact second positional token: "profile, self, ..."
            positional = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*,\s*([A-Za-z_]+)\b", raw)
            if positional:
                kind = _TARGET_KIND_ALIASES.get(positional.group(1), positional.group(1))
                if kind in {"self", "manager_of_self", "focus", "previous_focus", "relation_subject", "relation_object"}:
                    data["targets"] = [{"kind": kind, "name": ""}]

    fields_match = re.search(r"\bfields\s*=\s*\[([^\]]*)\]", raw, re.I)
    if fields_match:
        fields: list[str] = []
        field_aliases = {
            "latest": "all", "latest_news": "all", "latest_updates": "all",
            "types": "all", "document_types": "all",
        }
        for item in fields_match.group(1).split(","):
            field = item.strip().strip("'\"")
            field = field_aliases.get(field, field)
            if field and field not in fields:
                fields.append(field)
        data["fields"] = fields

    for key in ("query", "role_query", "department_query", "period", "time_query"):
        match = re.search(rf"\b{key}\s*=\s*['\"]([^'\"]*)['\"]", raw, re.I)
        if match:
            data[key] = match.group(1).strip()

    depth_match = re.search(r"\brelation_depth\s*=\s*(\d+)", raw, re.I)
    if depth_match:
        data["relation_depth"] = min(5, max(1, int(depth_match.group(1))))

    return data


def _message_content(response: Any) -> Any:
    if not isinstance(response, dict):
        return ""
    message = response.get("message")
    if isinstance(message, dict):
        return message.get("content")
    return ""


def safe_state_for_parser(state: dict[str, Any]) -> dict[str, Any]:
    """Expose only structural dialogue context to the NLU model.

    Names, usernames, IDs and enterprise facts stay out of the parser context.
    The model only learns whether a verified focus exists and what kind of
    operation happened previously, so it can understand elliptical follow-ups.
    """
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    current = focus.get("current") if isinstance(focus.get("current"), dict) else {}
    previous = focus.get("previous") if isinstance(focus.get("previous"), dict) else {}
    relation = state.get("relation") if isinstance(state.get("relation"), dict) else {}
    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    last_request = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
    directory_context = state.get("directory_context") if isinstance(state.get("directory_context"), dict) else {}

    return {
        "has_focus": bool(current.get("username") or current.get("userid")),
        "has_previous_focus": bool(previous.get("username") or previous.get("userid")),
        "last_intent": str(last_request.get("intent") or ""),
        "last_fields": list(last_request.get("fields") or [])[:8],
        "last_period": str(last_request.get("period") or "none"),
        "last_had_time_query": bool(last_request.get("time_query")),
        "relation_predicate": str(relation.get("predicate") or ""),
        "last_list_intent": str(list_state.get("intent") or ""),
        "last_list_count": len(list_state.get("items") or []) if isinstance(list_state.get("items"), list) else 0,
        "has_general_department_context": bool(directory_context.get("general_department_key")),
        "last_directory_context_type": str(directory_context.get("type") or ""),
        "recent_dialogue": [
            {
                "role": str(turn.get("role") or ""),
                "content": str(turn.get("content") or "")[:1200],
            }
            for turn in (state.get("_recent_dialogue") or [])[-10:]
            if isinstance(turn, dict)
            and str(turn.get("role") or "") in {"user", "assistant"}
            and str(turn.get("content") or "").strip()
        ],
        "task_flow": {
            "kind": str((state.get("task_flow") or {}).get("kind") or ""),
            "stage": str((state.get("task_flow") or {}).get("stage") or ""),
            "period": str((state.get("task_flow") or {}).get("period") or "none"),
            "time_window": str((state.get("task_flow") or {}).get("time_window") or ""),
            "duration_minutes": int((state.get("task_flow") or {}).get("duration_minutes") or 0),
        } if isinstance(state.get("task_flow"), dict) else {},
    }



_INTENT_ALIASES = {
    "employee_total": "employee_count",
    "employees_count": "employee_count",
    "employee_count_total": "employee_count",
    "department_counts": "general_department_counts",
    "general_department_count": "general_department_counts",
    "department_count": "general_department_counts",
    "department_members": "general_department_members",
    "general_department_employees": "general_department_members",
    "department_employees": "general_department_members",
    "role_search": "employee_role_search",
    "employee_role": "employee_role_search",
    "employee_profile": "employee_details",
    "employee": "employee_search",
    "search_employee": "employee_search",
    "manager_of_self": "manager",
    "my_manager": "manager",
    "leave": "leave_balance",
    "availability": "employee_availability",
    "employee_calendar": "employee_availability",
    "employee_schedule": "meetings",
    "meeting": "meetings",
    "calendar": "meetings",
    "attendance": "attendance_today",
    "news_updates": "news",
    "news_feed": "news",
    "latest_news": "news",
    "employee_leave_types": "leave_types",
    "vacation_types": "leave_types",
    "leave_type_list": "leave_types",
    "hr_letter_type": "hr_letter_types",
    "hr_letters_types": "hr_letter_types",
    "hr_letter_recipient": "hr_letter_recipients",
    "benefits": "mazaya",
    "offers": "mazaya",
    "policy_list": "policy_search",
    "policies": "policy_search",
    "announcement_list": "announcements",
}

_OPERATION_ALIASES = {
    "count": "calculate",
    "get": "lookup",
    "find": "search",
    "show": "details",
    "detail": "details",
    "details_lookup": "details",
    "check": "lookup",
    "view": "details",
    "query": "lookup",
    "get_info": "details",
    "get_profile": "details",
}

_TARGET_KIND_ALIASES = {
    "current_user": "self",
    "me": "self",
    "current_employee": "self",
    "current_focus": "focus",
    "manager": "manager_of_self",
}


def _coerce_semantic_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Make harmless schema variations from the local LLM parseable.

    This does not decide the user's intent. It only normalizes common JSON label
    variants so one word such as operation='count' cannot make the entire
    semantic parse fail validation.
    """
    data = dict(value or {})

    intent = str(data.get("intent") or "").strip()
    if intent in _INTENT_ALIASES:
        data["intent"] = _INTENT_ALIASES[intent]

    operation = str(data.get("operation") or "").strip()
    if operation in _OPERATION_ALIASES:
        data["operation"] = _OPERATION_ALIASES[operation]

    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list):
        raw_targets = []
    targets: list[dict[str, str]] = []
    for item in raw_targets[:8]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "none").strip()
        kind = _TARGET_KIND_ALIASES.get(kind, kind)
        targets.append({
            "kind": kind,
            "name": str(item.get("name") or ""),
        })
    data["targets"] = targets

    if not isinstance(data.get("fields"), list):
        data["fields"] = []

    # The local model sometimes returns harmless schema-shape variants.
    # Coerce them instead of throwing away the entire semantic request.
    for key in (
        "query", "role_query", "department_query", "start_date",
        "end_date", "time_query", "clarification_reason",
    ):
        value = data.get(key)
        if value is None or isinstance(value, bool):
            data[key] = ""
        elif not isinstance(value, str):
            data[key] = str(value)

    if data.get("period") in (None, ""):
        data["period"] = "none"
    elif str(data.get("period") or "").strip() == "upcoming":
        data["period"] = "none"

    try:
        depth = int(data.get("relation_depth") or 1)
    except (TypeError, ValueError):
        depth = 1
    data["relation_depth"] = min(5, max(1, depth))

    try:
        ordinal = int(data.get("ordinal") or 0)
    except (TypeError, ValueError):
        ordinal = 0
    data["ordinal"] = min(100, max(0, ordinal))

    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    data["confidence"] = min(1.0, max(0.0, confidence))

    return data


def _extract_general_department_literal(message: str) -> str:
    """Extract a literal GENERAL DEPARTMENT name without trailing request text."""
    raw = " ".join(str(message or "").strip().split())
    match = re.search(
        r"(?:الإدارة|الادارة|الاداره|الإدراة|الادراة|إدارة|ادارة|اداره|إدراة|ادراة)\s+"
        r"(?:العامة|العامه|عامة|عامه)(?:\s+[^؟?!.،,]+)?",
        raw,
        flags=re.I,
    )
    if not match:
        return ""

    value = match.group(0).strip(" ؟?!.،,")
    value = re.split(
        r"\s+(?:و\s*)?(?:أسمائهم|اسمائهم|أسماؤهم|اسماؤهم|أسماءهم|اسماءهم|أسماء|اسماء|أسامي|اسامي|مين|من\s+هم)\b",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]
    return value.strip()



def _extract_department_literal(message: str) -> str:
    """Extract any explicit organisational unit name from the current message.

    Unlike the older helper, this intentionally supports both general and
    lower-level departments because SearchEmployee exposes the employee's
    actual department. The directory layer is responsible for verifying the
    name against API data; this function only extracts the literal phrase.
    """
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return ""

    # Prefer the longest explicit Arabic organisational phrase. Stop before
    # request-tail words instead of swallowing "بكرة", "وأسمائهم", etc.
    match = re.search(
        r"((?:الإدارة|الادارة|الاداره|إدارة|ادارة|اداره)"
        r"(?:\s+(?:العامة|العامه|عامة|عامه))?\s+"
        r".+?)(?="
        r"\s+(?:وأسمائهم|واسمائهم|أسمائهم|اسمائهم|أسماءهم|اسماءهم|أسماء|اسماء|"
        r"ومناصبهم|ومسمياتهم|كم\s+عدد|كم\s+موظف|بكرة|بكره|اليوم|غدا|غداً)\b"
        r"|[؟?!.،,]|$)",
        raw,
        flags=re.I,
    )
    if match:
        value = match.group(1).strip(" ؟?!.،,")
        # Remove list/request words accidentally attached before the unit.
        value = re.sub(
            r"^(?:في|داخل|ضمن|حق|حقت|موظفين|موظفي|موظفو|قائمة\s+(?:ب)?موظفين)\s+",
            "",
            value,
            flags=re.I,
        ).strip()
        return value

    # English department names are left as a verified search phrase when the
    # message explicitly uses Department/Division/Office/Sector.
    match = re.search(
        r"((?:[A-Za-z][A-Za-z&/\- ]{1,80})\s+(?:department|division|office|sector))"
        r"(?=\s+(?:employees?|staff|tomorrow|today)\b|[?!.]|$)",
        raw,
        flags=re.I,
    )
    return match.group(1).strip() if match else ""



_ARABIC_DIGIT_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _ascii_digits(value: str) -> str:
    return str(value or "").translate(_ARABIC_DIGIT_TRANS)


def _has_explicit_period_cue(message: str) -> bool:
    raw = _ascii_digits(str(message or ""))
    norm = _normalize_for_semantic_guards(raw)
    return bool(
        re.search(
            r"(?:اليوم|بكره|بكرة|غدا|غداً|امس|أمس|هذا\s+الاسبوع|هالاسبوع|"
            r"الاسبوع\s+الماضي|الأسبوع\s+الماضي|هذا\s+الشهر|الشهر\s+الماضي|"
            r"today|tomorrow|yesterday|this\s+week|last\s+week|this\s+month|last\s+month)",
            norm, re.I,
        )
        or re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", raw)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]20\d{2}\b", raw)
    )


def _availability_period(message: str) -> tuple[str, str, str]:
    raw = _ascii_digits(str(message or ""))
    norm = _normalize_for_semantic_guards(raw)
    if re.search(r"\b(?:بكره|بكرة|غدا|غداً|tomorrow)\b", norm, re.I):
        return "tomorrow", "", ""
    if re.search(r"\b(?:امس|أمس|yesterday)\b", raw, re.I):
        return "yesterday", "", ""
    if re.search(
        r"(?:الاسبوع\s+(?:الماضي|اللي\s+راح|الي\s+راح|اللي\s+فات|الي\s+فات)|"
        r"الأسبوع\s+(?:الماضي|اللي\s+راح|الي\s+راح|اللي\s+فات|الي\s+فات)|last\s+week)",
        raw,
        re.I,
    ):
        return "last_week", "", ""
    if re.search(r"\b(?:هذا الاسبوع|هالاسبوع|الأسبوع الحالي|الاسبوع الحالي|this week)\b", norm, re.I):
        return "this_week", "", ""
    if re.search(r"\b(?:اليوم|today)\b", norm, re.I):
        return "today", "", ""
    iso = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
    if iso:
        try:
            date = datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))).date().isoformat()
            return "custom", date, date
        except ValueError:
            pass
    dmy = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", raw)
    if dmy:
        try:
            date = datetime(int(dmy.group(3)), int(dmy.group(2)), int(dmy.group(1))).date().isoformat()
            return "custom", date, date
        except ValueError:
            pass
    return "today", "", ""


def _clock_value(hour: str, minute: str | None, marker: str | None) -> str:
    try:
        h = int(hour)
        m = int(minute or 0)
    except ValueError:
        return ""
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return ""
    mark = _normalize_for_semantic_guards(marker or "").replace(".", "")
    pm = mark in {"م", "مساء", "مساءا", "pm", "p m"} or "مساء" in mark or "pm" in mark
    am = mark in {"ص", "صباح", "صباحا", "am", "a m"} or "صباح" in mark or "am" in mark
    if pm and h < 12:
        h += 12
    elif am and h == 12:
        h = 0
    elif not pm and not am and 1 <= h <= 7:
        # Workplace shorthand: "الساعة 2" normally means 2 PM.
        h += 12
    return f"{h:02d}:{m:02d}"


def _extract_time_query(message: str) -> str:
    raw = _ascii_digits(str(message or ""))
    marker = r"(?:ص|م|صباح(?:ا|ًا)?|مساء(?:ا|ً)?|am|pm|a\.m\.|p\.m\.)?"
    range_re = re.compile(
        rf"(?:من\s+|from\s+)?(?:الساعة|الساعه|at)?\s*(\d{{1,2}})(?::(\d{{1,2}}))?\s*({marker})\s*"
        rf"(?:الى|إلى|لـ|حتى|to|[-–—])\s*(?:الساعة|الساعه)?\s*(\d{{1,2}})(?::(\d{{1,2}}))?\s*({marker})",
        re.I,
    )
    match = range_re.search(raw)
    if match:
        start = _clock_value(match.group(1), match.group(2), match.group(3))
        end = _clock_value(match.group(4), match.group(5), match.group(6) or match.group(3))
        if start and end:
            return f"{start}-{end}"
    single_patterns = [
        re.compile(rf"(?:الساعة|الساعه)\s*(\d{{1,2}})(?::(\d{{1,2}}))?\s*({marker})", re.I),
        re.compile(rf"\bat\s+(\d{{1,2}})(?::(\d{{1,2}}))?\s*({marker})", re.I),
        # Marked standalone clock time, useful for contextual English follow-ups
        # such as "what about 3 PM?". The AM/PM marker is mandatory here so a
        # random number is never treated as a time.
        re.compile(r"\b(\d{1,2})(?::(\d{1,2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", re.I),
    ]
    for pattern in single_patterns:
        match = pattern.search(raw)
        if match:
            return _clock_value(match.group(1), match.group(2), match.group(3))
    return ""


def _extract_leave_amount(message: str) -> float | None:
    raw = _ascii_digits(str(message or ""))
    norm = _normalize_for_semantic_guards(raw)
    if re.search(r"\b(?:يومين|يومان)\b", norm):
        return 2.0
    numeric = re.search(r"(\d+(?:\.\d+)?)\s*(?:يوم|ايام|أيام|day|days)\b", raw, re.I)
    if numeric:
        try:
            return float(numeric.group(1))
        except ValueError:
            pass
    words = {
        "يوم واحد": 1, "واحد يوم": 1, "ثلاث ايام": 3, "ثلاثه ايام": 3,
        "اربع ايام": 4, "اربعه ايام": 4, "خمس ايام": 5, "خمسه ايام": 5,
        "ست ايام": 6, "سته ايام": 6, "سبع ايام": 7, "سبعه ايام": 7,
        "ثمان ايام": 8, "ثمانيه ايام": 8, "تسع ايام": 9, "تسعه ايام": 9,
        "عشر ايام": 10, "عشره ايام": 10,
        "one day": 1, "two days": 2, "three days": 3, "four days": 4,
        "five days": 5, "six days": 6, "seven days": 7, "eight days": 8,
        "nine days": 9, "ten days": 10,
    }
    for phrase, amount in words.items():
        if phrase in norm:
            return float(amount)

    # "إذا أخذت إجازة يوم الأحد كم يبقى؟" is one requested leave day even
    # though the user expressed it as a weekday rather than "يوم واحد".
    # Do not infer a multi-day span here; only one explicitly named weekday.
    weekday_pattern = (
        r"(?:الاحد|الأحد|الاثنين|الإثنين|الثلاثاء|الاربعاء|الأربعاء|"
        r"الخميس|الجمعه|الجمعة|السبت|sunday|monday|tuesday|wednesday|"
        r"thursday|friday|saturday)"
    )
    leave_cue = bool(re.search(r"\b(?:اجازه|اجازة|إجازة|vacation|leave|day off)\b", norm, re.I))
    weekdays = re.findall(weekday_pattern, raw, flags=re.I)
    range_cue = bool(re.search(r"(?:\bمن\b|\bfrom\b).+(?:\bالى\b|\bإلى\b|\bto\b)", raw, re.I))
    if leave_cue and len(weekdays) == 1 and not range_cue:
        return 1.0
    return None


def _clean_availability_person(value: str) -> str:
    name = " ".join(str(value or "").strip().split())
    if not name:
        return ""
    # Remove conversational/date/request prefixes that are not part of a name.
    name = re.sub(
        r"^(?:طيب|تمام|اوكي|أوكي|ممكن|لو سمحت|بليز|متى|when)\s+",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(
        r"^(?:(?:بكره|بكرة|غدا|غداً|اليوم|tomorrow|today)\s+)+",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(
        r"^(?:ابي|أبي|ابغى|أبغى)\s+(?:اعرف|أعرف|اشوف|أشوف)?\s*",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"^(?:اجتمع\s+مع|meeting\s+with)\s+", "", name, flags=re.I)
    name = re.sub(r"\s+(?:متى|when)\s*$", "", name, flags=re.I)
    name = re.sub(r"^(?:هل|هو|هي|is)\s+", "", name, flags=re.I)
    # Run interrogative cleanup again after removing a leading date.
    # Example: "اليوم متى فاضي؟" used to become the fake employee name "متى".
    name = re.sub(r"^(?:متى|when)\s*", "", name, flags=re.I)
    name = name.strip(" ؟?!.,،")

    # A pure date/time/interrogative word is never an employee name. This matters for
    # contextual follow-ups such as "اليوم فاضي؟" after a verified manager.
    temporal_only = _normalize_for_semantic_guards(name)
    if temporal_only in {
        "اليوم", "بكره", "بكرة", "غدا", "غداً", "امس", "أمس",
        "today", "tomorrow", "yesterday", "متى", "when",
    }:
        return ""
    return name


def _extract_availability_person(message: str) -> str:
    raw = " ".join(str(message or "").strip().split())
    patterns = [
        # Availability question: optional filler/day prefix, then the person.
        r"^(?:(?:طيب|تمام|ممكن)\s+)?(?:(?:بكره|بكرة|غدا|غداً|اليوم|tomorrow|today)\s+)?(.+?)\s+(?:فاضي|فاضية|متاح|متاحة|مشغول|مشغولة)(?=\s|[؟?!.,]|$)",
        r"^(?:(?:طيب|تمام|ممكن)\s+)?(?:(?:بكره|بكرة|غدا|غداً|اليوم|tomorrow|today)\s+)?(.+?)\s+(?:free|available|busy)(?=\s|[؟?!.,]|$)",
        r"^(?:(?:طيب|تمام)\s+)?(?:ابي\s+اعرف\s+اذا\s+|أبي\s+أعرف\s+إذا\s+)?(.+?)\s+(?:عنده|عندها)\s+(?:اجتماع|اجتماعات|موعد|مواعيد|meeting|meetings)",
        # "اجتماعات أسامه بكرة" / "جدول اجتماعات شاهه" / "جدول شاهه".
        r"^(?:ابي\s+|أبي\s+|ابغى\s+|أبغى\s+)?(?:اجتماعات|مواعيد|جدول(?:\s+اجتماعات)?|ميتنق|ميتنقات|meetings|calendar|schedule)\s+(?:حق\s+|حقت\s+|for\s+|of\s+)?(.+?)(?=\s+(?:اليوم|بكره|بكرة|غدا|غداً|الساعة|الساعه|يوم|today|tomorrow|at|on|حقت|حق)\b|[؟?!.,]|$)",
        r"^is\s+(.+?)\s+(?:free|available|busy)(?=\s|[?!.]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.I)
        if not match:
            continue
        name = _clean_availability_person(str(match.group(1) or ""))
        norm = _normalize_for_semantic_guards(name)
        if name and norm not in {"انا", "اني", "me", "i", "هذا الشخص", "هالشخص", "مديري", "my manager"}:
            return name
    return ""


def _availability_mode(message: str) -> str:
    norm = _normalize_for_semantic_guards(_ascii_digits(str(message or "")))
    if re.search(r"(?:متى\s+(?:هو\s+|هي\s+)?(?:فاضي|فاضية|متاح|متاحة)|(?:فاضي|فاضية|متاح|متاحة)|\bfree\b|\bavailable\b)", norm, re.I):
        return "free"
    if re.search(r"(?:اجتماعات|مواعيد|جدول|ميتنق|ميتنقات|\bmeetings?\b|\bcalendar\b|\bschedule\b|مشغول|مشغولة|\bbusy\b)", norm, re.I):
        return "busy"
    return "busy"

def _repair_required_feature_grammar(
    request: ParsedRequest,
    message: str,
    state: dict[str, Any] | None,
) -> ParsedRequest:
    """Deterministic safety net for employee totals, roles and GENERAL DEPARTMENT queries."""
    req = request.model_copy(deep=True)
    raw = " ".join(str(message or "").strip().split())
    norm = _normalize_for_semantic_guards(raw).strip(" ؟?!.،,")
    has_focus = _has_verified_focus(state)

    employee_word_pattern = r"(?:الموظف(?:ين|ون)?|موظف(?:ين|ون)?)"

    # Context-aware short follow-ups. We reuse only structural state plus the
    # already-verified person focus; no employee fact is inferred by the model.
    last_request = state.get("last_request") if isinstance(state, dict) and isinstance(state.get("last_request"), dict) else {}
    last_intent = str(last_request.get("intent") or "")
    followup_time = _extract_time_query(raw)
    explicit_period = bool(re.search(
        r"\b(?:اليوم|بكره|بكرة|غدا|غداً|هذا\s+الاسبوع|هالاسبوع|today|tomorrow|this\s+week)\b|\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]20\d{2}\b",
        _ascii_digits(raw), re.I,
    ))
    explicit_availability_person = _extract_availability_person(raw)
    manager_availability_ref = bool(re.search(r"\b(?:مديري|المدير\s+حقي|المدير\s+المباشر\s+حقي|my\s+manager)\b", norm, re.I))
    availability_followup_shape = bool(
        followup_time
        or explicit_period
        or re.search(r"\b(?:فاضي|فاضية|متاح|متاحة|مشغول|مشغولة|free|available|busy)\b", raw, re.I)
    )
    meeting_ctx = state.get("meeting_context") if isinstance(state, dict) and isinstance(state.get("meeting_context"), dict) else {}
    meeting_employee = meeting_ctx.get("employee") if isinstance(meeting_ctx.get("employee"), dict) else {}
    meeting_name = str(meeting_employee.get("name_ar") or meeting_employee.get("name") or meeting_employee.get("name_en") or "").strip()

    # ------------------------------------------------------------------
    # v68 critical grammar: these high-frequency enterprise/self follow-ups
    # must never depend on the LLM. Keep this BEFORE generic employee totals,
    # generic name extraction, and open-ended semantic parsing.
    # ------------------------------------------------------------------
    critical_norm = norm.strip(" ؟?!.،,")
    critical_norm = re.sub(r"^(?:(?:طيب|تمام|اوكي|أوكي|please|pls)\s+)+", "", critical_norm, flags=re.I).strip()

    # Strip harmless chat/bullet prefixes for command routing only.  The original
    # CURRENT_MESSAGE is still used for name grounding afterwards.
    command_raw = re.sub(r"^[\s\-–—•·*]+", "", raw).strip()
    command_raw = re.sub(
        r"^(?:(?:طيب|تمام|اوكي|أوكي|الحين|الآن|الان|ممكن|لو\s+سمحت|please|pls)\s+)+",
        "", command_raw, flags=re.I,
    ).strip()
    command_norm = _normalize_for_semantic_guards(command_raw).strip(" ؟?!.،,")

    # ------------------------------------------------------------------
    # Employee profile image. Keep this deterministic so common Arabic
    # phrasings such as "ابي صورتي الشخصية" never fall through to general.
    # ------------------------------------------------------------------
    self_image_norm = re.sub(
        r"^(?:(?:ابي|أبي|ابغى|أبغى|جيب|هات|اعرض|أعرض|ورني|ارني|أرني)\s+)+",
        "", command_norm, flags=re.I,
    ).strip()
    if re.fullmatch(
        r"(?:صورتي(?:\s+الشخصيه|\s+الشخصية|\s+الوظيفيه|\s+الوظيفية)?|"
        r"(?:الصوره|الصورة)\s+(?:الشخصيه|الشخصية|الوظيفيه|الوظيفية)\s+(?:حقتي|لي)|"
        r"(?:الصوره|الصورة)\s+حقتي|"
        r"(?:صوره|صورة)\s+ملفي|"
        r"my\s+(?:profile\s+)?(?:photo|image|picture))",
        self_image_norm, re.I,
    ):
        req.intent = "employee_image"
        req.operation = "lookup"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["all"]
        req.query = "profile:self_image"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Named/focused employee image. "صورته" / "صورتها" follows the verified
    # conversational focus; explicit names are resolved through the directory.
    if has_focus and re.fullmatch(
        r"(?:(?:ابي|أبي|ابغى|أبغى|جيب|هات|اعرض|أعرض|ورني|ارني|أرني)\s+)?"
        r"(?:صورته|صورتها|الصوره\s+حقته|الصورة\s+حقته|الصوره\s+حقتها|الصورة\s+حقتها)",
        command_norm, re.I,
    ):
        req.intent = "employee_image"
        req.operation = "lookup"
        req.targets = [TargetRef(kind="focus", name="")]
        req.fields = ["all"]
        req.query = "profile:focus_image"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    named_image = re.fullmatch(
        r"(?:(?:ابي|أبي|ابغى|أبغى|جيب|هات|اعرض|أعرض|ورني|ارني|أرني)\s+)?"
        r"(?:صوره|صورة|الصوره|الصورة)\s+(?:الموظف\s+)?(.+)",
        command_raw.rstrip(" ؟?!.،,"), re.I,
    )
    if named_image:
        employee_name = " ".join(str(named_image.group(1) or "").split()).strip()
        if employee_name and employee_name not in {"الشخصيه", "الشخصية", "الوظيفيه", "الوظيفية"}:
            req.intent = "employee_image"
            req.operation = "lookup"
            req.targets = [TargetRef(kind="employee", name=employee_name)]
            req.fields = ["all"]
            req.query = "profile:employee_image"
            req.clarification_reason = ""
            req.confidence = 0.99
            return req

    # Explicit writing/advice/explanation requests are general conversation even
    # when they mention enterprise nouns such as a manager or email. They must
    # never be converted into profile/contact lookups.
    if re.search(
        r"(?:اكتب(?:\s+لي)?|صيغ(?:\s+لي)?|صياغ|اعد\s+صياغ|أعد\s+صياغ|ترجم|اشرح|وضح|لخص|اختصر|وش\s+رايك|ايش\s+رايك|رأيك|رايك|انصحني|نصيحه|نصيحة|كيف\s+اتعامل|وش\s+اسوي|ايش\s+اسوي|كيف\s+ارد|وش\s+ارد|write|draft|rewrite|rephrase|translate|explain|summarize|brainstorm|advice)",
        command_norm,
        re.I,
    ):
        req.intent = "general"
        req.operation = "chat"
        req.targets = []
        req.fields = []
        req.query = raw
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Natural recognition variants should preserve the conversational meaning.
    if re.fullmatch(
        r"(?:(?:لسا|لسه|مازلت|ما زلت|بعدك|بعدك\s+)?\s*)?(?:تعرفني|تذكرني)|(?:do\s+you\s+still\s+(?:know|remember)\s+me)",
        command_norm,
        re.I,
    ):
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name", "job_title", "department"]
        req.query = "profile:recognition"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Signed-in user's direct manager. These common conversational variants must
    # never depend on Qwen formatting and must never fall into general chat, where
    # an LLM could invent an internal manager name.
    manager_self_only = bool(re.fullmatch(
        r"(?:(?:هل|طيب|تمام|اوكي|أوكي)\s+)?(?:"
        r"(?:تعرف|تعرفين|تعرفون)(?:\s+مين)?\s+(?:مديري|المدير\s+حقي|المدير\s+المباشر\s+حقي)"
        r"|(?:ابي|أبي|ابغى|أبغى|عطني|اعطني|ورني|اعرض\s+لي)\s+(?:مديري|المدير\s+حقي|المدير\s+المباشر\s+حقي)"
        r"|(?:مين|من)(?:\s+هو)?\s+(?:مديري|مديري\s+المباشر|المدير\s+حقي|المدير\s+المباشر\s+حقي)"
        r"|(?:مديري|المدير\s+حقي|المدير\s+المباشر\s+حقي)"
        r"|(?:do\s+you\s+know\s+my\s+manager|who\s+is\s+my\s+manager|show\s+me\s+my\s+manager|my\s+manager)"
        r")",
        command_norm,
        re.I,
    ))
    if manager_self_only:
        req.intent = "manager"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name"]
        req.query = "manager:self"
        req.relation_depth = 1
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Tarasol transactions: explicit self vs another employee is security-significant.
    if re.search(r"(?:تراسل|معامل(?:تي|اتي|ه|ة|ات)?|المعامل(?:ه|ة|ات)?|transaction)", command_norm, re.I):
        if re.search(r"(?:مديري|المدير\s+حقي|my\s+manager)", command_norm, re.I):
            req.intent = "tarasol_details"
            req.operation = "details"
            req.targets = [TargetRef(kind="manager_of_self", name="")]
            req.fields = ["all"]
            req.query = "tarasol:other_private"
            req.clarification_reason = ""
            req.confidence = 0.99
            return req
        if re.search(r"(?:تفاصيل|معاملتي|معاملاتي|حقتي|حقي|عندي|my\s+transactions?|details)", command_norm, re.I):
            req.intent = "tarasol_details"
            req.operation = "details"
            req.targets = [TargetRef(kind="self", name="")]
            req.fields = ["all"]
            req.query = "tarasol:self_details"
            req.clarification_reason = ""
            req.confidence = 0.99
            return req
        if re.search(r"(?:كم|عدد|count|how\s+many)", command_norm, re.I):
            req.intent = "tarasol_count"
            req.operation = "lookup"
            req.targets = [TargetRef(kind="self", name="")]
            req.fields = ["count"]
            req.clarification_reason = ""
            req.confidence = 0.99
            return req

    # Department directory meta-queries. "كم إدارة؟" is NEVER a person name.
    # Treat singular/plural and common typing variants of "department" as an
    # organisation-level question when the wording explicitly asks for a count
    # or the whole list. This must run before employee-name extraction.
    unit_word = r"(?:ال)?(?:اداره|ادارات)"
    all_units_count = bool(re.fullmatch(
        rf"(?:(?:كم|ما\s+عدد|عدد)\s+(?:عدد\s+)?{unit_word}(?:\s+(?:في|بال|ب)\s*(?:الهيئه|الهييه))?|how\s+many\s+departments?(?:\s+are\s+there)?)",
        command_norm,
        re.I,
    ))
    all_units_list = bool(
        re.fullmatch(r"(?:كل|جميع)\s+(?:ال)?(?:اداره|ادارات)", command_norm, re.I)
        or (
            re.search(r"(?:قائمه|ليست|كل|جميع|اعرض|ابي|ابغى|اعطني|عطني|هات)", command_norm, re.I)
            and re.search(r"(?:^|\s)(?:ب|با)?\s*(?:ال)?ادارات(?:\s|$)", command_norm, re.I)
        )
        or bool(re.fullmatch(r"all\s+departments?", command_norm, re.I))
    )
    last_req = state.get("last_request") if isinstance(state, dict) and isinstance(state.get("last_request"), dict) else {}
    last_query = str(last_req.get("query") or "")
    all_units_followup = bool(
        last_query.startswith("org_units:")
        and re.fullmatch(r"(?:كلها|جميعها|الكل|كلهم|all|all\s+of\s+them)", command_norm, re.I)
    )
    if all_units_count:
        req.intent = "general_department_counts"
        req.operation = "calculate"
        req.targets = []
        req.fields = ["count"]
        req.query = "org_units:count"
        req.department_query = ""
        req.clarification_reason = ""
        req.confidence = 0.99
        return req
    if all_units_list or all_units_followup:
        req.intent = "general_department_counts"
        req.operation = "list"
        req.targets = []
        req.fields = ["name", "count"]
        req.query = "org_units:list"
        req.department_query = ""
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Department members by official name OR common acronym (IT/HR/PMO...).
    dept_alias = _extract_department_alias(command_raw)
    dept_literal = _extract_department_literal(command_raw)
    employee_list_cue = bool(re.search(r"(?:موظف|موظفين|موظفي|موظفو|staff|employees?)", command_norm, re.I))
    if employee_list_cue and (dept_alias or dept_literal):
        req.intent = "general_department_members"
        req.operation = "list"
        req.targets = []
        req.fields = ["name", "job_title"]
        req.department_query = dept_literal or dept_alias
        req.query = "department:members"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # After asking for department members, a bare department/acronym is a safe
    # structural continuation (e.g. "إدارة المشاريع" after "موظفين PMO").
    if str(last_req.get("intent") or "") == "general_department_members" and (dept_alias or dept_literal):
        req.intent = "general_department_members"
        req.operation = "list"
        req.targets = []
        req.fields = ["name", "job_title"]
        req.department_query = dept_literal or dept_alias
        req.query = "department:members_followup"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Supported list/service intents that should not depend on Qwen formatting.
    if re.search(r"(?:^|\s)(?:وش|ما|ايش|إيش)?\s*(?:انواع|أنواع)\s+(?:ال)?(?:اجازات|إجازات|اجازه|إجازة|leave\s+types?|vacation\s+types?)(?=\s|[؟?!.،,]|$)", command_raw, re.I):
        req.intent = "leave_types"
        req.operation = "list"
        req.targets = []
        req.fields = ["all"]
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    if re.search(r"(?:انواع|أنواع|قائمه|قائمة|وش)\s+(?:خطابات|خطاب)\s+(?:الموارد\s+البشريه|الموارد\s+البشرية|hr)|(?:hr\s+letter\s+types?)", command_raw, re.I):
        req.intent = "hr_letter_types"
        req.operation = "list"
        req.targets = []
        req.fields = ["all"]
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    if re.search(r"(?:لمين|لمن|الى\s+مين|إلى\s+مين|جهات?|مستلمين|recipients?)", command_raw, re.I) and re.search(r"(?:خطاب|خطابات|hr\s+letter)", command_raw, re.I):
        req.intent = "hr_letter_recipients"
        req.operation = "list"
        req.targets = []
        req.fields = ["all"]
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    if re.search(r"(?:اخر|آخر|احدث|أحدث|جديد|latest|recent)", command_raw, re.I) and re.search(r"(?:اخبار|أخبار|news)", command_raw, re.I):
        req.intent = "news"
        req.operation = "list"
        req.targets = []
        req.fields = ["all"]
        req.query = ""
        req.clarification_reason = ""
        req.confidence = 0.99
        return req
    if re.fullmatch(r"(?:وش\s+)?(?:ال)?(?:اخبار|أخبار)(?:\s+الهيئه|\s+الهيئة)?", command_norm, re.I):
        req.intent = "news"
        req.operation = "list"
        req.targets = []
        req.fields = ["all"]
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    if re.search(r"(?:اعلانات|إعلانات|الاعلانات|الإعلانات|تعاميم|التعاميم|announcements?)", command_raw, re.I):
        req.intent = "announcements"
        req.operation = "list"
        req.targets = []
        req.fields = ["all"]
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Private leave balance about another person is still parsed correctly so
    # the authorization layer can return a privacy response instead of a false
    # "I did not understand" message.
    if re.search(r"(?:رصيد).*(?:اجاز|إجاز|leave|vacation)", command_raw, re.I) and re.search(r"(?:مديري|مدير\s+حقي|my\s+manager)", command_raw, re.I):
        req.intent = "leave_balance"
        req.operation = "lookup"
        req.targets = [TargetRef(kind="manager_of_self", name="")]
        req.fields = ["balance"]
        req.clarification_reason = ""
        req.confidence = 0.99
        return req
    if re.search(r"(?:مديري|مدير\s+حقي|my\s+manager)", command_raw, re.I) and re.search(r"(?:رصيد).*(?:اجاز|إجاز|leave|vacation)|(?:اجازات|إجازات|leave).*(?:رصيد)", command_raw, re.I):
        req.intent = "leave_balance"
        req.operation = "lookup"
        req.targets = [TargetRef(kind="manager_of_self", name="")]
        req.fields = ["balance"]
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Direct named-person manager request: "مين مدير عماد الهاجري؟"
    manager_person_match = re.fullmatch(
        r"(?:(?:مين|من)(?:\s+هو|\s+هي)?\s+)?(?:ال)?مدير(?:\s+المباشر)?\s+(?:ل|لـ)?\s*(.+)",
        command_raw.rstrip("؟?!.،,"),
        flags=re.I,
    )
    if manager_person_match:
        person_text = " ".join(str(manager_person_match.group(1) or "").split()).strip()
        if person_text and not _looks_like_organizational_role(person_text):
            req.intent = "manager"
            req.operation = "details"
            req.targets = [TargetRef(kind="employee", name=person_text)]
            req.fields = ["name"]
            req.relation_depth = 1
            req.clarification_reason = ""
            req.confidence = 0.99
            return req

    # Explicit employee information with or without "عن":
    # "معلومات عماد", "ابي معلومات عن عماد", "تفاصيله".
    if has_focus and (
        re.fullmatch(
            r"(?:(?:ابي|ابغى|عطني|اعطني|show\s+me)\s+)?(?:معلومات|بيانات|تفاصيل)(?:ه|ها|\s+(?:عنه|عنها|حقه|حقها))",
            command_norm, flags=re.I,
        )
        or re.fullmatch(
            r"(?:(?:ابي|ابغى|عطني|اعطني)\s+)?(?:كل\s+)?(?:معلوماته|معلوماتها|بياناته|بياناتها|تفاصيله|تفاصيلها)(?:\s+(?:اللي|الي)\s+(?:تقدر|تقدرين|تقدرون)\s+عليها)?",
            command_norm, flags=re.I,
        )
    ):
        req.intent = "employee_details"
        req.operation = "details"
        req.targets = [TargetRef(kind="focus", name="")]
        req.fields = ["name", "job_title", "department", "email", "mobile", "extension"]
        req.query = "employee:focus_summary"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req


    # ------------------------------------------------------------------
    # Verified content-list detail follow-up.
    # Examples:
    #   "تفاصيل 5"
    #   "ابي تفاصيل 5 ميزه"
    #   "تفاصيل الخامس"
    #   "ابي تفاصيل العرض رقم 3"
    #
    # This MUST run before generic employee-details parsing so a phrase
    # like "5 ميزه" is never interpreted as an employee name.
    # ------------------------------------------------------------------
    content_list_state = (
        state.get("list_state")
        if isinstance(state, dict)
        and isinstance(state.get("list_state"), dict)
        else {}
    )

    content_list_intent = str(
        content_list_state.get("intent") or ""
    ).strip()

    content_detail_map = {
        "news": "news_details",
        "announcements": "announcement_details",
        "mazaya": "mazaya_details",
        "policy_search": "policy_details",
    }

    if content_list_intent in content_detail_map:
        asks_details = bool(
            re.search(
                r"(?:تفاصيل|تفصيل|اعرف\s+عن|أعرف\s+عن|معلومات|details?|more\s+info)",
                command_norm,
                re.I,
            )
        )

        ordinal = 0
        ordinal_patterns = (
            (1, r"(?:الاول|الأول|اول|أول|first)"),
            (2, r"(?:الثاني|ثاني|second)"),
            (3, r"(?:الثالث|ثالث|third)"),
            (4, r"(?:الرابع|رابع|fourth)"),
            (5, r"(?:الخامس|خامس|fifth)"),
            (6, r"(?:السادس|سادس|sixth)"),
            (7, r"(?:السابع|سابع|seventh)"),
            (8, r"(?:الثامن|ثامن|eighth)"),
            (9, r"(?:التاسع|تاسع|ninth)"),
            (10, r"(?:العاشر|عاشر|tenth)"),
        )

        for number, pattern in ordinal_patterns:
            if re.search(pattern, command_norm, re.I):
                ordinal = number
                break

        if not ordinal:
            number_match = re.search(
                r"(?:رقم\s*|number\s*|#\s*)?(\d{1,2})",
                command_norm,
                re.I,
            )
            if number_match:
                ordinal = int(number_match.group(1))

        items = [
            row
            for row in (content_list_state.get("items") or [])
            if isinstance(row, dict)
        ]

        if asks_details and ordinal > 0 and ordinal <= len(items):
            req.intent = content_detail_map[content_list_intent]
            req.operation = "details"
            req.targets = []
            req.fields = ["all"]
            req.ordinal = ordinal
            req.query = ""
            req.clarification_reason = ""
            req.confidence = 0.99
            return req

    direct_info_match = re.fullmatch(
        r"(?:(?:ابي|ابغى|عطني|اعطني|show\s+me)\s+)?(?:معلومات|بيانات|تفاصيل|information|details)(?:\s+(?:عن|about))?\s+(.+)",
        command_raw.rstrip("؟?!.،,"),
        flags=re.I,
    )
    if direct_info_match:
        person_text = " ".join(str(direct_info_match.group(1) or "").split()).strip()
        if person_text and not _looks_like_organizational_role(person_text):
            req.intent = "employee_details"
            req.operation = "details"
            req.targets = [TargetRef(kind="employee", name=person_text)]
            req.fields = ["name", "job_title", "department", "email", "mobile", "extension"]
            req.query = "employee:summary"
            req.clarification_reason = ""
            req.confidence = 0.99
            return req

    # Direct employee search with normal conversational prefixes/bullets.
    command_search = re.fullmatch(
        r"(?:(?:ابي|أبي|ابغى|أبغى)\s+)?(?:ابحث|أبحث|دور|دوّر|دورلي|دور\s+لي|ابحث\s+لي|أبحث\s+لي|find|search(?:\s+for)?)\s+(?:لي\s+)?(?:عن\s+)?(.+)",
        command_raw.rstrip("؟?!.،,"),
        flags=re.I,
    )
    if command_search:
        employee_query = " ".join(str(command_search.group(1) or "").split()).strip()
        if employee_query and not _looks_like_organizational_role(employee_query):
            req.intent = "employee_search"
            req.operation = "search"
            req.targets = []
            req.query = employee_query
            req.fields = ["name", "job_title", "department"]
            req.clarification_reason = ""
            req.confidence = 0.99
            return req

    # Explicit self references always mean the authenticated employee, even
    # when conversational focus currently points at a manager/colleague.
    if critical_norm in {
        "انا", "اني", "مين انا", "من انا", "who am i", "me",
    }:
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name", "job_title", "department"]
        req.query = "profile:identity"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    if critical_norm in {
        "معلوماتي", "ابي معلوماتي", "ابغى معلوماتي", "عطني معلوماتي", "اعطني معلوماتي",
        "بياناتي", "ابي بياناتي", "ابغى بياناتي", "my profile", "my information", "my info",
        "show my profile", "show me my profile", "show my information", "show my info",
    }:
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name", "job_title", "department", "email", "mobile", "extension"]
        req.query = "profile:summary"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    critical_self_fields: list[str] = []
    for field_name, pattern in (
        ("name", r"(?:وش\s+اسمي|ايش\s+اسمي|إيش\s+اسمي|ما\s+اسمي|اسمي\s+وش|اسمي\s+ايش|what(?:'s|\s+is)\s+my\s+name|my\s+name)"),
        ("email", r"(?:ايميلي|ايميل\s*ي|بريدي(?:\s+الالكتروني)?|my\s+email)"),
        ("mobile", r"(?:رقم\s+جوالي|جوالي|رقم\s+موبايلي|موبايلي|my\s+(?:mobile|phone)(?:\s+number)?)"),
        ("extension", r"(?:تحويلتي|رقم\s+تحويلتي|my\s+extension)"),
        ("job_title", r"(?:مسماي|منصبي|مسمى\s+وظيفتي|my\s+(?:job\s+)?title)"),
        ("department", r"(?:ادارتي|الاداره\s+حقتي|my\s+department)"),
        ("username", r"(?:يوزري|اسم\s+المستخدم\s+حقي|my\s+username)"),
    ):
        if re.search(pattern, critical_norm, re.I) and field_name not in critical_self_fields:
            critical_self_fields.append(field_name)
    if critical_self_fields:
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = critical_self_fields
        req.query = "profile:self_fields"
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Gender requests are a distinct employee-count mode. SearchEmployee is
    # the only directory source; if it exposes no verified gender field, the
    # orchestrator will explicitly say the count is unavailable instead of
    # silently returning the overall 1024 total.
    critical_female = bool(re.search(r"(?:الاناث|اناث|female|females|women)", critical_norm, re.I))
    critical_male = bool(re.search(r"(?:الذكور|ذكور|male|males|men)", critical_norm, re.I))
    if (critical_female or critical_male) and re.search(r"(?:عدد|كم|count|how\s+many)", critical_norm, re.I) and re.search(r"(?:موظف|الموظفين|employee|employees|staff)", critical_norm, re.I):
        req.intent = "employee_count"
        req.operation = "calculate"
        req.targets = []
        req.fields = ["count"]
        req.query = "gender:both" if critical_female and critical_male else ("gender:female" if critical_female else "gender:male")
        req.clarification_reason = ""
        req.confidence = 0.99
        return req

    # Deterministic explicit employee-information request.
    info_match = re.match(
        r"^(?:ابي|ابغى|عطني|اعطني|show\s+me)?\s*(?:معلومات|بيانات|تفاصيل|information|details)\s+(?:عن|about)\s+(.+?)[؟?!.،,]*$",
        critical_norm,
        re.I,
    )
    if info_match:
        person_text = " ".join(str(info_match.group(1) or "").split()).strip()
        if person_text:
            req.intent = "employee_details"
            req.operation = "details"
            req.targets = [TargetRef(kind="employee", name=person_text)]
            req.fields = ["name", "job_title", "department", "email", "mobile", "extension"]
            req.query = "employee:summary"
            req.clarification_reason = ""
            req.confidence = 0.99
            return req

    # Select a unique row from the immediately preceding verified employee
    # search by a short partial phrase such as "بن فهد".
    critical_list = state.get("list_state") if isinstance(state, dict) and isinstance(state.get("list_state"), dict) else {}
    if str(critical_list.get("intent") or "") == "employee_search":
        rows = [x for x in (critical_list.get("items") or []) if isinstance(x, dict)]
        sel = _normalize_for_semantic_guards(re.sub(r"[—–\-|]+", " ", critical_norm))
        sel_tokens = [x for x in sel.split() if len(x) >= 2]
        if sel_tokens and len(sel_tokens) <= 5 and not re.search(r"^(?:ابي|ابغى|مين|وش|كم|هل|متى|ابحث|دور|who|what|when|search|find)\b", sel, re.I):
            hits: list[dict[str, Any]] = []
            for row in rows:
                hay = _normalize_for_semantic_guards(" ".join(str(row.get(k) or "") for k in ("name", "name_ar", "name_en", "job_title", "job_title_ar", "job_title_en", "username")))
                hay_tokens = set(hay.split())
                if all(token in hay_tokens for token in sel_tokens):
                    hits.append(row)
            if len(hits) == 1:
                selected = hits[0]
                selected_name = str(selected.get("name_ar") or selected.get("name") or selected.get("name_en") or "").strip()
                if selected_name:
                    req.intent = "employee_details"
                    req.operation = "details"
                    req.targets = [TargetRef(kind="employee", name=selected_name)]
                    req.fields = ["name", "job_title", "department"]
                    req.query = "employee:selected_from_search"
                    req.clarification_reason = ""
                    req.confidence = 0.99
                    return req

    # Natural person follow-up after a verified directory/manager result:
    #   "مين مديري" -> ماجد
    #   "اليوم فاضي؟" -> availability of the current verified focus (ماجد)
    # Do this before generic name extraction so date words can never become names.
    focus_only_availability = bool(
        has_focus
        and not explicit_availability_person
        and not manager_availability_ref
        and re.search(r"\b(?:فاضي|فاضية|متاح|متاحة|مشغول|مشغولة|free|available|busy)\b", raw, re.I)
        and (
            explicit_period
            or followup_time
            or re.fullmatch(
                r"(?:(?:طيب|تمام)\s+)?(?:اليوم|بكره|بكرة|غدا|غداً|today|tomorrow)?\s*"
                r"(?:فاضي|فاضية|متاح|متاحة|مشغول|مشغولة|free|available|busy)[؟?!.،,]*",
                raw,
                re.I,
            )
        )
    )
    if focus_only_availability:
        period, start_date, end_date = _availability_period(raw)
        req.intent = "employee_availability"
        req.operation = "lookup"
        req.targets = [TargetRef(kind="focus", name="")]
        req.fields = ["start_time", "end_time"]
        req.period = period
        req.start_date = start_date
        req.end_date = end_date
        req.time_query = followup_time
        req.query = _availability_mode(raw)
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Explicit correction back to the signed-in user's own calendar.  This must
    # beat any stale employee calendar context, e.g. after asking about the
    # manager: "لا اجتماعاتي مو مديري".
    own_calendar_correction = bool(re.search(
        r"(?:اجتماعاتي|مواعيدي|جدول\s+اجتماعاتي|ميتنقاتي|my\s+meetings|my\s+calendar|my\s+schedule)",
        raw, re.I,
    )) and bool(re.search(r"(?:^|\s)(?:لا|مو\s+مديري|مش\s+مديري|not\s+my\s+manager)(?:\s|$)", raw, re.I))
    if own_calendar_correction:
        if explicit_period:
            period, start_date, end_date = _availability_period(raw)
        else:
            period = str(last_request.get("period") or "today")
            start_date = str(last_request.get("start_date") or "")
            end_date = str(last_request.get("end_date") or "")
        req.intent = "meetings"
        req.operation = "list"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["subject", "start_time", "end_time"]
        req.period = period if period in {"today", "tomorrow", "yesterday", "this_week", "last_week", "custom"} else "today"
        req.start_date = start_date
        req.end_date = end_date
        req.time_query = followup_time
        req.query = ""
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # A short period follow-up after OWN meetings stays on the signed-in user.
    # Never let an older employee-availability context (e.g. the manager)
    # hijack "بكره؟" after "ابي جدول اجتماعاتي".
    own_calendar_followup = bool(
        (last_intent in {"meetings", "meeting_list"} or str(meeting_ctx.get("mode") or "") == "self_meetings")
        and not explicit_availability_person
        and not manager_availability_ref
        and (
            explicit_period
            or followup_time
            or re.fullmatch(r"(?:طيب\s+)?(?:اليوم|بكره|بكرة|غدا|غداً|today|tomorrow)[؟?!.،,]*", raw, re.I)
        )
    )
    if own_calendar_followup:
        if explicit_period:
            period, start_date, end_date = _availability_period(raw)
        else:
            period = str(last_request.get("period") or "today")
            start_date = str(last_request.get("start_date") or "")
            end_date = str(last_request.get("end_date") or "")
        req.intent = "meetings"
        req.operation = "list"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["subject", "start_time", "end_time"]
        req.period = period if period in {"today", "tomorrow", "yesterday", "this_week", "last_week", "custom"} else "today"
        req.start_date = start_date
        req.end_date = end_date
        req.time_query = followup_time or str(last_request.get("time_query") or "")
        req.query = ""
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Short attendance-period follow-ups stay in attendance, not calendar.
    if last_intent in {"attendance_today", "attendance_history", "working_hours"} and explicit_period and not explicit_availability_person:
        period, start_date, end_date = _availability_period(raw)
        req.intent = "working_hours" if last_intent == "working_hours" else ("attendance_today" if period == "today" else "attendance_history")
        req.operation = "lookup"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["all"]
        req.period = period
        req.start_date = start_date
        req.end_date = end_date
        req.query = str(last_request.get("query") or "attendance:summary")
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Correction after a calendar turn: "لا شاهه العقيل" replaces the previous
    # calendar person while preserving the requested day/time.
    correction = re.fullmatch(r"(?:لا|قصدي|اقصد|أقصد)\s+(.+?)[؟?!.،,]*", raw, flags=re.I)
    if last_intent == "employee_availability" and correction:
        corrected_name = _clean_availability_person(correction.group(1))
        if corrected_name:
            req.intent = "employee_availability"
            req.operation = "lookup"
            req.targets = [TargetRef(kind="employee", name=corrected_name)]
            req.fields = ["start_time", "end_time"]
            req.period = str(last_request.get("period") or meeting_ctx.get("period") or "today")
            req.start_date = str(last_request.get("start_date") or meeting_ctx.get("start_date") or "")
            req.end_date = str(last_request.get("end_date") or meeting_ctx.get("end_date") or "")
            req.time_query = str(last_request.get("time_query") or meeting_ctx.get("time_query") or "")
            req.query = str(last_request.get("query") or meeting_ctx.get("mode") or "busy")
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
            return req

    # Short calendar follow-ups ("بكرة", "طيب الساعة 3؟") are tied to the
    # verified calendar subject, not to the generic conversational focus.
    if last_intent == "employee_availability" and meeting_name and availability_followup_shape and not explicit_availability_person and not manager_availability_ref:
        if explicit_period:
            period, start_date, end_date = _availability_period(raw)
        else:
            period = str(meeting_ctx.get("period") or last_request.get("period") or "today")
            start_date = str(meeting_ctx.get("start_date") or last_request.get("start_date") or "")
            end_date = str(meeting_ctx.get("end_date") or last_request.get("end_date") or "")
        req.intent = "employee_availability"
        req.operation = "lookup"
        req.targets = [TargetRef(kind="employee", name=meeting_name)]
        req.fields = ["start_time", "end_time"]
        req.period = period if period in {"today","tomorrow","this_week","custom"} else "today"
        req.start_date = start_date
        req.end_date = end_date
        req.time_query = followup_time or str(meeting_ctx.get("time_query") or last_request.get("time_query") or "")
        req.query = _availability_mode(raw) if re.search(r"(?:فاضي|متاح|مشغول|busy|free|available|جدول|اجتماع|meeting)", raw, re.I) else str(meeting_ctx.get("mode") or last_request.get("query") or "busy")
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Leave arithmetic follow-up such as "طيب لو 3 أيام؟" after a leave-balance turn.
    followup_leave_amount = _extract_leave_amount(raw)
    if last_intent == "leave_balance" and followup_leave_amount is not None and re.search(
        r"(?:طيب|لو|اذا|إذا|اخذت|أخذت|باخذ|بأخذ|what\s+about|if\s+i\s+take|days?)", raw, re.I
    ):
        req.intent = "leave_balance"
        req.operation = "calculate"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["balance"]
        req.amount = followup_leave_amount
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Authenticated-user identity/profile. Keep the exact conversational
    # purpose so the renderer can answer naturally instead of using one fixed
    # profile template for every wording.
    profile_norm = re.sub(r"^(?:اهلا|هلا|مرحبا|hello|hi)\s+", "", norm, flags=re.I).strip()
    profile_norm = re.sub(r"^(?:(?:طيب|تمام|اوكي|أوكي|please|pls)\s+)+", "", profile_norm, flags=re.I).strip()

    recognition_patterns = {
        "تعرفني", "تعرف مين انا", "انا تعرفني",
        "do you know me", "do u know me", "you know me",
    }
    identity_patterns = {
        "مين انا", "من انا", "who am i", "who i am",
    }
    summary_patterns = {
        "وش معلوماتي", "ايش معلوماتي", "ما هي معلوماتي", "معلوماتي",
        "ابي معلوماتي", "أبي معلوماتي", "ابغى معلوماتي", "أبغى معلوماتي",
        "عطني معلوماتي", "اعطني معلوماتي", "اعرض معلوماتي",
        "بياناتي", "ابي بياناتي", "أبي بياناتي", "ابغى بياناتي", "أبغى بياناتي",
        "عطني بياناتي", "اعطني بياناتي", "اعرض بياناتي", "اعرض لي بياناتي",
        "ورني بياناتي", "كل بياناتي", "بياناتي كاملة", "بياناتي كامله",
        "my profile", "show my profile", "show me my profile",
        "my information", "my info", "show my information", "show my info",
        "what are my details", "show me my details",
    }

    if profile_norm in recognition_patterns:
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name", "job_title", "department"]
        req.query = "profile:recognition"
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    if profile_norm in identity_patterns:
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name", "job_title", "department"]
        req.query = "profile:identity"
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    if profile_norm in summary_patterns:
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name", "job_title", "department", "email", "mobile", "extension"]
        req.query = "profile:summary"
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Explicit first-person profile fields always refer to the signed-in user,
    # regardless of the current employee focus (for example after asking about
    # the manager). Support multiple requested fields in one turn.
    self_profile_fields: list[str] = []
    self_field_patterns = (
        ("name", r"(?:وش\s+اسمي|ايش\s+اسمي|إيش\s+اسمي|ما\s+اسمي|اسمي\s+وش|اسمي\s+ايش|what(?:'s|\s+is)\s+my\s+name|my\s+name)"),
        ("email", r"(?:ايميلي|إيميلي|ايميل(?:ي)?|إيميل(?:ي)?|بريدي(?:\s+الالكتروني|\s+الإلكتروني)?|my\s+email)"),
        ("mobile", r"(?:رقم\s+جوالي|جوالي|رقم\s+موبايلي|موبايلي|my\s+(?:mobile|phone)(?:\s+number)?)"),
        ("extension", r"(?:تحويلتي|رقم\s+تحويلتي|my\s+extension)"),
        ("job_title", r"(?:مسماي|مسمّاي|منصبي|مسمى\s+وظيفتي|my\s+job\s+title|my\s+title)"),
        ("department", r"(?:ادارتي|إدارتي|الاداره\s+حقتي|الإدارة\s+حقتي|my\s+department)"),
        ("username", r"(?:اسم\s+المستخدم\s+حقي|يوزري|my\s+username)"),
    )
    for field_name, pattern in self_field_patterns:
        if re.search(pattern, raw, flags=re.I) and field_name not in self_profile_fields:
            self_profile_fields.append(field_name)
    if self_profile_fields:
        req.intent = "profile"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = self_profile_fields
        req.query = "profile:self_fields"
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Preserve the user's own wording for other profile requests parsed by the
    # semantic model. This lets the renderer choose a suitable response style
    # without changing the profile facts or authorization behavior.
    if req.intent == "profile" and not str(req.query or "").strip():
        req.query = raw

    # Organization-wide employee count by explicit gender field from SearchEmployee.
    # Never infer gender from names or titles.
    gender_text = _normalize_for_semantic_guards(raw)
    asks_female = bool(re.search(r"(?:الاناث|الإناث|اناث|إناث|female|females|women)", gender_text, re.I))
    asks_male = bool(re.search(r"(?:الذكور|ذكور|male|males|men)", gender_text, re.I))
    gender_count_cue = bool(re.search(r"(?:عدد|كم|count|how\s+many)", gender_text, re.I)) and bool(re.search(r"(?:موظف|الموظفين|employees?|staff)", gender_text, re.I))
    if gender_count_cue and (asks_female or asks_male):
        req.intent = "employee_count"
        req.operation = "lookup"
        req.targets = []
        req.fields = ["count"]
        req.query = "gender:both" if (asks_female and asks_male) else ("gender:female" if asks_female else "gender:male")
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # First-person manager must never depend on conversational focus or on an
    # LLM interpretation. "مين مديري؟" always means the signed-in user's manager.
    manager_norm = re.sub(r"^(?:اهلا|هلا|مرحبا)\s+", "", norm).strip()
    if re.fullmatch(
        r"(?:(?:طيب|تمام|اوكي|أوكي)\s+)?(?:مين|من)(?:\s+هو)?\s+(?:مديري|مديري\s+المباشر|المدير\s+حقي|المدير\s+المباشر\s+حقي)",
        manager_norm,
    ):
        req.intent = "manager"
        req.operation = "details"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["name"]
        req.relation_depth = 1
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req


    # Deterministic leave-balance arithmetic for the signed-in employee.
    leave_norm = _normalize_for_semantic_guards(_ascii_digits(raw))
    leave_amount = _extract_leave_amount(raw)
    leave_words = bool(re.search(r"\b(?:اجازه|اجازة|اجازات|إجازة|إجازات|vacation|leave|days? off)\b", leave_norm, re.I))
    remaining_words = bool(re.search(r"\b(?:يبقى|يبقي|يتبقى|يتبقي|باقي|رصيدي|remaining|remain|left|balance)\b", leave_norm, re.I))
    if leave_words and leave_amount is not None and remaining_words:
        req.intent = "leave_balance"
        req.operation = "calculate"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["balance"]
        req.amount = leave_amount
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req
    if re.search(r"\b(?:رصيد\s+اجازاتي|رصيد\s+إجازاتي|رصيدي\s+من\s+الاجازات|my\s+(?:leave|vacation)\s+balance)\b", leave_norm, re.I):
        req.intent = "leave_balance"
        req.operation = "lookup"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["balance"]
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Leave calculation phrased without repeating the word "leave".
    implicit_leave_amount = _extract_leave_amount(raw)
    if (
        implicit_leave_amount is not None
        and re.search(r"(?:اخذت|أخذت|باخذ|بأخذ|لو\s+اخذ|لو\s+أخذ|if\s+i\s+take)", raw, re.I)
        and re.search(r"(?:يبقى|يبقي|يتبقى|يتبقي|باقي|remain|remaining|left)", raw, re.I)
    ):
        req.intent = "leave_balance"
        req.operation = "calculate"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["balance"]
        req.amount = implicit_leave_amount
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Resolve Mazaya before availability: "المزايا المتاحة" contains "متاحة".
    if re.search(
        r"(?:\bالمزايا\b|\bمزايا\b|\bخصومات\b|\bالعروض\b|\bbenefits?\b|\bemployee\s+offers?\b|\bdiscounts?\b)",
        raw,
        re.I,
    ):
        req.intent = "mazaya"
        req.operation = "list"
        req.targets = []
        req.fields = ["all"]
        req.query = ""
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Attendance/self-service.  Parse the requested period and the requested
    # attendance fact independently.  The current user is the default subject;
    # an explicit manager reference is still parsed as the manager so the
    # authorization layer can correctly deny private attendance for others.
    attendance_norm = _normalize_for_semantic_guards(raw)
    attendance_cue = bool(re.search(
        r"(?:حضوري|الحضور|سجل\s+حضوري|جدول\s+حضوري|جدول\s+حضور|بصمتي|دوامي|"
        r"حضرت|دخلت|طلعت|خرجت|بصمت|وقت\s+دخولي|وقت\s+خروجي|متى\s+(?:حضرت|دخلت|جيت|وصلت|طلعت|خرجت)|"
        r"ساعات\s+(?:دوامي|عملي)|my\s+attendance|attendance|check[- ]?in|check[- ]?out|"
        r"working\s+hours|when\s+did\s+i\s+(?:arrive|check\s*in|leave|check\s*out))",
        attendance_norm,
        re.I,
    ))
    if attendance_cue:
        period, start_date, end_date = _availability_period(raw)
        working_hours_cue = bool(re.search(
            r"(?:ساعات\s+(?:دوامي|عملي)|كم\s+ساعه\s+(?:داومت|دوامي|عملت)|working\s+hours)",
            attendance_norm,
            re.I,
        ))
        check_out_cue = bool(re.search(
            r"(?:متي\s+(?:طلعت|خرجت)|وقت\s+خروجي|check[- ]?out|when\s+did\s+i\s+(?:leave|check\s*out))",
            attendance_norm, re.I,
        ))
        check_in_cue = bool(re.search(
            r"(?:متي\s+(?:حضرت|دخلت|جيت|وصلت)|وقت\s+دخولي|check[- ]?in|when\s+did\s+i\s+(?:arrive|check\s*in))",
            attendance_norm, re.I,
        ))

        target = TargetRef(kind="self", name="")
        if re.search(r"\b(?:مديري|المدير\s+حقي|المدير\s+المباشر\s+حقي|my\s+manager)\b", attendance_norm, re.I):
            target = TargetRef(kind="manager_of_self", name="")

        if working_hours_cue:
            req.intent = "working_hours"
            req.query = "attendance:working_hours"
        elif period == "today":
            req.intent = "attendance_today"
            req.query = "attendance:check_out" if check_out_cue else ("attendance:check_in" if check_in_cue else "attendance:summary")
        else:
            req.intent = "attendance_history"
            req.query = "attendance:check_out" if check_out_cue else ("attendance:check_in" if check_in_cue else "attendance:summary")
        req.operation = "lookup"
        req.targets = [target]
        req.fields = ["all"]
        req.period = period
        req.start_date = start_date
        req.end_date = end_date
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Own meetings: deterministic because this route is frequently demoed and
    # must not depend on an LLM parse. Own calendar may show meeting subjects.
    own_meeting_cue = bool(re.search(
        r"(?:اجتماعاتي|مواعيدي|جدولي|جدول\s+اجتماعاتي|ميتنقاتي|my\s+meetings|my\s+calendar|my\s+schedule)",
        raw, re.I,
    ))
    if own_meeting_cue:
        period, start_date, end_date = _availability_period(raw)
        req.intent = "meetings"
        req.operation = "list"
        req.targets = [TargetRef(kind="self", name="")]
        req.fields = ["subject", "start_time", "end_time"]
        req.period = period
        req.start_date = start_date
        req.end_date = end_date
        req.time_query = _extract_time_query(raw)
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Privacy-preserving employee calendar availability. Other employees expose
    # busy/free TIMES only; meeting subjects are never returned by this intent.
    availability_cue = bool(re.search(
        r"(?:فاضي|فاضية|متاح|متاحة|مشغول|مشغولة|اجتماعات|مواعيد|جدول|ميتنق|ميتنقات|\bfree\b|\bavailable\b|\bbusy\b|\bcalendar\b|\bschedule\b|\bmeetings?\b)",
        raw,
        re.I,
    ))
    own_meetings = bool(re.search(r"\b(?:اجتماعاتي|مواعيدي|جدولي|جدول\s+اجتماعاتي|my\s+meetings|my\s+calendar|my\s+schedule)\b", raw, re.I))
    if availability_cue and not own_meetings:
        person = _extract_availability_person(raw)
        target: TargetRef | None = None
        norm_personal = _normalize_for_semantic_guards(raw)
        if re.search(r"\b(?:مديري|المدير\s+حقي|المدير\s+المباشر\s+حقي|my\s+manager)\b", norm_personal, re.I):
            target = TargetRef(kind="manager_of_self", name="")
        elif re.search(r"\b(?:انا\s+فاضي|انا\s+متاح|هل\s+انا\s+فاضي|am\s+i\s+(?:free|available|busy))\b", norm_personal, re.I):
            target = TargetRef(kind="self", name="")
        elif person and _focus_matches_person(state, person):
            target = TargetRef(kind="focus", name="")
        elif person:
            target = TargetRef(kind="employee", name=person)
        elif meeting_name and _is_anaphoric_reference(raw):
            target = TargetRef(kind="employee", name=meeting_name)
        elif has_focus and (re.search(r"\b(?:هو|هي|هذا\s+الشخص|هالشخص|فاضي|فاضية|متاح|متاحة|مشغول|مشغولة)\b", norm_personal) or _is_anaphoric_reference(raw)):
            target = TargetRef(kind="focus", name="")

        if target is not None:
            period, start_date, end_date = _availability_period(raw)
            req.intent = "employee_availability"
            req.operation = "lookup"
            req.targets = [target]
            req.fields = ["start_time", "end_time"]
            req.period = period
            req.start_date = start_date
            req.end_date = end_date
            req.time_query = _extract_time_query(raw)
            req.query = _availability_mode(raw)
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
            return req

    # Closed employee-detail grammar for natural short follow-ups.
    if has_focus and re.fullmatch(
        r"(?:وش|ايش|ما)?\s*(?:منصب(?:ه|ها)|مسم(?:اه|اها)|المسمى\s+الوظيفي(?:\s+له|\s+لها)?)",
        norm,
    ):
        req.intent = "employee_details"
        req.operation = "details"
        req.targets = [TargetRef(kind="focus", name="")]
        req.fields = ["job_title"]
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    explicit_detail_patterns = (
        (r"^(?:ابي|ابغى|اعطني|عطني|وش|ايش|ما)?\s*(?:ايميل|إيميل|بريد)\s+(.+)$", "email"),
        (r"^(?:ابي|ابغى|اعطني|عطني|وش|ايش|ما)?\s*(?:جوال|موبايل|رقم\s+(?:الجوال|جوال|الموبايل|موبايل|الهاتف|هاتف)|رقم\s+التواصل)\s+(.+)$", "mobile"),
        (r"^(?:ابي|ابغى|اعطني|عطني|وش|ايش|ما)?\s*(?:تحويله|تحويلة|التحويله|التحويلة)\s+(.+)$", "extension"),
        (r"^(?:وش|ايش|ما|ابي|ابغى)?\s*(?:منصب|مسمى|المسمى\s+الوظيفي)\s+(.+)$", "job_title"),
    )
    for pattern, field in explicit_detail_patterns:
        match = re.match(pattern, raw, flags=re.I)
        if match:
            person_text = re.sub(r"[؟?!.،,]+$", "", str(match.group(1) or "")).strip()
            if person_text and not _looks_like_organizational_role(person_text):
                req.intent = "employee_details"
                req.operation = "details"
                req.targets = [TargetRef(kind="employee", name=person_text)]
                req.fields = [field]
                req.role_query = ""
                req.clarification_reason = ""
                req.confidence = max(float(req.confidence or 0.0), 0.99)
                return req

    # Narrow a previous verified employee-search result using a partial name,
    # title, or a pasted result line (e.g. "بن فهد").
    list_ctx = state.get("list_state") if isinstance(state, dict) and isinstance(state.get("list_state"), dict) else {}
    if str(list_ctx.get("intent") or "") == "employee_search":
        rows = [x for x in (list_ctx.get("items") or []) if isinstance(x, dict)]
        selection_text = _normalize_for_semantic_guards(re.sub(r"[—–\-|]+", " ", raw))
        selection_tokens = [t for t in selection_text.split() if len(t) >= 2]
        # Do not treat a fresh semantic question as a list selection.
        fresh_question = bool(re.search(r"^(?:ابي|ابغى|مين|من|وش|ايش|كم|هل|متى|اعطني|عطني|ابحث|دور|show|find|search|who|what|when|how)\b", selection_text, re.I))
        if selection_tokens and not fresh_question and len(selection_tokens) <= 12:
            matched_rows: list[dict[str, Any]] = []
            for row in rows:
                hay = _normalize_for_semantic_guards(" ".join(str(row.get(k) or "") for k in (
                    "name", "name_ar", "name_en", "job_title", "job_title_ar", "job_title_en",
                    "department", "department_ar", "department_en", "username",
                )))
                hay_tokens = set(hay.split())
                if all(token in hay_tokens for token in selection_tokens):
                    matched_rows.append(row)
            if len(matched_rows) == 1:
                selected = matched_rows[0]
                selected_name = str(selected.get("name_ar") or selected.get("name") or selected.get("name_en") or "").strip()
                if selected_name:
                    req.intent = "employee_details"
                    req.operation = "details"
                    req.targets = [TargetRef(kind="employee", name=selected_name)]
                    req.fields = ["name", "job_title", "department"]
                    req.query = "employee:selected_from_search"
                    req.clarification_reason = ""
                    req.confidence = max(float(req.confidence or 0.0), 0.99)
                    return req

    # Person-scoped team/direct-report lists. This must run before the global
    # employee-list grammar so "الموظفين اللي تحت أوس" cannot become 1024 rows.
    reporting_patterns = (
        r"(?:قائمه|قائمة|قايمه)?\s*(?:ب)?(?:الموظفين|موظفين|موظفي)\s+(?:اللي|الي|الذين)?\s*(?:تحت|عند|تابعين\s+ل|يتبعون)\s+(.+)$",
        r"(?:مين|من)\s+(?:الموظفين\s+)?(?:اللي|الي)?\s*(?:تحت|يتبع|يتبعون)\s+(.+)$",
        r"(?:فريق|team\s+of)\s+(.+)$",
    )
    for pattern in reporting_patterns:
        match = re.search(pattern, raw, flags=re.I)
        if match:
            person_text = re.sub(r"[؟?!.،,]+$", "", str(match.group(1) or "")).strip()
            if person_text and not _looks_like_organizational_role(person_text):
                req.intent = "reporting_employees"
                req.operation = "list"
                req.targets = [TargetRef(kind="employee", name=person_text)]
                req.fields = ["name", "job_title"]
                req.query = ""
                req.clarification_reason = ""
                req.confidence = max(float(req.confidence or 0.0), 0.99)
                return req

    # Direct employee search should never depend on the LLM for simple phrasing.
    search_match = re.match(
        r"^(?:ابي\s+|أبي\s+|ابغى\s+|أبغى\s+)?(?:ابحث|أبحث|دور|دوّر|دورلي|دور\s+لي|ابحث\s+لي|أبحث\s+لي|find|search(?:\s+for)?)\s+(?:لي\s+)?(?:عن\s+)?(.+?)[؟?!.،,]*$",
        raw,
        flags=re.I,
    )
    if search_match:
        employee_query = " ".join(str(search_match.group(1) or "").split()).strip()
        if employee_query and not _looks_like_organizational_role(employee_query):
            req.intent = "employee_search"
            req.operation = "search"
            req.targets = []
            req.query = employee_query
            req.fields = ["name", "job_title", "department"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
            return req

    pending = state.get("pending") if isinstance(state, dict) and isinstance(state.get("pending"), dict) else {}
    if pending.get("kind") == "missing_employee_target":
        bare_name = re.sub(r"[؟?!.،,]+$", "", raw).strip()
        bare_name_norm = _normalize_for_semantic_guards(bare_name)
        if (
            bare_name
            and 1 <= len(bare_name_norm.split()) <= 5
            and not re.search(r"\b(?:مين|من|وش|ايش|ما|كم|ابي|ابغى|اعطني|عطني)\b", bare_name_norm)
        ):
            req.intent = "employee_search"
            req.operation = "search"
            req.targets = [TargetRef(kind="employee", name=bare_name)]
            req.fields = ["name"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
            return req

    directory_ctx = state.get("directory_context") if isinstance(state, dict) and isinstance(state.get("directory_context"), dict) else {}
    remembered_general_department = str(directory_ctx.get("general_department") or "").strip()
    if remembered_general_department and re.fullmatch(
        r"(?:مين|من)(?:\s+هو|\s+هي)?\s+(?:ال)?مدير\s+العام\s+فيها",
        norm,
    ):
        req.intent = "employee_role_search"
        req.operation = "search"
        req.targets = []
        req.role_query = f"مدير عام {remembered_general_department}"
        req.fields = ["name", "job_title", "department"]
        req.department_query = remembered_general_department
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Bare organisational role/title wording must be a role lookup even when
    # the user omits "مين". Example:
    #   "مدير قسم اول الابتكار واحتضان التقنيات؟"
    # This rule intentionally requires an organisational-unit word so a phrase
    # such as "مدير خالد" is not mistaken for a job-title lookup.
    bare_role_raw = re.sub(r"[؟?!.،,]+$", "", raw).strip()
    bare_role_norm = _normalize_for_semantic_guards(bare_role_raw)
    if (
        re.match(r"^(?:ال)?مدير(?:\s+عام(?:\s+اول)?)?\s+", bare_role_norm)
        and re.search(
            r"\b(?:ال)?(?:قسم|اداره|قطاع|وحده|مكتب|مركز|برنامج|مشروع)\b",
            bare_role_norm,
        )
    ):
        req.intent = "employee_role_search"
        req.operation = "search"
        req.targets = []
        req.role_query = bare_role_raw
        req.fields = ["name", "job_title", "department"]
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Closed manager follow-up anchored to the VERIFIED current focus.
    # Examples after a verified role/person result: "مين مديرها؟", "مديره؟".
    # This must not use a stale relation object from an older turn.
    if has_focus and re.fullmatch(
        r"(?:(?:مين|من)(?:\s+هو|\s+هي)?\s+)?(?:ال)?مدير(?:\s+المباشر)?(?:ه|ها|ته|تها)",
        manager_norm,
    ):
        req.intent = "manager"
        req.operation = "details"
        req.targets = [TargetRef(kind="focus", name="")]
        req.fields = ["name"]
        req.relation_depth = 1
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Global employee-list phrasing. Execution is SearchEmployee-only.
    global_employee_list = bool(
        re.search(
            r"\b(?:قايمه|قائمه|قائمة)\s+(?:ب)?(?:(?:جميع|كل|كافه|كافة)\s+)?(?:الموظفين|موظفين|موظفي)\b",
            norm,
        )
        or re.search(
            r"\b(?:ابي|ابغى|اعرض|اعطني)\s+(?:قايمه|قائمه|قائمة)\s+(?:ب)?(?:(?:جميع|كل|كافه|كافة)\s+)?(?:الموظفين|موظفين|موظفي)\b",
            norm,
        )
        or re.search(r"\b(?:employee list|list of employees|all employees)\b", norm)
    )
    exhaustive_employee_list = bool(
        re.search(r"\b(?:جميع|كل|كافه|كافة|كاملين|كلهم|جميعهم)\b", norm)
        or re.search(r"\b(?:all|everyone|entire list|full list)\b", norm)
    )

    last_list_state = state.get("list_state") if isinstance(state, dict) and isinstance(state.get("list_state"), dict) else {}
    last_list_intent = str(last_list_state.get("intent") or "")
    list_all_followup = (
        last_list_intent == "employee_list"
        and bool(
            re.fullmatch(
                r"(?:طيب\s+)?(?:ابي|ابغى|اعرض|اعطني)?\s*(?:هم\s+)?(?:كلهم|جميعهم|الكل|كل\s+الموظفين|جميع\s+الموظفين)",
                norm,
            )
            or re.fullmatch(r"(?:show|list)\s+all(?:\s+of\s+them)?", norm)
        )
    )

    if (global_employee_list or list_all_followup) and not (_extract_department_literal(raw) or _extract_department_alias(raw)):
        req.intent = "employee_list"
        req.operation = "list"
        req.targets = []
        req.fields = ["name", "job_title"]
        if exhaustive_employee_list or list_all_followup:
            req.fields.append("all")
        req.query = ""
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Direct organisational role wording, including conversational prefixes:
    # "بسالك مين مدير عام الإدارة العامة للتقنية والرقمنة"
    direct_role_match = re.search(
        r"(?:^|\s)(?:مين|من)(?:\s+هو|\s+هي)?\s+"
        r"((?:ال)?مدير(?:\s+عام)?\s+.+?)[؟?!.]*$",
        raw,
        flags=re.I,
    )
    if direct_role_match:
        role_query = str(direct_role_match.group(1) or "").strip()
        role_norm = _normalize_for_semantic_guards(role_query)
        if re.search(
            r"\b(?:ال)?(?:اداره|قسم|قطاع|وحده|مكتب|مركز|برنامج|مشروع)\b",
            role_norm,
        ):
            req.intent = "employee_role_search"
            req.operation = "search"
            req.targets = []
            req.role_query = role_query
            if not req.fields:
                req.fields = ["name", "job_title", "department"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
            return req

    # Explicit role-holder wording. This runs before department counting so
    # "مين ماسك منصب مدير إدارة ..." never becomes an employee/department list.
    role_match = re.search(
        r"(?:مين|من)\s+(?:هو\s+)?(?:اللي\s+)?(?:ماسك|يشغل|يتولى)\s+(?:منصب|دور)\s+(.+?)[؟?!.]*$",
        raw,
        flags=re.I,
    )
    if not role_match:
        role_match = re.search(
            r"\bwho\s+(?:holds|occupies)\s+(?:the\s+)?(?:role|position)\s+(?:of\s+)?(.+?)[?!.]*$",
            raw,
            flags=re.I,
        )
    if role_match:
        role_query = str(role_match.group(1) or "").strip()
        if role_query:
            req.intent = "employee_role_search"
            req.operation = "search"
            req.targets = []
            req.role_query = role_query
            if not req.fields:
                req.fields = ["name", "job_title", "department"]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)
            return req

    literal_department = _extract_department_literal(raw) or _extract_department_alias(raw)

    # "لا أبي بس إدارة معينة" does not name an actual department. Never turn
    # "معينة" into a department query; ask the user to name it instead.
    if re.search(r"\b(?:اداره|الاداره)\s+معينه\b", norm) and not literal_department:
        req.intent = "general_department_counts"
        req.operation = "calculate"
        req.targets = []
        req.fields = ["count"]
        req.department_query = ""
        req.clarification_reason = "missing_general_department"
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    all_general_departments = bool(
        re.search(r"\bكل\s+اداره\s+عامه\b", norm)
        or re.search(r"\bكل\s+الادارات\s+العامه\b", norm)
        or re.search(r"\bجميع\s+الادارات\s+العامه\b", norm)
        or re.search(r"\bالادارات\s+العامه\b", norm)
        or re.search(r"\beach\s+general\s+department\b", norm)
        or re.search(r"\ball\s+general\s+departments\b", norm)
    )

    count_request = bool(
        (re.search(r"\b(?:كم|عدد|اجمالي)\b", norm) and re.search(employee_word_pattern, norm))
        or re.search(r"\b(?:how many employees|employee count|total employees)\b", norm)
    )
    names_request = bool(
        re.search(r"\b(?:اسماء|اسامي|اسمائهم|اسماؤهم|اسماءهم)\b", norm)
        or re.search(r"\b(?:names|who are they)\b", norm)
    )

    # GENERAL DEPARTMENT count, optionally with names.
    if count_request and (all_general_departments or literal_department):
        req.intent = "general_department_counts"
        req.operation = "calculate"
        req.targets = []
        req.fields = ["count"] + (["name"] if names_request else [])
        req.department_query = "" if all_general_departments else literal_department
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Global employee total. Must run after department count handling.
    global_count_request = bool(
        re.search(rf"\b(?:{employee_word_pattern}\s+كاملين|كل\s+{employee_word_pattern}|جميع\s+{employee_word_pattern})\b", norm)
        or re.search(rf"\b(?:كم\s+(?:عدد\s+)?{employee_word_pattern}|عدد\s+{employee_word_pattern}|اجمالي\s+(?:عدد\s+)?{employee_word_pattern})\b", norm)
        or re.search(r"\b(?:how many employees|employee count|total employees)\b", norm)
    )
    if global_count_request and not literal_department and not all_general_departments and not req.targets:
        req.intent = "employee_count"
        req.operation = "calculate"
        req.targets = []
        req.fields = ["count"]
        req.department_query = ""
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Explicit member request for one GENERAL DEPARTMENT.
    member_request = bool(
        re.search(r"\b(?:مين|من)\b", norm) and re.search(employee_word_pattern, norm)
    ) or bool(re.search(r"(?:^|\s)ب?(?:موظفين|موظفو|موظفي)(?=\s|$)", norm)) \
      or bool(re.search(r"\b(?:الي|اللي|الذين)\s+في\b", norm))
    if literal_department and member_request and not count_request:
        req.intent = "general_department_members"
        req.operation = "list"
        req.targets = []
        req.fields = ["name"]
        req.department_query = literal_department
        req.clarification_reason = ""
        req.confidence = max(float(req.confidence or 0.0), 0.99)
        return req

    # Plain possessive profile/contact follow-up after a verified person focus.
    # Collect every requested field, e.g. "ابي ايميله ورقم جواله".
    if has_focus:
        requested_fields: list[str] = []
        if re.search(r"(?:ايميل(?:ه|ها|ة)?|إيميل(?:ه|ها|ة)?|ايميله|ايميلة|إيميله|إيميلة|بريد(?:ه|ها)?|email)", raw, re.I):
            requested_fields.append("email")
        if re.search(r"(?:جوال(?:ه|ها)?|موبايل(?:ه|ها)?|رقم\s+(?:جوال(?:ه|ها)?|موبايل(?:ه|ها)?)|رقم\s+جواله|رقم\s+جوالها|mobile|phone)", raw, re.I):
            requested_fields.append("mobile")
        if re.search(r"(?:تحويل(?:ته|تها|ه|ها)?|extension)", raw, re.I):
            requested_fields.append("extension")

        if requested_fields and _is_anaphoric_reference(message):
            req.intent = "employee_details"
            req.operation = "details"
            req.fields = requested_fields
            req.targets = [
                TargetRef(
                    kind="relation_object" if _explicit_manager_object_reference(message) else "focus",
                    name="",
                )
            ]
            req.clarification_reason = ""
            req.confidence = max(float(req.confidence or 0.0), 0.99)

    return req


def _model_intent_has_evidence(
    request: ParsedRequest,
    message: str,
    state: dict[str, Any] | None,
) -> ParsedRequest:
    """Reject model-only enterprise guesses that are not grounded in the text.

    Deterministic enterprise grammar runs before the LLM. This guard handles the
    remaining open-ended model path: if Qwen labels ordinary conversation as an
    employee/system request without any matching linguistic evidence, the turn is
    returned to general conversation instead of touching enterprise tools.
    """
    req = request.model_copy(deep=True)
    if req.intent in {"general", "web_search", "summarize_previous"}:
        return req

    norm = _normalize_for_semantic_guards(message)
    has_focus = _has_verified_focus(state)

    evidence: dict[str, tuple[str, ...]] = {
        "profile": (
            r"\b(?:مين\s+انا|من\s+انا|تعرفني|تذكرني|معلوماتي|بياناتي|ايميلي|إيميلي|بريدي|جوالي|تحويلتي|منصبي|مسماي|ادارتي|إدارتي)\b",
            r"\b(?:who\s+am\s+i|do\s+you\s+know\s+me|remember\s+me|my\s+(?:profile|email|mobile|phone|extension|job\s+title|department))\b",
        ),
        "employee_details": (
            r"\b(?:معلومات|بيانات|تفاصيل|ايميل|إيميل|بريد|جوال|موبايل|تحويل|منصب|مسمى|اداره|إدارة)\b",
            r"\b(?:employee|email|mobile|phone|extension|job\s+title|department|details|information)\b",
        ),
        "employee_search": (r"\b(?:ابحث|أبحث|دور|دورلي|موظف)\b", r"\b(?:find|search|employee)\b"),
        "employee_role_search": (r"\b(?:مدير|رئيس|مستشار|نائب|منصب|قسم|اداره|إدارة)\b", r"\b(?:manager|director|head|chief|role|position)\b"),
        "employee_count": (r"\b(?:كم|عدد|اجمالي).*\bموظف", r"\b(?:how\s+many|count|total).*\bemployees?\b"),
        "general_department_counts": (r"\b(?:اداره|إدارة|الادارات|الإدارات|قسم|وحده|مكتب|قطاع)\b", r"\b(?:department|division|office|sector)\b"),
        "general_department_members": (r"\b(?:موظف|موظفين|موظفي|اداره|إدارة|قسم|وحده|مكتب|قطاع)\b", r"\b(?:employees?|staff|department|division|office|sector)\b"),
        "manager": (r"\b(?:مدير|مديري|مديره|مديرها)\b", r"\bmanager\b"),
        "leave_balance": (r"\b(?:اجاز|إجاز|رصيد)\b", r"\b(?:leave|vacation|balance)\b"),
        "leave_types": (r"\b(?:اجاز|إجاز)\b", r"\b(?:leave|vacation)\b"),
        "leave_history": (r"\b(?:اجاز|إجاز)\b", r"\b(?:leave|vacation)\b"),
        "vacations_list": (r"\b(?:اجاز|إجاز)\b", r"\b(?:leave|vacation)\b"),
        "meetings": (r"\b(?:اجتماع|اجتماعات|ميتنج)\b", r"\bmeetings?\b"),
        "meeting_list": (r"\b(?:اجتماع|اجتماعات|ميتنج)\b", r"\bmeetings?\b"),
        "attendance_today": (r"\b(?:حضور|حضوري|دوام|دخلت|خرجت)\b", r"\b(?:attendance|check.?in|check.?out)\b"),
        "attendance_history": (r"\b(?:حضور|حضوري|دوام)\b", r"\battendance\b"),
        "working_hours": (r"\b(?:ساعات\s+عملي|ساعات\s+دوامي|دوام)\b", r"\bworking\s+hours\b"),
        "reporting_employees": (r"\b(?:موظفين|موظفي|فريق|يتبعون|تحت|عند)\b", r"\b(?:team|direct\s+reports|reports\s+to)\b"),
        "team_attendance": (r"\b(?:فريقي|موظفيني|حضور\s+الفريق)\b", r"\bteam\s+attendance\b"),
        "tarasol_count": (r"\b(?:تراسل|معامل|معاملات)\b", r"\b(?:tarasol|transactions?)\b"),
        "tarasol_details": (r"\b(?:تراسل|معامل|معاملات|معاملتي)\b", r"\b(?:tarasol|transactions?)\b"),
        "hr_letter_recipients": (r"\b(?:خطاب|خطابات|الموارد\s+البشريه|الموارد\s+البشرية)\b", r"\bhr\s+letters?\b"),
        "hr_letter_types": (r"\b(?:خطاب|خطابات|الموارد\s+البشريه|الموارد\s+البشرية)\b", r"\bhr\s+letters?\b"),
        "policy_search": (r"\b(?:سياس|إجراء|اجراء)\b", r"\b(?:policy|policies|procedure)\b"),
        "policy_details": (r"\b(?:سياس|إجراء|اجراء)\b", r"\b(?:policy|policies|procedure)\b"),
        "news": (r"\b(?:خبر|اخبار|أخبار)\b", r"\bnews\b"),
        "news_details": (r"\b(?:خبر|اخبار|أخبار)\b", r"\bnews\b"),
        "announcements": (r"\b(?:اعلان|إعلان|اعلانات|إعلانات|تعميم|تعاميم)\b", r"\bannouncements?\b"),
        "announcement_details": (r"\b(?:اعلان|إعلان|اعلانات|إعلانات|تعميم|تعاميم)\b", r"\bannouncements?\b"),
        "mazaya": (r"\b(?:مزايا|ميزة|عروض)\b", r"\b(?:benefits?|offers?|mazaya)\b"),
        "mazaya_details": (r"\b(?:مزايا|ميزة|عروض)\b", r"\b(?:benefits?|offers?|mazaya)\b"),
    }

    patterns = evidence.get(req.intent)
    if patterns and any(re.search(pattern, norm, re.I) for pattern in patterns):
        return req

    # A genuine anaphoric employee follow-up can be very short ("ايميله؟").
    # The evidence rules above catch the field words; focus alone is never enough
    # to turn arbitrary chat into an employee request.
    if req.intent == "employee_details" and has_focus and _is_anaphoric_reference(message):
        return req

    return ParsedRequest(
        intent="general",
        operation="chat",
        targets=[],
        fields=[],
        query=str(message or "").strip(),
        confidence=0.99,
        clarification_reason="",
    )



def _v138_reference_safety(request: ParsedRequest, message: str, state: dict[str, Any] | None) -> ParsedRequest:
    """Prevent third-person/focus wording from silently becoming self."""
    req = request.model_copy(deep=True)
    if not isinstance(state, dict):
        return req

    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    current = focus.get("current") if isinstance(focus.get("current"), dict) else {}
    if not current:
        return req

    msg = _normalize_for_semantic_guards(message)
    third_person = bool(re.search(
        r"(?:اجازاته|إجازاته|اجازتها|إجازتها|حضوره|حضورها|دوامه|دوامها|"
        r"ايميله|إيميله|ايميلها|إيميلها|جواله|جوالها|تحويلته|تحويلتها|"
        r"مديره|مديرها|منصبه|منصبها|ادارته|إدارته|ادارتها|إدارتها|"
        r"\bhis\b|\bher\b|\bhe\b|\bshe\b)",
        msg,
        re.I,
    ))
    if third_person and req.targets and any(t.kind == "self" for t in req.targets):
        req.targets = [
            TargetRef(kind="focus", name="") if t.kind == "self" else t
            for t in req.targets
        ]
        req.confidence = max(float(req.confidence or 0.0), 0.90)
    return req



def _person_slot_target(text: str, state: dict[str, Any] | None = None) -> tuple[TargetRef, float]:
    """
    Compatibility helper used by orchestrator.py to resolve short person slots.

    Returns:
      (TargetRef, confidence_bonus)

    Rules are intentionally conservative:
    - first-person/self wording -> self
    - "مديري" / my manager -> manager_of_self
    - clear pronouns/possessives -> current focus when available
    - otherwise -> employee(name=<literal text>)
    """
    raw = str(text or "").strip()
    norm = _normalize_for_semantic_guards(raw).strip(" ؟?!.,،")

    state = state if isinstance(state, dict) else {}
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    current = focus.get("current") if isinstance(focus.get("current"), dict) else {}
    has_focus = bool(current)

    # Authenticated self.
    if re.fullmatch(
        r"(?:انا|أنا|اني|إني|عني|لي|حقي|نفسي|myself|me|my)",
        norm,
        re.I,
    ):
        return TargetRef(kind="self", name=""), 0.99

    # Authenticated user's manager.
    if re.fullmatch(
        r"(?:مديري|مديرتي|مديري المباشر|my manager|my direct manager)",
        norm,
        re.I,
    ):
        return TargetRef(kind="manager_of_self", name=""), 0.99

    # Clear deictic / possessive reference to current verified focus.
    focus_forms = {
        "هو", "هي", "هذا", "هذه", "هذي", "ذا",
        "عنه", "عنها", "له", "لها", "حقه", "حقها",
        "ايميله", "إيميله", "ايميلها", "إيميلها",
        "جواله", "جوالها", "تحويلته", "تحويلتها",
        "منصبه", "منصبها", "ادارته", "إدارته", "ادارتها", "إدارتها",
        "مديره", "مديرها", "حضوره", "حضورها", "اجازاته", "إجازاته",
        "اجازتها", "إجازتها", "his", "her", "him", "she", "he",
    }
    if has_focus and norm in focus_forms:
        return TargetRef(kind="focus", name=""), 0.99

    # Also support phrases that are effectively just a focus possessive,
    # e.g. "بياناته" / "معلوماته".
    if has_focus and re.fullmatch(
        r"(?:بياناته|بياناتها|معلوماته|معلوماتها|تفاصيله|تفاصيلها|"
        r"email him|email her|his details|her details)",
        norm,
        re.I,
    ):
        return TargetRef(kind="focus", name=""), 0.95

    # Empty text with a valid focus means the omitted subject is the focus.
    if not norm and has_focus:
        return TargetRef(kind="focus", name=""), 0.90

    # Fallback: treat the literal as an employee name candidate.
    return TargetRef(kind="employee", name=raw), 0.0


class SemanticParser:
    def __init__(self, *, retries: int = 2) -> None:
        self.retries = max(1, min(int(retries), 3))

    async def canonicalize(
        self,
        message: str,
        state: dict[str, Any],
        language: str,
    ) -> str | None:
        """
        Compatibility hook required by orchestrator.py.

        v141 intentionally disables free-form canonical rewriting.
        Returning None makes the orchestrator parse the ORIGINAL user message,
        which avoids meaning-changing rewrites such as:
          "إجازاتي السابقة" -> "معلومات عن اجازاتي"
          "مين مدير الهيئة" -> "مين مديري الهيئة"

        Deterministic critical/preflight routing still runs before semantic
        parsing, and the strengthened semantic prompt handles open phrasing.
        """
        return None

    @staticmethod
    def _clean_request(request: ParsedRequest) -> ParsedRequest:
        request.fields = request.normalized_fields()
        cleaned_targets: list[TargetRef] = []
        for target in request.targets:
            name = " ".join(str(target.name or "").split())
            if target.kind != "employee":
                name = ""
            cleaned_targets.append(TargetRef(kind=target.kind, name=name))
        request.targets = cleaned_targets
        request.department_query = " ".join(str(request.department_query or "").split())[:300]
        request.time_query = " ".join(str(request.time_query or "").split())[:80]
        request.clarification_reason = " ".join(str(request.clarification_reason or "").split())[:500]
        return request

    async def _run_json(self, messages: list[dict[str, str]]) -> ParsedRequest | None:
        for attempt in range(self.retries):
            raw = ""
            try:
                response = await asyncio.to_thread(chat, messages)
                raw = str(_message_content(response) or "")
                parsed = _extract_json(raw)
                if not isinstance(parsed, dict):
                    # Qwen 2.5 sometimes returns a compact classifier form such as
                    # `manager, self, fields=[name]`. Recover ONLY the explicit
                    # closed schema here; normal grounding/evidence guards still run
                    # afterwards, so casual chat can never become an enterprise call
                    # merely because the model guessed an intent.
                    parsed = _extract_relaxed_semantic(raw)
                    if isinstance(parsed, dict):
                        print(
                            f"[Semantic Parser Relaxed Parse] attempt={attempt + 1} raw={raw[:700]!r}",
                            file=sys.stderr,
                        )
                if isinstance(parsed, dict):
                    parsed = _coerce_semantic_payload(parsed)
                    return self._clean_request(ParsedRequest.model_validate(parsed))
                print(
                    f"[Semantic Parser Invalid JSON] attempt={attempt + 1} raw={raw[:700]!r}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"[Semantic Parser Validation Error] attempt={attempt + 1} error={exc} raw={raw[:700]!r}",
                    file=sys.stderr,
                )
            messages = list(messages) + [{
                "role": "system",
                "content": "Your previous output was invalid. Return exactly one valid JSON object matching the required schema and nothing else.",
            }]
        return None

    async def parse(self, message: str, state: dict[str, Any], *, language: str) -> ParsedRequest:
        """
        Model-first semantic understanding.

        Old builds routed deterministic regex/preflight BEFORE the language model.
        That made supported examples reliable but made natural conversation brittle.
        v148 reverses that:
          1) Qwen sees current message + bounded recent dialogue + verified structure.
          2) We apply grounding/reference invariants.
          3) Deterministic grammar is used only as a fallback when the model fails
             or incorrectly returns general for an obvious enterprise request.

        Authorization and API execution remain backend-owned.
        """
        payload = {
            "CURRENT_MESSAGE": str(message or "").strip(),
            "LANGUAGE": language,
            "DIALOGUE_CONTEXT": safe_state_for_parser(state),
        }
        primary_messages: list[dict[str, str]] = [
            {"role": "system", "content": SEMANTIC_PARSER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]

        first = await self._run_json(primary_messages)

        if first is not None:
            first = _ground_employee_names(first, message)
            first = _apply_semantic_invariants(first, message, state)
            first = _v138_reference_safety(first, message, state)
            first = _model_intent_has_evidence(first, message, state)

            # A second semantic pass is useful only on genuinely conversational /
            # ambiguous turns. It is NOT an execution or authorization decision.
            msg_norm = _normalize_for_semantic_guards(message)
            conversational_turn = bool(
                len(str(message or "").split()) <= 5
                or re.search(
                    r"(?:^|\s)(?:طيب|لا|قصدي|يعني|هو|هي|هذا|هذه|حقه|حقها|مديره|مديرها|"
                    r"الأول|الاول|الثاني|ساعه|ساعة|ساعتين|نص|تمام|اي|ايوه|ايوة|yes|no|him|her)(?:\s|$)",
                    msg_norm,
                    flags=re.I,
                )
            )
            if conversational_turn and (
                first.confidence < 0.88
                or bool(first.clarification_reason)
                or first.intent == "general"
            ):
                review_payload = {
                    "CURRENT_MESSAGE": str(message or "").strip(),
                    "LANGUAGE": language,
                    "DIALOGUE_CONTEXT": safe_state_for_parser(state),
                    "FIRST_DRAFT": first.model_dump(),
                }
                reviewed = await self._run_json([
                    {"role": "system", "content": SEMANTIC_REVIEW_PROMPT + """

Use RECENT_DIALOGUE for conversational meaning only.
A short answer may complete the immediately preceding question.
Do not invent enterprise facts from dialogue text.
Prefer the user's current explicit correction over previous context.
"""},
                    {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False, default=str)},
                ])
                if reviewed is not None:
                    reviewed = _ground_employee_names(reviewed, message)
                    reviewed = _apply_semantic_invariants(reviewed, message, state)
                    reviewed = _v138_reference_safety(reviewed, message, state)
                    reviewed = _model_intent_has_evidence(reviewed, message, state)
                    # Use the review when it resolves ambiguity rather than adding it.
                    if (
                        reviewed.intent != "general"
                        and reviewed.confidence >= first.confidence
                        and not reviewed.clarification_reason
                    ):
                        first = reviewed

        # Deterministic grammar is now a FALLBACK, not the primary brain.
        fallback = ParsedRequest(
            intent="general",
            operation="chat",
            targets=[],
            fields=[],
            query="",
            role_query="",
            department_query="",
            period="none",
            start_date="",
            end_date="",
            ordinal=0,
            relation_depth=1,
            amount=None,
            time_query="",
            confidence=0.0,
            clarification_reason="",
        )
        fallback = _repair_required_feature_grammar(fallback, message, state)


        # A numbered detail request against the immediately previous verified
        # content list is deterministic and must override model misclassification.
        content_detail_intents = {
            "news_details": "news",
            "announcement_details": "announcements",
            "mazaya_details": "mazaya",
            "policy_details": "policy_search",
        }

        if fallback.intent in content_detail_intents and fallback.ordinal > 0:
            list_state = (
                state.get("list_state")
                if isinstance(state.get("list_state"), dict)
                else {}
            )
            expected_list_intent = content_detail_intents[fallback.intent]

            if str(list_state.get("intent") or "") == expected_list_intent:
                print(
                    "[Verified List Detail Override] "
                    f"list={expected_list_intent!r} "
                    f"intent={fallback.intent!r} "
                    f"ordinal={fallback.ordinal}",
                    file=sys.stderr,
                )
                return self._clean_request(fallback)

        if first is None:
            if fallback.intent != "general":
                print(f"[Semantic Fallback] {fallback.model_dump()}", file=sys.stderr)
                return self._clean_request(fallback)
            return ParsedRequest(
                intent="general",
                operation="chat",
                targets=[],
                fields=[],
                confidence=0.0,
                clarification_reason="semantic_parser_failed",
            )

        # If Qwen said "general" but the closed fallback recognizes a supported
        # enterprise feature, use fallback. Otherwise keep the model's semantics.
        if first.intent == "general" and fallback.intent != "general":
            print(
                f"[Semantic Fallback Override] model=general fallback={fallback.intent!r}",
                file=sys.stderr,
            )
            return self._clean_request(fallback)

        print(f"[Semantic Model First] {first.model_dump()}", file=sys.stderr)
        return self._clean_request(first)

