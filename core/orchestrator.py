from __future__ import annotations

import ast
import hashlib
import asyncio
import json
import math
import operator
import re
import secrets
import sys
from datetime import datetime, timezone, timedelta
from copy import deepcopy
from typing import Any

from core.leave_overlap_response import render_leave_overlap
from core.overlap_dates import parse_overlap_dates
from core.response_quality import polish_response
from core.date_dialogue import date_year_turn, dated_message_needs_year
from core.config import settings
from core.conversation_memory import ConversationMemory
from core.employee_api import (
    EmployeeAPIClient,
    EmployeeAPIError,
    EmployeeAuthenticationError,
    EmployeeAuthorizationError,
)
from core.employee_tools import EMPLOYEE_TOOL_NAMES, EmployeeContext, execute_employee_tool
from core.llm_client import chat
from core.prompts import (
    GENERAL_PROMPT,
    STRICT_LANGUAGE_REPAIR_PROMPT,
    WEB_PLAN_PROMPT,
    WEB_RESPONSE_PROMPT,
)
from core.reference_resolver import ReferenceResolver, restore_pending_selection, restore_pending_target
from core.semantic_models import ParsedRequest, ResolvedTarget, TargetRef
from core.request_validator import validate_request
from core.response_renderer import (
    FRIENDLY_RENDERER_BUILD,
    render_attendance,
    render_availability,
    render_content_list,
    render_content_detail,
    render_generic,
    render_generic_list,
    render_hr_letter_types,
    render_hr_letter_recipients,
    render_employee_search_results,
    render_leave_balance,
    render_leave_history,
    render_meetings,
    localize_tarasol_data,
    render_profile,
    render_reporting_employees,
    render_tarasol,
    render_tarasol_count,
    summarize_verified,
)
from core.semantic_parser import (
    SEMANTIC_PARSER_BUILD,
    SemanticParser,
    detect_language,
    _extract_leave_amount,
    _person_slot_target,
    _extract_time_query,
    _availability_period,
    _attendance_period,
    _has_explicit_period_cue,
)
from core.intent_map import DETAIL_BASE_INTENT, get_policy
from core.directory_queries import DirectoryQueryService, bounded_employee as directory_employee, employee_name as directory_employee_name
from core.authorization import AuthorizationService
from core.action_workflows import ActionWorkflowService
from core.runtime_cache import TTLCache
from mcp_layer.client import MCPToolClient
# RCRC-FIX-v1: guard rails for the no-tools general mode.
from core.general_guard import (
    GENERAL_ONLY_PROMPT,
    enterprise_clarification,
    is_content_explain_request,
    looks_fabricated,
    sanitize_general_history,
)


ORCHESTRATOR_BUILD = "rcrc-v248-selection-language-fix-2026-10-15"


# ---------------------------------------------------------------------------
# Tiny deterministic utilities. These are closed grammars, not NLU keyword
# routing. Everything open-ended goes through SemanticParser.
# ---------------------------------------------------------------------------

_GREETINGS = {
    "ar": {
        "هلا": "هلا! كيف أقدر أساعدك؟",
        "اهلا": "هلا! كيف أقدر أساعدك؟",
        "اهلين": "أهلين! كيف أقدر أساعدك؟",
        "أهلين": "أهلين! كيف أقدر أساعدك؟",
        "أهلا": "هلا! كيف أقدر أساعدك؟",
        "مرحبا": "مرحبًا! كيف أقدر أساعدك؟",
        "السلام عليكم": "وعليكم السلام! كيف أقدر أساعدك؟",
    },
    "en": {
        "hi": "Hi! How can I help?",
        "hello": "Hello! How can I help?",
        "hey": "Hey! How can I help?",
    },
}

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
ARABIC_META_RE = re.compile(r"[\u0600-\u06ff]")


def _safe_calculation(text: str) -> str | None:
    expression = str(text or "").strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not expression or len(expression) > 120:
        return None
    if not re.fullmatch(r"[0-9\s+\-*/().%]+", expression):
        return None
    # RCRC-FIX-v2: "2" alone is a list selection, not a calculation.
    if not re.search(r"[+\-*/%]", expression.lstrip("+- ")):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError
            value = _ALLOWED_BINOPS[type(node.op)](left, right)
            if not math.isfinite(value) or abs(value) > 1e15:
                raise ValueError
            return float(value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return float(_ALLOWED_UNARY[type(node.op)](evaluate(node.operand)))
        raise ValueError

    try:
        result = evaluate(tree)
    except Exception:
        return None
    return str(int(result)) if result.is_integer() else str(round(result, 10)).rstrip("0").rstrip(".")


def _strip_conversation_fillers(value: str) -> str:
    """Remove harmless conversational lead-ins without changing the intent.

    Examples:
        "طيب مين انت" -> "مين انت"
        "اوكي وش تقدر تساعدني فيه" -> "وش تقدر تساعدني فيه"
    """
    text = _norm_role_text(value).strip()
    text = re.sub(r"[؟?!.،,]+$", "", text).strip()
    text = re.sub(
        r"^(?:(?:طيب|طب|تمام|اوكي|أوكي|اوك|حلو|ممتاز|ماشي|يب|ايوه|اها|يعني|بس|لو سمحت|ممكن|ok|okay|well|so)\s+)+",
        "",
        text,
        flags=re.I,
    ).strip()
    return text


def _arabic_gender_salutation(profile: dict[str, Any]) -> str:
    """Return أبشر/أبشري from verified employee profile only."""
    raw = " ".join(
        str(profile.get(key) or "")
        for key in ("gender_ar", "gender_en", "genderAr", "genderEn", "gender")
    ).strip().casefold()

    if not raw:
        return ""

    if raw in {"أنثى", "انثى", "female", "woman", "f"} or "أنثى" in raw or "انثى" in raw or "female" in raw:
        return "أبشري"
    if raw in {"ذكر", "male", "man", "m"} or "ذكر" in raw or "male" in raw:
        return "أبشر"
    return ""


def _assistant_identity_reply(message: str, language: str) -> str | None:
    normalized = _strip_conversation_fillers(message)

    assistant_ar = {
        "انت مين", "انتي مين", "مين انت", "مين انتي", "من انت", "من انتي",
        "وش انت", "وش انتي", "ايش انت", "ايش انتي",
        "عرفني بنفسك", "عرف بنفسك", "عرف عن نفسك",
    }
    assistant_en = {
        "who are you", "who r u", "who are u", "introduce yourself",
        "tell me about yourself", "what are you",
    }

    if normalized in assistant_ar or normalized.casefold() in assistant_en:
        if language == "ar":
            return (
                "أنا سند 😊 "
                "أساعدك في معلومات الموظفين، الإجازات، الحضور، الاجتماعات، تراسل، "
                "المزايا، الأخبار والسياسات."
            )
        return (
            "I'm Sanad 😊 I can help with employee information, "
            "leave, attendance, meetings, Tarasol, benefits, news, and policies."
        )
    return None


def _assistant_capabilities_reply(message: str, language: str) -> str | None:
    normalized = _strip_conversation_fillers(message)
    ar_patterns = (
        r"^(?:وش|ايش|إيش|ماذا)\s+(?:تقدر|تقدرين|تستطيع)\s+(?:تساعدني\s+فيه|تسوي|تعمل)$",
        r"^(?:وش|ايش|إيش)\s+(?:خدماتك|قدراتك)$",
        r"^(?:كيف|بوش)\s+(?:تقدر|تقدرين)\s+تساعدني$",
    )
    en_patterns = (
        r"^what can you do$", r"^how can you help me$", r"^what can you help me with$",
    )
    if any(re.fullmatch(pattern, normalized, re.I) for pattern in ar_patterns) or any(
        re.fullmatch(pattern, normalized.casefold(), re.I) for pattern in en_patterns
    ):
        if language == "ar":
            return (
                "أقدر أساعدك في معلومات الموظفين، الإجازات والحضور، الاجتماعات، تراسل، "
                "المزايا والعروض، الأخبار، الإعلانات والسياسات، بالإضافة للمساعدة في الصياغة والشرح 😊"
            )
        return (
            "I can help with employee information, leave and attendance, meetings, Tarasol, "
            "benefits, news, announcements, policies, and general writing or explanations."
        )
    return None


# ===========================================================================
# RCRC-FIX-v1 (تعديل): قبل كانت أي رسالة 7 كلمات أو أقل بدون كلمة مفتاحية
# تروح للموديل العام بدون API، فيخترع ("مين عماد؟"، "اوس"، "رقمه؟").
# الحين الافتراضي هو الـ API، والموديل العام فقط للمجاملة والكتابة الصريحة.
# ===========================================================================
def _obvious_general_conversation(message: str, state: dict[str, Any]) -> bool:
    """Fast-path detector for obvious small talk or explicit writing/advice.

    This helper is intentionally conservative. The semantic parser is the primary
    authority for open-ended general knowledge, technical explanations, coding,
    drafting, translation, and other normal assistant questions."""
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return False
    norm = _norm_role_text(raw)
    if not norm:
        return False

    small_talk = {
        "شكرا", "شكرا لك", "شكرا جزيلا", "مشكور", "مشكوره", "تسلم", "تسلمين",
        "يعطيك العافيه", "الله يعطيك العافيه", "جزاك الله خير", "الله يجزاك خير",
        "تمام", "اوكي", "اوك", "طيب", "زين", "حلو", "ممتاز", "رائع", "كفو",
        "ok", "okay", "thanks", "thank you", "thx", "great", "nice", "cool", "perfect",
    }
    if norm in small_talk or re.fullmatch(r"(?:ه){2,}|(?:ha){2,}h?|lol|lmao", norm, re.I):
        return True

    # Any open clarification (which employee? which item?) -> never general.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "").strip():
        return False

    # First-person identity forms are enterprise profile requests.
    if re.search(
        r"(?:مين\s+انا|من\s+انا|تعرفني|تذكرني|معلوماتي|بياناتي|ايميلي|بريدي|جوالي|تحويلتي|منصبي|ادارتي|who\s+am\s+i|do\s+you\s+know\s+me)",
        norm, re.I,
    ):
        return False

    # "لخصها / اختصرها" after a verified list = summarize_previous, not general.
    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    if list_state.get("items") and re.search(r"(?:لخص|اختصر|summari[sz]e|shorten)", norm, re.I):
        return False

    # Explicit writing/advice/explanation is general, unless it is about RCRC
    # content (policies/news/announcements/benefits) that must come from APIs.
    explicit_general_cues = (
        r"(?:اكتب(?:\s+لي)?|صيغ(?:\s+لي)?|صياغ|اعد\s+صياغ|ترجم|اشرح|وضح|لخص|اختصر|"
        r"وش\s+رايك|ايش\s+رايك|رايك|انصحني|نصيحه|كيف\s+اتعامل|وش\s+اسوي|ايش\s+اسوي|"
        r"كيف\s+ارد|وش\s+ارد|write|draft|rewrite|rephrase|translate|explain|summarize|brainstorm|advice)"
    )
    if re.search(explicit_general_cues, norm, re.I) and not is_content_explain_request(norm):
        return True

    # Everything else (names, pronouns, short follow-ups, unknown wording)
    # goes to the enterprise preflight/parser. Default is NOT general.
    return False


# ===========================================================================
# RCRC-FIX-v2 (إضافة): اختيار من القائمة اللي انعرضت ("2" / "الثاني" / "ابي عدنان")
# والرد على سؤال توضيح بـ "انا".
# ===========================================================================
_ORDINAL_WORDS = {
    "الاول": 1, "اول": 1, "الثاني": 2, "ثاني": 2, "الثالث": 3, "ثالث": 3,
    "الرابع": 4, "رابع": 4, "الخامس": 5, "خامس": 5,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}
_SELF_REPLIES = {
    "انا", "انا نفسي", "نفسي", "لي", "لي انا", "حقي", "لي انا نفسي",
    "me", "myself", "mine", "for me",
}
_PERSON_SCOPED_INTENTS = {
    "leave_balance", "leave_history", "attendance_today", "attendance_history",
    "working_hours", "tarasol_count", "tarasol_details", "meetings", "meeting_list",
    "employee_details", "manager", "employee_availability", "reporting_employees",
    "employee_image",
}


# ===========================================================================
# RCRC-FIX-v10 (إضافة): مطابقة الأسماء العربية بمرونة:
#   "السلطان" = "سلطان" = "بن سلطان" ، و"الماجد" = "ماجد".
# ===========================================================================
_NAME_PARTICLES = {"بن", "بنت", "ابن", "ال", "bin", "bint", "al"}


def _person_name_tokens(text: str) -> list[str]:
    tokens = []
    for token in _norm_role_text(text).split():
        if token in _NAME_PARTICLES:
            continue
        if token.startswith("ال") and len(token) >= 5:
            token = token[2:]
        if token.startswith("al") and len(token) >= 5 and token.isascii():
            token = token[2:]
        tokens.append(token)
    return tokens


def _row_name_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("name_ar", "name", "name_en"):
        tokens.update(_person_name_tokens(str(row.get(key) or "")))
    return tokens


def _known_people(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Every verified person already seen in this conversation."""
    rows: list[Any] = list(state.get("entity_history") or [])
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    relation = state.get("relation") if isinstance(state.get("relation"), dict) else {}
    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    role = state.get("role_lookup") if isinstance(state.get("role_lookup"), dict) else {}
    meeting = state.get("meeting_context") if isinstance(state.get("meeting_context"), dict) else {}
    rows += [focus.get("current"), focus.get("previous"), relation.get("subject"), relation.get("object"),
             role.get("employee"), meeting.get("employee")]
    rows += list(list_state.get("items") or [])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _uid(row).casefold() or str(row.get("name_ar") or row.get("name") or "").casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _match_known_person(name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Unique verified person from this conversation whose name contains all typed tokens."""
    query = _person_name_tokens(name)
    if not query:
        return {}
    hits = [row for row in _known_people(state) if all(t in _row_name_tokens(row) for t in query)]
    return hits[0] if len(hits) == 1 else {}


def _name_search_variants(name: str) -> list[str]:
    """"ماجد السلطان" -> ["ماجد بن سلطان", "ماجد سلطان"] for a second directory search."""
    words = " ".join(str(name or "").split()).split()
    if len(words) < 2:
        return []
    variants: list[str] = []
    for index, word in enumerate(words):
        norm = _norm_role_text(word)
        if norm.startswith("ال") and len(norm) >= 5 and index > 0:
            bare = word[2:]
            variants.append(" ".join(words[:index] + ["بن", bare] + words[index + 1:]))
            variants.append(" ".join(words[:index] + [bare] + words[index + 1:]))
    if any(_norm_role_text(w) in {"بن", "ابن"} for w in words):
        variants.append(" ".join(w for w in words if _norm_role_text(w) not in {"بن", "ابن"}))
    out: list[str] = []
    for variant in variants:
        if variant and variant != name and variant not in out:
            out.append(variant)
    return out


def _selection_text(norm: str) -> str:
    text = str(norm or "").translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(
        r"^(?:(?:لا|قصدي|اقصد|ابي|ابغى|اريد|اختار|اخترت|عطني|اعطني|هات|الموظف|رقم|الرقم|number|no\.?|#)\s*)+",
        "", text, flags=re.I,
    ).strip()
    return text


def _pending_or_list_selection(norm: str, state: dict[str, Any]) -> ParsedRequest | None:
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    kind = str(pending.get("kind") or "")
    raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}

    # 1) "انا" as the answer to "تقصد لك أنت ولا لموظف معيّن؟"
    if norm in _SELF_REPLIES and raw_req and str(raw_req.get("intent") or "") in _PERSON_SCOPED_INTENTS:
        try:
            restored = ParsedRequest.model_validate(raw_req)
        except Exception:
            restored = None
        if restored is not None:
            restored.targets = [TargetRef(kind="self", name="")]
            restored.confidence = 1.0
            restored.clarification_reason = ""
            return restored

    # 2) Candidate rows: an open selection first, else the list shown last turn.
    rows: list[dict[str, Any]] = []
    if kind in {"role_selection", "employee_disambiguation"}:
        rows = [x for x in (pending.get("candidates") or []) if isinstance(x, dict)]
    if not rows:
        list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
        last = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
        list_intent = str(list_state.get("intent") or "")
        if list_intent in {"employee_role_search", "employee_search", "reporting_employees"} and list_intent == str(last.get("intent") or ""):
            rows = [x for x in (list_state.get("items") or []) if isinstance(x, dict)]
    if not rows:
        return None

    text = _selection_text(norm)
    chosen: dict[str, Any] = {}
    index = int(text) if re.fullmatch(r"\d{1,2}", text) else _ORDINAL_WORDS.get(text, 0)
    # RCRC-FIX-v10: a copied list line ("5 ماجد عائض ناصر العسيري — مدير ...")
    leading = re.match(r"^(\d{1,2})(?:\s|$)", text)
    if not index and leading and 1 <= int(leading.group(1)) <= len(rows):
        candidate = rows[int(leading.group(1)) - 1]
        rest_tokens = set(_person_name_tokens(text[leading.end():]))
        if not rest_tokens or _row_name_tokens(candidate) & rest_tokens:
            index = int(leading.group(1))
    if not index:
        # the full name of exactly one candidate appears in the message
        text_tokens = set(_person_name_tokens(text))
        full_hits = [r for r in rows if _row_name_tokens(r) and _row_name_tokens(r) <= text_tokens]
        if len(full_hits) == 1:
            chosen = full_hits[0]
    if chosen:
        pass
    elif index:
        if 1 <= index <= len(rows):
            chosen = rows[index - 1]
    else:
        tokens = [t for t in _person_name_tokens(text) if len(t) >= 2]
        if not tokens or len(tokens) > 5:
            return None
        if re.match(r"^(?:مين|من|وش|ايش|كم|هل|متى|who|what|when|how)\b", text, re.I):
            return None
        hits = []
        for row in rows:
            hay_tokens = _row_name_tokens(row) | set(_norm_role_text(" ".join(str(row.get(k) or "") for k in (
                "job_title", "job_title_ar", "job_title_en", "username",
            ))).split())
            if all(t in hay_tokens for t in tokens):
                hits.append(row)
        if len(hits) == 1:
            chosen = hits[0]
    if not chosen:
        return None

    selected_name = str(chosen.get("name_ar") or chosen.get("name") or chosen.get("name_en") or "").strip()
    if not selected_name:
        return None

    if kind == "employee_disambiguation" and raw_req:
        try:
            restored = ParsedRequest.model_validate(raw_req)
            restored.targets = [TargetRef(kind="employee", name=selected_name)]
            restored.confidence = 1.0
            restored.clarification_reason = ""
            return restored
        except Exception:
            pass

    wanted = [f for f in (raw_req.get("fields") or []) if f not in {"name", "job_title", "department"}] if raw_req else []
    print(f"[Selection] picked={selected_name!r} from {len(rows)} rows", file=sys.stderr)
    return ParsedRequest(
        intent="employee_details", operation="details",
        targets=[TargetRef(kind="employee", name=selected_name)],
        fields=wanted or ["name", "job_title", "department"],
        query="employee:selected_from_list", confidence=1.0,
    )


# ===========================================================================
# RCRC-FIX-v3 (إضافة): الـ endpoint القديم للموظفين التابعين انشال.
# القائمة الحين موجودة داخل رد myProfile / employeeFullDetails، فنلقاها
# تلقائيًا داخل الرد ونوحّد شكل الصفوف.
# ===========================================================================
_REPORT_KEY_HINT = re.compile(
    r"(?:sub|report|team|direct|staff|employees|members|children|subordinat|مرؤوس|تابع)", re.I
)
_REPORT_KEY_EXCLUDE = re.compile(r"(?:manager|mngr|boss|supervisor|parent|مدير)", re.I)
_PERSON_NAME_KEYS = (
    "namea", "namel", "nameAr", "nameEn", "name_ar", "name_en", "name",
    "employeeName", "employeeNameAr", "employeeNameEn", "fullName", "empName",
    "arabicName", "englishName",
)


def _first_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "") and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text and text.casefold() not in {"none", "null", "n/a", "-"}:
                return text
    return ""


def _looks_like_person_row(row: Any) -> bool:
    return isinstance(row, dict) and any(_first_value(row, key) for key in _PERSON_NAME_KEYS)


def _find_reporting_rows(data: Any) -> tuple[list[dict[str, Any]], str]:
    """RCRC-FIX-v13: return the best list of people anywhere in the payload.

    Lists under report/team/employees-like keys win; a list under any other key
    is still accepted (the key name in the real response is not known)."""
    candidates: list[tuple[int, list[dict[str, Any]], str]] = []

    def walk(node: Any, path: str, key: str, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(node, list):
            people = [x for x in node if _looks_like_person_row(x)]
            if people and len(people) * 2 >= len(node):
                score = len(people)
                if _REPORT_KEY_HINT.search(key or ""):
                    score += 1000
                if _REPORT_KEY_EXCLUDE.search(key or "") and not _REPORT_KEY_HINT.search(key or ""):
                    score -= 500
                candidates.append((score, people, path))
            for index, item in enumerate(node[:50]):
                if isinstance(item, (dict, list)):
                    walk(item, f"{path}[{index}]", key, depth + 1)
        elif isinstance(node, dict):
            for child_key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value, f"{path}.{child_key}" if path else str(child_key), str(child_key), depth + 1)

    walk(data, "", "", 0)
    if not candidates:
        return [], ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, rows, path = candidates[0]
    return rows, path


def _payload_outline(data: Any, depth: int = 0) -> Any:
    """Structure of a response for the log: keys and list sizes only, no values."""
    if depth > 3:
        return "…"
    if isinstance(data, dict):
        return {str(k): _payload_outline(v, depth + 1) for k, v in list(data.items())[:25]
                if isinstance(v, (dict, list))} or f"dict({len(data)} fields)"
    if isinstance(data, list):
        first = _payload_outline(data[0], depth + 1) if data and isinstance(data[0], (dict, list)) else ""
        return f"list[{len(data)}] {first}".strip()
    return "value"

def _has_empty_people_list(data: Any) -> bool:
    """True when the payload has an empty list under a reports-like key."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and not value and _REPORT_KEY_HINT.search(str(key)) and not _REPORT_KEY_EXCLUDE.search(str(key)):
                return True
            if isinstance(value, (dict, list)) and _has_empty_people_list(value):
                return True
    elif isinstance(data, list):
        return any(_has_empty_people_list(item) for item in data[:20])
    return False


def _normalize_report_row(raw: dict[str, Any], language: str) -> dict[str, Any]:
    name_ar = _first_value(raw, "namea", "nameAr", "name_ar", "employeeNameAr", "arabicName")
    name_en = _first_value(raw, "namel", "nameEn", "name_en", "employeeNameEn", "englishName")
    generic_name = _first_value(raw, "name", "employeeName", "fullName", "empName")
    title_ar = _first_value(raw, "jodesca", "jobTitleAr", "job_title_ar", "positionAr", "jobDescAr")
    title_en = _first_value(raw, "jodesce", "hodesce", "jobTitleEn", "job_title_en", "positionEn", "jobDescEn")
    generic_title = _first_value(raw, "job_title", "jobTitle", "position", "jobDesc", "title")
    dept_ar = _first_value(raw, "dpDesca", "departmentAr", "department_ar")
    dept_en = _first_value(raw, "dpDescl", "departmentEn", "department_en")
    generic_dept = _first_value(raw, "department", "departmentName")
    username = _first_value(raw, "userid", "username", "userCode", "userName", "userId")
    ar = language == "ar"
    row = {
        "username": username or None,
        "userid": username or None,
        "name": (name_ar or generic_name or name_en) if ar else (name_en or generic_name or name_ar),
        "name_ar": name_ar or (generic_name if ARABIC_META_RE.search(generic_name) else "") or None,
        "name_en": name_en or None,
        "job_title": (title_ar or generic_title or title_en) if ar else (title_en or generic_title or title_ar),
        "job_title_ar": title_ar or None,
        "job_title_en": title_en or None,
        "department": (dept_ar or generic_dept or dept_en) if ar else (dept_en or generic_dept or dept_ar),
        "department_ar": dept_ar or None,
        "department_en": dept_en or None,
        "email": _first_value(raw, "email", "emailAddress", "mail") or None,
        "mobile": _first_value(raw, "mobile", "mobileNo", "mobileNumber") or None,
        "extension": _first_value(raw, "ext", "extension") or None,
    }
    return {k: v for k, v in row.items() if v not in (None, "")}


def _render_reporting_rows(rows: list[dict[str, Any]], language: str, *, subject_name: str, self_subject: bool) -> str:
    if not rows:
        if language == "ar":
            return (
                "حاليًا ما عندك موظفين تابعين لك مباشرة 👥😊"
                if self_subject
                else f"حاليًا ما يظهر موظفين تابعين مباشرة لـ {subject_name} 👥😊"
            )
        return (
            "The system doesn't show any employees reporting directly to you."
            if self_subject
            else f"The system doesn't show any employees reporting directly to {subject_name}."
        )
    if language == "ar":
        header = (
            f"أكيد 👥 هذولا الموظفين التابعين لك مباشرة ({len(rows)}):"
            if self_subject
            else f"أكيد 👥 هذولا الموظفين التابعين مباشرة لـ {subject_name} ({len(rows)}):"
        )
    else:
        header = (
            f"Employees reporting directly to you ({len(rows)}):"
            if self_subject
            else f"Employees reporting directly to {subject_name} ({len(rows)}):"
        )
    lines = [header]
    for index, row in enumerate(rows, start=1):
        name = str(row.get("name") or row.get("name_ar") or row.get("name_en") or "").strip()
        title = str(row.get("job_title") or "").strip()
        if language == "en" and ARABIC_META_RE.search(title):
            title = ""
        lines.append(f"{index}. {name}" + (f" — {title}" if title and title != name else ""))
    return "\n".join(lines)


def _contextual_reply_kind(message: str) -> str:
    """
    Classify a short conversational reply to a pending assistant offer.

    This is intentionally meaning-based rather than an exact sentence matcher:
    a pending action should survive natural acknowledgements such as
    "ايوا", "تمام ورني", "أوكي يلا", "sure", or "go ahead".

    Returns: "affirmative", "negative", or "".
    """
    raw = str(message or "").strip()
    if not raw:
        return ""

    norm = _norm_role_text(raw)
    words = [w for w in re.split(r"\s+", norm) if w]

    # Long/structured messages are treated as a new user request, not a reply
    # to the previous offer.
    if len(words) > 5 or len(norm) > 45:
        return ""

    negative = bool(re.search(
        r"(?:^|\s)(?:لا|لأ|مو\s+لازم|مش\s+لازم|خلاص|الغ|الغي|"
        r"no|nope|cancel|never\s*mind)(?:$|\s)",
        norm,
        re.I,
    ))
    if negative:
        return "negative"

    # A real new question/topic should not be swallowed by the pending offer.
    topic_change = bool(re.search(
        r"(?:\?|؟|مين|متى|وين|وش|ايش|كيف|كم|ليش|"
        r"اجتماع|موظف|مدير|اجاز|إجاز|حضور|دوام|سياس|خبر|اعلان|إعلان|مزايا|"
        r"who|when|where|what|how|why|meeting|employee|manager|leave|"
        r"attendance|policy|news|announcement|benefit)",
        norm,
        re.I,
    ))
    if topic_change and not re.search(
        r"(?:ورني|اعرض|أعرض|طلع|طلّع|show|go\s+ahead)",
        norm,
        re.I,
    ):
        return ""

    affirmative_signal = bool(re.search(
        r"(?:ايوا|ايوه|ايه|إيوا|أيوه|إيه|نعم|تمام|اوكي|أوكي|اوك|"
        r"اكيد|أكيد|يلا|ورني|اعرض(?:ها|هم)?|أعرض(?:ها|هم)?|"
        r"طلعها|طلّعها|yes|yeah|yep|sure|okay|ok|show(?:\s+me|\s+them)?|go\s+ahead)",
        norm,
        re.I,
    ))
    if affirmative_signal:
        return "affirmative"

    # When سند has just asked an explicit yes/no continuation question,
    # a very short non-question reply is normally conversational acceptance.
    # This avoids requiring a phrase dictionary for every dialect variant.
    if len(words) <= 2 and not topic_change:
        return "affirmative"

    return ""


def _context_engine_followup(
    message: str,
    state: dict[str, Any],
) -> ParsedRequest | None:
    """
    Resolve an elliptical follow-up from structural conversation context.

    This intentionally handles *modifiers*, not new topics. Examples:
      meetings -> "طيب الخميس الماضي؟"
      attendance -> "الثلاثاء؟"
      Tawasul events -> "طيب أمس"
      vacation history -> "الشهر الماضي"

    A message containing a new enterprise noun is left to the semantic parser.
    """
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return None

    norm = raw.casefold().translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    }))
    norm = re.sub(r"[؟?!.،,]+$", "", norm).strip()
    norm = re.sub(
        r"^(?:(?:طيب|طب|تمام|اوكي|اوك|حلو|ممتاز|ماشي|يعني|"
        r"then|okay|ok|and)\s+)+",
        "",
        norm,
        flags=re.I,
    ).strip()

    last = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
    ctx = state.get("conversation_context") if isinstance(state.get("conversation_context"), dict) else {}
    if not last:
        return None

    last_intent = str(last.get("intent") or ctx.get("active_domain") or "")
    if last_intent in {"", "general", "web_search", "summarize_previous"}:
        return None

    # A clear new domain noun means this is a new request, not an elliptical
    # modifier. Date/time words are deliberately absent from this guard.
    new_topic = bool(re.search(
        r"(?:موظف|موظفين|مدير|اداره|ادارة|سياس|خبر|اخبار|إعلان|اعلان|"
        r"مزايا|معامل|تراسل|رصيد|اجاز|إجاز|حضور|دوام|اجتماع|ميتنج|"
        r"employee|manager|department|policy|news|announcement|benefit|"
        r"transaction|leave|attendance|meeting)",
        norm,
        re.I,
    ))

    # If the message only names/modifies a date/time, inherit the domain.
    has_period = _has_explicit_period_cue(raw)
    selected_time = _extract_time_query(raw)

    modifier_only = bool(
        has_period
        or selected_time
        or re.fullmatch(
            r"(?:اليوم|امس|أمس|بكره|بكرة|غدا|غداً|"
            r"الاحد|الأحد|الاثنين|الإثنين|الثلاثاء|الاربعاء|الأربعاء|"
            r"الخميس|الجمعه|الجمعة|السبت|"
            r"الصباح|صباح|الظهر|ظهر|العصر|عصر|المساء|مساء|الليل|ليل|"
            r"today|yesterday|tomorrow|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"morning|afternoon|evening|tonight)",
            norm,
            re.I,
        )
    )

    # Explicit new-topic nouns normally win. The only exception is when the
    # noun merely repeats the currently active domain ("اجتماعات الخميس").
    active_domain_nouns = {
        "meetings": r"(?:اجتماع|اجتماعات|ميتنج|meeting)",
        "meeting_list": r"(?:اجتماع|اجتماعات|ميتنج|meeting)",
        "attendance_today": r"(?:حضور|دوام|attendance)",
        "attendance_history": r"(?:حضور|دوام|attendance)",
        "working_hours": r"(?:حضور|دوام|ساعات|attendance|working)",
        "vacation_history": r"(?:اجاز|إجاز|vacation|leave)",
    }
    if new_topic:
        own = active_domain_nouns.get(last_intent, "")
        if not own or not re.search(own, norm, re.I):
            return None

    if not modifier_only:
        return None

    try:
        restored = ParsedRequest.model_validate(last)
    except Exception:
        return None

    if has_period:
        try:
            if last_intent in {"attendance_today", "attendance_history", "working_hours"}:
                period, start_date, end_date = _attendance_period(raw)
                if last_intent in {"attendance_today", "attendance_history"}:
                    restored.intent = "attendance_today" if period == "today" else "attendance_history"
            else:
                period, start_date, end_date = _availability_period(raw)

            restored.period = period
            restored.start_date = start_date
            restored.end_date = end_date
        except Exception:
            return None

    if selected_time:
        restored.time_query = selected_time

    # Preserve the active sub-view, e.g. meetings:tawasul.
    active_view = str(ctx.get("active_view") or last.get("query") or "")
    if active_view:
        restored.query = active_view

    restored.clarification_reason = ""
    restored.confidence = 1.0
    return restored


def _list_selection_index(message: str) -> int | None:
    """
    Resolve a user choice from the last displayed numbered list.
    Supports:
      "1", "رقم 1", "الأول", "الثاني", ... "العاشر"
    """
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return None

    # Arabic/English digits.
    arabic_digits = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    norm = raw.translate(arabic_digits).strip().casefold()

    m = re.fullmatch(r"(?:رقم\s*)?(\d{1,3})", norm)
    if m:
        value = int(m.group(1))
        return value - 1 if value >= 1 else None

    ordinal_map = {
        "الاول": 0, "الأول": 0, "اول": 0, "أول": 0,
        "الثاني": 1, "ثاني": 1,
        "الثالث": 2, "ثالث": 2,
        "الرابع": 3, "رابع": 3,
        "الخامس": 4, "خامس": 4,
        "السادس": 5, "سادس": 5,
        "السابع": 6, "سابع": 6,
        "الثامن": 7, "ثامن": 7,
        "التاسع": 8, "تاسع": 8,
        "العاشر": 9, "عاشر": 9,
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
    }
    return ordinal_map.get(norm)


def _employee_name_from_row(row: dict[str, Any]) -> str:
    return str(
        row.get("name_ar")
        or row.get("name")
        or row.get("name_en")
        or row.get("employeeName")
        or row.get("employee_name")
        or ""
    ).strip()


def _employee_username_from_row(row: dict[str, Any]) -> str:
    return str(
        row.get("username")
        or row.get("userName")
        or row.get("user_name")
        or row.get("employeeUsername")
        or row.get("employee_username")
        or row.get("userid")
        or row.get("userId")
        or ""
    ).strip()


def _critical_orchestrator_preflight(message: str, state: dict[str, Any], language: str) -> ParsedRequest | None:
    """Second safety boundary for critical follow-ups before Qwen.

    This intentionally covers only closed, high-value grammar: authenticated
    self references, gender-count mode, unique selection from the last verified
    search list, and focus-only availability/date follow-ups.
    """
    def make(**kwargs: Any) -> ParsedRequest:
        base = dict(
            intent="general", operation="chat", targets=[], fields=[], query="",
            role_query="", department_query="", period="none", start_date="",
            end_date="", ordinal=0, relation_depth=1, amount=None, time_query="",
            confidence=0.99, clarification_reason="",
        )
        base.update(kwargs)
        return ParsedRequest(**base)


    raw = " ".join(str(message or "").strip().split())

    # v243: numbered selection from the last displayed employee list.
    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    if (str(list_state.get("kind") or "") == "department_employees" or str(list_state.get("intent") or "") == "general_department_members"):
        selected_index = _list_selection_index(raw)
        items = list_state.get("items") if isinstance(list_state.get("items"), list) else []

        if selected_index is not None and 0 <= selected_index < len(items):
            selected_row = items[selected_index] if isinstance(items[selected_index], dict) else {}
            selected_name = _employee_name_from_row(selected_row)
            selected_username = _employee_username_from_row(selected_row)

            # Set the selected employee as the current focus.
            focus_state = state.setdefault("focus", {})
            previous_current = (
                focus_state.get("current")
                if isinstance(focus_state.get("current"), dict)
                else {}
            )
            if previous_current:
                focus_state["previous"] = previous_current

            selected_focus = dict(selected_row)
            selected_focus.update({
                "username": selected_username,
                "name": selected_name,
                "source": "department_list_selection",
            })
            focus_state["current"] = selected_focus

            # Keep list_state for a later different numeric choice, but route
            # this turn directly to verified employee details.
            print(
                f"[List Selection] index={selected_index + 1} "
                f"name={selected_name!r} username={selected_username!r}",
                file=sys.stderr,
            )

            # A numbered choice is a REQUEST FOR FULL DETAILS, not an ACK.
            # Preserve the exact displayed row and verified employee code.
            selected_code = str(
                selected_row.get("emp_code")
                or selected_row.get("userCode")
                or selected_row.get("empCode")
                or selected_row.get("employeeCode")
                or ""
            ).strip()
            selected_focus["emp_code"] = selected_code
            focus_state["current"] = selected_focus
            state["selected_list_employee"] = {
                "name": selected_name,
                "username": selected_username,
                "emp_code": selected_code,
                "index": selected_index + 1,
            }
            # Route through authorization, target resolution and real API
            # retrieval. Never claim full details from a brief list row.
            return make(
                intent="employee_details",
                operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=["name", "job_title", "department", "email", "mobile", "extension"],
                query="employee:selected_from_department_list",
                confidence=1.0,
            )

        # A numeric choice outside the current list should not silently map
        # to a different person.
        if selected_index is not None and items:
            return make(
                intent="general",
                operation="chat",
                targets=[],
                fields=[],
                query="general:list_selection_out_of_range",
                confidence=1.0,
                clarification_reason="list_selection_out_of_range",
            )

    norm = raw.casefold()
    norm = norm.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
    norm = (
        norm.replace("الادراه", "الاداره").replace("الادراة", "الاداره")
        .replace("ادراه", "اداره").replace("ادراة", "اداره")
        .replace("الادرارات", "الادارات").replace("ادرارات", "ادارات")
        .replace("الادراات", "الادارات").replace("ادراات", "ادارات")
    )
    norm = re.sub(r"[؟?!.،,]+$", "", norm).strip()
    norm = re.sub(r"^[\-–—•]+\s*", "", norm).strip()
    raw = re.sub(r"^[\-–—•]+\s*", "", raw).strip()

    # INTEGRATED FIX: an explicit what-if leave balance question cannot route
    # to attendance because of the ambiguous Arabic word "يوم".
    # Uses existing verified leave balance and NEVER submits a leave request.
    leave_signal = bool(re.search(r"(?:اجاز|إجاز|leave|vacation|time\s+off|days?\s+off)", norm, re.I))
    remain_signal = bool(re.search(r"(?:كم\s+(?:يبق|باق)|يبق|يتبق|باقي|المتبقي|remaining|how\s+many.*left|\bleft\b)", norm, re.I))
    take_signal = bool(re.search(r"(?:خذت|اخذت|اخذ|باخذ|ناخذ|خصمت|استخدمت|لو|اذا|if\s+i\s+take|take)", norm, re.I))
    leave_ctx = state.get("leave_context") if isinstance(state.get("leave_context"), dict) else {}
    last_request_ctx = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
    prior_leave = bool(leave_ctx.get("displayed_balance") is not None or last_request_ctx.get("intent") == "leave_balance")
    # Short follow-ups such as "طيب لو يوم" keep the *current* leave calculation.
    # A previous leave balance in storage alone is not enough: the latest
    # request must still belong to leave_balance to prevent cross-topic leakage.
    short_leave_followup = bool(re.fullmatch(
        r"(?:طيب\s*)?(?:لو|اذا|if)\s*(?:يوم|يومين|\d+\s*(?:يوم|ايام|days?|day)|one\s+day|two\s+days?)\s*(?:كم\s*(?:يبقى|باقي|يتبقى))?",
        norm.strip(), re.I,
    ))
    active_leave = str(last_request_ctx.get("intent") or "") == "leave_balance"
    if (remain_signal and take_signal and (leave_signal or prior_leave)) or (short_leave_followup and active_leave and prior_leave):
        amount_match = re.search(r"(?<!\d)([0-9٠-٩]+(?:\.[0-9]+)?)\s*(?:يوم|ايام|أيام|day|days)", norm, re.I)
        amount = None
        if amount_match:
            amount = float(amount_match.group(1).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
        elif re.search(r"(?:يومين|يومَين|two\s+days?|two\s+days?\s+off)", norm, re.I):
            amount = 2.0
        elif re.search(r"(?:ثلاث\s*ايام|ثلاثة\s*أيام|three\s+days?)", norm, re.I):
            amount = 3.0
        elif re.search(r"(?:\bيوم\b|\bday\b|one\s+day)", norm, re.I):
            amount = 1.0
        if amount is not None and amount > 0:
            return make(intent="leave_balance", operation="calculate",
                        targets=[TargetRef(kind="self", name="")],
                        fields=["balance"], amount=amount,
                        query="leave:self_hypothetical", confidence=1.0)

    # RCRC-FIX-v136: narrow department acronym normalization.
    # Supports natural mixed-language wording without changing any API routing:
    #   الHR / بالHR / HR  -> hr
    #   الIT / بالIT / IT  -> it
    #   "ال هف"            -> it  (IT typed while Arabic keyboard is active)
    norm = re.sub(
        r"(?<![a-z])(?:ال|بال|لل)\s*(hr|it|ict|pmo)(?![a-z])",
        r" \1 ",
        norm,
        flags=re.I,
    )
    norm = re.sub(
        r"(?:^|\s)(?:ال\s*)?هف(?:\s|$)",
        " it ",
        norm,
        flags=re.I,
    )
    norm = re.sub(r"\s+", " ", norm).strip()

    # RCRC-FIX-v2: more conversational fillers ("حلو مين مدير شاهه").
    norm = re.sub(r"^(?:(?:طيب|طب|تمام|اوكي|اوك|حلو|ممتاز|ماشي|يب|ايوه|اها|please|pls)\s+)+", "", norm, flags=re.I).strip()
    # Common demo typos/spacing in English should not turn a clear request into
    # an employee-name search. Keep this intentionally tiny and meaning-safe.
    norm = re.sub(r"\biwant\b", "i want", norm, flags=re.I)
    norm = re.sub(r"\bmanger\b", "manager", norm, flags=re.I)


    # A negation corrects the requested field, not the referenced person.
    # Never treat "لا" / "no" as an employee name.
    if re.fullmatch(
        r"(?:لا|مو|no)[\s،,]+(?:ابي\s+|ابغى\s+)?(?:رقم\s+)?(?:جواله|جوالها|الجوال|جوال|موبايله|mobile|mobile number)",
        norm, re.I,
    ):
        current = (state.get("focus") or {}).get("current") or {}
        if _uid(current):
            return make(
                intent="employee_details", operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=["mobile"], query="employee:correction_mobile",
                confidence=1.0,
            )

    pending_now = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if (
        str(pending_now.get("kind") or "") == "missing_employee_target"
        and norm in {"انا", "أنا", "لي", "حقي", "me", "myself", "for me"}
        and isinstance(pending_now.get("request"), dict)
    ):
        try:
            restored = ParsedRequest.model_validate(pending_now["request"])
            restored.targets = [TargetRef(kind="self", name="")]
            restored.clarification_reason = ""
            restored.confidence = 1.0
            return restored
        except Exception:
            pass

    # If سند just offered the selected employee's job title/department,
    # a natural "اي / نعم / yes" continues with that same verified focus.
    if (
        str(pending_now.get("kind") or "") == "employee_more_details"
        and norm in {"اي", "ايوه", "أيوه", "نعم", "يب", "yes", "yeah", "yep"}
    ):
        return make(
            intent="employee_details",
            operation="details",
            targets=[TargetRef(kind="focus", name="")],
            fields=["job_title", "department"],
            query="employee:focus_more_details",
            confidence=1.0,
        )

    # Bare contact/work-field follow-ups belong to the currently verified
    # employee focus. This prevents phrases like "ابي تحويلة" from drifting to
    # an older person or being interpreted as a new employee name.
    focus_state = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    focus_current = focus_state.get("current") if isinstance(focus_state.get("current"), dict) else {}
    has_verified_focus = bool(_uid(focus_current))
    if has_verified_focus:
        bare_field_text = re.sub(
            r"^(?:(?:ابي|أبي|ابغى|أبغى|عطني|اعطني|هات|ممكن|وش|ايش|what(?:'s| is)?|show me|give me)\s+)+",
            "",
            norm,
            flags=re.I,
        ).strip()
        bare_field_map = {
            "ايميل": "email", "إيميل": "email", "البريد": "email", "البريد الالكتروني": "email",
            "جوال": "mobile", "الجوال": "mobile", "رقم الجوال": "mobile", "رقم جوال": "mobile",
            "تحويله": "extension", "تحويلة": "extension", "التحويله": "extension", "التحويلة": "extension",
            "المسمى": "job_title", "المسمى الوظيفي": "job_title", "مسمى وظيفي": "job_title",
            "الاداره": "department", "الادارة": "department", "إدارته": "department", "ادارته": "department",
            "email": "email", "mobile": "mobile", "phone": "mobile", "extension": "extension",
            "job title": "job_title", "department": "department",
        }
        bare_field = bare_field_map.get(bare_field_text)
        if bare_field:
            return make(
                intent="employee_details",
                operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=[bare_field],
                query="employee:focus_field",
                confidence=1.0,
            )

    if re.fullmatch(
        r"(?:تستعبط|تستهبل|تستسعبط|غبي|غبيه|غباء|يا\s+غبي|يا\s+غبيه|"
        r"are\s+you\s+kidding|stupid)",
        norm,
        re.I,
    ):
        return make(
            intent="general",
            operation="chat",
            targets=[],
            fields=[],
            query="general:frustration",
            confidence=1.0,
        )

    # Contextual self-meeting continuation. A short time-window reply must
    # preserve the day/date that was selected on the previous turn. Pending is
    # the primary source; meeting_context is a structural fallback so a state
    # cleanup cannot silently reset "Tuesday morning" back to today.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    pending_kind = str(pending.get("kind") or "")
    meeting_context = state.get("meeting_context") if isinstance(state.get("meeting_context"), dict) else {}
    selected_time = _extract_time_query(raw)

    if pending_kind == "meeting_time_window" or (
        selected_time
        and str(meeting_context.get("subject_kind") or "") == "self"
        and str(meeting_context.get("mode") or "").startswith("self_meetings")
    ):
        raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}
        try:
            if raw_req:
                restored = ParsedRequest.model_validate(raw_req)
            else:
                restored = ParsedRequest.model_validate({
                    "intent": "meetings",
                    "operation": "list",
                    "targets": [{"kind": "self", "name": ""}],
                    "fields": [
                        "subject", "start_time", "end_time", "location",
                        "organizer_name", "my_response", "teams_link",
                    ],
                    "query": "meetings:self",
                    "role_query": "",
                    "department_query": "",
                    "period": str(meeting_context.get("period") or "none"),
                    "start_date": str(meeting_context.get("start_date") or ""),
                    "end_date": str(meeting_context.get("end_date") or ""),
                    "ordinal": 0,
                    "relation_depth": 1,
                    "amount": None,
                    "time_query": str(meeting_context.get("time_query") or ""),
                    "confidence": 1.0,
                    "clarification_reason": "",
                })

            # The same reply may explicitly change the day/date.
            if _has_explicit_period_cue(raw):
                period, start_date, end_date = _availability_period(raw)
                restored.period = period
                restored.start_date = start_date
                restored.end_date = end_date

            if selected_time:
                restored.time_query = selected_time

            if selected_time or _has_explicit_period_cue(raw):
                restored.clarification_reason = ""
                restored.confidence = 1.0
                return restored
        except Exception:
            pass

    # Follow-up after سند mentions hidden Tawasul events/invitations.
    # A short "اي / نعم / ورني" means "show those Tawasul items" and must not
    # drift into generic chat.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "") == "tawasul_events_offer":
        raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}
        reply_kind = _contextual_reply_kind(raw)

        if reply_kind == "affirmative" and raw_req:
            try:
                restored = ParsedRequest.model_validate(raw_req)
                restored.query = "meetings:tawasul"
                restored.clarification_reason = ""
                restored.confidence = 1.0

                # Natural contextual refinement:
                # "اي، أمس" / "تمام بكرة" / "ورني اليوم" keeps the same
                # Tawasul intent while changing only the requested date.
                if _has_explicit_period_cue(raw):
                    try:
                        period, start_date, end_date = _availability_period(raw)
                        restored.period = period
                        restored.start_date = start_date
                        restored.end_date = end_date
                    except Exception:
                        pass

                print(
                    f"[Context Continuation] pending='tawasul_events_offer' "
                    f"text={raw!r} -> meetings:tawasul period={restored.period!r}",
                    file=sys.stderr,
                )
                print(
                    "[Context Continuation] reusing same MeetingList request scope; "
                    "tool cache should HIT while TTL is active",
                    file=sys.stderr,
                )
                return restored
            except Exception as exc:
                print(f"[Context Continuation Error] {exc}", file=sys.stderr)

        if reply_kind == "negative":
            _clear_pending(state)
            return make(
                intent="general", operation="chat", targets=[], fields=[],
                query="general:acknowledgement", confidence=1.0,
            )

    # Follow-up to a meeting result. When سند explicitly asks whether to check
    # the next time window, a natural yes/no reply must continue that same
    # meeting request instead of becoming generic chat.
    # Follow-up after سند offers to show a department employee list.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "") == "department_members_offer":
        raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}
        reply_kind = _contextual_reply_kind(raw)

        if reply_kind == "affirmative" and raw_req:
            try:
                restored = ParsedRequest.model_validate(raw_req)
                _clear_pending(state)
                print(
                    f"[Context Continuation] pending='department_members_offer' "
                    f"text={raw!r} -> department:members",
                    file=sys.stderr,
                )
                return restored
            except Exception as exc:
                print(f"[Department Context Continuation Error] {exc}", file=sys.stderr)

        if reply_kind == "negative":
            _clear_pending(state)
            return make(
                intent="general",
                operation="chat",
                targets=[],
                fields=[],
                query="general:acknowledgement",
                confidence=1.0,
            )

    # Follow-up after سند offers to show Tarasol transaction details.
    # This is context-based: "اي / ايوا / ورني / اعرضها" means execute the
    # exact offer that سند just made, not a generic chat response.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "") == "tarasol_details_offer":
        raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}
        reply_kind = _contextual_reply_kind(raw)

        if reply_kind == "affirmative" and raw_req:
            try:
                restored = ParsedRequest.model_validate(raw_req)
                restored.intent = "tarasol_details"
                restored.operation = "details"
                restored.targets = [TargetRef(kind="self", name="")]
                restored.fields = ["all"]
                restored.query = "tarasol:self_details"
                restored.clarification_reason = ""
                restored.confidence = 1.0
                _clear_pending(state)
                print(
                    f"[Context Continuation] pending='tarasol_details_offer' "
                    f"text={raw!r} -> tarasol:self_details",
                    file=sys.stderr,
                )
                return restored
            except Exception as exc:
                print(f"[Tarasol Context Continuation Error] {exc}", file=sys.stderr)

        if reply_kind == "negative":
            _clear_pending(state)
            return make(
                intent="general",
                operation="chat",
                targets=[],
                fields=[],
                query="general:acknowledgement",
                confidence=1.0,
            )

    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "") == "meeting_followup_window":
        raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}
        affirmative = bool(re.fullmatch(
            r"(?:اي|إي|ايوه|أيوه|ايه|إيه|نعم|yes|yeah|yep|تمام|اوكي|أوكي)",
            norm, re.I,
        ))
        negative = bool(re.fullmatch(
            r"(?:لا|لأ|no|nope|خلاص|مو لازم|مش لازم)",
            norm, re.I,
        ))
        if affirmative and raw_req:
            try:
                restored = ParsedRequest.model_validate(raw_req)
                restored.clarification_reason = ""
                restored.confidence = 1.0
                return restored
            except Exception:
                pass
        if negative:
            return make(
                intent="general", operation="chat", targets=[], fields=[],
                query="general:acknowledgement", confidence=1.0,
            )

    # RCRC-FIX-v121: finish a previous date-range clarification using a short
    # period-only reply such as "الشهر الماضي" without forcing the user to
    # repeat the original leave/vacation request.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "") == "missing_date_range":
        raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}

        # v178: month/date follow-ups must complete the ORIGINAL enterprise
        # request deterministically. Examples after "ابي جدول حضوري":
        #   "ابي شهر اوقست", "اغسطس الحالي", "شهر 8", "August 2026".
        # Do not send these short replies back to Qwen, where they can drift to
        # a generic calendar question. The semantic date resolver is already
        # authoritative for these closed date expressions.
        if raw_req and _has_explicit_period_cue(raw):
            try:
                restored = ParsedRequest.model_validate(raw_req)
                if restored.intent in {"attendance_today", "attendance_history", "working_hours"}:
                    selected_period, selected_start, selected_end = _attendance_period(raw)
                else:
                    selected_period, selected_start, selected_end = _availability_period(raw)
                if restored.intent in {"attendance_today", "attendance_history"}:
                    restored.intent = "attendance_today" if selected_period == "today" else "attendance_history"
                    restored.operation = "lookup"
                    restored.targets = [TargetRef(kind="self", name="")]
                    restored.fields = ["all"]
                    restored.query = str(restored.query or "attendance:summary")
                restored.period = selected_period
                restored.start_date = selected_start
                restored.end_date = selected_end
                restored.clarification_reason = ""
                restored.confidence = 1.0
                return restored
            except Exception:
                pass

        period_map = (
            ("today", r"^(?:اليوم|today)$"),
            ("tomorrow", r"^(?:بكره|بكرة|غدا|غداً|tomorrow)$"),
            ("yesterday", r"^(?:امس|أمس|yesterday)$"),
            ("this_week", r"^(?:هذا\s+الاسبوع|الأسبوع\s+هذا|الاسبوع\s+هذا|this\s+week)$"),
            ("last_week", r"^(?:الاسبوع\s+الماضي|الأسبوع\s+الماضي|last\s+week)$"),
            ("this_month", r"^(?:هذا\s+الشهر|الشهر\s+هذا|this\s+month)$"),
            ("last_month", r"^(?:الشهر\s+الماضي|last\s+month)$"),
        )
        selected_period = ""
        for value, pattern in period_map:
            if re.fullmatch(pattern, norm, re.I):
                selected_period = value
                break
        if selected_period and raw_req:
            try:
                restored = ParsedRequest.model_validate(raw_req)
                restored.period = selected_period
                restored.start_date = ""
                restored.end_date = ""
                restored.clarification_reason = ""
                restored.confidence = 1.0
                return restored
            except Exception:
                pass

    # v232: generic structural follow-up inheritance.
    contextual_request = _context_engine_followup(raw, state)
    if contextual_request is not None:
        print(
            f"[Context Engine] text={raw!r} -> "
            f"intent={contextual_request.intent!r} "
            f"period={contextual_request.period!r} "
            f"start={contextual_request.start_date!r} "
            f"end={contextual_request.end_date!r} "
            f"query={contextual_request.query!r}",
            file=sys.stderr,
        )
        return contextual_request

    # RCRC-FIX-v118: HR-letter read-only routing must be resolved before
    # canonical rewriting. Phrases such as "لمن أقدر أوجه خطاب الموارد البشرية؟"
    # must never be rewritten as a manager lookup, and "أنواع خطابات الموارد
    # البشرية" must never leak profile fields into the HR metadata validator.
    hr_letter_domain = bool(re.search(
        r"(?:خطاب|خطابات|تعريف\s+(?:راتب|بالراتب)|hr\s*letters?|salary\s+letters?|employment\s+letters?)",
        norm, re.I,
    ))
    if hr_letter_domain:
        hr_types = bool(re.search(
            r"(?:انواع|الانواع|نوع\s+الخطاب|وش\s+الانواع|ايش\s+الانواع|"
            r"what\s+(?:types?|kinds?)|letter\s+types?|types?\s+of\s+(?:hr\s+)?letters?)",
            norm, re.I,
        ))
        hr_recipients = bool(re.search(
            r"(?:لمن|لمين|الى\s+من|الي\s+من|اوجه|اوجّه|موجه|موجّه|"
            r"جهه|جهات|الجهه|الجهات|recipient|recipients|to\s+whom|"
            r"who\s+can\s+i\s+address|addressed\s+to)",
            norm, re.I,
        ))
        if hr_types:
            return make(
                intent="hr_letter_types", operation="list",
                targets=[], fields=[], query="hr_letter:types",
            )
        if hr_recipients:
            return make(
                intent="hr_letter_recipients", operation="list",
                targets=[], fields=[], query="hr_letter:recipients",
            )

    # v188: first-person possessive contact/profile requests are FINAL self-profile
    # grammar and must be resolved before the generic explicit-named-contact rule.
    # Without this guard, Arabic possessive endings such as "ايميلي" / "جوالي"
    # can be stripped down to stray letters (e.g. "ي ي") and then misread as a
    # person name by the older contact fallback.
    self_contact_fields: list[str] = []
    for field_name, pattern in (
        ("email", r"(?:ايميلي|إيميلي|ايميل\s*ي|بريدي(?:\s+الالكتروني)?|my\s+email)"),
        ("mobile", r"(?:رقم\s+جوالي|جوالي|رقم\s+موبايلي|موبايلي|my\s+(?:mobile|phone)(?:\s+number)?)"),
        ("extension", r"(?:تحويلتي|رقم\s+تحويلتي|my\s+extension)"),
        ("job_title", r"(?:مسماي|منصبي|مسمى\s+وظيفتي|my\s+(?:job\s+)?title)"),
        ("department", r"(?:ادارتي|إدارتي|الاداره\s+حقتي|الإدارة\s+حقتي|my\s+department)"),
        ("username", r"(?:يوزري|اسم\s+المستخدم\s+حقي|my\s+username)"),
    ):
        if re.search(pattern, norm, re.I) and field_name not in self_contact_fields:
            self_contact_fields.append(field_name)
    if self_contact_fields:
        return make(
            intent="profile", operation="details",
            targets=[TargetRef(kind="self", name="")],
            fields=self_contact_fields, query="profile:self_fields",
        )

    # Explicit named contact request with one or more fields.
    # Current-turn employee name always wins over self/focus context.
    # Examples:
    #   "ابي ايميل ورقم جوال اوس الدعجان"
    #   "عطني ايميل ماجد السلطان"
    explicit_contact_fields: list[str] = []
    if re.search(r"(?:ايميل|إيميل|ايمميل|بريد(?:\s+الكتروني)?|email)", norm, re.I):
        explicit_contact_fields.append("email")
    if re.search(r"(?:رقم\s+(?:جوال|موبايل)|جوال|موبايل|phone|mobile)", norm, re.I):
        explicit_contact_fields.append("mobile")
    if re.search(r"(?:تحويل|تحويله|تحويلة|extension)", norm, re.I):
        explicit_contact_fields.append("extension")

    if explicit_contact_fields:
        explicit_person = norm
        explicit_person = re.sub(
            r"^(?:(?:تمام|طيب|ايوا|ايوه|ايه|إيه|اوكي|أوكي|okay|ok)\s+)*(?:(?:ابي|أبي|ابغى|أبغى|اريد|أريد|عطني|اعطني|أعطني|هات|ممكن|please|give\s+me)\s+)+",
            "",
            explicit_person,
            flags=re.I,
        ).strip()
        # Remove the COMPLETE requested field phrase, including Arabic
        # possessive suffixes.  The old pattern removed only the stem
        # (e.g. "ايميل" from "ايميله") and left a stray "ه" behind.
        # That residue was then misread as an explicit employee name ("ه ه")
        # and overrode an already verified conversational focus.
        explicit_person = re.sub(
            r"(?:ايميل(?:ه|ها|ة)?|إيميل(?:ه|ها|ة)?|ايمميل(?:ه|ها|ة)?|"
            r"بريد(?:ه|ها)?(?:\s+الكتروني)?|email(?:\s+address)?|"
            r"رقم\s+(?:جوال|موبايل)(?:ه|ها)?|"
            r"(?:جوال|موبايل)(?:ه|ها)?|phone(?:\s+number)?|mobile(?:\s+number)?|"
            r"تحويلت(?:ه|ها)|تحويل(?:ه|ها|ة)?|extension)",
            " ",
            explicit_person,
            flags=re.I,
        )
        explicit_person = re.sub(r"(?:^|\s)(?:و|and|&)(?=\s|$)", " ", explicit_person, flags=re.I)
        explicit_person = " ".join(explicit_person.split()).strip(" ؟?!.,،")

        # A real current-turn name beats any old conversational focus.
        if (
            explicit_person
            and bool(explicit_person.split())
            and explicit_person not in {"انا", "أنا", "مديري", "مديره", "مديرها", "ي", "لي", "حقي"}
        ):
            return make(
                intent="employee_details",
                operation="details",
                targets=[TargetRef(kind="employee", name=explicit_person)],
                fields=list(dict.fromkeys(explicit_contact_fields)),
                query="employee:explicit_named_contact",
            )

    named_contact = re.fullmatch(
        r"(?:(?:ابي|ابغى|اريد|عطني|اعطني|هات|ممكن)\s+)?"
        r"(?P<field>ايميل|إيميل|بريد(?:\s+الكتروني)?|جوال|موبايل|رقم\s+(?:جوال|موبايل)|email|mobile|phone)"
        r"\s+(?P<name>.+)",
        norm, re.I,
    )
    if named_contact:
        person = str(named_contact.group("name") or "").strip(" ؟?!.,،")
        if person and person not in {"ي", "لي", "حقي", "مديري", "مديره", "مديرها"}:
            f = str(named_contact.group("field") or "")
            field = "email" if re.search(r"(?:ايميل|إيميل|بريد|email)", f, re.I) else "mobile"
            return make(
                intent="employee_details", operation="details",
                targets=[TargetRef(kind="employee", name=person)],
                fields=[field], query="employee:explicit_contact",
            )

    if re.search(
        r"(?:رصيد\s+(?:اجازتي|إجازتي|اجازاتي|إجازاتي)|"
        r"كم\s+رصيد\s+(?:اجازتي|إجازتي|اجازاتي|إجازاتي)|"
        r"(?:رصيد|كم)\s+.*(?:اجازاتي|إجازاتي)|"
        r"my\s+(?:leave|vacation)\s+balance)",
        norm,
        re.I,
    ):
        return make(
            intent="leave_balance", operation="lookup",
            targets=[TargetRef(kind="self", name="")],
            fields=["balance"], query="leave:self_balance",
        )
    if re.search(r"(?:رصيد\s+الاجازه\s+السنويه|رصيد\s+الإجازة\s+السنوية|annual\s+leave\s+balance)", norm, re.I):
        return make(
            intent="leave_balance", operation="lookup",
            targets=[TargetRef(kind="self", name="")],
            fields=["balance"], query="leave:self_annual_balance",
        )

    if re.search(
        r"(?:وش|ايش|ما|what).*(?:اجازات|إجازات|اجازه|إجازة).*(?:اقدر|أقدر|يمكنني|اقدم|أقدم|available|apply)",
        norm, re.I,
    ):
        return make(intent="leave_types", operation="list", targets=[], fields=["all"], query="leave:types")

    if re.search(r"(?:مين|من|عطني|اعطني|ابي|ابغى).*(?:موظف|الموظفين|مين).*(?:في\s+)?(?:اجازه|إجازة|اجازة)", norm, re.I):
        period = "today" if re.search(r"(?:اليوم|today)", norm, re.I) else (
            "tomorrow" if re.search(r"(?:بكره|بكرة|tomorrow)", norm, re.I) else (
                "this_week" if re.search(r"(?:هذا\s+الاسبوع|الأسبوع\s+هذا|this\s+week)", norm, re.I) else "none"
            )
        )
        return make(
            intent="vacations_list", operation="list", targets=[], fields=["all"],
            period=period, query="vacations:list",
            clarification_reason="" if period != "none" else "missing_date_range",
        )

    if re.search(
        r"(?:مين|من|عطني|اعطني|ابي|ابغى|وش).*(?:الموظفين|موظفين|الفريق|فريقي).*(?:التابعين\s+لي|يتبعونني|تحتي|عندي|direct\s+reports?|report\s+to\s+me)",
        norm, re.I,
    ):
        return make(
            intent="reporting_employees", operation="list",
            targets=[TargetRef(kind="self", name="")],
            fields=["name", "job_title"], query="reporting:self",
        )

    if re.search(r"(?:كم|عدد|how\s+many).*(?:معامله|معاملة|معاملات|تراسل|tarasol|transactions?)", norm, re.I):
        return make(
            intent="tarasol_count", operation="lookup",
            targets=[TargetRef(kind="self", name="")],
            fields=["count"], query="tarasol:self_count",
        )

    # RCRC-FIX-v119: explicit self-meeting requests must be resolved before
    # canonical rewriting / semantic parsing. This prevents phrases such as
    # "عطني اجتماعاتي اليوم" from being misclassified as working hours.
    meeting_domain = bool(re.search(
        r"(?:اجتماع|اجتماعات|مواعيدي|جدولي|meeting|meetings|calendar)",
        norm, re.I,
    ))
    if meeting_domain:
        # Only route clearly self-owned meeting/calendar requests here. Named
        # employee availability remains handled by the existing verified-focus
        # availability flow below.
        self_meeting = bool(
            re.search(
                r"(?:اجتماعاتي|اجتماعتي|اجتماعي|مواعيدي|جدولي|"
                r"جدول\s+(?:اجتماعاتي|اجتماعتي)|قائمه\s+(?:اجتماعاتي|اجتماعتي)|قائمة\s+(?:اجتماعاتي|اجتماعتي)|"
                r"مواعيد\s+اجتماعاتي|my\s+(?:meetings?|calendar|schedule))",
                norm, re.I,
            )
            or (
                re.search(r"(?:عطني|اعطني|ابي|ابغى|وش|ايش|اعرض|show|list|what)", norm, re.I)
                and re.search(r"(?:اجتماع|اجتماعات|meeting|meetings)", norm, re.I)
                and not re.search(r"(?:اجتماعاته|اجتماعاتها|his\s+meetings|her\s+meetings)", norm, re.I)
            )
        )
        if self_meeting:
            # Resolve dates and weekdays deterministically. This lets natural
            # wording such as "جدول اجتماعاتي الثلاثاء" map to the actual next
            # Tuesday date instead of falling through to the LLM.
            if _has_explicit_period_cue(raw):
                period, start_date, end_date = _availability_period(raw)
            else:
                period, start_date, end_date = "none", "", ""

            time_query = _extract_time_query(raw)

            return make(
                intent="meetings", operation="list",
                targets=[TargetRef(kind="self", name="")],
                fields=[
                    "subject", "start_time", "end_time", "location",
                    "organizer_name", "my_response", "teams_link",
                ],
                period=period, start_date=start_date, end_date=end_date,
                time_query=time_query, query="meetings:self",
                clarification_reason="" if period != "none" else "missing_date_range",
            )

    # RCRC-FIX-v120: deterministic routing for enterprise requests that Qwen
    # previously rewrote into unrelated employee/profile intents. Keep these
    # rules narrow and scoped to explicit enterprise domains.

    # ------------------------------------------------------------------
    # Leave / vacation read-only requests
    # ------------------------------------------------------------------
    # Exact/general vacation-list wording must beat canonical rewrite to
    # "leave types". This is the organisation-visible vacations list.
    if re.fullmatch(
        r"(?:(?:عطني|اعطني|ابي|ابغى|اعرض|هات)\s+)?(?:قائمه|قائمة|list)\s+(?:ال)?(?:اجازات|إجازات|اجاز|إجاز|vacations?|leaves?)",
        norm, re.I,
    ):
        return make(
            intent="vacations_list", operation="list", targets=[],
            fields=["all"], period="none", query="vacations:list",
            clarification_reason="missing_date_range",
        )

    leave_domain = bool(re.search(r"(?:اجاز|إجاز|vacation|leave)", norm, re.I))
    if leave_domain:
        # "إجازاتي السابقة" / "سجل إجازاتي" needs an explicit period.
        # Never silently default a historical request to this month.
        if re.search(
            r"(?:اجازاتي\s+(?:السابقه|الماضيه)|سجل\s+اجازاتي|تاريخ\s+اجازاتي|"
            r"previous\s+(?:leave|vacation)s?|leave\s+history|vacation\s+history)",
            norm, re.I,
        ):
            return make(
                intent="leave_history", operation="list",
                targets=[TargetRef(kind="self", name="")],
                fields=["all"], period="none", query="leave:self_history",
                clarification_reason="missing_date_range",
            )

        # A full overlap range is authoritative: do not ask for dates again.
        if re.search(
            r"(?:متداخل|متداخله|متقاط|تعارض|تقاطع|overlap|intersect|conflict)",
            norm, re.I,
        ):
            overlap_start, overlap_end, overlap_issue = parse_overlap_dates(raw)
            if not overlap_issue:
                overlap_period = "custom"
            elif overlap_issue == "missing_range" and _has_explicit_period_cue(raw):
                overlap_period, overlap_start, overlap_end = _availability_period(raw)
                # Month-only without year is ambiguous for leave overlap.
                if re.search(r"(?:\b20\d{2}\b)", raw) is None and overlap_period == "custom":
                    overlap_issue = "missing_year"
                else:
                    overlap_issue = ""
            return make(
                intent="leave_overlap", operation="lookup",
                targets=[TargetRef(kind="self", name="")],
                fields=["all"], period="custom" if not overlap_issue and overlap_start else "none",
                start_date=overlap_start, end_date=overlap_end,
                query="leave:self_overlap",
                clarification_reason="" if not overlap_issue else (
                    "missing_year" if overlap_issue == "missing_year" else "missing_date_range"
                ),
            )

        # "قائمة الإجازات" means the organisation-visible vacation list,
        # while "أنواع الإجازات" is handled by the existing leave-types route.
        if (
            re.search(r"(?:قائمه|قائمة|list)\s+(?:ال)?(?:اجاز|إجاز|vacation|leave)", norm, re.I)
            and not re.search(r"(?:انواع|أنواع|types?|kinds?)", norm, re.I)
        ):
            return make(
                intent="vacations_list", operation="list",
                targets=[], fields=["all"], period="today", query="vacations:list",
            )

    # ------------------------------------------------------------------
    # Team attendance
    # ------------------------------------------------------------------
    if (
        re.search(
            r"(?:حضور|دوام|تأخير|تاخير|أوفر\s*تايم|اوفر\s*تايم|"
            r"ساعات\s+إضافية|ساعات\s+اضافية|نقص|فرق\s+الساعات|"
            r"إجازة|اجازة|نوع\s+اليوم|بصم|attendance|lateness|late|"
            r"overtime|extra\s+hours|off\s+day|weekend|leave|vacation)",
            norm, re.I,
        )
        and re.search(
            r"(?:الموظفين\s+(?:التابعين|اللي\s+يتبعون|اللي\s+تحتي)|موظفيني|فريقي|"
            r"team|direct\s+reports?|reporting\s+employees)",
            norm, re.I,
        )
    ):
        period = ""
        if re.search(r"(?:اليوم|today)", norm, re.I):
            period = "today"
        elif re.search(r"(?:امس|أمس|yesterday)", norm, re.I):
            period = "yesterday"
        elif re.search(r"(?:هذا\s+الاسبوع|الاسبوع\s+(?:هذا|الحالي)|this\s+week)", norm, re.I):
            period = "this_week"
        if re.search(r"(?:تاخير|تأخير|متأخر|lateness|late|delay)", norm, re.I):
            team_mode = "attendance:lateness"
        elif re.search(r"(?:اوفر\s*تايم|أوفر\s*تايم|ساعات\s+(?:اضافية|إضافية)|overtime|extra\s+hours)", norm, re.I):
            team_mode = "attendance:overtime"
        elif re.search(r"(?:ناقص|نقص|فرق\s+(?:الدوام|الساعات)|difference|shortfall)", norm, re.I):
            team_mode = "attendance:difference"
        elif re.search(r"(?:الساعات\s+(?:المطلوبه|المطلوبة|الرسمية)|regular\s+hours|required\s+hours)", norm, re.I):
            team_mode = "attendance:regular_hours"
        elif re.search(r"(?:اجازه|اجازة|إجازة|اوف\s*داي|أوف\s*داي|off\s+day|weekend|نوع\s+اليوم|يوم\s+دوام)", norm, re.I):
            team_mode = "attendance:day_type"
        elif re.search(r"(?:بصم|بصمة|fingerprint|punch)", norm, re.I):
            team_mode = "attendance:punch_status"
        elif re.search(r"(?:كم\s+ساع|مدة\s+الدوام|ساعات\s+الدوام|working\s+hours|how\s+long)", norm, re.I):
            team_mode = "attendance:working_hours"
        else:
            team_mode = "attendance:summary"

        return make(
            intent="team_attendance", operation="list",
            targets=[TargetRef(kind="self", name="")],
            fields=["all"], period=period or "none", query=team_mode,
            clarification_reason="" if period else "missing_date_range",
        )

    # ------------------------------------------------------------------
    # Authenticated employee profile image
    # ------------------------------------------------------------------
    if re.fullmatch(
        r"(?:(?:جيب|هات|اعرض|ورني|ابي|ابغى)\s+)?(?:صورتي|الصوره\s+حقتي|الصورة\s+حقتي|"
        r"صوره\s+ملفي|صورة\s+ملفي|my\s+(?:profile\s+)?(?:photo|image|picture))",
        norm, re.I,
    ):
        return make(
            intent="employee_image", operation="lookup",
            targets=[TargetRef(kind="self", name="")], fields=["all"],
            query="profile:self_image",
        )

    # Named employee profile image.
    # Resolve the employee name through the verified directory first; never
    # pass a guessed username directly to EmpImage.
    named_image = re.fullmatch(
        r"(?:(?:جيب|هات|اعرض|ورني|ابي|ابغى)\s+)?(?:صوره|صورة|الصوره|الصورة)\s+(.+)",
        norm, re.I,
    )
    if named_image:
        employee_name = str(named_image.group(1) or "").strip()
        if employee_name:
            return make(
                intent="employee_image", operation="lookup",
                targets=[TargetRef(kind="employee", name=employee_name)],
                fields=["all"],
                query="profile:employee_image",
            )

    # ------------------------------------------------------------------
    # Content-detail follow-ups from the last verified list.
    # Never send "first news/announcement/benefit" to employee search.
    # ------------------------------------------------------------------
    detail_cue = bool(re.search(r"(?:تفاصيل|detail|details|افتح|open|اشرح)", norm, re.I))
    if detail_cue:
        ordinal = 0
        ordinal_patterns = (
            (1, r"(?:الاول|اول|الأول|أول|first)"),
            (2, r"(?:الثاني|ثاني|second)"),
            (3, r"(?:الثالث|ثالث|third)"),
            (4, r"(?:الرابع|رابع|fourth)"),
            (5, r"(?:الخامس|خامس|fifth)"),
        )
        for number, pattern in ordinal_patterns:
            if re.search(pattern, norm, re.I):
                ordinal = number
                break
        if not ordinal:
            number_match = re.search(r"(?:رقم|number|#)?\s*(\d{1,2})", norm, re.I)
            if number_match:
                ordinal = int(number_match.group(1))

        detail_intent = ""
        if re.search(r"(?:خبر|اخبار|الأخبار|الاخبار|news)", norm, re.I):
            detail_intent = "news_details"
        elif re.search(r"(?:اعلان|إعلان|اعلانات|إعلانات|announcement|announcements)", norm, re.I):
            detail_intent = "announcement_details"
        elif re.search(r"(?:ميزه|ميزة|مزايا|عرض|benefit|offer|mazaya)", norm, re.I):
            detail_intent = "mazaya_details"
        elif re.search(r"(?:سياسه|سياسة|سياسات|policy|procedure|اجراء|إجراء)", norm, re.I):
            detail_intent = "policy_details"

        # For short follow-ups such as "تفاصيل الأول", infer ONLY from the
        # verified list_state domain. Never infer an ID.
        if not detail_intent:
            list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
            detail_by_list = {
                "news": "news_details",
                "announcements": "announcement_details",
                "mazaya": "mazaya_details",
                "policy_search": "policy_details",
            }
            detail_intent = detail_by_list.get(str(list_state.get("intent") or ""), "")

        if detail_intent:
            return make(
                intent=detail_intent, operation="details", targets=[], fields=["all"],
                ordinal=ordinal, query="",
            )

    # RCRC-FIX-v2: the user answers the previous clarification or picks from
    # the list that was just shown ("2", "الثاني", "ابي عدنان", "انا").
    selection_request = _pending_or_list_selection(norm, state)
    if selection_request is not None:
        return selection_request

    # Final stability firewall: organisation-directory meta queries and
    # Tarasol privacy-sensitive requests are resolved before generic person-field
    # extraction, so words such as "كم إدارة" or "معاملتي" can never become
    # employee names.
    # Organisation-level count/list. Include singular "إدارة" and common
    # spacing/typing variants, but require explicit all/list/count language so a
    # named department like "إدارة المشاريع" is never swallowed here.
    unit_word = r"(?:ال)?(?:اداره|ادارات)"
    all_units_count = bool(re.fullmatch(
        rf"(?:(?:كم|ما\s+عدد|عدد)\s+(?:عدد\s+)?{unit_word}(?:\s+(?:في|بال|ب)\s*(?:الهيئه|الهييه))?|how\s+many\s+departments?(?:\s+are\s+there)?)",
        norm,
        re.I,
    ))
    all_units_list = bool(
        re.fullmatch(r"(?:كل|جميع)\s+(?:ال)?(?:اداره|ادارات)", norm, re.I)
        or (
            re.search(r"(?:قائمه|ليست|كل|جميع|اعرض|ابي|ابغى|اعطني|عطني|هات)", norm, re.I)
            and re.search(r"(?:^|\s)(?:ب|با)?\s*(?:ال)?ادارات(?:\s|$)", norm, re.I)
        )
        or bool(re.fullmatch(r"all\s+departments?", norm, re.I))
    )
    last_req_for_units = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
    if str(last_req_for_units.get("query") or "").startswith("org_units:") and re.fullmatch(r"(?:كلها|جميعها|الكل|all|all\s+of\s+them)", norm, re.I):
        all_units_list = True
    if all_units_count:
        return make(intent="general_department_counts", operation="calculate", fields=["count"], query="org_units:count")
    if all_units_list:
        return make(intent="general_department_counts", operation="list", fields=["name", "count"], query="org_units:list")

    # Centralized routing for explicit organization questions. Runs before the
    # generic person-name and department-list fallbacks. Actual membership is
    # always checked by the verified directory backend.
    from core.directory_intent import route_department_message
    org_request = route_department_message(raw)
    if org_request is not None:
        if org_request.action == "manager":
            # Job-role lookup is not the same as listing department employees.
            # Do not invent a manager from a matching staff list.
            return make(
                intent="employee_role_search", operation="search",
                role_query=f"مدير {org_request.department}",
                fields=["name", "job_title", "department"],
                query="org:manager_role_search",
            )
        if org_request.action == "members":
            return make(
                intent="general_department_members", operation="list",
                department_query=org_request.department,
                fields=["name", "job_title"], query="department:members",
            )
        if org_request.action == "count":
            return make(
                intent="general_department_counts", operation="calculate",
                department_query=org_request.department,
                fields=["count"], query="department:count",
            )
        if org_request.action == "units":
            # Our directory's employee department labels do not prove an org
            # hierarchy. Ask for a verified organization chart rather than
            # presenting employee names as child departments.
            return make(
                intent="general", operation="chat",
                query="general:org_hierarchy_unverified",
            )

    # RCRC-FIX-v136: explicit job-title questions must be role searches, not
    # department-member queries or employee-name searches.
    #
    # Only closed title shapes are handled here. Open-ended language still goes
    # through the existing semantic parser/orchestrator logic.
    role_candidate = raw.strip(" ؟?!.,،")
    # Normalize a very common typing error only inside role parsing.
    role_candidate = role_candidate.replace("ادراة", "ادارة").replace("إدراة", "إدارة")

    # Most-specific prefixes MUST run before generic "مين / من".
    role_candidate = re.sub(
        r"^(?:"
        r"(?:مين|من)\s+(?:اللي\s+)?(?:ماسك|يشغل|شاغل)\s+(?:منصب\s+)?|"
        r"(?:مين|من)\s+صاحب\s+منصب\s+|"
        r"(?:ابي|أبي|ابغى|أبغى)\s+(?:اعرف|أعرف)\s+(?:مين|من)\s+(?:اللي\s+)?"
        r"(?:ماسك|يشغل|شاغل)\s+(?:منصب\s+)?|"
        r"who\s+(?:holds|occupies)\s+(?:the\s+)?(?:position|role)\s+(?:of\s+)?|"
        r"(?:مين|من)(?:\s+هو|\s+هي)?\s+|"
        r"who\s+is\s+(?:the\s+)?"
        r")",
        "",
        role_candidate,
        flags=re.I,
    ).strip()

    role_candidate_norm = role_candidate.casefold().translate(
        str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه"})
    )

    explicit_role_shape = bool(re.match(
        r"^(?:"
        r"(?:ال)?مدير\s+عام|"
        r"مدير\s+(?:اداره|قسم)|"
        r"مدير\s+تنفيذي|"
        r"رئيس\s+(?:قسم|اداره|المراجعه)|"
        r"نائب\s+الرئيس(?:\s+التنفيذي)?|"
        r"مساعد\s+الرئيس(?:\s+التنفيذي)?|"
        r"المشرف\s+العام|"
        r"مستشار(?:\s+|$)|"
        r"كبير\s+تقني(?:ي)?|"
        r"كبير\s+[^\s]+|"
        r"general\s+manager|director|head\s+of|chief|manager"
        r")(?:\s+.+)$",
        role_candidate_norm,
        re.I,
    ))

    if explicit_role_shape:
        return make(
            intent="employee_role_search",
            operation="search",
            targets=[],
            fields=["name", "job_title", "department"],
            role_query=role_candidate,
            query="role:explicit",
        )

    # HARD CONTENT FIREWALL: full-detail content phrases are never employee names.
    # This also protects conversations that started before selected content was
    # persisted in list_state. If no exact selected item exists yet, the detail
    # executor will ask for the item number within the SAME content domain.
    _content_full_cue = bool(re.search(
        r"(?:التفاصيل\s+(?:كامله|كاملة)(?:\s+(?:للخبر|للاعلان|للإعلان|للميزه|للميزة|للسياسه|للسياسة))?|"
        r"كامل\s+التفاصيل|كل\s+التفاصيل|التفاصيل\s+بالكامل|"
        r"الخبر\s+كامل|كامل\s+الخبر|الاعلان\s+كامل|الإعلان\s+كامل|"
        r"كامل\s+الاعلان|كامل\s+الإعلان|full\s+details?|full\s+article)",
        norm, re.I,
    ))
    if _content_full_cue:
        _ls = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
        _base_to_detail = {
            "news": "news_details",
            "announcements": "announcement_details",
            "mazaya": "mazaya_details",
            "policy_search": "policy_details",
        }
        _detail_intent = str(_ls.get("selected_detail_intent") or "")
        if _detail_intent not in DETAIL_BASE_INTENT:
            # Explicit noun in CURRENT_MESSAGE wins; otherwise inherit list domain.
            if re.search(r"(?:خبر|اخبار|أخبار|news|article)", norm, re.I):
                _detail_intent = "news_details"
            elif re.search(r"(?:اعلان|إعلان|announcement)", norm, re.I):
                _detail_intent = "announcement_details"
            elif re.search(r"(?:ميزه|ميزة|مزايا|عرض|benefit|offer)", norm, re.I):
                _detail_intent = "mazaya_details"
            elif re.search(r"(?:سياسه|سياسة|policy|procedure)", norm, re.I):
                _detail_intent = "policy_details"
            else:
                _detail_intent = _base_to_detail.get(str(_ls.get("intent") or ""), "")
        if _detail_intent in DETAIL_BASE_INTENT:
            return make(
                intent=_detail_intent, operation="details", targets=[], fields=["all"],
                ordinal=0, query="content:full",
            )

    # Direct employee contact/profile requests bypass the unreliable general LLM route.
    # Do not match service requests such as leave/attendance/meetings.
    from core.direct_employee_intent import direct_employee_lookup
    direct_lookup = direct_employee_lookup(raw)
    if direct_lookup:
        mode, person_query, wanted_fields = direct_lookup
        if mode == "email_lookup":
            return make(intent="employee_search", operation="search", fields=wanted_fields,
                        query=person_query, confidence=1.0)
        return make(intent="employee_details", operation="details",
                    targets=[TargetRef(kind="employee", name=person_query)],
                    fields=wanted_fields, query="employee:direct_lookup", confidence=1.0)

    # Explicit employee-name search. Keep this narrow so policies/news/content
    # searches are not mistaken for people-directory searches.
    employee_search_match = re.fullmatch(
        r"(?:(?:طيب|طب|تمام|اوكي|أوكي|اوك|حلو|ممكن|بس)\s+)*"
        r"(?:(?:ابي|أبي|ابغى|أبغى)\s+)?"
        r"(?:ابحث|أبحث|دور|دوّر|دورلي|دور\s+لي|find|search)"
        r"(?:\s+لي)?\s+(?:عن|على|for)?\s*(.+)",
        norm,
        re.I,
    )
    if employee_search_match:
        employee_query = str(employee_search_match.group(1) or "").strip(" ؟?!.,،")
        blocked_domains = bool(re.search(
            r"(?:سياسه|سياسة|خبر|اخبار|أخبار|اعلان|إعلان|مزايا|مزايا|"
            r"policy|news|announcement|mazaya)",
            employee_query,
            re.I,
        ))
        plural_directory = bool(re.search(
            r"(?:موظفين|الموظفين|الموظفات|employees|staff)",
            employee_query,
            re.I,
        ))
        if employee_query and not blocked_domains and not plural_directory:
            focus_state = state.setdefault("focus", {})
            previous_current = focus_state.get("current") if isinstance(focus_state.get("current"), dict) else {}
            if previous_current:
                focus_state["previous"] = previous_current
            focus_state["current"] = {}
            _clear_pending(state)
            print(
                f"[Explicit Employee Search] query={employee_query!r} cleared_old_focus=True",
                file=sys.stderr,
            )
            return make(
                intent="employee_search",
                operation="search",
                targets=[],
                fields=["name", "job_title", "department"],
                query=employee_query,
            )

    alias_map = {
        "it": "التقنية والرقمنة",
        "ict": "التقنية والرقمنة",
        "hr": "الموارد البشرية",
        "pmo": "المشاريع",
        "cybersecurity": "الأمن السيبراني",
        "cyber security": "الأمن السيبراني",
    }
    dept_query = ""
    for alias, canonical in alias_map.items():
        if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", norm, re.I):
            dept_query = canonical
            break
    if not dept_query and re.search(
        r"(?:التقنيه|التقنية)\s*(?:و|وال)?\s*(?:الرقمنه|الرقمنة)|"
        r"(?:technology|digitalization|digitization)",
        norm,
        re.I,
    ):
        dept_query = "التقنية والرقمنة"

    if not dept_query:
        m_dep = re.search(
            r"((?:ال)?(?:اداره|ادارة|إدارة|ادراة)(?:\s+(?:ال)?(?:عامه|عامة))?\s+[^؟?!.،,]+)",
            norm,
            re.I,
        )
        if m_dep:
            dept_query = str(m_dep.group(1) or "").strip()
    # Natural directory-list wording, including common spelling variants:
    # "قائمة موظفين إدارة التقنية", "موظفي ادراة التقنية", "staff in IT".
    department_member_cue = bool(re.search(
        r"(?:موظف|موظفين|موظفي|موظفو|الموظفين|الموظفات|employees?|staff)",
        norm,
        re.I,
    ))

    # Department employee requests are intentionally two-step by default:
    #
    #   "ابي موظفين الـ IT"          -> summary/count + offer to show list
    #   "كم عدد موظفين الـ IT؟"      -> count only
    #   "ابي قائمة موظفين الـ IT"    -> list immediately
    #   "اعرض موظفين الـ IT"         -> list immediately
    #
    # This keeps large employee lists from flooding the chat unless the user
    # explicitly asks to see them.
    if department_member_cue and dept_query:
        explicit_count = bool(re.search(
            r"(?:كم|عدد|اجمالي|إجمالي|how\s+many|count)",
            norm,
            re.I,
        ))
        explicit_list = bool(re.search(
            r"(?:قائمه|قائمة|قايمه|قايمة|ليست|اعرض|أعرض|ورني|وريني|هات\s+القائمه|"
            r"هات\s+القائمة|هات\s+القايمه|عطني\s+القائمه|عطني\s+القائمة|عطني\s+القايمه|"
            r"اعطني\s+القائمه|اعطني\s+القائمة|اعطني\s+القايمه|"
            r"list|show|show\s+me)",
            norm,
            re.I,
        ))

        if explicit_count:
            names_too = bool(re.search(
                r"(?:اسماء|أسماء|اسامي|اسمائهم|names)",
                norm,
                re.I,
            ))
            return make(
                intent="general_department_counts",
                operation="calculate",
                targets=[],
                fields=["count"] + (["name"] if names_too else []),
                department_query=dept_query,
                query="department:count",
            )

        if explicit_list:
            return make(
                intent="general_department_members",
                operation="list",
                targets=[],
                fields=["name", "job_title"],
                department_query=dept_query,
                query="department:members",
            )

        # Default/general mention = two-step summary.
        return make(
            intent="general_department_counts",
            operation="lookup",
            targets=[],
            fields=["count"],
            department_query=dept_query,
            query="department:summary",
        )
    if (str(last_req_for_units.get("intent") or "") == "general_department_members" and dept_query and len(norm.split()) <= 8 and re.search(r"(?:موظف|موظفين|موظفي|ادار|قسم|وحد|قطاع|موظفو|employees?|staff|departments?|divisions?|teams?)", norm, re.I) and not re.search(r"(?:بصم|داوم|دوام|حضور|تأخير|attendance|check.?in|working hours|اجاز|إجاز|رصيد|اجتماع|meeting)", norm, re.I)):
        return make(
            intent="general_department_members", operation="list", targets=[],
            fields=["name", "job_title"], department_query=dept_query,
            query="department:members_followup",
        )

    # RCRC-FIX-v136: descriptive organization questions are general knowledge
    # questions, not employee names. This list is intentionally narrow and does
    # NOT include "مزايا", "أخبار", "إعلانات", "سياسات", etc. because those have
    # dedicated enterprise APIs below.
    if (
        re.search(r"(?:الهيئه|الهييه|الهيئة|rcrc)", norm, re.I)
        and re.search(
            r"(?:انجازات|انجاز|إنجازات|إنجاز|مشاريع|مشروع|اهداف|هدف|أهداف|"
            r"رؤيه|رؤية|رساله|رسالة|اختصاصات|مهام|"
            r"achievements?|projects?|goals?|vision|mission)",
            norm,
            re.I,
        )
    ):
        return make(
            intent="general",
            operation="chat",
            targets=[],
            fields=[],
            query=raw,
            confidence=0.99,
            clarification_reason="",
        )

    if re.search(r"(?:تراسل|معامل(?:تي|اتي|ه|ة|ات)?|المعامل(?:ه|ة|ات)?|transaction)", norm, re.I):
        if re.search(r"(?:مديري|المدير\s+حقي|my\s+manager)", norm, re.I):
            return make(
                intent="tarasol_details", operation="details",
                targets=[TargetRef(kind="manager_of_self", name="")],
                fields=["all"], query="tarasol:other_private",
            )
        # RCRC-FIX-v4: "كم/عدد" means a count even with "عندي".
        if re.search(r"(?:كم|عدد|count|how\s+many)", norm, re.I) and not re.search(r"(?:تفاصيل|details)", norm, re.I):
            return make(
                intent="tarasol_count", operation="lookup",
                targets=[TargetRef(kind="self", name="")], fields=["count"],
            )
        if re.search(r"(?:تفاصيل|معاملتي|معاملاتي|حقتي|حقي|عندي|details|my\s+transactions?)", norm, re.I):
            return make(
                intent="tarasol_details", operation="details",
                targets=[TargetRef(kind="self", name="")], fields=["all"],
                query="tarasol:self_details",
            )
        if re.search(r"(?:كم|عدد|count|how\s+many)", norm, re.I):
            return make(
                intent="tarasol_count", operation="lookup",
                targets=[TargetRef(kind="self", name="")], fields=["count"],
            )

    # ------------------------------------------------------------------
    # v69 demo-stability firewall. These are small closed grammars that
    # must never be reinterpreted by Qwen or stale employee focus.
    # ------------------------------------------------------------------
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    current_focus = focus.get("current") if isinstance(focus.get("current"), dict) else {}
    has_focus = bool(current_focus.get("username") or current_focus.get("userid"))
    last = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
    last_intent = str(last.get("intent") or "")

    relation_state = state.get("relation") if isinstance(state.get("relation"), dict) else {}
    relation_subject = relation_state.get("subject") if isinstance(relation_state.get("subject"), dict) else {}
    relation_object = relation_state.get("object") if isinstance(relation_state.get("object"), dict) else {}

    def _person_tokens(value: str) -> list[str]:
        value = str(value or "").casefold()
        value = value.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
        value = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", value)
        return [
            token for token in value.split()
            if token not in {"بن", "ابن", "بنت", "al", "bin", "bint"} and len(token) >= 2
        ]

    def _mentions_employee(text: str, employee: dict[str, Any]) -> bool:
        if not employee:
            return False
        query_tokens = _person_tokens(text)
        if not query_tokens:
            return False
        employee_text = " ".join(str(employee.get(k) or "") for k in (
            "name", "name_ar", "name_en", "username", "userid"
        ))
        employee_tokens = set(_person_tokens(employee_text))
        return all(token in employee_tokens for token in query_tokens)

    def recent_unique_employee(name_text: str) -> dict[str, Any]:
        """Resolve a short name against the immediately previous verified list."""
        list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
        if str(list_state.get("intent") or "") not in {"reporting_employees", "employee_search", "employee_role_search"}:
            return {}
        query = str(name_text or "").casefold().translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
        tokens = [t for t in re.findall(r"[0-9a-z\u0600-\u06ff]+", query) if len(t) >= 2]
        if not tokens:
            return {}
        matches: list[dict[str, Any]] = []
        for row in [x for x in (list_state.get("items") or []) if isinstance(x, dict)]:
            hay = " ".join(str(row.get(k) or "") for k in ("name","name_ar","name_en","job_title","job_title_ar","job_title_en","username")).casefold()
            hay = hay.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
            if all(re.search(rf"(?:^|\s){re.escape(t)}(?:\s|$)", hay) for t in tokens):
                matches.append(row)
        return matches[0] if len(matches) == 1 else {}

    def recent_employee_mentioned(message_text: str) -> dict[str, Any]:
        """Return one recently verified employee explicitly mentioned now.

        This is intentionally limited to the immediately previous verified
        employee/result list. It lets phrases such as "منصب عدنان وايميله"
        bind back to the verified عدنان result without treating "منصب عدنان"
        as a literal employee name.
        """
        list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
        if str(list_state.get("intent") or "") not in {"employee_search", "employee_role_search", "reporting_employees"}:
            return {}
        text_norm = str(message_text or "").casefold().translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
        text_tokens = set(re.findall(r"[0-9a-z\u0600-\u06ff]+", text_norm))
        if not text_tokens:
            return {}
        hits: list[dict[str, Any]] = []
        for row in [x for x in (list_state.get("items") or []) if isinstance(x, dict)]:
            name = str(row.get("name_ar") or row.get("name") or row.get("name_en") or "").strip()
            if not name:
                continue
            name_norm = name.casefold().translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
            name_tokens = [t for t in re.findall(r"[0-9a-z\u0600-\u06ff]+", name_norm) if t not in {"بن","بنت","ابن","al","bin","bint"}]
            if not name_tokens:
                continue
            overlap = [t for t in name_tokens if t in text_tokens]
            # One explicit unique first/family-name token is enough only inside
            # the already verified recent list; otherwise require two tokens.
            if len(overlap) >= min(2, len(name_tokens)) or (len(overlap) == 1 and len(list_state.get("items") or []) == 1):
                hits.append(row)
        return hits[0] if len(hits) == 1 else {}

    # RCRC-FIX-v5: my manager's team ("الموظفين اللي تحت مديري", "فريق مديري").
    if re.search(
        # RCRC-FIX-v6: "تحت/عند/يتبعون" is REQUIRED (before, "مين مديري" matched this rule).
        r"(?:(?:الموظفين|موظفين|موظفي|مين|الناس)\s+(?:اللي\s+|الي\s+)?(?:تحت|عند|يتبعون|تابعين\s+ل)\s*مديري(?:\s|$)"
        r"|(?:^|\s)فريق\s+مديري(?:\s|$)"
        r"|my\s+manager'?s\s+team|(?:team|employees)\s+(?:of|under)\s+my\s+manager)",
        norm, re.I,
    ) and not re.search(r"(?:مدير\s+مديري)", norm):
        return make(
            intent="reporting_employees", operation="list",
            targets=[TargetRef(kind="manager_of_self", name="")], fields=["name", "job_title"],
            query="reporting:manager_of_self",
        )

    # RCRC-FIX-v3: the signed-in user's own team.
    if re.search(
        r"(?:فريقي|موظفيني|(?:الموظفين|موظفين|مين)\s+(?:اللي\s+|الي\s+)?(?:تحتي|يتبعوني|يتبعني|تابعين\s+لي)|"
        r"my\s+team|my\s+direct\s+reports|who\s+reports\s+to\s+me)",
        norm, re.I,
    ):
        return make(
            intent="reporting_employees", operation="list",
            targets=[TargetRef(kind="self", name="")], fields=["name", "job_title"],
            query="reporting:self",
        )

    # Natural self-recognition variants. These are first-person identity
    # questions, never employee-directory searches.
    if norm in {
        "you know me", "do you know me", "you knew me", "do you remember me",
        "تعرفني", "تعرفني انا", "تذكرني",
    }:
        return make(
            intent="profile", operation="details",
            targets=[TargetRef(kind="self", name="")],
            fields=["name", "job_title", "department"], query="profile:recognition",
        )

    # ------------------------------------------------------------------
    # v72 demo routing guard. Domain switches and relational follow-ups
    # are resolved here before contact-name parsing or Qwen.
    # ------------------------------------------------------------------

    # RCRC-FIX-v1 (إضافة): "لخصها / اختصرها" بعد قائمة موثقة تلخص القائمة من
    # البيانات الحقيقية (summarize_previous) بدل ما تروح للموديل العام ويخترع.
    _fix_list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    if _fix_list_state.get("items") and re.fullmatch(
        r"(?:(?:طيب|تمام|ممكن|لو\s+سمحت)\s+)?(?:لخص|اختصر)(?:ها|ه|هم)?(?:\s+لي)?|"
        r"(?:please\s+)?(?:summari[sz]e|shorten)(?:\s+(?:it|them|that))?",
        norm, re.I,
    ):
        return make(intent="summarize_previous", operation="summarize", query="summarize:previous_list")

    # Writing/drafting must win even when the text mentions a manager or an
    # email. Example: "اكتب لي ايميل شكر برسله لمديري" is a writing task,
    # not an employee-directory email lookup.
    writing_cue_early = bool(re.search(
        r"(?:اكتب(?:\s+لي)?|صيغ(?:\s+لي)?|صياغه|صياغة|اعد\s+صياغ|أعد\s+صياغ|"
        r"اختصر(?:ها|ه)?|خله\s+رسمي|خليها\s+رسمي|ترجم(?:ها|ه)?|"
        r"write\s+(?:me\s+)?|draft\s+(?:me\s+)?|rewrite|rephrase|translate|make\s+it\s+(?:shorter|formal|professional))",
        norm, re.I,
    ))
    # RCRC-FIX-v1 (تعديل): "اشرح/لخص سياسة أو خبر" لازم يروح للـ API.
    if writing_cue_early and not is_content_explain_request(norm):
        return make(intent="general", operation="chat", query="general:writing")

    # Mazaya / employee benefits are an enterprise content domain.
    # Route natural variants to the verified Mazaya API instead of general chat.
    # Examples:
    #   "ايش المميزات الي تقدمها لنا الهيئة؟"
    #   "وش المزايا المتاحة؟"
    #   "ابي الامتيازات والعروض"
    #   "what benefits does RCRC offer?"
    mazaya_cue = bool(re.search(
        r"(?:مزايا|المزايا|ميزه|ميزة|مميزات|المميزات|امتياز|امتيازات|الامتيازات|"
        r"benefit|benefits|perk|perks|mazaya)",
        norm,
        re.I,
    ))
    mazaya_org_cue = bool(
        re.search(r"(?:الهيئه|الهيئة|rcrc|لنا|للموظفين|employee)", norm, re.I)
        or re.search(r"(?:تقدم|توفر|تعطي|offer|provide)", norm, re.I)
    )
    if mazaya_cue and mazaya_org_cue:
        return make(
            intent="mazaya",
            operation="list",
            targets=[],
            fields=["all"],
            query="",
        )

    # Policies/procedures are their own enterprise domain. Never let a phrase
    # such as "ابحث عن سياسة العمل عن بعد" fall through to employee search.
    policy_cue = bool(re.search(
        r"(?:سياس(?:ه|ة|ات)|اجراء(?:ات)?|إجراء(?:ات)?|"
        r"قانون|قوانين|لائحه|لائحة|لوائح|نظام|انظمه|أنظمة|"
        r"policy|policies|procedure|procedures|rules?|regulations?)",
        norm, re.I,
    ))
    if policy_cue:
        # General/list-style policy requests retrieve all available policies.
        if _is_generic_policy_request(raw):
            return make(
                intent="policy_search", operation="search", targets=[], fields=[],
                query="",
            )

        policy_query = raw
        policy_query = re.sub(
            r"^(?:(?:ابحث(?:\s+لي)?\s+عن|دور\s+على|اعرض(?:\s+لي)?|ابي|ابغى|وش|ايش|"
            r"search\s+(?:for\s+)?|find\s+|show\s+(?:me\s+)?|what\s+(?:are|is)\s+the\s+))",
            "", policy_query, flags=re.I,
        ).strip()
        policy_query = re.sub(r"^(?:هي|the)\s+", "", policy_query, flags=re.I).strip()
        policy_query = re.sub(
            r"^(?:ال)?(?:سياس(?:ه|ة|ات)|اجراء(?:ات)?|إجراء(?:ات)?|policy|policies|procedure|procedures)\s*",
            "", policy_query, flags=re.I,
        ).strip(" ؟?!.,،")
        # Generic list words are not search terms. "وش السياسات المتاحة؟"
        # should list policies instead of searching literally for "المتاحة".
        if re.fullmatch(
            r"(?:المتاحه|المتاحة|الموجوده|الموجودة|كلها|جميعها|available|all|list)?",
            policy_query.casefold(), re.I,
        ):
            policy_query = ""
        return make(
            intent="policy_search", operation="search", targets=[], fields=[],
            query=policy_query,
        )

    # Department follow-ups must respect the PERSON explicitly mentioned in the
    # current message, even when the previous manager lookup moved focus to that
    # person's manager. Example:
    #   "مين مدير عبدالمحسن؟" -> focus becomes the manager
    #   "عبدالمحسن تحت اي ادارة" -> relation_subject, NOT current manager focus
    department_match = re.search(
        r"\s+(?:تحت\s+(?:اي|أي)\s+اداره|في\s+(?:اي|أي)\s+اداره|"
        r"تابع\s+(?:(?:لاي|لأي|اي|أي)\s+)?اداره|"
        r"يتبع\s+(?:(?:لاي|لأي|اي|أي)\s+)?اداره|"
        r"under\s+which\s+department|which\s+department)\s*$",
        norm, re.I,
    )
    if department_match:
        person_text = norm[:department_match.start()].strip()
        person_text = re.sub(
            r"^(?:(?:ابي|ابغى|اريد|اعرف|ابحث\s+عن|ابحث\s+لي\s+عن|"
            r"وش|ايش|what\s+about)\s+)+",
            "", person_text, flags=re.I,
        ).strip()

        if person_text in {"انا", "أنا", "me", "myself"}:
            target = TargetRef(kind="self", name="")
        elif person_text in {"هو", "هي", "هذا", "هذي", "هذاك", "ذي", "he", "she"} and has_focus:
            target = TargetRef(kind="focus", name="")
        elif person_text and _mentions_employee(person_text, relation_subject):
            target = TargetRef(kind="relation_subject", name="")
        elif person_text and _mentions_employee(person_text, relation_object):
            target = TargetRef(kind="relation_object", name="")
        elif person_text and _mentions_employee(person_text, current_focus):
            target = TargetRef(kind="focus", name="")
        elif person_text:
            target = TargetRef(kind="employee", name=person_text)
        elif has_focus:
            target = TargetRef(kind="focus", name="")
        else:
            target = TargetRef(kind="self", name="")

        return make(
            intent="employee_details", operation="details",
            targets=[target], fields=["department"],
            query="employee:department",
        )

    # Public contact fields for the signed-in user's direct manager. This must
    # beat generic contact-name parsing: "my manager email" is a relation, not
    # a person literally named "manager".
    manager_self_ref = bool(re.search(r"(?:مديري|المدير\s+حقي|my\s+manager(?:'s)?)", norm, re.I))
    if manager_self_ref:
        manager_fields: list[str] = []
        if re.search(r"(?:\bاسم\b|اسمه|name)", norm, re.I):
            manager_fields.append("name")
        if re.search(r"(?:ايميل|ايمميل|إيميل|بريد|email)", norm, re.I):
            manager_fields.append("email")
        if re.search(r"(?:رقم\s+(?:جوال|موبايل)|جوال|موبايل|phone|mobile|number)", norm, re.I):
            manager_fields.append("mobile")
        if re.search(r"(?:تحويل|extension)", norm, re.I):
            manager_fields.append("extension")
        if re.search(r"(?:منصب|المسمى\s+الوظيفي|المسمي\s+الوظيفي|مسمى|مسمي|job\s+title|position|title)", norm, re.I):
            manager_fields.append("job_title")
        if re.search(r"(?:ادارته|إدارته|اداره|إدارة|department)", norm, re.I):
            manager_fields.append("department")
        if manager_fields:
            return make(
                intent="manager", operation="details",
                targets=[TargetRef(kind="self", name="")], fields=list(dict.fromkeys(manager_fields)),
                relation_depth=1, query="manager:self_fields",
            )

    # Explicit named-manager relation: "مين مدير ماجد السلطان"
    # The employee named in the current message always beats old focus.
    named_manager_match = re.match(
        r"^(?:مين|من|who\s+is)\s+(?:هو\s+|هي\s+)?(?:مدير|مديرة|manager(?:\s+of)?)\s+(.+?)$",
        norm,
        re.I,
    )
    if named_manager_match:
        person_name = re.sub(
            r"[؟?!.،,]+$",
            "",
            str(named_manager_match.group(1) or ""),
        ).strip()
        if person_name and person_name not in {"ي", "لي", "حقي", "انا", "أنا", "me", "myself"}:
            return make(
                intent="manager",
                operation="details",
                targets=[TargetRef(kind="employee", name=person_name)],
                fields=["name"],
                relation_depth=1,
                query="manager:named_employee",
            )

    # Natural self leave-balance family.
    # Covers wording such as:
    #   "كم عندي من رصيد اجازات؟"
    #   "وش باقي لي اجازات؟"
    #   "كم باقي من اجازتي؟"
    #   "how much leave do I have left?"
    self_leave_balance = bool(
        re.search(
            r"(?:رصيد\s*(?:ال)?اجاز|رصيد\s*(?:ال)?إجاز|"
            r"(?:كم|وش|ايش|ما)\s+(?:باقي|متبقي|متبقى|عندي).*?(?:اجاز|إجاز)|"
            r"(?:اجاز|إجاز).*?(?:باقي|متبقي|متبقى|رصيد)|"
            r"(?:leave|vacation).*?(?:balance|remaining|left)|"
            r"(?:how\s+much|what).*?(?:leave|vacation).*?(?:have|left|remaining))",
            norm,
            re.I,
        )
    )
    if self_leave_balance and not re.search(
        r"(?:مديري|manager|اجازاته|إجازاته|اجازتها|إجازتها|his\s+leave|her\s+leave)",
        norm,
        re.I,
    ):
        return make(
            intent="leave_balance",
            operation="lookup",
            targets=[TargetRef(kind="self", name="")],
            fields=["balance"],
            query="leave:self_balance",
        )

    # RCRC-FIX-v137: third-person leave wording must preserve the referenced
    # employee. Never silently convert "رصيد اجازاته" / "إجازاتها" to self.
    third_person_leave = bool(
        has_focus
        and re.search(r"(?:اجازاته|إجازاته|اجازتها|إجازتها|رصيد(?:\s+)?اجازاته|رصيد(?:\s+)?إجازاته|his\s+leave|her\s+leave)", norm, re.I)
    )
    if third_person_leave:
        if re.search(r"(?:رصيد|balance)", norm, re.I):
            return make(
                intent="leave_balance",
                operation="lookup",
                targets=[TargetRef(kind="focus", name="")],
                fields=["balance"],
                query="leave:focus_private",
            )
        return make(
            intent="leave_history",
            operation="list",
            targets=[TargetRef(kind="focus", name="")],
            fields=["all"],
            period="none",
            query="leave:focus_history_private",
            clarification_reason="missing_date_range",
        )

    # A comparison with another employee's leave balance can never expose the
    # other employee's private balance. Fetch only the signed-in user's balance
    # and explain the privacy boundary in the renderer path.
    if (
        re.search(r"(?:قارن|قارنه|compare)", norm, re.I)
        and re.search(r"(?:رصيد|balance)", norm, re.I)
        and re.search(r"(?:اجاز|إجاز|leave|vacation)", norm, re.I)
        and re.search(r"(?:مديري|manager)", norm, re.I)
    ):
        return make(
            intent="leave_balance", operation="lookup",
            targets=[TargetRef(kind="self", name="")], fields=["balance"],
            query="leave:compare_manager_private",
        )

    # Simple planned leave such as "اجازه بكره كيف" is a self-service leave
    # question and must not inherit a manager's calendar context.
    if (
        re.search(r"(?:اجازه|اجازة|إجازة|leave|vacation|day\s+off)", norm, re.I)
        and re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I)
        and not re.search(r"(?:عماد|شاهه|اوس|أسامه|ماجد|مديري|مديره|مديرها|employee|manager)", norm, re.I)
    ):
        return make(
            intent="leave_balance", operation="calculate",
            targets=[TargetRef(kind="self", name="")], fields=["balance"], amount=1.0,
            query="leave:planned_tomorrow",
        )

    # RCRC-FIX-v137: "حضوره / حضورها / متى حضر؟" after a verified employee
    # refers to that employee. Keep the target so authorization can deny or allow
    # correctly; never answer with the signed-in user's attendance.
    if has_focus and re.search(
        r"(?:حضوره|حضورها|دوامه|دوامها|متى\s+(?:حضر|حضرت|دخل|دخلت|وصل|وصلت|خرج|خرجت|انصرف|انصرفت)|"
        r"his\s+attendance|her\s+attendance|when\s+did\s+(?:he|she)\s+(?:arrive|check[- ]?in|check[- ]?out|leave))",
        norm,
        re.I,
    ) and not re.search(r"(?:مديري|my\s+manager)", norm, re.I):
        period = "yesterday" if re.search(r"(?:امس|أمس|yesterday)", norm, re.I) else "today"
        checkout = bool(re.search(r"(?:خرج|خرجت|انصرف|انصرفت|check[- ]?out|leave)", norm, re.I))
        return make(
            intent="attendance_history" if period == "yesterday" else "attendance_today",
            operation="lookup",
            targets=[TargetRef(kind="focus", name="")],
            fields=["all"],
            period=period,
            query="attendance:check_out" if checkout else "attendance:check_in",
        )

    # Attendance wording about the signed-in user's manager is attendance, not
    # calendar availability. Authorization will correctly deny the private data.
    if (
        re.search(r"(?:حضر|حضوره|حضور|دخل|خرج|انصرف|دوام|attendance|check[- ]?in|check[- ]?out)", norm, re.I)
        and re.search(r"(?:مديري|my\s+manager)", norm, re.I)
    ):
        period = "yesterday" if re.search(r"(?:امس|أمس|yesterday)", raw, re.I) else "today"
        checkout = bool(re.search(r"(?:خرج|انصرف|check[- ]?out)", norm, re.I))
        query = "attendance:check_out" if checkout else "attendance:check_in"
        intent = "attendance_history" if period == "yesterday" else "attendance_today"
        return make(
            intent=intent, operation="lookup",
            targets=[TargetRef(kind="manager_of_self", name="")],
            fields=["all"], period=period, query=query,
        )

    # "إيميل مديره" means the direct manager of the current verified person,
    # then the requested public directory field(s). It is not a person named
    # "مديره".
    if has_focus and re.search(r"(?:مديره|مديرها|his\s+manager|her\s+manager)", norm, re.I):
        relation_fields: list[str] = []
        if re.search(r"(?:ايميل|إيميل|بريد|email)", norm, re.I):
            relation_fields.append("email")
        if re.search(r"(?:رقم\s+(?:جوال|موبايل)|جوال|موبايل|phone|mobile|ورقم(?:ها|ه)?)", norm, re.I):
            relation_fields.append("mobile")
        if re.search(r"(?:تحويل|extension)", norm, re.I):
            relation_fields.append("extension")
        return make(
            intent="manager", operation="details",
            targets=[TargetRef(kind="focus", name="")], fields=relation_fields or ["name"],
            relation_depth=1, query="manager:contact" if relation_fields else "manager:relation",
        )

    def _explicit_iso_date(text: str) -> str:
        m = re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[-/](\d{4})(?!\d)", text)
        if m:
            day, month, year = map(int, m.groups())
            try:
                from datetime import date as _date
                return _date(year, month, day).isoformat()
            except Exception:
                return ""
        m = re.search(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", text)
        if m:
            year, month, day = map(int, m.groups())
            try:
                from datetime import date as _date
                return _date(year, month, day).isoformat()
            except Exception:
                return ""
        # English natural dates used in demo prompts, e.g.
        # "September 24, 2026" or "24 September 2026".
        month_names = (
            "january|february|march|april|may|june|july|august|"
            "september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
        )
        candidates = []
        m = re.search(rf"\b(({month_names})\s+\d{{1,2}},?\s+\d{{4}})\b", text, re.I)
        if m:
            candidates.append(m.group(1))
        m = re.search(rf"\b((\d{{1,2}})\s+({month_names})\s+\d{{4}})\b", text, re.I)
        if m:
            candidates.append(m.group(1))
        if candidates:
            from datetime import datetime as _datetime
            for candidate in candidates:
                clean = re.sub(r"\s+", " ", candidate.replace(",", " ")).strip()
                for fmt in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
                    try:
                        return _datetime.strptime(clean, fmt).date().isoformat()
                    except Exception:
                        pass
        return ""

    meeting_context = state.get("meeting_context") if isinstance(state.get("meeting_context"), dict) else {}

    # Another employee's meetings are privacy-safe only as busy time slots.
    # Possessive wording follows the current verified employee and preserves an
    # explicit date such as 24-09-2026.
    if has_focus and re.search(r"(?:اجتماعاته|اجتماعاتها|جدول\s+اجتماعاته|جدول\s+اجتماعاتها|his\s+meetings|her\s+meetings)", norm, re.I):
        explicit_date = _explicit_iso_date(raw)
        if explicit_date:
            return make(
                intent="employee_availability", operation="lookup",
                targets=[TargetRef(kind="focus", name="")], fields=["start_time", "end_time"],
                period="custom", start_date=explicit_date, end_date=explicit_date, query="busy",
            )
        period = "tomorrow" if re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I) else "today"
        return make(
            intent="employee_availability", operation="lookup",
            targets=[TargetRef(kind="focus", name="")], fields=["start_time", "end_time"],
            period=period, query="busy",
        )

    # "متى فاضي؟" after a calendar lookup inherits the last employee AND the
    # exact date/range unless the current message explicitly changes the date.
    availability_followup = bool(re.fullmatch(
        r"(?:"
        r"(?:(?:متى|متي|when)\s+)?(?:فاضي|فاضيه|متاح|متاحه|free|available)"
        r"|when\s+is\s+(?:he|she)\s+(?:free|available)(?:\s+on\s+that\s+day)?"
        r")",
        norm, re.I,
    ))
    if has_focus and availability_followup:
        period = str(meeting_context.get("period") or "today")
        start_date = str(meeting_context.get("start_date") or "")
        end_date = str(meeting_context.get("end_date") or "")
        if re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I):
            period, start_date, end_date = "tomorrow", "", ""
        elif re.search(r"(?:اليوم|today)", norm, re.I):
            period, start_date, end_date = "today", "", ""
        return make(
            intent="employee_availability", operation="lookup",
            targets=[TargetRef(kind="focus", name="")], fields=["start_time", "end_time"],
            period=period, start_date=start_date, end_date=end_date, query="free",
        )

    # A bare period after another employee's calendar continues that employee,
    # not the signed-in user's meetings.
    if meeting_context.get("employee") and re.fullmatch(
        r"(?:اليوم|بكره|بكرة|غدا|غداً|today|tomorrow)(?:\s+طيب)?", norm, re.I
    ):
        period = "tomorrow" if re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I) else "today"
        mode = str(meeting_context.get("mode") or "busy")
        return make(
            intent="employee_availability", operation="lookup",
            targets=[TargetRef(kind="employee", name=str((meeting_context.get("employee") or {}).get("name_ar") or (meeting_context.get("employee") or {}).get("name") or ""))],
            fields=["start_time", "end_time"], period=period, query=mode,
        )

    # Natural text selection from a pending employee disambiguation. Preserve
    # the ORIGINAL intent (for example employee details) instead of letting a
    # short correction such as "لا سليمان الماجد" become calendar availability.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if pending.get("kind") == "employee_disambiguation":
        candidates = [x for x in (pending.get("candidates") or []) if isinstance(x, dict)]
        selection = re.sub(
            r"^(?:(?:لا|قصدي|اقصد|أقصد|اي|ايوه|نعم|طيب|تمام|ابي|أبي|ابغى|أبغى|اريد|أريد|عطني|اعطني|هات|الموظف)\s+)+",
            "", norm, flags=re.I,
        ).strip()
        selection_tokens = [t for t in selection.split() if len(t) >= 2]
        hits = []
        if selection_tokens:
            for row in candidates:
                hay = " ".join(str(row.get(k) or "") for k in ("name","name_ar","name_en","job_title","job_title_ar","job_title_en","username")).casefold()
                hay = hay.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
                if all(re.search(rf"(?:^|\s){re.escape(t)}(?:\s|$)", hay) for t in selection_tokens):
                    hits.append(row)
        if len(hits) == 1:
            raw_request = pending.get("request") if isinstance(pending.get("request"), dict) else {}
            try:
                restored = ParsedRequest.model_validate(raw_request)
                selected_name = str(hits[0].get("name_ar") or hits[0].get("name") or hits[0].get("name_en") or "").strip()
                if selected_name:
                    restored.targets = [TargetRef(kind="employee", name=selected_name)]
                    restored.confidence = 1.0
                    restored.clarification_reason = ""
                    return restored
            except Exception:
                pass

    # Contact follow-up for the current VERIFIED person. This runs before
    # named-person extraction so short phrases such as "رقم جواله" or
    # "ابي ايميله ورقم جواله وتحويلته" never become fake employee searches.
    if has_focus:
        focus_contact_patterns: tuple[tuple[str, str], ...] = (
            ("email", r"(?:ايميل(?:ها|ه|ة)|إيميل(?:ها|ه|ة)|بريد(?:ها|ه)|his\s+email|her\s+email)"),
            ("mobile", r"(?:رقم\s+(?:جوال|موبايل)(?:ها|ه)|(?:جوال|موبايل)(?:ها|ه)|his\s+(?:mobile|phone)(?:\s+number)?|her\s+(?:mobile|phone)(?:\s+number)?)"),
            ("extension", r"(?:تحويلت(?:ها|ه)|تحويل(?:ها|ه)|his\s+extension|her\s+extension)"),
        )
        focus_contact_fields: list[str] = []
        contact_residue = re.sub(
            r"^(?:(?:ابي|ابغى|عطني|اعطني|هات|ممكن|please|give\s+me|show\s+me)\s+)+",
            "", norm, flags=re.I,
        ).strip()
        for field_name, pattern in focus_contact_patterns:
            if re.search(pattern, norm, re.I):
                if field_name not in focus_contact_fields:
                    focus_contact_fields.append(field_name)
                contact_residue = re.sub(pattern, " ", contact_residue, flags=re.I)
        # Arabic conjunction "و" is commonly attached to each requested field
        # (for example: "ايميله ورقم جواله وتحويلته").  After removing the
        # field phrases this can leave standalone "و" tokens.  Treat those as
        # separators, not as an employee name.
        contact_residue = re.sub(r"[؟?!.،,]+", " ", contact_residue)
        contact_residue = " ".join(
            token for token in contact_residue.split()
            if token.casefold() not in {"و", "and", "&"}
        )
        if focus_contact_fields and not contact_residue:
            return make(
                intent="employee_details", operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=focus_contact_fields,
                query="employee:focus_fields",
            )

    # Focus-only profile follow-up with one or more requested fields.
    # Examples:
    #   "ووش منصبه وابي ايميله" -> current verified focus, [job_title, email]
    #   "وش ادارته وتحويلته"     -> current verified focus, [department, extension]
    # A real person token left after removing field/filler words means this is
    # NOT a focus-only follow-up and must continue to the named-person path.
    if has_focus:
        focus_detail_fields: list[str] = []
        focus_field_patterns: tuple[tuple[str, str], ...] = (
            ("email", r"(?:ايميل(?:ها|ه|ة)?|ايميله|ايميلة|إيميل(?:ها|ه|ة)?|بريد(?:ها|ه)?|email(?:\s+address)?)"),
            ("mobile", r"(?:رقم\s+(?:جوال|موبايل)(?:ها|ه)?|(?:جوال|موبايل)(?:ها|ه)?|phone|mobile(?:\s+number)?)"),
            ("extension", r"(?:تحويلت(?:ها|ه)|تحويل(?:ها|ه|ة)?|extension)"),
            ("job_title", r"(?:منصب(?:ها|ه)?|مسماه|مسماها|مسمى|مسمي|المسمى\s+الوظيفي|المسمي\s+الوظيفي|job\s+title|position)"),
            ("department", r"(?:ادارت(?:ها|ه)|إدارت(?:ها|ه)|اداره|إدارة|department)"),
        )
        focus_residue = norm
        for field_name, pattern in focus_field_patterns:
            if re.search(pattern, focus_residue, re.I):
                if field_name not in focus_detail_fields:
                    focus_detail_fields.append(field_name)
                focus_residue = re.sub(pattern, " ", focus_residue, flags=re.I)
        focus_residue = re.sub(
            r"(?:^|\s)(?:و)?(?:طيب|تمام|ووش|وش|ايش|ابي|ابغى|اريد|عطني|اعطني|هات|ممكن|please|give\s+me|show\s+me)(?=\s|$)",
            " ", focus_residue, flags=re.I,
        )
        focus_residue = re.sub(r"(?:^|\s)(?:و|and|&)(?:\s|$)", " ", focus_residue, flags=re.I)
        focus_residue = re.sub(r"[؟?!.،,؛;:]+", " ", focus_residue)
        focus_residue = " ".join(focus_residue.split()).strip()
        if focus_detail_fields and not focus_residue:
            return make(
                intent="employee_details", operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=focus_detail_fields,
                query="employee:focus_profile_fields",
            )

    # --------------------------------------------------------------
    # Explicit named contact lookup. A named person in the CURRENT message
    # always beats self/focus memory. This fixes "ايميل شاهه", "ايميل عماد",
    # and combined requests such as "ايميل شاهه ورقمها".
    # --------------------------------------------------------------
    contact_fields: list[str] = []
    if re.search(r"(?:ايميل|ايمميل|إيميل|بريد(?:\s+الكتروني)?|email(?:\s+address)?)", norm, re.I):
        contact_fields.append("email")
    if re.search(r"(?:رقم\s+(?:جوال|موبايل)|جوال|موبايل|phone|mobile)", norm, re.I) or re.search(r"(?:ورقم(?:ها|ه)?)(?:\s|$)", norm, re.I):
        if "mobile" not in contact_fields:
            contact_fields.append("mobile")
    if re.search(r"(?:تحويلت(?:ها|ه)|تحويل(?:ها|ه)|تحويله|تحويلة|extension)", norm, re.I):
        contact_fields.append("extension")

    # English pronoun follow-up firewall. When a verified employee is already
    # in focus, words such as "his" / "her" are references, never names.
    # Run this BEFORE named-person extraction so Critical Fallback cannot turn
    # "can you give me his email and phone number" into an employee search.
    if has_focus and re.search(r"\b(?:his|her|him|he|she)\b", norm, re.I):
        pronoun_fields = list(contact_fields)
        if re.search(r"(?:job\s+title|position|role)", norm, re.I) and "job_title" not in pronoun_fields:
            pronoun_fields.append("job_title")
        if re.search(r"(?:department|division|unit)", norm, re.I) and "department" not in pronoun_fields:
            pronoun_fields.append("department")
        if re.search(r"(?:name|who\s+is\s+(?:he|she))", norm, re.I) and "name" not in pronoun_fields:
            pronoun_fields.append("name")
        if pronoun_fields:
            return make(
                intent="employee_details", operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=list(dict.fromkeys(pronoun_fields)),
                query="employee:focus_fields",
            )

    # Role-title requests can ask for contact fields too. Handle them before
    # named-person extraction so "ايميل مدير إدارة ..." stays a role lookup.
    raw_role_lead = re.sub(
        r"^(?:(?:مين|من)(?:\s+هو|\s+هي)?\s+|who\s+is\s+(?:the\s+)?)",
        "", raw, flags=re.I,
    ).strip()
    raw_role_lead = raw_role_lead.strip(" ؟?!.,")
    # RCRC-FIX-v2: "ابحث لي عن مدير إدارة ..." is also a role lookup.
    raw_role_lead = re.sub(
        r"^(?:(?:ابي|أبي|ابغى|أبغى)\s+)?(?:ابحث|أبحث|دور|دوّر)(?:\s+لي)?\s+(?:عن|على)\s+",
        "", raw_role_lead, flags=re.I,
    ).strip()
    raw_role_lead = re.sub(
        r"^(?:(?:ابي|ابغى|عطني|اعطني)\s+)?(?:ايميل|ايمميل|إيميل|بريد(?:\s+إلكتروني|\s+الكتروني)?|رقم\s+(?:جوال|موبايل)|جوال|موبايل|تحويله|تحويلة)\s+",
        "",
        raw_role_lead,
        flags=re.I,
    ).strip()
    role_detect = raw_role_lead.casefold().translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه"}))
    # A few bilingual role labels used by the mobile directory are normalized to
    # their verified Arabic titles. This is translation only; employee identity
    # still comes from SearchEmployee.
    role_aliases = {
        "head of procurement and contracts": "رئيس قسم المشتريات والعقود",
        "director of benefits and compensation": "مدير إدارة المزايا والتعويضات",
        "executive vice president of the special zones sector": "نائب الرئيس التنفيذي لقطاع المناطق الخاصة",
        "general manager of technology and digitization": "مدير عام الإدارة العامة للتقنية والرقمنة",
        "general manager of technology and digitalization": "مدير عام الإدارة العامة للتقنية والرقمنة",
    }
    alias_key = re.sub(r"[^a-z0-9 ]+", " ", role_detect).strip()
    alias_key = re.sub(r"\s+", " ", alias_key)
    if alias_key in role_aliases:
        raw_role_lead = role_aliases[alias_key]
        role_detect = raw_role_lead.casefold().translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه"}))
    # Common shorthand: IT means the verified Technology & Digitization general
    # department in this directory. Canonicalize only this organizational alias.
    if re.search(r"(?:المدير\s+العام|مدير\s+عام).*(?:\bit\b|اداره\s+ال\s*it)", role_detect, re.I):
        raw_role_lead = "مدير عام الإدارة العامة للتقنية والرقمنة"
        role_detect = "مدير عام الاداره العامه للتقنيه والرقمنه"
    elif re.search(r"general\s+manager.*\bit\b", role_detect, re.I):
        # Search the authoritative Arabic title for reliability; the final
        # response still uses the verified English job title when the user asks
        # in English.
        raw_role_lead = "مدير عام الإدارة العامة للتقنية والرقمنة"
        role_detect = "مدير عام الاداره العامه للتقنيه والرقمنه"
    role_shape = bool(re.match(
        r"^(?:(?:ال)?مدير\s+عام|نائب\s+الرئيس(?:\s+التنفيذي)?|مساعد\s+الرئيس(?:\s+التنفيذي)?|"
        r"المشرف\s+العام|رئيس\s+(?:قسم|المراجعه)|مدير\s+(?:اداره|قسم)|"
        r"مدير\s+تنفيذي|مستشار\s+|executive\s+vice\s+president|vice\s+president|"
        r"general\s+manager|director|head\s+of|chief|manager).+",
        role_detect,
        re.I,
    ))
    if role_shape and not re.search(r"^(?:مديره|مديرها|مديرهم)$", role_detect, re.I):
        fields = contact_fields or ["name", "job_title", "department"]
        return make(
            intent="employee_role_search",
            operation="search",
            targets=[],
            fields=fields,
            role_query=raw_role_lead,
            query="role:explicit",
        )

    # If there is no verified focus, a possessive-only contact phrase still is
    # not an employee name. Ask for the person instead of searching the literal
    # words as a directory query.
    if contact_fields and not has_focus:
        contact_only = norm
        contact_only = re.sub(r"^(?:(?:ابي|ابغى|عطني|اعطني|هات|ممكن)\s+)+", "", contact_only, flags=re.I).strip()
        for pattern in (
            r"(?:ايميل(?:ها|ه)|إيميل(?:ها|ه)|بريد(?:ها|ه))",
            r"(?:رقم\s+(?:جوال|موبايل)(?:ها|ه)|(?:جوال|موبايل)(?:ها|ه))",
            r"(?:تحويلت(?:ها|ه)|تحويل(?:ها|ه))",
        ):
            contact_only = re.sub(pattern, " ", contact_only, flags=re.I)
        contact_only = re.sub(r"(?:\s+و\s*|^\s*و\s*)", " ", contact_only)
        contact_only = " ".join(contact_only.split())
        if not contact_only:
            return make(
                intent="general", operation="chat",
                clarification_reason="missing_contact_subject",
            )

    # A generic contact phrase is never an employee name.  If the wording looks
    # like a contact request but does not contain a real person name, either use
    # the verified focus or ask for the subject instead of searching the literal
    # phrase in the directory.  This also protects minor typos around "جوال".
    # RCRC-FIX-v4: first-person forms ("تحويلتي", "ايميلي") are self fields, handled below.
    self_contact_form = bool(re.search(r"(?:ايميلي|بريدي|جوالي|موبايلي|تحويلتي|my\s+(?:email|phone|mobile|extension))", norm, re.I))
    if not contact_fields and not self_contact_form and re.search(r"(?:جوال|موبايل|تحويل|ايميل|إيميل|بريد|phone|mobile|extension|email)", norm, re.I):
        generic_contact = re.sub(
            r"^(?:(?:ابي|ابغى|عطني|اعطني|هات|ممكن|وش|ايش|what(?:'s| is)?|please)\s+)+",
            "", norm, flags=re.I,
        ).strip()
        if re.fullmatch(r"(?:رقم\s+)?(?:جوال\S*|موبايل\S*|phone\S*|mobile\S*|تحويل\S*|ايميل\S*|إيميل\S*|بريد\S*)", generic_contact, re.I):
            if has_focus:
                # Ambiguous possessive typo: prefer the verified conversational
                # subject rather than inventing a directory search.
                guessed_field = "mobile" if re.search(r"(?:جوال|موبايل|phone|mobile)", generic_contact, re.I) else (
                    "extension" if re.search(r"(?:تحويل|extension)", generic_contact, re.I) else "email"
                )
                return make(
                    intent="employee_details", operation="details",
                    targets=[TargetRef(kind="focus", name="")],
                    fields=[guessed_field], query="employee:focus_fields",
                )
            return make(
                intent="general", operation="chat",
                clarification_reason="missing_contact_subject",
            )

    # Named employee + one or more directory fields. Field words are not part
    # of the employee name. This handles natural combinations such as:
    #   "ابي منصب عدنان وايميلة"
    #   "وش إدارة عماد وإيميله"
    named_detail_fields = list(contact_fields)
    if re.search(r"(?:منصب|المسمى\s+الوظيفي|المسمي\s+الوظيفي|مسمى\s+وظيفي|مسمي\s+وظيفي|job\s+title|position)", norm, re.I):
        if "job_title" not in named_detail_fields:
            named_detail_fields.append("job_title")
    if re.search(r"(?:\bاداره\b|\bإدارة\b|\bdepartment\b)", raw, re.I):
        if "department" not in named_detail_fields:
            named_detail_fields.append("department")

    if named_detail_fields and not re.search(r"(?:ايميلي|بريدي|جوالي|موبايلي|تحويلتي|منصبي|مسماي|ادارتي|إدارتي|my\s+email|my\s+(?:phone|mobile)|my\s+(?:job\s+)?title|my\s+department)", norm, re.I):
        # Parse an explicitly typed employee in the CURRENT message before using
        # recent conversational focus. Example: "ابي ايميل اوس الدعجان" after
        # "مين مديري" must refer to أوس, not the manager.
        detail_name = norm
        detail_name = re.sub(r"^(?:(?:طيب|تمام|ووش|وش|ايش|ابي|ابغى|عطني|اعطني|هات|ممكن|please|give\s+me)\s+)+", "", detail_name, flags=re.I).strip()
        # Remove field phrases wherever they appear, including attached Arabic
        # conjunction forms such as "وايميلة".
        field_noise = (
            r"(?:و)?(?:منصب(?:ها|ه)?|المسمى\s+الوظيفي|المسمي\s+الوظيفي|مسمى\s+وظيفي|مسمي\s+وظيفي|"
            r"ايميل(?:ها|ه|ة)?|إيميل(?:ها|ه|ة)?|بريد(?:ها|ه)?|رقم\s+(?:جوال|موبايل)(?:ها|ه)?|"
            r"جوال(?:ها|ه)?|موبايل(?:ها|ه)?|تحويلت(?:ها|ه)|تحويل(?:ها|ه|ة)?|"
            r"ادارت(?:ها|ه)?|إدارت(?:ها|ه)?|job\s+title|position|email|mobile|phone|extension|department)"
        )
        detail_name = re.sub(field_noise, " ", detail_name, flags=re.I)
        # Remove request filler even when Arabic "و" is attached ("وابي").
        detail_name = re.sub(
            r"(?:^|\s)(?:و)?(?:ابي|ابغى|عطني|اعطني|هات|ممكن|وش|ووش|please|give\s+me)(?=\s|$)",
            " ", detail_name, flags=re.I,
        )
        detail_name = re.sub(r"(?:^|\s)(?:و|and|&)(?:\s|$)", " ", detail_name, flags=re.I)
        detail_name = " ".join(detail_name.split()).strip(" ؟?!.,،")
        significant_name_tokens = _person_tokens(detail_name)
        # RCRC-FIX-v5: "منصب مديري" / "ايميل مديري" -> my manager, not a person named "مديري".
        slot_target, _slot_bonus = _person_slot_target(detail_name, state)
        if detail_name and slot_target.kind != "employee":
            return make(
                intent="profile" if slot_target.kind == "self" else "employee_details",
                operation="details", targets=[slot_target],
                fields=list(dict.fromkeys(named_detail_fields)),
                query="employee:slot_fields",
            )
        if detail_name and significant_name_tokens and detail_name not in {"ه", "ها", "هو", "هي", "عنه", "عنها", "له", "لها"}:
            recent = recent_unique_employee(detail_name)
            explicit_name = str(recent.get("name_ar") or recent.get("name") or recent.get("name_en") or detail_name).strip() if recent else detail_name
            return make(
                intent="employee_details", operation="details",
                targets=[TargetRef(kind="employee", name=explicit_name)],
                fields=list(dict.fromkeys(named_detail_fields)),
                query="employee:explicit_fields",
            )
        if named_detail_fields and has_focus:
            return make(
                intent="employee_details", operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=list(dict.fromkeys(named_detail_fields)),
                query="employee:focus_fields",
            )

    # Named employee + directory field(s). Strip the requested field words and
    # keep only the literal person text. Possessive-only forms ("ايميلها") are
    # intentionally excluded here and handled by focus below.
    if contact_fields and not re.search(r"(?:ايميلي|بريدي|جوالي|موبايلي|تحويلتي|my\s+email|my\s+(?:phone|mobile))", norm, re.I):
        name_candidate = norm
        name_candidate = re.sub(r"^(?:(?:ابي|ابغى|عطني|اعطني|please|give\s+me)\s+)+", "", name_candidate, flags=re.I).strip()
        name_candidate = re.sub(r"^(?:ايميل|ايمميل|إيميل|بريد(?:\s+الكتروني)?|email(?:\s+address)?|رقم\s+(?:جوال|موبايل)|جوال|موبايل|phone|mobile|تحويله|تحويلة|extension)\s+", "", name_candidate, flags=re.I).strip()
        # Remove a trailing second-field phrase: "شاهه ورقمها" -> "شاهه".
        name_candidate = re.sub(
            r"\s+و(?:ايميل(?:ها|ه)?|بريد(?:ها|ه)?|رقم(?:\s+(?:جوال|موبايل))?(?:ها|ه)?|جوال(?:ها|ه)?|موبايل(?:ها|ه)?|تحويل(?:ها|ه)?|تحويله|تحويلة)\s*$",
            "",
            name_candidate,
            flags=re.I,
        ).strip()
        # Do not turn a pronoun or one-letter stripping artifact into a fake
        # employee search. A contact-only follow-up belongs to verified focus.
        significant_candidate_tokens = _person_tokens(name_candidate)
        if name_candidate and significant_candidate_tokens and name_candidate not in {"ه", "ها", "هو", "هي", "عنه", "عنها", "له", "لها"} and not re.fullmatch(r"(?:ايميل(?:ها|ه)|جوال(?:ها|ه)|رقم(?:ها|ه)|تحويل(?:ها|ه))", name_candidate, re.I):
            recent = recent_unique_employee(name_candidate)
            explicit_name = str(recent.get("name_ar") or recent.get("name") or recent.get("name_en") or name_candidate).strip() if recent else name_candidate
            return make(
                intent="employee_details",
                operation="details",
                targets=[TargetRef(kind="employee", name=explicit_name)],
                fields=contact_fields,
                query="employee:explicit_contact",
            )
        if contact_fields and has_focus:
            return make(
                intent="employee_details",
                operation="details",
                targets=[TargetRef(kind="focus", name="")],
                fields=contact_fields,
                query="employee:focus_fields",
            )

    # RCRC-FIX-v7: "مين الموظفين الي تحته" / "وش فريقه" / "الناس اللي يشتغلون عنده"
    if has_focus and re.search(
        r"(?:^|\s)(?:(?:(?:مين|من|وش|ايش|ابي|ابغى|عطني|اعطني|اعرض|هات)\s+)*(?:قائمه\s+|قائمة\s+)?(?:ب)?(?:الموظفين|الموظفون|موظفين|الناس|الاشخاص|مين)\s+(?:اللي\s+|الي\s+|الذين\s+)?(?:تحتها|تحته|عندها|عنده|يتبعونها|يتبعونه|تابعين\s+لها|تابعين\s+له|يشتغلون\s+تحتها|يشتغلون\s+تحته|يشتغلون\s+عنده|يشتغلون\s+عندها)|(?:فريقها|فريقه|موظفينها|موظفينه))(?:\\s|$)",
        norm, re.I,
    ):
        return make(
            intent="reporting_employees", operation="list",
            targets=[TargetRef(kind="focus", name="")], fields=["name", "job_title"],
            query="reporting:focus",
        )

    # Direct-report/team follow-up. A pronoun here refers to the current
    # VERIFIED focus, not to an old manager relation or the global directory.
    if has_focus and (
        re.fullmatch(
            r"(?:(?:ابي|ابغى|عطني|اعطني)\s+)?(?:قائمه\s+|قائمة\s+)?(?:الموظفين|الموظفون|فريقه|فريقها)(?:\s+(?:الي|اللي)\s+)?(?:تحته|تحتها|عنده|عندها)?",
            norm, re.I,
        )
        or re.fullmatch(
            r"(?:(?:show\s+me|list)\s+)?(?:the\s+)?(?:employees|people|staff|team|direct\s+reports?)\s+(?:reporting\s+to|under)\s+(?:him|her)",
            norm, re.I,
        )
        or re.fullmatch(r"who\s+reports\s+to\s+(?:him|her)", norm, re.I)
    ):
        return make(intent="reporting_employees", operation="list", targets=[TargetRef(kind="focus", name="")], fields=["name", "job_title"])

    last_req = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}

    # v178: keep short explicit date/month replies inside attendance even when
    # there is no pending clarification (for example after a UI retry).
    if str(last_req.get("intent") or "") in {"attendance_today", "attendance_history", "working_hours"} and _has_explicit_period_cue(raw):
        try:
            selected_period, selected_start, selected_end = _attendance_period(raw)
            last_attendance_intent = str(last_req.get("intent") or "attendance_history")
            restored_intent = (
                "working_hours"
                if last_attendance_intent == "working_hours"
                else ("attendance_today" if selected_period == "today" else "attendance_history")
            )
            return make(
                intent=restored_intent,
                operation="lookup",
                targets=[TargetRef(kind="self", name="")],
                fields=["all"],
                period=selected_period,
                start_date=selected_start,
                end_date=selected_end,
                query=str(last_req.get("query") or ("attendance:working_hours" if restored_intent == "working_hours" else "attendance:summary")),
                confidence=1.0,
            )
        except Exception:
            pass

    # Explicit Tawasul calendar request.
    if (
        re.search(r"(?:تواصل|tawasul)", norm, re.I)
        and re.search(
            r"(?:فعالي|دعوات?|دعوه|دعوة|اجتماعات?|events?|invitations?)",
            norm, re.I,
        )
    ):
        period = "none"
        start_date = ""
        end_date = ""
        if _has_explicit_period_cue(raw):
            try:
                period, start_date, end_date = _availability_period(raw)
            except Exception:
                period = "none"
        else:
            last = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
            if str(last.get("intent") or "") in {"meetings", "meeting_list"}:
                period = str(last.get("period") or "none")
                start_date = str(last.get("start_date") or "")
                end_date = str(last.get("end_date") or "")

        return make(
            intent="meetings",
            operation="list",
            targets=[TargetRef(kind="self", name="")],
            fields=[
                "subject", "start_time", "end_time", "location",
                "organizer_name", "my_response", "teams_link",
            ],
            period=period,
            start_date=start_date,
            end_date=end_date,
            query="meetings:tawasul",
            clarification_reason="" if period != "none" else "missing_date_range",
            confidence=1.0,
        )

    period_only_map = (
        ("today", r"^(?:اليوم|today)$"),
        ("tomorrow", r"^(?:بكره|بكرة|غدا|غداً|tomorrow)$"),
        ("yesterday", r"^(?:امس|أمس|yesterday)$"),
        ("this_week", r"^(?:هذا\s+الاسبوع|الأسبوع\s+هذا|الاسبوع\s+هذا|this\s+week)$"),
        ("last_week", r"^(?:الاسبوع\s+الماضي|الأسبوع\s+الماضي|last\s+week)$"),
        ("this_month", r"^(?:هذا\s+الشهر|الشهر\s+هذا|this\s+month)$"),
        ("last_month", r"^(?:الشهر\s+الماضي|last\s+month)$"),
    )
    period_only = ""
    for _pval, _ppat in period_only_map:
        if re.fullmatch(_ppat, norm, re.I):
            period_only = _pval
            break
    if period_only and str(last_req.get("intent") or "") in {"meetings", "meeting_list", "employee_availability", "leave_history"}:
        try:
            restored = ParsedRequest.model_validate({
                "intent": last_req.get("intent"),
                "operation": last_req.get("operation") or "list",
                "targets": last_req.get("targets") or [{"kind":"self","name":""}],
                "fields": last_req.get("fields") or ["all"],
                "query": last_req.get("query") or "",
                "role_query": last_req.get("role_query") or "",
                "department_query": last_req.get("department_query") or "",
                "period": period_only,
                "start_date": "",
                "end_date": "",
                "ordinal": 0,
                "relation_depth": 1,
                "amount": last_req.get("amount"),
                "time_query": "",
                "confidence": 1.0,
                "clarification_reason": "",
            })
            return restored
        except Exception:
            pass

    if re.search(r"(?:عناوين|مواضيع|اسماء|أسماء|subjects?|titles?).*(?:الاجتماعات|اجتماعات|meetings?)", norm, re.I):
        if str(last_req.get("intent") or "") in {"meetings", "meeting_list"}:
            return make(
                intent="meetings", operation="list",
                targets=[TargetRef(kind="self", name="")],
                fields=["subject", "start_time", "end_time"],
                period=str(last_req.get("period") or "none"),
                start_date=str(last_req.get("start_date") or ""),
                end_date=str(last_req.get("end_date") or ""),
                query="meetings:self",
                clarification_reason="" if str(last_req.get("period") or "none") != "none" else "missing_date_range",
            )

    # Period-only attendance continuation, e.g. "أمس" after "متى حضرت اليوم".
    attendance_context = state.get("attendance_context") if isinstance(state.get("attendance_context"), dict) else {}
    if attendance_context and re.fullmatch(r"(?:اليوم|امس|أمس|today|yesterday)", raw.strip(" ؟?!."), re.I):
        day_norm = norm
        period = "yesterday" if re.search(r"(?:امس|yesterday)", day_norm, re.I) else "today"
        mode = str(attendance_context.get("mode") or "attendance:summary")
        intent = "attendance_today" if period == "today" else "attendance_history"
        return make(intent=intent, operation="lookup", targets=[TargetRef(kind="self", name="")], fields=["all"], period=period, query=mode)

    # Writing / drafting / rewriting / translation is ordinary LLM work, not
    # an enterprise lookup. Route it directly to the general assistant.
    writing_cue = bool(re.search(
        r"(?:اكتب(?:\s+لي)?|صيغ(?:\s+لي)?|صياغه|صياغة|اعد\s+صياغ|أعد\s+صياغ|"
        r"اختصر(?:ها|ه)?|خله\s+رسمي|خليها\s+رسمي|ترجم(?:ها|ه)?|"
        r"write\s+(?:me\s+)?|draft\s+(?:me\s+)?|rewrite|rephrase|translate|make\s+it\s+(?:shorter|formal|professional))",
        norm, re.I,
    ))
    # RCRC-FIX-v1 (تعديل)
    if writing_cue and not is_content_explain_request(norm):
        return make(intent="general", operation="chat", query="general:writing")

    # Tarasol is authenticated self-service. A generic employee focus must
    # never hijack "my transaction" or "its details" after a Tarasol count.
    if re.search(r"(?:تراسل|tarasol)", norm, re.I) and re.search(r"(?:تفاصيل|معاملت(?:ي|ي)|معاملاتي|transactions?|details?)", norm, re.I):
        if re.search(r"(?:تفاصيل|details?)", norm, re.I):
            return make(intent="tarasol_details", operation="details", targets=[TargetRef(kind="self", name="")], fields=["all"])
    if last_intent in {"tarasol_count", "tarasol_details"} and re.fullmatch(
        r"(?:(?:ابي|ابغى|عطني|اعطني)\s+)?(?:تفاصيل(?:ها|ه)?|details?|show\s+(?:me\s+)?(?:the\s+)?details)",
        norm, re.I,
    ):
        return make(intent="tarasol_details", operation="details", targets=[TargetRef(kind="self", name="")], fields=["all"])

    # Leave calculations are always about the signed-in employee. Handle the
    # demo wording "today and tomorrow" deterministically as two leave days.
    leave_text = norm
    # RCRC-FIX-v2: "اذا خذت يوم كم يبقي لي" has no word "اجازة" but is the same request.
    leave_word = bool(re.search(r"(?:اجازه|اجازة|إجازة|leave|vacation|days?\s+off|time\s+off|take\b.*\boff)", leave_text, re.I))
    take_word = bool(re.search(r"(?:خذت|اخذت|اخذ|باخذ|ناخذ|if\s+i\s+take)", leave_text, re.I)) and bool(re.search(r"(?:يوم|ايام|days?|\d)", leave_text, re.I))
    if (leave_word or take_word) and re.search(r"(?:يبقى|يبقي|يتبقى|يتبقي|باقي|remain|remaining|left|have\s+left)", leave_text, re.I):
        amount = None
        if re.search(r"(?:اليوم.*(?:بكره|بكرة|غدا)|(?:بكره|بكرة|غدا).*اليوم|today.*tomorrow|tomorrow.*today)", leave_text, re.I):
            amount = 2.0
        else:
            m = re.search(r"(?:^|\s)(\d+(?:\.\d+)?)\s*(?:يوم|ايام|أيام|days?)", leave_text, re.I)
            if m:
                amount = float(m.group(1))
            elif re.search(r"(?:يوم\s+(?:الاحد|الأحد|الاثنين|الإثنين|الثلاثاء|الاربعاء|الأربعاء|الخميس)|on\s+(?:sunday|monday|tuesday|wednesday|thursday))", leave_text, re.I):
                amount = 1.0
        if amount is None:
            amount = _extract_leave_amount(raw)
        if amount is not None:
            return make(intent="leave_balance", operation="calculate", targets=[TargetRef(kind="self", name="")], fields=["balance"], amount=amount)

    # Explicit attendance request about another named employee must keep that
    # employee as the target so authorization can correctly deny private data.
    # Never silently answer with the signed-in user's attendance.
    named_attendance = None
    m = re.search(r"when\s+did\s+(.+?)\s+(check[- ]?in|check[- ]?out|arrive|leave)(?:\s+(?:today|yesterday))?$", norm, re.I)
    if m:
        candidate = str(m.group(1) or "").strip()
        if candidate not in {"i", "me", "my manager", "he", "she"}:
            named_attendance = (candidate, str(m.group(2) or ""))
    if named_attendance is None:
        m = re.search(r"(?:متى|متي)\s+(?:حضر|دخل|وصل|خرج|انصرف)\s+(.+?)(?:\s+(?:اليوم|امس))?$", norm, re.I)
        if m:
            candidate = str(m.group(1) or "").strip()
            if candidate not in {"انا", "اني", "مديري"}:
                named_attendance = (candidate, "check-out" if re.search(r"(?:خرج|انصرف)", norm, re.I) else "check-in")
    if named_attendance:
        candidate, action = named_attendance
        period = "yesterday" if re.search(r"(?:امس|yesterday)", norm, re.I) else "today"
        intent = "attendance_history" if period == "yesterday" else "attendance_today"
        mode = "attendance:check_out" if re.search(r"(?:out|leave|خرج|انصرف)", action, re.I) else "attendance:check_in"
        return make(
            intent=intent, operation="lookup",
            targets=[TargetRef(kind="employee", name=candidate)], fields=["all"],
            period=period, query=mode,
        )

    # Attendance is also always self-service. Short follow-ups inherit the last
    # attendance period instead of accidentally following an employee focus.
    attendance_context = state.get("attendance_context") if isinstance(state.get("attendance_context"), dict) else {}
    attendance_followup = bool(re.search(
        r"(?:حضوري|الحضور|حضور|بصمتي|دوامي|جدول\s+حضوري|"
        r"متي\s+(?:حضرت|دخلت|جيت|وصلت|طلعت|خرجت|انصرفت)|"
        r"وقت\s+(?:دخولي|خروجي)|ساعات\s+(?:دوامي|عملي)|"
        r"تاخير|تأخير|تأخرت|تاخرت|متاخر|متأخر|lateness|late|delay|"
        r"اوفر\s*تايم|أوفر\s*تايم|اضافي|إضافي|وقت\s+اضافي|وقت\s+إضافي|overtime|"
        r"ناقصني|النقص|فرق\s+(?:الدوام|الساعات)|difference|shortfall|"
        r"الساعات\s+(?:المطلوبه|المطلوبة|الرسميه|الرسمية)|regular\s+hours|"
        r"اجازه|اجازة|إجازة|اوف\s*داي|أوف\s*داي|off\s+day|weekend|نوع\s+اليوم|يوم\s+دوام|"
        r"بصم|بصمة|fingerprint|punch|"
        r"check[- ]?in|check[- ]?out|attendance|working\s+hours)",
        norm, re.I,
    ))
    if attendance_followup:
        has_explicit_time = _has_explicit_period_cue(raw)
        selected_period, selected_start, selected_end = ("none", "", "")
        if has_explicit_time:
            selected_period, selected_start, selected_end = _attendance_period(raw)
            if selected_period == "none":
                has_explicit_time = False

        # A full new attendance request only asks for a date when the user truly
        # omitted it. Any resolved day/range is final and must not be clarified
        # again later in the turn.
        full_attendance_request = bool(re.search(
            r"(?:حضوري|الحضور|حضور|بصمتي|دوامي|جدول\s+حضوري|"
            r"ساعات\s+(?:دوامي|عملي)|تاخير|تأخير|تأخرت|تاخرت|lateness|late|delay|"
            r"اوفر\s*تايم|أوفر\s*تايم|اضافي|إضافي|overtime|"
            r"ناقصني|النقص|فرق\s+(?:الدوام|الساعات)|difference|shortfall|"
            r"الساعات\s+(?:المطلوبه|المطلوبة|الرسميه|الرسمية)|regular\s+hours|"
            r"اجازه|اجازة|إجازة|اوف\s*داي|أوف\s*داي|off\s+day|weekend|نوع\s+اليوم|يوم\s+دوام|"
            r"بصم|بصمة|fingerprint|punch|"
            r"attendance|working\s+hours)",
            norm,
            re.I,
        ))
        inherited_period = str(attendance_context.get("period") or "today") if not full_attendance_request else ""
        inherited_start = str(attendance_context.get("start_date") or "") if not full_attendance_request else ""
        inherited_end = str(attendance_context.get("end_date") or "") if not full_attendance_request else ""

        period = selected_period if has_explicit_time else inherited_period
        start_date = selected_start if has_explicit_time else inherited_start
        end_date = selected_end if has_explicit_time else inherited_end

        checkout = bool(re.search(r"(?:طلعت|خرجت|انصرفت|خروجي|check[- ]?out|leave)", norm, re.I))
        checkin = bool(re.search(r"(?:حضرت|دخلت|جيت|وصلت|دخولي|check[- ]?in|arrive)", norm, re.I))
        working = bool(re.search(r"(?:ساعات\s+(?:دوامي|عملي)|working\s+hours)", norm, re.I))
        lateness = bool(re.search(r"(?:تاخير|تأخير|تأخرت|تاخرت|متاخر|متأخر|lateness|late|delay)", norm, re.I))
        overtime = bool(re.search(r"(?:اوفر\s*تايم|أوفر\s*تايم|وقت\s+(?:اضافي|إضافي)|ساعات\s+(?:اضافيه|إضافية)|overtime|extra\s+hours)", norm, re.I))
        difference = bool(re.search(r"(?:ناقصني|النقص|فرق\s+(?:الدوام|الساعات)|difference|shortfall)", norm, re.I))
        regular_hours = bool(re.search(r"(?:الساعات\s+(?:المطلوبه|المطلوبة|الرسميه|الرسمية)|regular\s+hours|required\s+hours)", norm, re.I))
        day_type = bool(re.search(
            r"(?:اجازه|اجازة|إجازة|اوف\s*داي|أوف\s*داي|off\s+day|weekend|"
            r"نوع\s+اليوم|يوم\s+دوام|كان\s+دوام|holiday|leave\s+day)",
            norm, re.I,
        ))
        punch_status = bool(re.search(
            r"(?:بصم|بصمة|بصمات|fingerprint|punch|نسي\s+(?:البصمه|البصمة)|"
            r"دخول\s+وخروج|check[- ]?in\s+and\s+check[- ]?out)",
            norm, re.I,
        ))

        today_iso = datetime.now(timezone(timedelta(hours=3))).date().isoformat()
        is_today = bool(start_date and start_date == end_date == today_iso) or period == "today"
        intent = "working_hours" if working else ("attendance_today" if is_today else "attendance_history")

        if lateness:
            query = "attendance:lateness"
        elif overtime:
            query = "attendance:overtime"
        elif difference:
            query = "attendance:difference"
        elif regular_hours:
            query = "attendance:regular_hours"
        elif day_type:
            query = "attendance:day_type"
        elif punch_status:
            query = "attendance:punch_status"
        elif working:
            query = "attendance:working_hours"
        elif checkout:
            query = "attendance:check_out"
        elif checkin:
            query = "attendance:check_in"
        else:
            query = "attendance:summary"
        return make(
            intent=intent, operation="lookup", targets=[TargetRef(kind="self", name="")],
            fields=["all"], period=period or "none", start_date=start_date, end_date=end_date, query=query,
            clarification_reason="missing_date_range" if full_attendance_request and not has_explicit_time else "",
            confidence=1.0 if has_explicit_time else 0.99,
        )

    # Natural content follow-up. If the last verified list contains exactly
    # one item, phrases such as "ابي تفاصيلها" / "افتحها" / "ورني اكثر"
    # automatically refer to that single content item. This runs BEFORE the
    # employee-focus fallback so a previous employee focus cannot hijack it.
    generic_content_detail = bool(re.fullmatch(
        r"(?:(?:طيب|تمام|اوكي|أوكي|ok|okay)\s+)?"
        r"(?:(?:ابي|ابغى|أبي|أبغى|عطني|اعطني|أعطني|ورني|وريني|افتح|افتحها|show|open)\s+)?"
        r"(?:تفاصيل(?:ها|ه)?|التفاصيل(?:\s+كامله|\s+كاملة)?|"
        r"معلومات(?:ها|ه)?|المزيد|اكثر|أكثر|more|details?|full\s+details?)",
        norm,
        re.I,
    ))
    if generic_content_detail:
        _ls = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
        _items = [x for x in (_ls.get("items") or []) if isinstance(x, dict)]
        _base_to_detail = {
            "news": "news_details",
            "announcements": "announcement_details",
            "mazaya": "mazaya_details",
            "policy_search": "policy_details",
        }
        _detail_intent = _base_to_detail.get(str(_ls.get("intent") or ""), "")
        if _detail_intent and len(_items) == 1:
            return make(
                intent=_detail_intent,
                operation="details",
                targets=[],
                fields=["all"],
                ordinal=1,
                query="content:single_auto",
                confidence=1.0,
            )

    # Pronouns such as "his email" always mean the current VERIFIED employee,
    # while "my email" remains self. Run this before the self-field rules.
    if has_focus:
        focus_fields: list[str] = []
        for field_name, pattern in (
            ("email", r"(?:ايميل(?:ها|ه|ة)|إيميل(?:ها|ه|ة)|بريد(?:ها|ه)|his\s+email|her\s+email)"),
            ("mobile", r"(?:رقم\s+جوال(?:ها|ه)|جوال(?:ها|ه)|رقم\s+موبايل(?:ها|ه)|موبايل(?:ها|ه)|his\s+(?:mobile|phone)|her\s+(?:mobile|phone))"),
            ("extension", r"(?:تحويلت(?:ها|ه)|تحويل(?:ها|ه)|his\s+extension|her\s+extension)"),
            ("job_title", r"(?:منصب(?:ها|ه)|مسما(?:ها|ه)|his\s+(?:job\s+)?title|her\s+(?:job\s+)?title)"),
            ("department", r"(?:ادارت(?:ها|ه)|إدارت(?:ها|ه)|تحت\s+(?:اي|أي)\s+اداره|(?:باي|بأي)\s+اداره|his\s+department|her\s+department|which\s+department|what(?:'s|\s+is)\s+(?:his|her)\s+department)"),
        ):
            if re.search(pattern, norm, re.I) and field_name not in focus_fields:
                focus_fields.append(field_name)
        if focus_fields:
            return make(intent="employee_details", operation="details", targets=[TargetRef(kind="focus", name="")], fields=focus_fields, query="employee:focus_fields")
        if (
            re.fullmatch(r"(?:(?:لا|طيب|تمام)\s+)?(?:(?:ابي|ابغى|عطني|اعطني)\s+)?(?:معلومات|بيانات|تفاصيل)(?:\s+(?:عنه|عنها|حقه|حقها))", norm, re.I)
            or re.fullmatch(r"(?:(?:لا|طيب|تمام)\s+)?(?:(?:ابي|ابغى|عطني|اعطني)\s+)?(?:كل\s+)?(?:معلوماته|معلوماتها|بياناته|بياناتها|تفاصيله|تفاصيلها)(?:\s+(?:اللي|الي)\s+(?:تقدر|تقدرين|تقدرون)\s+عليها)?", norm, re.I)
        ):
            return make(intent="employee_details", operation="details", targets=[TargetRef(kind="focus", name="")], fields=["name","job_title","department","email","mobile","extension"], query="employee:focus_summary")

    if norm in {"انا", "اني", "مين انا", "من انا", "who am i", "me"}:
        return make(intent="profile", operation="details", targets=[TargetRef(kind="self", name="")], fields=["name","job_title","department"], query="profile:identity")

    if norm in {"معلوماتي","ابي معلوماتي","ابغى معلوماتي","عطني معلوماتي","اعطني معلوماتي","بياناتي","ابي بياناتي","my profile","my information","my info","show my profile","show me my profile"}:
        return make(intent="profile", operation="details", targets=[TargetRef(kind="self", name="")], fields=["name","job_title","department","email","mobile","extension"], query="profile:summary")

    self_fields: list[str] = []
    for field_name, pattern in (
        ("name", r"(?:اسمي|إسمي|وش\s+اسمي|ما\s+اسمي|my\s+name)"),
        ("email", r"(?:ايميلي|ايميل\s*ي|بريدي(?:\s+الالكتروني)?|my\s+email)"),
        ("mobile", r"(?:رقم\s+جوالي|جوالي|رقم\s+موبايلي|موبايلي|my\s+(?:mobile|phone)(?:\s+number)?)"),
        ("extension", r"(?:تحويلتي|رقم\s+تحويلتي|my\s+extension)"),
        ("job_title", r"(?:مسماي|منصبي|مسمى\s+وظيفتي|my\s+(?:job\s+)?title)"),
        ("department", r"(?:ادارتي|الاداره\s+حقتي|my\s+department)"),
        ("username", r"(?:يوزري|اسم\s+المستخدم\s+حقي|my\s+username)"),
    ):
        if re.search(pattern, norm, re.I) and field_name not in self_fields:
            self_fields.append(field_name)
    if self_fields:
        # Self contact facts come from PersonalInformation/employeeFullDetails
        # through get_employee_profile. Never route "رقم جوالي" or "تحويلتي"
        # to the directory EmployeeDetails endpoint.
        return make(
            intent="profile", operation="details",
            targets=[TargetRef(kind="self", name="")],
            fields=self_fields, query="profile:self_fields",
        )

    female = bool(re.search(r"(?:الاناث|اناث|female|females|women)", norm, re.I))
    male = bool(re.search(r"(?:الذكور|ذكور|male|males|men)", norm, re.I))
    if (female or male) and re.search(r"(?:عدد|كم|count|how\s+many)", norm, re.I) and re.search(r"(?:موظف|الموظفين|employee|employees|staff)", norm, re.I):
        mode = "gender:both" if female and male else ("gender:female" if female else "gender:male")
        return make(intent="employee_count", operation="calculate", fields=["count"], query=mode)

    # Short unique selection from the last employee search.
    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    if str(list_state.get("intent") or "") == "employee_search":
        tokens = [x for x in norm.split() if len(x) >= 2]
        if tokens and len(tokens) <= 5 and not re.search(r"^(?:ابي|ابغى|مين|وش|كم|هل|متى|ابحث|دور|who|what|when|search|find)\b", norm, re.I):
            hits = []
            for row in [x for x in (list_state.get("items") or []) if isinstance(x, dict)]:
                hay = " ".join(str(row.get(k) or "") for k in ("name","name_ar","name_en","job_title","job_title_ar","job_title_en","username")).casefold()
                hay = hay.translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
                if all(re.search(rf"(?:^|\s){re.escape(t)}(?:\s|$)", hay) for t in tokens):
                    hits.append(row)
            if len(hits) == 1:
                row = hits[0]
                name = str(row.get("name_ar") or row.get("name") or row.get("name_en") or "").strip()
                if name:
                    return make(intent="employee_details", operation="details", targets=[TargetRef(kind="employee", name=name)], fields=["name","job_title","department"], query="employee:selected_from_search")

    info = re.match(r"^(?:ابي|ابغى|عطني|اعطني|show\s+me)?\s*(?:معلومات|بيانات|تفاصيل|information|details)\s+(?:عن|about)\s+(.+)$", norm, re.I)
    if info:
        name = str(info.group(1) or "").strip()
        # RCRC-FIX-v5: "معلومات عن مديري" / "معلومات عني"
        info_slot, _ = _person_slot_target(name, state)
        if name and info_slot.kind != "employee":
            return make(
                intent="profile" if info_slot.kind == "self" else "employee_details",
                operation="details", targets=[info_slot],
                fields=["name", "job_title", "department", "email", "mobile", "extension"],
                query="employee:summary",
            )
        if name:
            recent = recent_unique_employee(name)
            if recent:
                name = str(recent.get("name_ar") or recent.get("name") or recent.get("name_en") or name).strip()
            return make(intent="employee_details", operation="details", targets=[TargetRef(kind="employee", name=name)], fields=["name","job_title","department","email","mobile","extension"], query="employee:summary")

    # "اليوم متى فاضي؟" / "بكره فاضي؟" = current verified person.
    current = current_focus
    availability_shape = bool(re.fullmatch(
        r"(?:(?:اليوم|بكره|بكرة|غدا|غداً|today|tomorrow)\s+)?(?:(?:متى|متي|when)\s+)?(?:فاضي|فاضيه|متاح|متاحه|مشغول|مشغوله|free|available|busy)(?:\s+(?:اليوم|بكره|بكرة|غدا|غداً|today|tomorrow))?",
        norm,
        re.I,
    ))
    if has_focus and availability_shape:
        period = "tomorrow" if re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I) else "today"
        mode = "busy" if re.search(r"(?:مشغول|مشغوله|busy)", norm, re.I) else "free"
        return make(intent="employee_availability", operation="lookup", targets=[TargetRef(kind="focus", name="")], fields=["start_time","end_time"], period=period, query=mode)

    # Period-only follow-up after own meetings.
    last = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
    if str(last.get("intent") or "") in {"meetings", "meeting_list"} and re.fullmatch(r"(?:اليوم|بكره|بكرة|غدا|غداً|today|tomorrow)", norm, re.I):
        period = "tomorrow" if re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I) else "today"
        return make(intent="meetings", operation="list", targets=[TargetRef(kind="self", name="")], fields=["subject","start_time","end_time"], period=period)

    return None


def _log_nlu_miss(message: str, reason: str, state: dict[str, Any]) -> None:
    """RCRC-FIX-v4: every message the assistant did not understand goes to
    logs/nlu_misses.jsonl so real phrasings can be reviewed and added."""
    try:
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        path = _Path(__file__).resolve().parent.parent / "logs" / "nlu_misses.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        last = state.get("last_request") if isinstance(state.get("last_request"), dict) else {}
        record = {
            "time": _dt.now().isoformat(timespec="seconds"),
            "message": message,
            "reason": reason,
            "last_intent": str(last.get("intent") or ""),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[NLU Miss Log] {exc}", file=sys.stderr)


def _friendly_clarification(message: str, language: str) -> str:
    """A short, friendly experimental-stage clarification."""
    if language != "ar":
        return "Sorry 😅 I'm still in my experimental phase and I didn't understand that request this time. Could you rephrase it a little?"
    return "آسف 😅 أنا لسه بمرحلتي التجريبية وما فهمت طلبك هالمرة. ممكن توضحه لي شوي؟"


def _greeting(message: str, language: str) -> str | None:
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return None

    if language == "ar":
        norm = _norm_role_text(raw)
        # Friendly greeting variants such as "هيلووو", "هاااي", and "هلووو"
        # should never be sent to the enterprise semantic parser.
        compact = re.sub(r"\s+", "", norm)
        if (
            re.fullmatch(r"ه+ا+ي+", compact)
            or re.fullmatch(r"ه+ل+و+", compact)
            or re.fullmatch(r"ه+ي+ل+و+", compact)
        ):
            return "هلا فيك 👋 وش أقدر أساعدك فيه؟"

        greeting_words = {
            "هلا",
            "اهلا",
            "اهلين",
            "أهلين",
            "مرحبا",
            "هلا فيك",
            "اهلا فيك",
            "اهلا وسهلا",
            "أهلا وسهلا",
            "السلام عليكم",
            "صباح الخير",
            "مساء الخير",
            # Common Arabic-script transliterations of hi/hello.
            "هاي",
            "هلو",
            "هيلو",
            "هيلو",
        }

        small_talk = {
            "كيف حالك",
            "كيفك",
            "شلونك",
            "علومك",
            "اخبارك",
        }

        conversation_openers = {
            "بسالك",
            "بسالك سوال",
            "بسالك سؤال",
            "ابي اسالك",
            "ابي اسألك",
            "ممكن اسالك",
            "ممكن اسألك",
            "عندي سوال",
            "عندي سؤال",
        }

        if norm in conversation_openers:
            return "طبعًا، اسألني 😊"

        if norm in greeting_words:
            return "هلا فيك 👋 وش أقدر أساعدك فيه؟"

        has_greeting = any(
            norm == item
            or norm.startswith(item + " ")
            for item in greeting_words
        )

        has_small_talk = any(
            phrase in norm
            for phrase in small_talk
        )

        if has_greeting and has_small_talk:
            return "هلا فيك 👋 بخير الحمدلله 😊 وش أقدر أساعدك فيه؟"

        # Do not swallow a real request just because it starts with a greeting.
        # Examples that must continue to NLU:
        #   "اهلا تعرفني؟"
        #   "هلا مين مديري؟"
        # Pure greetings were already handled by `norm in greeting_words` above.
        if has_small_talk and len(norm.split()) <= 5:
            return "بخير الحمدلله 😊 وش أقدر أساعدك فيه؟"

        return None

    norm = raw.casefold().strip()
    norm = re.sub(r"[?!.]+$", "", norm).strip()
    compact = re.sub(r"\s+", "", norm)
    if (
        re.fullmatch(r"h+i+", compact)
        or re.fullmatch(r"he+y+", compact)
        or re.fullmatch(r"he+l+o+", compact)
    ):
        return "Hi! How can I help?"

    if norm in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening",
    }:
        return "Hi! How can I help?"

    if any(
        norm.startswith(item + " ")
        for item in ("hi", "hello", "hey")
    ):
        if any(
            phrase in norm
            for phrase in (
                "how are you",
                "how's it going",
                "how are things",
            )
        ):
            return "Hi! I'm doing well. How can I help?"

        return "Hi! How can I help?"

    if norm in {
        "how are you",
        "how's it going",
        "how are things",
    }:
        return "I'm doing well. How can I help?"

    return None

def _uid(row: dict[str, Any]) -> str:
    return str(row.get("username") or row.get("userid") or "").strip()


def _bounded_employee(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    keys = (
        "username", "userid", "name", "name_ar", "name_en", "job_title", "job_title_ar", "job_title_en",
        "department", "department_ar", "department_en",
        "general_department", "general_department_source",
        "sector_ar", "sector_en", "email", "mobile", "extension",
    )
    return {k: row.get(k) for k in keys if row.get(k) not in (None, "")}


def _clean_employee_discovery_query(value: Any) -> str:
    """Remove search-command filler while preserving the actual directory query."""
    raw = " ".join(str(value or "").strip().split())
    if not raw:
        return ""

    filler = {
        "ابحث", "ابحثي", "دور", "دوري", "دورلي", "دور لي", "لقي", "لقّي",
        "عن", "لي", "ابي", "أبي", "أبغى", "ابغى", "باسم", "اسم", "اسمه", "اسمهـا",
        "موظف", "موظفه", "موظفة", "موظفين", "الموظف", "الموظفين", "الموظفات",
        "employee", "employees", "find", "search", "for", "show", "me",
    }

    tokens = [token for token in raw.split() if token.casefold() not in {x.casefold() for x in filler}]
    cleaned = " ".join(tokens).strip()
    return cleaned or raw


def _normalize_directory_discovery_row(raw: Any, language: str) -> dict[str, Any]:
    """Normalize one verified SearchEmployee row for browse-style employee search."""
    if not isinstance(raw, dict):
        return {}

    name_ar = str(raw.get("namea") or raw.get("nameAr") or raw.get("name_ar") or "").strip()
    name_en = str(raw.get("namel") or raw.get("nameEn") or raw.get("name_en") or "").strip()
    generic_name = str(raw.get("name") or raw.get("employeeName") or "").strip()

    title_ar = str(raw.get("jodesca") or raw.get("jobTitleAr") or raw.get("job_title_ar") or "").strip()
    title_en = str(raw.get("jodesce") or raw.get("hodesce") or raw.get("jobTitleEn") or raw.get("job_title_en") or "").strip()
    generic_title = str(raw.get("job_title") or raw.get("jobTitle") or raw.get("position") or "").strip()

    department_ar = str(raw.get("dpDesca") or raw.get("departmentAr") or raw.get("department_ar") or "").strip()
    department_en = str(raw.get("dpDescl") or raw.get("departmentEn") or raw.get("department_en") or "").strip()
    generic_department = str(raw.get("department") or raw.get("departmentName") or "").strip()

    username = str(
        raw.get("userid") or raw.get("username") or raw.get("userCode") or raw.get("userName") or ""
    ).strip()
    if not username:
        return {}

    email = raw.get("email") or raw.get("mail") or raw.get("emailAddress") or raw.get("email_address")
    mobile = (
        raw.get("mobile") or raw.get("mobileNo") or raw.get("mobileNumber") or raw.get("mobilePhone")
        or raw.get("mobilephone") or raw.get("cellPhone") or raw.get("cellphone") or raw.get("cellNo")
        or raw.get("cellNumber") or raw.get("phoneMobile") or raw.get("mobile_number") or raw.get("mobile_no")
    )
    extension = raw.get("ext") or raw.get("extension") or raw.get("phoneExt") or raw.get("extensionNumber")

    prefer_ar = language == "ar"
    row = {
        "username": username,
        "userid": username,
        "name": (name_ar or generic_name or name_en) if prefer_ar else (name_en or generic_name or name_ar),
        "name_ar": name_ar or None,
        "name_en": name_en or None,
        "job_title": (title_ar or generic_title or title_en) if prefer_ar else (title_en or generic_title or title_ar),
        "job_title_ar": title_ar or None,
        "job_title_en": title_en or None,
        "department": (department_ar or generic_department or department_en) if prefer_ar else (department_en or generic_department or department_ar),
        "department_ar": department_ar or None,
        "department_en": department_en or None,
        "email": email or None,
        "mobile": mobile or None,
        "extension": extension or None,
    }
    return _bounded_employee(row)


def _directory_discovery_match(row: dict[str, Any], query: str) -> bool:
    """Conservative verification across employee name/title/department fields.

    Short Latin queries such as ``IT`` must never be treated as arbitrary
    substrings. Otherwise ``it`` incorrectly matches words such as
    ``internal`` and ``facilities``. For known organisational acronyms we
    resolve only against verified department aliases / exact acronyms.
    """
    q = _norm_role_text(query)
    if not q:
        return False
    tokens = [token for token in q.split() if len(token) >= 2]
    if not tokens:
        return False

    name_fields = [
        _norm_role_text(value)
        for value in (row.get("name"), row.get("name_ar"), row.get("name_en"))
        if str(value or "").strip()
    ]
    title_fields = [
        _norm_role_text(value)
        for value in (row.get("job_title"), row.get("job_title_ar"), row.get("job_title_en"))
        if str(value or "").strip()
    ]
    department_fields = [
        _norm_role_text(value)
        for value in (
            row.get("department"), row.get("department_ar"), row.get("department_en"),
            row.get("general_department"),
        )
        if str(value or "").strip()
    ]
    org_fields = title_fields + department_fields
    fields = name_fields + org_fields
    words = set(" ".join(fields).split())

    def acronym(value: str) -> str:
        parts = re.findall(r"[a-z]+", value.casefold())
        return "".join(part[0] for part in parts if part)

    acronyms = {acronym(value) for value in org_fields if re.search(r"[a-z]", value)}

    # RCRC organisational alias used by users conversationally. The verified
    # directory names the unit as Technology and Digitization / التقنية والرقمنة,
    # so "IT" should resolve to that department instead of substring matches.
    it_department_aliases = (
        "technology and digitization",
        "technology digitization",
        "information technology",
        "التقنيه والرقمنه",
        "تقنيه والرقمنه",
        "التقنيه",
    )

    for token in tokens:
        latin_short = bool(re.fullmatch(r"[a-z0-9]{2,5}", token))
        if latin_short:
            if token == "it":
                if token in words or token in acronyms:
                    continue
                if any(
                    any(alias in field for alias in it_department_aliases)
                    for field in department_fields
                ):
                    continue
                return False

            # Other short Latin terms are accepted only as complete words or
            # verified acronyms. Never use arbitrary substring matching.
            if token in words or token in acronyms:
                continue
            return False

        if any(token in field for field in fields):
            continue
        return False
    return True


def _merge_employee(base: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    """Merge employee data without letting an unverified API row replace identity.

    SearchEmployee / resolver identity is the verified subject. Some deployed
    EmployeeDetails responses have been observed to return a different/generic
    userid. When that happens, keep the verified identity and ignore identity
    fields from the mismatched details payload.
    """
    base_row = _bounded_employee(base)
    detail_row = _bounded_employee(details)
    merged = dict(base_row)

    base_uid = _uid(base_row).strip().casefold()
    detail_uid = _uid(detail_row).strip().casefold()
    identity_keys = {"username", "userid", "name", "name_ar", "name_en"}
    mismatched_identity = bool(base_uid and detail_uid and base_uid != detail_uid)

    for key, value in detail_row.items():
        if value in (None, ""):
            continue
        if mismatched_identity and key in identity_keys:
            continue
        merged[key] = value

    # If a verified base identity exists, it always wins.
    if base_uid:
        verified_username = str(base_row.get("username") or base_row.get("userid") or "").strip()
        merged["username"] = verified_username
        merged["userid"] = verified_username
        for key in ("name", "name_ar", "name_en"):
            if base_row.get(key) not in (None, ""):
                merged[key] = base_row[key]

    if "userid" not in merged and merged.get("username"):
        merged["userid"] = merged["username"]
    if "username" not in merged and merged.get("userid"):
        merged["username"] = merged["userid"]
    return merged


def _name(row: dict[str, Any], lang: str) -> str:
    if lang == "en":
        return str(row.get("name_en") or row.get("name") or row.get("name_ar") or _uid(row)).strip()
    return str(row.get("name_ar") or row.get("name") or row.get("name_en") or _uid(row)).strip()


def _remember_entity(state: dict[str, Any], employee: dict[str, Any], *, set_focus: bool = True) -> None:
    row = _bounded_employee(employee)
    if not _uid(row):
        return
    history = [x for x in (state.get("entity_history") or []) if isinstance(x, dict) and _uid(x).casefold() != _uid(row).casefold()]
    history.append(row)
    state["entity_history"] = history[-12:]
    if set_focus:
        focus = state.setdefault("focus", {"current": {}, "previous": {}})
        current = _bounded_employee(focus.get("current"))
        if _uid(current).casefold() != _uid(row).casefold():
            focus["previous"] = current
        focus["current"] = row


def _role_rows(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    rows = data.get("employees")
    if not isinstance(rows, list):
        return []
    return [
        _bounded_employee(row)
        for row in rows
        if isinstance(row, dict) and _uid(_bounded_employee(row))
    ]


def _norm_role_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    text = text.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي",
        "ة": "ه", "ؤ": "و", "ئ": "ي",
    }))
    # Common Arabic typo normalization seen in natural employee queries.
    text = text.replace("الادراة", "الادارة").replace("ادراة", "ادارة")
    text = re.sub(r"[؟،؛:!?.,\-–—]+", " ", text)
    text = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_generic_policy_request(value: Any) -> bool:
    """True when the user wants the available policy/procedure list."""
    text = _norm_role_text(value)
    if not text:
        return True

    text = re.sub(
        r"\b(?:show|list|give|tell|what|are|is|the|me|all|available|latest|top|main|"
        r"policies|policy|procedures|procedure|rules|regulations|rcrc|organization)\b",
        " ",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"(?:\b(?:ابي|ابغى|اعطني|عطني|اعرض|ورني|هات|وش|ايش|ما|ماهي|هي|"
        r"ابرز|اهم|احدث|اخر|كل|جميع|المتاحه|الموجوده|عندكم|لدينا|لي)\b|"
        r"(?:في|بال|لل)?\s*ال?هي(?:ئ|ي)?ه)",
        " ",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\bو?(?:السياسات|سياسات|السياسه|سياسه|الاجراءات|اجراءات|الاجراء|اجراء|"
        r"القوانين|قوانين|القانون|قانون|الانظمه|انظمه|النظام|نظام|"
        r"اللوائح|لوائح|اللائحه|لائحه)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:و|and|or)\b", " ", text, flags=re.I)
    return not re.sub(r"\s+", " ", text).strip()


def _role_best_match(data: Any, role_query: str = "") -> dict[str, Any]:
    """Return a safe verified role holder.

    Prefer an API-provided best_match. For compatibility with an older
    employee_api.py, derive a winner only when the ranked rows clearly identify
    one person. Never reuse stale conversational focus as the role result.
    """
    if not isinstance(data, dict):
        return {}

    api_best = _bounded_employee(data.get("best_match"))
    if _uid(api_best):
        return api_best

    rows = _role_rows(data)
    if len(rows) == 1:
        return rows[0]
    if not rows:
        return {}

    query = _norm_role_text(role_query)
    q_tokens = set(query.split())
    if not q_tokens:
        return {}

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        title = _norm_role_text(
            row.get("job_title")
            or row.get("jobTitle")
            or row.get("job_title_ar")
            or row.get("position")
        )
        department = _norm_role_text(
            row.get("department")
            or row.get("department_ar")
            or row.get("departmentName")
            or row.get("org_unit")
        )
        combined = f"{title} {department}".strip()
        tokens = set(combined.split())
        matched = len(q_tokens & tokens)
        coverage = matched / max(1, len(q_tokens))

        score = coverage
        if query == title:
            score += 2.0
        elif query == combined:
            score += 2.5
        elif query and query in combined:
            score += 1.0

        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return {}

    top_score, top_row = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0

    # Deliberately conservative: only bind memory when one role-holder is
    # materially clearer than the rest.
    if top_score >= 0.80 and (len(scored) == 1 or top_score - second_score >= 0.20):
        return top_row

    return {}



def _render_role_rows(
    rows: list[dict[str, Any]],
    language: str,
    *,
    limit: int = 5,
    header: bool = True,
) -> str:
    clean_rows = [row for row in rows if isinstance(row, dict)][:limit]
    if not clean_rows:
        return "ما لقيت موظف مطابق للاسم هذا حاليًا 🔎😊 تأكد من الاسم وجرب مرة ثانية." if language == "ar" else "I couldn't find matching employees."

    lines = ([(("أكيد 😊 لقيت لك موظف مطابق:" if len(clean_rows) == 1 else "لقيت لك أكثر من موظف مطابق 👥😊 مين تقصد؟") if language == "ar" else "Matching results:")] if header else [])
    for index, row in enumerate(clean_rows, start=1):
        name = _name(row, language) or _uid(row)
        if language == "ar":
            title = str(row.get("job_title_ar") or row.get("job_title") or "").strip()
            department = str(row.get("general_department_ar") or row.get("general_department") or row.get("department_ar") or row.get("department") or "").strip()
        else:
            title = str(row.get("job_title_en") or row.get("job_title") or "").strip()
            department = str(row.get("general_department_en") or row.get("department_en") or row.get("department") or "").strip()
            if re.search(r"[\u0600-\u06ff]", title): title = ""
            if re.search(r"[\u0600-\u06ff]", department): department = ""

        parts: list[str] = []
        for value in (name, title, department):
            value = str(value or "").strip()
            if not value:
                continue
            if any(existing.casefold() == value.casefold() for existing in parts):
                continue
            parts.append(value)

        lines.append(f"{index}. " + " — ".join(parts))

    return "\n".join(lines)


def _first_name(value: str) -> str:
    clean = " ".join(str(value or "").split()).strip()
    return clean.split()[0] if clean else ""


def _generic_policy_navigation(message: str, language: str) -> str | None:
    """Generic policy requests in the app are navigation help, not policy API search."""
    norm = _norm_role_text(str(message or "").strip())
    generic_patterns = (
        r"(?:لا\s+)?(?:ابي|ابغى|أبي|أبغى|وين|وش|ماهي|ما\s+هي|اعطني|عطني|ورني|اظهر|أظهر)?\s*"
        r"(?:كل|جميع)?\s*(?:سياسات|السياسات)(?:\s+و?الإجراءات|\s+و?الاجراءات)?(?:\s+الهيئة)?",
        r"(?:وين|كيف\s+اوصل|كيف\s+أوصل|كيف\s+القى|كيف\s+ألقى)\s+(?:ل|إلى|الى)?\s*"
        r"(?:السياسات|سياسات)(?:\s+و?الإجراءات|\s+و?الاجراءات)?",
    )
    clean = norm.strip(" ؟?!.,،")
    if any(re.fullmatch(pattern, clean, re.I) for pattern in generic_patterns):
        if language == "ar":
            return "السياسات والإجراءات تلقينها داخل التطبيق من المزيد → السياسات والإجراءات. وإذا كنتِ تقصدين سياسة معيّنة، اكتبي اسمها وأنا أساعدك فيها."
        return "You can find Policies and Procedures in the app under **More → Policies and Procedures**. If you mean a specific policy, tell me its name and I’ll help."
    return None



def _semantic_message(message: str) -> str:
    """Remove harmless conversational filler only when it precedes a real question."""
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return raw

    # "طيب تعرف متى فاضي اليوم" -> "متى فاضي اليوم"
    # But keep "تعرف محمد؟" intact because there "تعرف" carries meaning.
    cleaned = re.sub(
        r"^(?:طيب\s+)?(?:تعرف|تدري|ممكن\s+تشوف|ممكن\s+تشيك|شوف\s+لي)\s+"
        r"(?=(?:متى|متي|هل|وين|وش|ايش|إيش|اذا|إذا|كم)\b)",
        "",
        raw,
        flags=re.I,
    ).strip()
    return cleaned or raw



def _availability_preference_turn(
    message: str,
    state: dict[str, Any],
    language: str,
) -> tuple[str, str] | None:
    """Continue an availability preference question naturally."""
    flow = state.get("task_flow") if isinstance(state.get("task_flow"), dict) else {}
    if str(flow.get("kind") or "") != "availability_preference":
        return None

    norm = _norm_role_text(str(message or "").strip()).strip(" ؟?!.,،")
    window = ""
    if re.search(r"(?:الصباح|صباح|بدري|morning)", norm, re.I):
        window = "morning"
    elif re.search(r"(?:الظهر|ظهر|بعد\s+الظهر|العصر|عصر|afternoon)", norm, re.I):
        window = "afternoon"
    elif re.search(r"(?:المساء|مساء|الليل|ليل|evening|night)", norm, re.I):
        window = "evening"

    # A clear new topic cancels the preference prompt.
    if not window and re.search(
        r"(?:اجاز|إجاز|حضور|سياس|اعلان|إعلان|خبر|مزايا|تراسل|ايميل|إيميل|email|موظف|مدير)",
        norm,
        re.I,
    ):
        state["task_flow"] = {}
        return None

    if not window:
        if language == "ar":
            return ("answer", "تقصد الصباح، الظهر/العصر، ولا المساء؟")
        return ("answer", "Do you mean morning, afternoon, or evening?")

    period = str(flow.get("period") or "today")
    target = flow.get("target") if isinstance(flow.get("target"), dict) else {}
    name = _name(target, language) or _name(target, "ar")
    first = _first_name(name) or ("الموظف" if language == "ar" else "them")

    state["availability_preference_window"] = window
    state["task_flow"] = {}

    day_ar = "اليوم" if period == "today" else "بكرة" if period == "tomorrow" else "الفترة المطلوبة"
    if language == "ar":
        return ("rewrite", f"متى {first} فاضي {day_ar}")
    day_en = "today" if period == "today" else "tomorrow" if period == "tomorrow" else "for the requested period"
    return ("rewrite", f"When is {first} free {day_en}?")


def _employee_meeting_request(message: str, state: dict[str, Any], language: str) -> str | None:
    """Start a meeting-planning conversation with the verified focused employee."""
    norm = _norm_role_text(str(message or "").strip()).strip(" ؟?!.,،")
    wants_meeting = bool(
        re.search(r"(?:ابي|ابغى|أبي|أبغى|ودي|احتاج|أحتاج|نبغى|نبي|want|need).*(?:اجتماع|meeting)", norm, re.I)
        and re.search(r"(?:معه|معاه|معها|معاها|مع\s+ه|with\s+him|with\s+her)", norm, re.I)
    )
    if not wants_meeting:
        return None

    meeting_ctx = state.get("meeting_context") if isinstance(state.get("meeting_context"), dict) else {}
    employee = meeting_ctx.get("employee") if isinstance(meeting_ctx.get("employee"), dict) else {}

    if not employee or not _uid(employee):
        focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
        employee = focus.get("current") if isinstance(focus.get("current"), dict) else {}

    if not employee or not _uid(employee):
        return None

    name = _name(employee, language) or _name(employee, "ar")
    first = _first_name(name) or ("الموظف" if language == "ar" else "them")
    period = str(meeting_ctx.get("period") or "none")
    if period not in {"today", "tomorrow"}:
        period = "none"

    state["task_flow"] = {
        "kind": "employee_meeting_helper",
        "stage": "duration" if period != "none" else "date",
        "period": period,
        "time_window": "",
        "duration_minutes": 0,
        "target": _bounded_employee(employee),
        "original_message": str(message or "")[:1000],
    }

    if language == "ar":
        if period == "today":
            return f"تمام، مع {first} اليوم. كم تقريبًا تتوقع مدة الاجتماع؟"
        if period == "tomorrow":
            return f"تمام، مع {first} بكرة. كم تقريبًا تتوقع مدة الاجتماع؟"
        return f"تمام، مع {first}. تبغيه اليوم ولا بكرة؟ وبعدها أضبط لك الوقت الأنسب."

    if period == "today":
        return f"Got it — with {first} today. Roughly how long will the meeting be?"
    if period == "tomorrow":
        return f"Got it — with {first} tomorrow. Roughly how long will the meeting be?"
    return f"Got it — with {first}. Do you want it today or tomorrow?"


def _parse_duration_minutes(text: str) -> int:
    norm = _norm_role_text(str(text or "").strip()).strip(" ؟?!.,،")
    if re.fullmatch(r"(?:1|١)", norm):
        return 60
    if re.fullmatch(r"(?:2|٢)", norm):
        return 120
    if re.fullmatch(r"(?:3|٣)", norm):
        return 180
    if re.search(r"(?:ساعة|ساعه)\s*(?:ونص|و\s*نص|ونصف)|1\.5\s*(?:hour|hours?)", norm, re.I):
        return 90
    if re.search(r"(?:نص|نصف)\s*(?:ساعة|ساعه)|30\s*(?:دقيقة|دقيقه|دقايق|minutes?|mins?)?", norm, re.I):
        return 30
    if re.search(r"(?:ربع)\s*(?:ساعة|ساعه)|15\s*(?:دقيقة|دقيقه|دقايق|minutes?|mins?)?", norm, re.I):
        return 15
    if re.search(r"(?:45)\s*(?:دقيقة|دقيقه|دقايق|minutes?|mins?)?", norm, re.I):
        return 45
    if re.search(r"(?:ساعتين|ساعتان|2\s*(?:ساعة|ساعه|hours?|hrs?))", norm, re.I):
        return 120
    if re.search(r"(?:ثلاث\s+ساعات|ثلاثة\s+ساعات|3\s*(?:ساعات|hours?|hrs?))", norm, re.I):
        return 180
    if re.search(r"(?:ساعة|ساعه|60\s*(?:دقيقة|دقيقه|دقايق|minutes?|mins?)?|an?\s+hour|1\s*hour)", norm, re.I):
        return 60

    m = re.search(r"\b(\d{1,3})\s*(?:دقيقة|دقيقه|دقايق|minutes?|mins?)\b", norm, re.I)
    if m:
        return max(5, min(int(m.group(1)), 480))
    h = re.search(r"\b(\d{1,2})\s*(?:ساعة|ساعه|ساعات|hours?|hrs?)\b", norm, re.I)
    if h:
        return max(15, min(int(h.group(1)) * 60, 480))
    return 0


def _frustration_reply(message: str, state: dict[str, Any], language: str) -> str | None:
    """Friendly recovery when the user is correcting the assistant, not making a new API request."""
    raw = str(message or "").strip()
    norm = _norm_role_text(raw).strip(" ؟?!.,،")
    frustrated = bool(re.search(
        r"(?:تستهبل|تستعبط|غبي|غباء|مو\s+كذا|مش\s+كذا|حوس|لخبط|ما\s+تفهم|ماتفهم|وش\s+فيك|ايش\s+فيك|wrong|stupid)",
        norm, re.I,
    ))
    if not frustrated:
        return None

    flow = state.get("task_flow") if isinstance(state.get("task_flow"), dict) else {}
    if flow:
        state["task_flow"] = {}

    if language == "ar":
        return "هههه حقك علي 😅"
    return "Fair 😅 I overcomplicated that. Say it naturally and I’ll continue from the context without forcing a scripted flow."


def _meeting_companion_turn(message: str, state: dict[str, Any], language: str) -> tuple[str, str] | None:
    """Continue only an employee-meeting planning flow. Self meeting lists never start this."""
    flow = state.get("task_flow") if isinstance(state.get("task_flow"), dict) else {}
    if str(flow.get("kind") or "") != "employee_meeting_helper":
        return None

    norm = _norm_role_text(str(message or "").strip()).strip(" ؟?!.,،")
    stage = str(flow.get("stage") or "")

    # A clear new domain request cancels the planning flow.
    if re.search(r"(?:اجاز|إجاز|حضور|سياس|اعلان|إعلان|خبر|مزايا|تراسل|email|ايميل|إيميل|رصيد)", norm, re.I):
        state["task_flow"] = {}
        return None

    target = flow.get("target") if isinstance(flow.get("target"), dict) else {}
    name = _name(target, language) or _name(target, "ar")
    first = _first_name(name) or ("الموظف" if language == "ar" else "them")

    if stage == "date":
        if re.search(r"(?:اليوم|today)", norm, re.I):
            flow["period"] = "today"
        elif re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I):
            flow["period"] = "tomorrow"
        else:
            return ("answer", "اليوم ولا بكرة؟" if language == "ar" else "Today or tomorrow?")
        flow["stage"] = "duration"
        state["task_flow"] = flow
        day = "اليوم" if flow["period"] == "today" else "بكرة"
        return ("answer", f"تمام، مع {first} {day}. كم تقريبًا مدة الاجتماع؟" if language == "ar"
                else f"Got it — with {first} {day}. Roughly how long?")

    if stage == "duration":
        minutes = _parse_duration_minutes(norm)
        if not minutes:
            return ("answer", "كم تقريبًا مدته؟ مثلاً نص ساعة، ساعة، أو ساعتين." if language == "ar"
                    else "Roughly how long — 30 minutes, an hour, or two hours?")

        flow["duration_minutes"] = minutes
        period = str(flow.get("period") or "none")
        state["availability_duration_minutes"] = minutes
        state["task_flow"] = {}

        # Keep the verified person as calendar focus; the normal availability path
        # will query the backend again rather than trusting old chat text.
        focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
        if target and _uid(target):
            focus["current"] = _bounded_employee(target)
            state["focus"] = focus

        day = "اليوم" if period == "today" else "بكرة"
        return ("rewrite", f"متى فاضي {day}")

    return None



_COMPOUND_START_RE = re.compile(
    r"(?:عطني|اعطني|أعطني|ابي|أبي|ابغى|أبغى|وش|ايش|إيش|مين|من|متى|هل|كم|"
    r"عندي|ورني|جيب|هات|ابحث|أبحث|شيك|شوف|show|give|find|who|what|when|how)\b",
    re.I,
)


def _split_compound_message(message: str) -> list[str]:
    """Conservatively split an explicitly multi-request turn.

    We intentionally do not split ordinary field conjunctions such as
    "ايميله ورقم جواله". A conjunction is split only when the right-hand side
    clearly starts another request. This keeps the existing single-request
    parser as the authority while allowing sequential, agentic execution.
    """
    raw = " ".join(str(message or "").strip().split())
    if not raw:
        return []

    # Keep a self-manager request and manager-owned fields in one semantic turn.
    # Example: "مين مديري المباشر وابي ايميله" should return the manager name
    # and email together, not execute two near-duplicate agent steps.
    manager_compound = _norm_role_text(raw)
    if (
        re.search(r"(?:مديري|my\s+manager)", manager_compound, re.I)
        and re.search(
            r"(?:ايميله|إيميله|بريده|جواله|رقم\s+جواله|تحويلته|منصبه|مسماه|"
            r"his\s+(?:email|phone|mobile|extension|title)|manager'?s\s+(?:email|phone|mobile|extension|title))",
            manager_compound,
            re.I,
        )
    ):
        return [raw]

    chunks = [x.strip() for x in re.split(r"[؟?؛;\n]+", raw) if x.strip()]
    output: list[str] = []
    for chunk in chunks:
        # "وبعدين" is always a new step.
        chunk = re.sub(r"\s+وبعدين\s+", " ||| ", chunk, flags=re.I)
        # Split at waw only when the next phrase clearly begins a new request.
        chunk = re.sub(
            r"\s+و(?=(?:عطني|اعطني|أعطني|ابي|أبي|ابغى|أبغى|وش|ايش|إيش|مين|متى|هل|كم|عندي|ورني|جيب|هات|ابحث|أبحث|شيك|شوف)\b)",
            " ||| ",
            chunk,
            flags=re.I,
        )
        output.extend(x.strip() for x in chunk.split("|||") if x.strip())

    # Manager relations are separate tool calls from profile/contact fields.
    # Split only when the same clause also asks for non-manager fields.
    expanded: list[str] = []
    for item in output:
        item_norm = _norm_role_text(item)
        self_manager = bool(re.search(r"(?:مديري|my\s+manager)", item_norm, re.I))
        focus_manager = bool(re.search(r"(?:مديره|مديرها|his\s+manager|her\s+manager)", item_norm, re.I))
        asks_other_fields = bool(re.search(
            r"(?:اسمي|منصبي|مسماي|ادارتي|إدارتي|ايميلي|إيميلي|بريدي|جوالي|موبايلي|"
            r"منصبه|مسماه|ادارته|إدارته|ايميله|إيميله|بريده|جواله|موبايله|"
            r"name|title|department|email|mobile|phone)", item_norm, re.I
        ))
        manager_owned_fields = bool(
            self_manager
            and re.search(r"(?:اسم\s+مديري|مديري.*(?:ايميله|إيميله|منصبه|مسماه|ادارته|إدارته)|(?:ايميل|إيميل|منصب|مسمى|اداره|إدارة)\s+مديري)", item_norm, re.I)
        )
        if manager_owned_fields:
            expanded.append(item)
        elif (self_manager or focus_manager) and asks_other_fields:
            first = item
            if self_manager:
                first = re.sub(r"(?:\s*و)?\s*(?:مين\s+)?مديري(?:\s+مين)?[؟?!.،,]*", " ", first, flags=re.I)
                manager_step = "مين مديري؟"
            else:
                first = re.sub(r"(?:\s*و)?\s*(?:مين\s+)?مدير(?:ه|ها)(?:\s+مين)?[؟?!.،,]*", " ", first, flags=re.I)
                manager_step = "مين مديره؟"
            first = re.sub(r"\s+", " ", first).strip(" ،,")
            if len(first) >= 2:
                expanded.append(first)
            expanded.append(manager_step)
        else:
            expanded.append(item)

    cleaned_expanded: list[str] = []
    for item in expanded:
        simple_num = re.fullmatch(r"\d{1,2}(?::\d{2})?\.?\s*", item)
        if simple_num and cleaned_expanded:
            number = re.sub(r"\D", "", item)
            if number and re.search(rf"(?<!\d){re.escape(number)}(?!\d)", cleaned_expanded[-1]):
                continue
        cleaned_expanded.append(item)

    # Compound mode is intentionally bounded.
    output = [x for x in cleaned_expanded if len(x) >= 2][:5]
    return output if len(output) > 1 else [raw]


def _combine_compound_answers(answers: list[str], language: str) -> str:
    clean: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        text = str(answer or "").strip()
        if not text:
            continue
        # Avoid repetitive conversational openers in one combined response.
        if clean and language == "ar":
            text = re.sub(r"^(?:أكيد|تمام|أبشر|ابشر)[،,:\s]+", "", text).strip()
        key_text = re.sub(
            r"^(?:(?:أكيد|تمام|أبشر|ابشر)\s*)?[😊👌🌷📧📱☎️\s]+",
            "",
            text,
            flags=re.I,
        ).strip()
        key = re.sub(r"\s+", " ", key_text).casefold()
        if key and key not in seen:
            seen.add(key)
            clean.append(text)
    return "\n\n".join(clean)


def _friendly_surface(answer: str, language: str) -> str:
    """Final cleanup only. Tone is owned by each renderer/prompt, not forced globally."""
    text = str(answer or "").strip()
    if not text:
        return text

    if language != "ar":
        acronyms = {"RCRC", "IT", "HR", "PMO", "ICT", "AI", "OCI", "API", "CAB", "SQL", "MCP", "SSO"}
        def _fix_caps(match: re.Match[str]) -> str:
            word = match.group(0)
            return word.upper() if word.upper() in acronyms else word.capitalize()

        text = re.sub(r"\b[A-Z]{3,}\b", _fix_caps, text)
        text = text.replace("**", "")
        return text

    # Keep user-facing text clean without forcing the same opener on every reply.
    # Preserve lightweight markdown emphasis; the frontend already renders it.
    text = re.sub(r"^(?:أكيد\s*[🌷😊👌]?\s*){2,}", "أكيد 😊 ", text).strip()

    # Frontend currently renders markdown markers literally.
    # Never expose ** in user-facing answers.
    text = text.replace("**", "")
    return text


def _clear_pending(state: dict[str, Any]) -> None:
    state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}


def _set_employee_more_details_pending(
    state: dict[str, Any],
    request: ParsedRequest,
    *,
    self_subject: bool,
) -> None:
    """Persist the renderer's "job title and department?" offer."""
    if self_subject or request.intent != "employee_details":
        return
    fields = [str(field) for field in (request.fields or [])]
    field_set = set(fields)
    contact_fields = {"email", "mobile", "extension"}
    if not (2 <= len(fields) <= 3):
        return
    if not field_set.intersection(contact_fields):
        return
    if field_set.intersection({"job_title", "department"}):
        return

    follow_req = ParsedRequest(
        intent="employee_details",
        operation="details",
        targets=[TargetRef(kind="focus", name="")],
        fields=["job_title", "department"],
        query="employee:focus_more_details",
        confidence=1.0,
    )
    state["pending"] = {
        "kind": "employee_more_details",
        "query": "",
        "candidates": [],
        "request": follow_req.model_dump(),
        "message": "after_employee_contact_card",
    }


def _record_request(state: dict[str, Any], request: ParsedRequest) -> None:
    # Keep only structural request context here. Verified employee facts stay in
    # focus/relation; this block lets short follow-ups reuse the last period/time
    # without asking the user to repeat themselves.
    state["last_request"] = {
        "intent": request.intent,
        "operation": request.operation,
        "fields": list(request.fields),
        "targets": [target.model_dump() for target in request.targets],
        "period": request.period,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "time_query": getattr(request, "time_query", ""),
        "query": str(request.query or ""),
        "role_query": str(request.role_query or ""),
        "department_query": str(request.department_query or ""),
        "amount": request.amount,
    }

    # v232 Context Engine: persist only structural conversation state.
    # No employee facts are copied here. This lets short follow-ups inherit the
    # active domain/view/date without replaying arbitrary old chat history.
    first_target = request.targets[0].model_dump() if request.targets else {}
    state["conversation_context"] = {
        "active_domain": str(request.intent or ""),
        "active_view": str(request.query or ""),
        "operation": str(request.operation or ""),
        "period": str(request.period or "none"),
        "start_date": str(request.start_date or ""),
        "end_date": str(request.end_date or ""),
        "time_query": str(getattr(request, "time_query", "") or ""),
        "fields": list(request.fields or [])[:12],
        "target_kind": str(first_target.get("kind") or ""),
        # name is intentionally omitted: verified identity remains in focus.
        "has_target": bool(first_target),
    }


def _prepare_memory_for_request(
    state: dict[str, Any],
    request: ParsedRequest,
) -> None:
    """Prevent stale verified context from hijacking a new independent turn."""
    if request.intent in {
        "general",
        "web_search",
        "summarize_previous",
    }:
        return

    if request.intent in DETAIL_BASE_INTENT:
        return

    contextual_kinds = {
        "focus",
        "previous_focus",
        "relation_subject",
        "relation_object",
    }

    if any(
        target.kind in contextual_kinds
        for target in request.targets
    ):
        return

    _clear_pending(state)

    state["relation"] = {
        "predicate": "",
        "subject": {},
        "object": {},
    }

    keep_directory_context = (
        request.intent == "general_department_members"
        and not str(
            request.department_query or ""
        ).strip()
    )

    if not keep_directory_context:
        state["directory_context"] = {
            "type": "",
            "general_department_key": "",
            "general_department": "",
            "count": 0,
        }

    if request.intent != "employee_role_search":
        state["role_lookup"] = {
            "role_query": "",
            "employee": {},
        }

    state["list_state"] = {
        "intent": "",
        "items": [],
        "query": "",
    }

    # Keep the last verified person in conversational focus until a NEW
    # employee is actually resolved successfully.
    #
    # Clearing focus before execution breaks natural follow-ups:
    #   "مين شاهه العقيل؟" -> focus=شاهه
    #   "مين مدير عام ...؟" -> if lookup fails, شاهه must still be remembered
    #   "طيب ابي ايميلها" -> must resolve to شاهه
    #
    # Successful employee/role resolution calls _remember_entity(), which moves
    # the old current focus to previous and installs the new verified employee.
    state.setdefault(
        "focus",
        {
            "current": {},
            "previous": {},
        },
    )


def _self_subject(target: ResolvedTarget) -> bool:
    return target.kind == "self"


def _requested_field_available(row: dict[str, Any], field: str, lang: str) -> bool:
    mapping = {
        "name": ("name_ar", "name", "name_en") if lang == "ar" else ("name_en", "name", "name_ar"),
        "name_en": ("name_en",),
        "job_title": ("job_title_ar", "job_title", "job_title_en") if lang == "ar" else ("job_title_en", "job_title", "job_title_ar"),
        "job_title_en": ("job_title_en",),
        "department": ("department_ar", "department", "department_en") if lang == "ar" else ("department_en", "department", "department_ar"),
        "department_en": ("department_en",),
        "sector": ("sector_ar", "sector_en"),
        "sector_en": ("sector_en",),
        "email": ("email",),
        "mobile": ("mobile",),
        "extension": ("extension",),
        "username": ("username", "userid"),
        "employee_type": ("employee_type",),
    }
    keys = mapping.get(field, (field,))
    return any(row.get(key) not in (None, "", " ") for key in keys)


def _verified_row_satisfies(row: dict[str, Any], fields: list[str], lang: str) -> bool:
    wanted = [field for field in fields if field != "all"]
    return bool(wanted) and all(_requested_field_available(row, field, lang) for field in wanted)


def _content_item_id(row: dict[str, Any]) -> str:
    for key in ("id", "Id", "ID", "item_id", "newsId", "announcementId", "policyId", "mazayaId"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _content_item_text(row: dict[str, Any]) -> str:
    for key in ("headerAr", "headerEn", "header", "title", "subject", "name", "name_ar", "name_en", "company"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _rows_from_data(data: Any) -> list[dict[str, Any]]:
    """
    Extract list rows from direct payloads and standard RCRC API wrappers.

    The important rule is that context/pending logic must see the same rows
    the renderer sees. MeetingList commonly returns:
        {"data": {"meetings": [...]}}
    """
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if not isinstance(data, dict):
        return []

    for key in (
        "employees", "items", "results", "rows", "meetings", "transactions",
        "attendance", "vacations", "policies", "news", "announcements", "benefits",
    ):
        value = data.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    nested = data.get("data")
    if isinstance(nested, list):
        return [row for row in nested if isinstance(row, dict)]
    if isinstance(nested, dict):
        return _rows_from_data(nested)

    return []


def _content_selection_number(message: str) -> int:
    raw = str(message or "").strip().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    raw = re.sub(r"[؟?!.،,]+", " ", raw)
    norm = " ".join(raw.casefold().split())
    direct = re.fullmatch(r"(?:(?:رقم|number|#)\s*)?(\d{1,2})", norm, re.I)
    if direct:
        return int(direct.group(1))
    ordinal_words = {
        "الاول": 1, "الأول": 1, "اول": 1, "أول": 1, "first": 1,
        "الثاني": 2, "ثاني": 2, "second": 2,
        "الثالث": 3, "ثالث": 3, "third": 3,
        "الرابع": 4, "رابع": 4, "fourth": 4,
        "الخامس": 5, "خامس": 5, "fifth": 5,
        "السادس": 6, "سادس": 6, "sixth": 6,
        "السابع": 7, "سابع": 7, "seventh": 7,
        "الثامن": 8, "ثامن": 8, "eighth": 8,
        "التاسع": 9, "تاسع": 9, "ninth": 9,
        "العاشر": 10, "عاشر": 10, "tenth": 10,
    }
    cleaned = re.sub(r"^(?:ابي|أبي|ابغى|أبغى|اختار|اخترت|رقم|number)\s+", "", norm, flags=re.I).strip()
    return ordinal_words.get(cleaned, 0)


def _is_email_drafting_request(message: str) -> bool:
    """True only when the user wants us to WRITE an email, not look one up."""
    norm = _norm_role_text(str(message or ""))
    has_email = bool(re.search(r"(?:ايميل|إيميل|ايميلًا|ايميلاً|رساله\s+ايميل|email|e-mail)", norm, re.I))
    has_write = bool(re.search(
        r"(?:اكتب(?:\s+لي)?|اكتبي(?:\s+لي)?|صيغ(?:\s+لي)?|صيغي(?:\s+لي)?|"
        r"صياغ(?:ه|ة)|جهز(?:\s+لي)?|حضّر(?:\s+لي)?|حضر(?:\s+لي)?|"
        r"ابي\s+(?:اكتب|ارسل)|ابغى\s+(?:اكتب|ارسل)|بكتب|برسل|"
        r"write|draft|compose|help\s+me\s+write)",
        norm, re.I,
    ))
    return has_email and has_write


def _writing_context(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("writing_context") if isinstance(state, dict) else {}
    return value if isinstance(value, dict) else {}


def _writing_followup_target_language(message: str, state: dict[str, Any]) -> str:
    """Resolve the output language for a revision/translation follow-up."""
    norm = _norm_role_text(str(message or ""))
    ctx = _writing_context(state)
    previous = str(ctx.get("language") or "").strip().lower()

    if re.search(r"(?:بالانجليزي|بالإنجليزي|انجليزي|إنجليزي|english|to\s+english|translate\s+to\s+english)", norm, re.I):
        return "en"
    if re.search(r"(?:بالعربي|بالعربية|عربي|العربيه|العربية|arabic|to\s+arabic|translate\s+to\s+arabic)", norm, re.I):
        return "ar"

    # Bare "ترجمه/translate it" means translate the current draft to the
    # other supported language. Other edit requests preserve draft language.
    bare_translate = bool(re.fullmatch(
        r"(?:ترجم(?:ه|ها|يه|يها|لي|ها\s+لي)?|ترجمة|translate(?:\s+it)?|translate)",
        norm.strip(" ؟?!.،,"), re.I,
    ))
    if bare_translate:
        return "en" if previous == "ar" else "ar"
    return previous if previous in {"ar", "en"} else detect_language(message, fallback="ar")


def _is_writing_followup_request(message: str, state: dict[str, Any]) -> bool:
    """Short natural edits that should modify the last generated draft."""
    ctx = _writing_context(state)
    if str(ctx.get("kind") or "") != "email" or not str(ctx.get("draft") or "").strip():
        return False

    norm = _norm_role_text(str(message or "")).strip()
    # Explicitly drafting a NEW email should use the new-draft hard route.
    if _is_email_drafting_request(message):
        return False

    patterns = (
        # Natural Gulf/Arabic follow-ups such as "ابيه لطيف اكثر" or
        # "أبيها أقصر" refer to the current draft when one exists.
        r"(?:خله|خليها|خليه|اجعله|اجعلها|ابيه|أبيه|ابيها|أبيها|ابغاه|أبغاه|ابغاها|أبغاها|make\s+it)",
        r"(?:رسمي|اكثر\s+رسمي|أكثر\s+رسم|formal|professional|لطيف|ألطف|الطف|ودود|ودي|ودّي|friendly|warm|warmer|مباشر|direct|واضح|أوضح|مهني|professional)",
        r"(?:اختصر|اختصره|اختصري|اقصر|أقصر|shorter|concise|مختصر)",
        r"(?:طوله|اطوله|أطوله|longer|وسع|وسعه)",
        r"(?:ترجم|ترجمه|ترجمها|ترجملي|ترجمي|ترجمة|translate|بالانجليزي|بالإنجليزي|بالعربي|بالعربية|english|arabic)",
        r"(?:غير|غيّر|عدل|عدّل|بدل|بدّل|change|revise|rewrite|edit)",
        r"(?:البداية|النهاية|العنوان|subject|opening|ending|last\s+paragraph|first\s+paragraph)",
        r"(?:شيل|احذف|حذف|remove|delete|اضف|أضف|زيد|add)",
        r"(?:لا\s+تذكر|لا\s+تحط|بدون|without)",
    )
    return any(re.search(p, norm, re.I) for p in patterns)


def _email_request_is_vague(message: str) -> bool:
    """A bare 'write me an email' needs the topic/purpose, not an API lookup."""
    norm = _norm_role_text(str(message or "")).strip(" ؟?!.،,")
    stripped = re.sub(
        r"(?:ابي|ابغى|ممكن|لو\s+سمحت|please|can\s+you|could\s+you|"
        r"اكتب(?:\s+لي)?|اكتبي(?:\s+لي)?|صيغ(?:\s+لي)?|صيغي(?:\s+لي)?|"
        r"جهز(?:\s+لي)?|حضّر(?:\s+لي)?|حضر(?:\s+لي)?|write(?:\s+me)?|draft(?:\s+me)?|compose|"
        r"ايميل|إيميل|email|e-mail)",
        " ", norm, flags=re.I,
    )
    stripped = " ".join(stripped.split()).strip(" ؟?!.،,")
    return len(stripped) < 3


def _friendly_endpoint_answer(answer: str, intent: str, language: str) -> str:
    """Give verified endpoint answers a warm, concise Sanad tone.

    This is presentation-only: it never changes facts, IDs, routing or tool data.
    Errors/empty-state messages keep their original wording so we do not prefix
    failures with an inappropriate "أكيد" / "Sure".
    """
    text = str(answer or "").strip()
    if not text:
        return text

    first = text.splitlines()[0].strip()
    if language == "ar":
        failure_starts = (
            "ما قدرت", "ما لقيت", "ما فيه", "للأسف", "آسف", "عذرًا",
            "هذي المعلومة", "هذه المعلومة", "سجّل", "تعذر", "غير متوفر",
        )
        if first.startswith(failure_starts):
            return text
        # Already conversational/friendly: do not duplicate a lead-in.
        if first.startswith(("أكيد", "أبشر", "أبشري", "حصلته", "لقيت", "تمام", "هذي", "هذا", "عندك", "ما عندك", "مرحب")):
            return text
        return f"أكيد 😊\n{text}"

    failure_starts = ("Sorry", "I couldn't", "I can’t", "No ", "There are no", "Please sign in")
    if first.startswith(failure_starts) or first.startswith(("Sure", "Of course", "Got it", "I found", "Here are", "You have", "You don't", "Your ")):
        return text
    return f"Sure 😊\n{text}"


def _restore_content_detail_selection(message: str, state: dict[str, Any]) -> ParsedRequest | None:
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "") != "content_detail_selection":
        return None
    raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}
    index = _content_selection_number(message)
    if not raw_req or index <= 0:
        return None
    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    items = [x for x in (list_state.get("items") or []) if isinstance(x, dict)]
    if not (1 <= index <= len(items)):
        return None
    try:
        restored = ParsedRequest.model_validate(raw_req)
    except Exception:
        return None
    restored.ordinal = index
    restored.operation = "details"
    restored.fields = ["all"]
    restored.query = ""
    restored.clarification_reason = ""
    restored.confidence = 1.0
    state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}
    return restored


def _restore_full_content_followup(message: str, state: dict[str, Any]) -> ParsedRequest | None:
    raw = str(message or "").strip()
    norm = re.sub(r"[؟?!.،,]+", " ", raw.casefold())
    norm = " ".join(norm.split())
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    pending_full = str(pending.get("kind") or "") == "content_full_details"
    confirmation = bool(re.fullmatch(
        r"(?:اي|ايوه|أيوه|نعم|تمام|طيب|اوكي|أوكي|ok|yes|yeah|yep|جيبه|جيبها|هات|اعرضه|اعرضها|كامل|كامله|كاملة)",
        norm, re.I,
    ))
    full_cue = bool(re.search(
        r"(?:التفاصيل\s+(?:كامله|كاملة)(?:\s+(?:للخبر|للاعلان|للإعلان|للميزه|للميزة|للسياسه|للسياسة))?|"
        r"كامل\s+التفاصيل|كل\s+التفاصيل|التفاصيل\s+بالكامل|"
        r"الخبر\s+كامل|كامل\s+الخبر|الاعلان\s+كامل|الإعلان\s+كامل|"
        r"كامل\s+الاعلان|كامل\s+الإعلان|full\s+details?|show\s+all\s+details?|full\s+article)",
        norm, re.I,
    ))
    if not full_cue and not (pending_full and confirmation):
        return None

    # A confirmation such as "اي" / "جيبه" after Sanad explicitly offered
    # the full verified item stays locked to the same content focus.
    if pending_full and confirmation:
        state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}

    # Primary source: explicit verified content focus.
    focus = state.get("content_focus") if isinstance(state.get("content_focus"), dict) else {}
    intent = str(focus.get("detail_intent") or "")

    # Persistent fallback: selected content is also stored inside list_state,
    # which survives normal conversation-memory cleanup. This prevents phrases
    # such as "التفاصيل الكاملة للخبر" from ever falling through to employee search.
    if intent not in DETAIL_BASE_INTENT:
        list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
        selected_intent = str(list_state.get("selected_detail_intent") or "")
        if selected_intent in DETAIL_BASE_INTENT:
            intent = selected_intent
        else:
            base_to_detail = {
                "news": "news_details",
                "announcements": "announcement_details",
                "mazaya": "mazaya_details",
                "policy_search": "policy_details",
            }
            if int(list_state.get("selected_index") or 0) > 0:
                intent = base_to_detail.get(str(list_state.get("intent") or ""), "")

    if intent not in DETAIL_BASE_INTENT:
        return None

    return ParsedRequest(
        intent=intent, operation="details", targets=[], fields=["all"],
        query="content:full", role_query="", department_query="",
        period="none", start_date="", end_date="", ordinal=0,
        relation_depth=1, amount=None, time_query="", confidence=1.0,
        clarification_reason="",
    )

def _choose_content_item(request: ParsedRequest, state: dict[str, Any]) -> dict[str, Any] | None:
    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
    items = [x for x in (list_state.get("items") or []) if isinstance(x, dict)]
    expected = DETAIL_BASE_INTENT.get(request.intent)
    if expected and str(list_state.get("intent") or "") != expected:
        return None
    if request.ordinal and 1 <= request.ordinal <= len(items):
        return items[request.ordinal - 1]
    query = " ".join(str(request.query or "").casefold().split())
    if query:
        matches = [row for row in items if query in _content_item_text(row).casefold()]
        if len(matches) == 1:
            return matches[0]
    if len(items) == 1:
        return items[0]
    return None


CJK_RE = re.compile(r"[\u3400-\u9fff]")
ARABIC_TEXT_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_TEXT_RE = re.compile(r"[A-Za-z]")


def _language_is_valid(text: str, language: str) -> bool:
    value = str(text or "").strip()
    if not value or CJK_RE.search(value):
        return False
    if language == "ar":
        return bool(ARABIC_TEXT_RE.search(value))
    return bool(LATIN_TEXT_RE.search(value)) and not bool(ARABIC_TEXT_RE.search(value))


class Orchestrator:
    def __init__(self) -> None:
        self.employee_api = EmployeeAPIClient()
        self.memory = ConversationMemory()
        self.parser = SemanticParser()
        self.resolver = ReferenceResolver(self.employee_api)
        self.directory = DirectoryQueryService(self.employee_api)
        self.authorization = AuthorizationService(self.employee_api)
        self.actions = ActionWorkflowService(self.employee_api)
        self.cache = TTLCache(max_entries=512)
        self.mcp_client = MCPToolClient()
        self._mcp_tool_names: set[str] = set()

    @property
    def tool_names(self) -> list[str]:
        return sorted(set(EMPLOYEE_TOOL_NAMES) | self._mcp_tool_names)

    async def start(self) -> None:
        if getattr(settings, "enable_tools", False):
            try:
                await self.mcp_client.connect()
                self._mcp_tool_names = {tool.name for tool in self.mcp_client.list_tools()}
            except Exception as exc:
                print(f"[MCP startup warning] {exc}", file=sys.stderr)
        print(f"[Orchestrator] Build: {ORCHESTRATOR_BUILD}", file=sys.stderr)
        print(f"[Renderer] Build: {FRIENDLY_RENDERER_BUILD}", file=sys.stderr)
        print(f"[Orchestrator] Enterprise tools: {self.tool_names}", file=sys.stderr)
        print(f"[Runtime] orchestrator_file={__file__}", file=sys.stderr)
        import core.response_renderer as _response_renderer_module
        print(f"[Runtime] response_renderer_file={getattr(_response_renderer_module, '__file__', '')}", file=sys.stderr)
        import core.semantic_parser as _semantic_parser_module
        print(f"[Runtime] semantic_parser_file={getattr(_semantic_parser_module, '__file__', '')}", file=sys.stderr)
        print(f"[Runtime] semantic_parser_build={SEMANTIC_PARSER_BUILD}", file=sys.stderr)

    async def stop(self) -> None:
        await self.employee_api.close()
        try:
            await self.mcp_client.close()
        except Exception:
            pass

    def reset_memory(self, username: str, conversation_id: str | None = None) -> None:
        username = str(username or "").strip()
        if not username:
            return
        if conversation_id:
            self.memory.reset(f"{username}:{conversation_id}")
        else:
            self.memory.reset_prefix(f"{username}:")

    async def welcome(self, context: EmployeeContext, language: str = "ar") -> dict[str, str]:
        """Authenticated welcome payload used when the chat screen opens."""
        language = "en" if str(language).lower() == "en" else "ar"
        result = await self._execute_tool("get_employee_profile", {}, context, target_username=context.username)
        if not result.get("success"):
            message = (
                "أهلًا وسهلًا 👋\nأنا سند، وحاليًا أنا في مرحلتي التجريبية 😊\nأقدر أساعدك في الوصول لخدمات ومعلومات الهيئة بسهولة، مثل الإجازات، الموظفين، الاجتماعات وغيرها.\nكيف أقدر أساعدك اليوم؟"
                if language == "ar"
                else "Hi 👋 I'm Sanad. I can help with RCRC services and information such as leave, employees, meetings, and more. I'm currently in my experimental phase 😊 How can I help today?"
            )
            return {"message": message, "name": ""}
        data = result.get("data") if isinstance(result.get("data"), dict) else {}

        # Never greet the signed-in user with another employee's profile.
        # Some POC API deployments have returned the token owner's profile even
        # when a different username was posted. Until production identity comes
        # from a validated token claim, fail closed to a generic greeting.
        returned_username = _uid(data).strip().casefold()
        expected_username = str(context.username or "").strip().casefold()
        if returned_username and expected_username and returned_username != expected_username:
            print(
                f"[Welcome Identity Guard] expected={context.username!r} returned={_uid(data)!r}",
                file=sys.stderr,
            )
            message = (
                "أهلًا وسهلًا 👋\nأنا سند، وحاليًا أنا في مرحلتي التجريبية 😊\nأقدر أساعدك في الوصول لخدمات ومعلومات الهيئة بسهولة، مثل الإجازات، الموظفين، الاجتماعات وغيرها.\nكيف أقدر أساعدك اليوم؟"
                if language == "ar"
                else "Hi 👋 I'm Sanad. I can help with RCRC services and information such as leave, employees, meetings, and more. I'm currently in my experimental phase 😊 How can I help today?"
            )
            return {"message": message, "name": ""}

        full_name = _name(data, language) or _name(data, "ar") or context.username
        first_name = str(full_name).strip().split()[0] if str(full_name).strip() else context.username
        message = (
            f"أهلًا وسهلًا {first_name} 👋\nأنا سند، وحاليًا أنا في مرحلتي التجريبية 😊\nأقدر أساعدك في الوصول لخدمات ومعلومات الهيئة بسهولة، مثل الإجازات، الموظفين، الاجتماعات وغيرها.\nكيف أقدر أساعدك اليوم؟"
            if language == "ar"
            else f"Hi {first_name} 👋 I'm Sanad. I can help with RCRC services and information such as leave, employees, meetings, and more. I'm currently in my experimental phase 😊 How can I help today?"
        )
        return {"message": message, "name": full_name}

    async def _is_sensitive(self, message: str) -> bool:
        if "check_safety" not in self._mcp_tool_names:
            return False
        try:
            result = await self.mcp_client.call_tool("check_safety", {"question": message})
            return str(result).strip().upper() == "SENSITIVE"
        except Exception:
            return False

    # =======================================================================
    # RCRC-FIX-v1 (تعديل): الموديل العام ما عنده أي بيانات. يستخدم برومبت خاص
    # يمنعه يخترع، وكل جواب يمر على حارس يمسك الإيميلات/الجوالات/التحويلات
    # و"مديرك هو..." المخترعة قبل ما توصل للمستخدم.
    # =======================================================================
    async def _writing_facts(self, message: str, context: EmployeeContext | None, language: str, state: dict[str, Any] | None = None) -> dict[str, str]:
        """RCRC-FIX-v2: verified names for drafting (user + direct manager)."""
        facts: dict[str, str] = {}
        if context is None:
            return facts

        # Friendly email drafting: use the currently VERIFIED recipient when the
        # user says "اكتب له/لها ايميل" or asks for an email draft in context.
        if isinstance(state, dict) and re.search(r"(?:اكتب|صيغ|صيغي|write|draft).*(?:ايميل|إيميل|email)|(?:ايميل|إيميل|email).*(?:اكتب|صيغ|write|draft)", message, re.I):
            focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
            current = focus.get("current") if isinstance(focus.get("current"), dict) else {}
            if current and _uid(current):
                recipient_name = _name(current, language) or _name(current, "ar")
                if recipient_name:
                    facts["recipient_name"] = recipient_name
                    facts["recipient_first_name"] = _first_name(recipient_name)
                recipient_email = str(current.get("email") or "").strip()
                if recipient_email:
                    facts["recipient_email"] = recipient_email

        try:
            result = await self._execute_tool("get_employee_profile", {}, context, target_username=context.username)
            if result.get("success") and isinstance(result.get("data"), dict):
                me = result["data"]
                if _uid(me).casefold() == str(context.username or "").casefold():
                    facts["user_name"] = _name(me, language)
            if re.search(r"(?:مدير|manager)", _norm_role_text(message), re.I):
                result = await self._execute_tool("get_manager_details", {}, context, target_username=context.username)
                if result.get("success") and isinstance(result.get("data"), dict):
                    manager_row = result["data"]
                    manager_name = _name(manager_row, language)
                    facts["user_direct_manager_name"] = manager_name
                    if re.search(r"(?:اكتب|صيغ|write|draft).*(?:مدير|manager)|(?:مدير|manager).*(?:ايميل|إيميل|email)", message, re.I):
                        if manager_name:
                            facts["recipient_name"] = manager_name
                            facts["recipient_first_name"] = _first_name(manager_name)
                        manager_email = str(manager_row.get("email") or manager_row.get("emailAddress") or "").strip()
                        if manager_email:
                            facts["recipient_email"] = manager_email
        except Exception as exc:
            print(f"[Writing Facts] {exc}", file=sys.stderr)
        return {k: v for k, v in facts.items() if v}

    async def _email_draft_answer(
        self,
        message: str,
        language: str,
        verified_facts: dict[str, str] | None = None,
    ) -> str:
        """Dedicated drafting path. It never searches the employee directory."""
        if _email_request_is_vague(message):
            return (
                "أكيد 😊 وش موضوع الإيميل ومين بترسله له؟ اكتب لي الفكرة حتى لو بكلمتين وأنا أصيغه لك."
                if language == "ar"
                else "Sure 😊 What is the email about, and who are you sending it to? Give me the idea and I’ll draft it for you."
            )

        system_prompt = """
You are Sanad's email-writing mode. Draft the email requested by the user.
Rules:
- Do NOT search for enterprise data and do NOT invent names, emails, titles, dates, approvals, or facts.
- You may use VERIFIED_FACTS only when supplied.
- If the user gives rough wording, turn it into a polished email while preserving their meaning.
- Match the requested language. For Arabic, use natural professional Arabic. For English, use natural professional English.
- If the user asks for concise/short, keep it concise.
- Return ONLY the ready-to-send email text. Include a subject only if useful or explicitly requested.
- Do not explain what you did.
""".strip()
        payload = {
            "CURRENT_LANGUAGE": language,
            "CURRENT_MESSAGE": message,
            "VERIFIED_FACTS": verified_facts or {},
        }
        try:
            response = await asyncio.to_thread(chat, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ])
            answer = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "").strip()
        except Exception as exc:
            print(f"[Email Draft Error] {exc}", file=sys.stderr)
            answer = ""

        if answer and _language_is_valid(answer, language):
            return answer
        return (
            "أكيد 😊 اكتب لي باختصار وش تبغى تقول في الإيميل ومين المستلم، وأنا أصيغه لك بشكل مرتب."
            if language == "ar"
            else "Sure 😊 Tell me briefly what you want the email to say and who it is for, and I’ll draft it."
        )

    async def _email_followup_answer(
        self,
        instruction: str,
        draft: str,
        target_language: str,
    ) -> str:
        """Revise/translate the last email draft without invoking enterprise search."""
        system_prompt = """
You are Sanad's email revision mode. The user is editing the LAST_EMAIL_DRAFT.
Apply only the user's latest instruction to that same draft.
Rules:
- Preserve the original meaning and verified names/facts unless the user explicitly asks to remove/change wording.
- Never invent new enterprise facts, names, emails, job titles, dates, approvals, or commitments.
- If asked to make it more formal/professional, improve tone and wording without becoming overly verbose.
- If asked to shorten it, preserve the key message.
- If asked to translate, translate the entire ready-to-send email naturally.
- Respect TARGET_LANGUAGE exactly: ar = Arabic, en = English.
- If asked to change only one part (subject/opening/ending/paragraph), keep the rest substantially unchanged.
- Return ONLY the revised ready-to-send email text. Do not explain the edits.
""".strip()
        payload = {
            "TARGET_LANGUAGE": target_language,
            "USER_INSTRUCTION": instruction,
            "LAST_EMAIL_DRAFT": draft,
        }
        try:
            response = await asyncio.to_thread(chat, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ])
            answer = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "").strip()
        except Exception as exc:
            print(f"[Email Followup Error] {exc}", file=sys.stderr)
            answer = ""

        if answer and _language_is_valid(answer, target_language):
            return answer
        return draft


    async def _general_answer(
        self,
        message: str,
        state: dict[str, Any],
        language: str,
        verified_facts: dict[str, str] | None = None,
    ) -> str:
        history = state.get("general_history") if isinstance(state.get("general_history"), list) else []
        reaction_norm = _norm_role_text(str(message or "").strip())
        # Short frustration/rejection is about the quality of the previous reply,
        # not about the last employee entity.
        isolated_reaction = bool(re.fullmatch(
            r"(?:لا+|لاا+|مو\s+كذا|غلط|خطا|خطأ|تستهبل|تستعبط|غبي|وش\s+ذا|ايش\s+ذا|what|no+)",
            reaction_norm, re.I,
        ))
        # Old assistant turns that contain enterprise-looking facts are dropped
        # so an old hallucination is never fed back to the model.
        history_for_model = [] if isolated_reaction else sanitize_general_history(history[-8:], state)

        messages = [
            {"role": "system", "content": GENERAL_PROMPT},
            {"role": "user", "content": json.dumps({
                "CURRENT_LANGUAGE": language,
                "GENERAL_HISTORY": history_for_model,
                "VERIFIED_FACTS": verified_facts or {},
                "CONVERSATION_CONTEXT": {
                    "task_flow": state.get("task_flow") if isinstance(state.get("task_flow"), dict) else {},
                    "current_focus_name": _name(((state.get("focus") or {}).get("current") or {}), language)
                        if isinstance(state.get("focus"), dict) else "",
                },
                "CURRENT_MESSAGE": message,
            }, ensure_ascii=False)},
        ]
        allowed_text = " ".join((verified_facts or {}).values())

        answer = ""
        try:
            response = await asyncio.to_thread(chat, messages)
            answer = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "").strip()
        except Exception as exc:
            print(f"[General Answer Error] {exc}", file=sys.stderr)
            answer = ""

        # The fabrication guard.
        blocked = False
        if answer and looks_fabricated(answer, state, message, allowed_text):
            print(
                f"[General Guard] blocked general answer (needs enterprise data or fabricated fact) "
                f"msg={message[:80]!r}",
                file=sys.stderr,
            )
            answer = (
                "آسف 😅 أنا لسه بمرحلتي التجريبية وما فهمت طلبك هالمرة. ممكن توضحه لي شوي؟"
                if language == "ar"
                else "Sorry 😅 I'm still in my experimental phase and I didn't understand that request this time. Could you rephrase it a little?"
            )
            blocked = True

        # Qwen occasionally leaks a different language. Repair once, then fall back safely.
        if answer and not blocked and not _language_is_valid(answer, language):
            try:
                response = await asyncio.to_thread(chat, [
                    {"role": "system", "content": STRICT_LANGUAGE_REPAIR_PROMPT},
                    {"role": "user", "content": json.dumps({"CURRENT_LANGUAGE": language, "CURRENT_MESSAGE": message, "DRAFT_ANSWER": answer}, ensure_ascii=False)},
                ])
                repaired = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "").strip()
                if _language_is_valid(repaired, language) and not looks_fabricated(repaired, state, message, allowed_text):
                    answer = repaired
                else:
                    answer = ""
            except Exception:
                answer = ""

        if not answer:
            answer = (
                "هههه حقك علي 😅"
                if language == "ar"
                else "I'm currently in my experimental phase 😊 I couldn't prepare a good response to that request just now. Try phrasing it another way and I'll try again."
            )

        history = list(history[-8:])
        history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        state["general_history"] = history[-10:]
        return answer

    async def _web_answer(self, message: str, language: str) -> tuple[str, list[str]]:
        if not getattr(settings, "enable_web_search", False) or "web_search" not in self._mcp_tool_names:
            return ("البحث الخارجي مو متاح عندي حاليًا 😊" if language == "ar" else "Web search is currently unavailable."), []
        plan = {"query": message, "topic": "general", "max_results": 5, "search_depth": "basic", "days": None}
        try:
            response = await asyncio.to_thread(chat, [
                {"role": "system", "content": WEB_PLAN_PROMPT},
                {"role": "user", "content": json.dumps({"CURRENT_MESSAGE": message}, ensure_ascii=False)},
            ])
            raw = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "")
            start, end = raw.find("{"), raw.rfind("}")
            parsed = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
            if isinstance(parsed, dict):
                plan["query"] = str(parsed.get("query") or message).strip()
                plan["topic"] = "news" if parsed.get("topic") == "news" else "general"
                plan["search_depth"] = "advanced" if parsed.get("search_depth") == "advanced" else "basic"
                plan["max_results"] = max(1, min(int(parsed.get("max_results") or 5), 8))
                if parsed.get("days") is not None:
                    plan["days"] = max(1, min(int(parsed["days"]), 365))
        except Exception:
            pass
        try:
            raw_result = await self.mcp_client.call_tool("web_search", plan)
            data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except Exception:
            data = None
        if not isinstance(data, dict) or data.get("success") is not True:
            return ("ما قدرت أوصل لنتائج البحث الخارجي هالمرة 😅" if language == "ar" else "I couldn't retrieve web results right now."), []
        results = [x for x in (data.get("results") or []) if isinstance(x, dict)][:8]
        if not results:
            return ("ما لقيت نتيجة مناسبة لطلبك في البحث الخارجي." if language == "ar" else "I couldn't find relevant web results."), []
        sources = [str(x.get("url") or "").strip() for x in results if str(x.get("url") or "").strip()]
        try:
            response = await asyncio.to_thread(chat, [
                {"role": "system", "content": WEB_RESPONSE_PROMPT},
                {"role": "user", "content": json.dumps({"CURRENT_LANGUAGE": language, "CURRENT_MESSAGE": message, "WEB_RESULTS": results}, ensure_ascii=False)},
            ])
            answer = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "").strip()
        except Exception:
            answer = ""
        if not answer:
            lines = ["هذي أبرز النتائج:" if language == "ar" else "Here are the main results:"]
            for item in results[:5]:
                title = str(item.get("title") or "").strip()
                if title:
                    lines.append(f"- {title}")
            answer = "\n".join(lines)
        return answer, list(dict.fromkeys(sources))

    @staticmethod
    def _smart_cache_ttl(tool_name: str, default_ttl: int) -> int:
        """
        Global enterprise cache policy.

        Every read-only API/tool is cached. Known tools receive a tuned TTL;
        future/unknown read tools automatically receive a safe 60-second TTL.

        Mutation tools are NEVER cached. This protects future actions such as
        create/update/delete/submit/approve/reject/cancel/send.
        """
        tool = str(tool_name or "").strip()
        tool_cf = tool.casefold()

        # Never cache mutations/actions.
        mutation_markers = (
            "create", "update", "delete", "remove", "submit", "approve",
            "reject", "cancel", "send", "post", "put", "patch", "set_",
            "add_", "upload", "save_", "execute_", "request_",
        )
        if any(marker in tool_cf for marker in mutation_markers):
            return 0

        ttl_by_tool = {
            # Meetings / calendar
            "get_meeting_list": 60,
            "get_my_meetings": 60,

            # Attendance / working hours
            "get_today_attendance": 20,
            "get_employee_attendance": 30,
            "get_subemployees_attendance": 30,
            "get_working_hours": 30,

            # Leave / vacation
            "get_leave_balance": 120,
            "get_vacation_history": 180,
            "get_vacations_list": 180,
            "get_intersected_vacations": 120,

            # Employee directory / organization
            "get_employee_profile": 300,
            "get_employee_details": 300,
            "get_manager_details": 300,
            "get_reporting_employees": 300,
            "search_employees": 300,
            "get_employee_image": 600,

            # Policies / content
            "search_policies": 1800,
            "get_policy_details": 1800,
            "get_news_list": 300,
            "get_news_details": 600,
            "get_announcements_list": 300,
            "get_announcement_details": 600,
            "get_mazaya_list": 600,
            "get_mazaya_details": 900,

            # Tarasol / operational transactions
            "get_tarasol_count": 30,
            "get_tarasol_transactions": 30,
        }

        selected = ttl_by_tool.get(tool)
        if selected is not None:
            return selected

        # Global fallback: cache all unknown/future read-only APIs too.
        try:
            configured = int(default_ttl)
        except Exception:
            configured = 0

        return configured if configured > 0 else 60


    @staticmethod
    def _cache_log_scope(arguments: dict[str, Any]) -> str:
        """Small, non-sensitive cache scope for logs."""
        args = arguments if isinstance(arguments, dict) else {}
        parts: list[str] = []

        for key in ("period", "start_date", "end_date", "time_query"):
            value = str(args.get(key) or "").strip()
            if value:
                parts.append(f"{key}={value!r}")

        # Search text is useful for content/directory cache debugging, but keep
        # it short and never log tokens/credentials.
        search = str(args.get("search") or "").strip()
        if search:
            clean = search.replace("\n", " ")[:60]
            parts.append(f"search={clean!r}")

        return " ".join(parts)


    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any], context: EmployeeContext, target_username: str | None = None) -> dict[str, Any]:
        target = str(target_username or context.username or "").strip()
        default_ttl = self.cache.ttl_for(tool_name)
        ttl = self._smart_cache_ttl(tool_name, default_ttl)
        cache_scope = self._cache_log_scope(arguments or {})
        if ttl <= 0:
            print(
                f"[Tool Cache BYPASS] tool={tool_name!r} reason='mutation_or_disabled' "
                f"{cache_scope}".rstrip(),
                file=sys.stderr,
            )
        # Scope cache entries to the authenticated token without storing the
        # token itself.  This is critical in the POC because context.username
        # can be a shared/dev placeholder (for example rcuser) while different
        # users authenticate with different bearer tokens.
        token_fingerprint = hashlib.sha256(
            str(context.access_token or "").encode("utf-8")
        ).hexdigest()[:16]
        requester_scope = f"{context.username}:{token_fingerprint}"
        cache_key = self.cache.key(
            requester=requester_scope,
            target=target,
            tool_name=tool_name,
            arguments=arguments or {},
        )
        if ttl > 0:
            cached = self.cache.get(cache_key)
            if cached is not None:
                print(
                    f"[Tool Cache HIT] source='cache' tool={tool_name!r} target={target!r} "
                    f"ttl={ttl} {cache_scope}".rstrip(),
                    file=sys.stderr,
                )
                return {"success": True, "data": cached, "cached": True}

        try:
            data = await execute_employee_tool(
                tool_name=tool_name,
                arguments=arguments,
                context=context,
                api_client=self.employee_api,
                target_username=target_username,
            )
            if ttl > 0:
                self.cache.set(cache_key, data, ttl)
                print(
                    f"[Tool Cache SET] tool={tool_name!r} target={target!r} "
                    f"ttl={ttl} {cache_scope}".rstrip(),
                    file=sys.stderr,
                )
            return {"success": True, "data": data}
        except EmployeeAuthenticationError:
            return {"success": False, "error_type": "auth", "error": "authentication"}
        except EmployeeAuthorizationError:
            return {"success": False, "error_type": "authorization", "error": "authorization"}
        except EmployeeAPIError as exc:
            print(
                f"[Tool API error] tool={tool_name!r} target={target!r} error={exc}",
                file=sys.stderr,
            )
            return {"success": False, "error_type": "api", "error": str(exc)}
        except Exception as exc:
            print(f"[Tool unexpected error] {tool_name}: {exc}", file=sys.stderr)
            return {"success": False, "error_type": "unexpected", "error": str(exc)}

    @staticmethod
    def _tool_error(
        result: dict[str, Any],
        language: str,
        *,
        intent: str = "",
    ) -> str:
        """Return a short, human error without exposing backend details.

        Keep authentication/authorization explicit, but make ordinary service
        failures specific to the user's active task. The experimental-phase
        sentence is intentionally not repeated on every transient failure.
        """
        kind = str(result.get("error_type") or "")

        if kind == "auth":
            return (
                "يبدو إن جلسة الدخول انتهت. سجّل الدخول مرة ثانية وجربني."
                if language == "ar"
                else "Your session seems to have expired. Please sign in again and try once more."
            )

        if kind == "authorization":
            employee_private_intents = {
                "employee_details",
                "attendance_today",
                "attendance_history",
                "working_hours",
                "leave_balance",
                "leave_history",
                "vacations_list",
                "team_attendance",
                "reporting_employees",
            }

            if intent in employee_private_intents:
                return (
                    "بعض معلومات الموظفين تكون غير متاحة حفاظًا على الخصوصية 😊\n"
                    "لكن أقدر أساعدك بالمعلومات المتاحة عن الموظف."
                    if language == "ar"
                    else
                    "Some employee information isn’t available due to privacy restrictions 😊\n"
                    "But I can still help with the details that are available."
                )

            return (
                "هذي المعلومة مو متاحة لك حسب صلاحيات الوصول الحالية."
                if language == "ar"
                else "This information isn’t available with your current access permissions."
            )

        if language != "ar":
            english = {
                "employee_details": "I couldn't retrieve the employee details right now. Please try again.",
                "manager": "I couldn't retrieve the manager details right now. Please try again.",
                "attendance_today": "I couldn't retrieve today's attendance right now. Please try again.",
                "attendance_history": "I couldn't retrieve the attendance history right now. Please try again.",
                "working_hours": "I couldn't retrieve the working hours right now. Please try again.",
                "meetings": "I couldn't retrieve the meetings right now. Please try again.",
                "meeting_list": "I couldn't retrieve the meetings right now. Please try again.",
                "leave_balance": "I couldn't retrieve the leave balance right now. Please try again.",
                "leave_history": "I couldn't retrieve the leave history right now. Please try again.",
                "tarasol_count": "I couldn't retrieve your Tarasol transactions right now. Please try again.",
                "tarasol_details": "I couldn't retrieve the Tarasol details right now. Please try again.",
            }
            return english.get(intent, "I couldn't retrieve that information right now. Please try again.")

        arabic = {
            "employee_details": "ما قدرت أوصل لمعلومات الموظف حاليًا. جرّبني مرة ثانية.",
            "profile": "ما قدرت أجيب معلوماتك حاليًا. جرّبني مرة ثانية.",
            "manager": "ما قدرت أجيب معلومات المدير حاليًا. جرّبني مرة ثانية.",
            "attendance_today": "ما قدرت أجيب بيانات حضورك اليوم حاليًا. جرّبني مرة ثانية.",
            "attendance_history": "ما قدرت أجيب سجل حضورك حاليًا. جرّبني مرة ثانية.",
            "working_hours": "ما قدرت أجيب ساعات دوامك حاليًا. جرّبني مرة ثانية.",
            "meetings": "ما قدرت أجيب اجتماعاتك حاليًا. جرّبني مرة ثانية.",
            "meeting_list": "ما قدرت أجيب اجتماعاتك حاليًا. جرّبني مرة ثانية.",
            "leave_balance": "ما قدرت أجيب رصيد إجازاتك حاليًا. جرّبني مرة ثانية.",
            "leave_history": "ما قدرت أجيب سجل إجازاتك حاليًا. جرّبني مرة ثانية.",
            "tarasol_count": "ما قدرت أوصل لمعاملات تراسل حاليًا. جرّبني مرة ثانية.",
            "tarasol_details": "ما قدرت أجيب تفاصيل معاملات تراسل حاليًا. جرّبني مرة ثانية.",
        }
        return arabic.get(intent, "صار عندي تعثر بسيط وما قدرت أجيب المعلومة حاليًا. جرّبني مرة ثانية.")

    def _tool_args(self, request: ParsedRequest, language: str) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if request.period != "none":
            args["period"] = request.period
        if request.start_date:
            args["start_date"] = request.start_date
        if request.end_date:
            args["end_date"] = request.end_date
        if request.intent in {"meetings", "employee_availability", "tarasol_details"}:
            args["language"] = language
        if request.intent == "employee_search":
            args["search"] = request.query
        if request.intent == "employee_role_search":
            args["role_query"] = request.role_query
        if request.intent == "policy_search":
            # Generic policy/procedure requests must list available policies, not
            # search literally for filler such as "والاجراءات". Keep a search
            # term only when the user actually named a specific policy/topic.
            policy_search = str(request.query or "").strip()
            normalized_policy_search = re.sub(
                r"[؟?!.،,]+", " ", policy_search.casefold()
            )
            normalized_policy_search = " ".join(normalized_policy_search.split())
            generic_policy_only = re.fullmatch(
                r"(?:(?:و|and|&)?\s*(?:ال)?(?:سياس(?:ه|ة|ات)|اجراء(?:ات)?|إجراء(?:ات)?|policy|policies|procedure|procedures)\s*)+",
                normalized_policy_search,
                re.I,
            )
            if (
                not normalized_policy_search
                or generic_policy_only
                or _is_generic_policy_request(policy_search)
            ):
                policy_search = ""
            args.update({"search": policy_search, "start_index": 0, "batch_size": 10})
        if request.intent == "news":
            # Generic news-list requests must not be sent upstream as literal
            # search text (e.g. "اخر اخبار الهيئة"), otherwise the API may
            # return zero rows while the user simply asked for the latest list.
            news_search = str(request.query or "").strip()
            generic_news = re.sub(r"[؟?!.،,]+", " ", news_search.casefold())
            generic_news = " ".join(generic_news.split())
            if (
                not generic_news
                or re.fullmatch(
                    r"(?:اخر|آخر|احدث|أحدث|ابرز|أبرز|اهم|أهم|latest|recent|top|highlights?)?\s*(?:ال)?(?:اخبار|أخبار|news)(?:\s+(?:ال)?(?:هيئه|هيئة|rcrc))?",
                    generic_news,
                    re.I,
                )
                or generic_news in {
                    "وش اخبار الهيئة", "وش أخبار الهيئة", "وش الاخبار", "وش الأخبار",
                    "اخبار الهيئة", "أخبار الهيئة", "ابرز اخبار الهيئة", "أبرز أخبار الهيئة", "اهم اخبار الهيئة", "أهم أخبار الهيئة", "الاخبار", "الأخبار",
                }
            ):
                news_search = ""
            args.update({"search": news_search, "start_index": 0, "batch_size": 5})
        if request.intent == "announcements":
            # A general request for the latest Tawasul announcements is a LIST,
            # not a literal keyword search. Pass an empty search to the API.
            # Preserve explicitly named topics for genuine filtered searches.
            announcement_search = str(request.query or "").strip()
            normalized_announcement = announcement_search.casefold()
            normalized_announcement = re.sub(r"[\u064b-\u065f\u0670]", "", normalized_announcement)
            normalized_announcement = re.sub(r"[إأآٱ]", "ا", normalized_announcement)
            normalized_announcement = re.sub(r"[؟?!.،,:\-]+", " ", normalized_announcement)
            normalized_announcement = " ".join(normalized_announcement.split())
            generic_announcement = re.fullmatch(
                r"(?:(?:ابي|ابغى|اريد|اعرض|ورني|هات|عطني|اعطني|وش|ايش|show|list|give|me|the|please)\s+)*"
                r"(?:(?:اخر|احدث|ابرز|اهم|جميع|كل|latest|recent|top|all|current|new)\s+)*"
                r"(?:(?:ال)?(?:اعلان|اعلانات|تعميم|تعاميم|announcement|announcements|circular|circulars))"
                r"(?:\s+(?:(?:ال)?(?:هيئه|هيئة|تواصل|tawasul|rcrc)))?",
                normalized_announcement,
                re.I,
            )
            if not normalized_announcement or generic_announcement:
                announcement_search = ""
            args.update({"search": announcement_search, "start_index": 0, "batch_size": 10})
        if request.intent == "mazaya":
            args.update({"search": request.query, "start_index": 0, "batch_size": 10})
        return args

    async def _resolve_detail_item(self, request: ParsedRequest, state: dict[str, Any], context: EmployeeContext, language: str) -> tuple[dict[str, Any] | None, str]:
        # A request for "full details" is locked to the last VERIFIED content item.
        # It never goes back through employee/person resolution.
        if str(request.query or "") == "content:full":
            focus = state.get("content_focus") if isinstance(state.get("content_focus"), dict) else {}
            row = None
            if str(focus.get("detail_intent") or "") == request.intent:
                full_data = focus.get("full_data")
                if isinstance(full_data, dict) and full_data:
                    return full_data, ""
                focused_row = focus.get("row") if isinstance(focus.get("row"), dict) else None
                if focused_row is not None:
                    row = focused_row

            # Fallback to persistent selection stored in list_state.
            if row is None:
                list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
                if str(list_state.get("selected_detail_intent") or "") == request.intent:
                    selected_full = list_state.get("selected_full_data")
                    if isinstance(selected_full, dict) and selected_full:
                        return selected_full, ""
                    selected_row = list_state.get("selected_item")
                    if isinstance(selected_row, dict):
                        row = selected_row
                    else:
                        items = [x for x in (list_state.get("items") or []) if isinstance(x, dict)]
                        selected_index = int(list_state.get("selected_index") or 0)
                        if 1 <= selected_index <= len(items):
                            row = items[selected_index - 1]
        else:
            row = _choose_content_item(request, state)

        # v185: an explicit noun + ordinal is independent from stale list_state.
        # Example: after reading news, "ابي تفاصيل اول ميزه" must open the first
        # Mazaya item, not the first news item. If the matching verified list is
        # not already in state, fetch that list first and then resolve ordinal.
        if row is None and request.ordinal > 0:
            base_intent = DETAIL_BASE_INTENT.get(request.intent)
            base_tools = {
                "news": "get_news_list",
                "announcements": "get_announcements_list",
                "mazaya": "get_mazaya_list",
                "policy_search": "search_policies",
            }
            base_tool = base_tools.get(str(base_intent or ""))
            if base_tool:
                list_args: dict[str, Any] = {"start_index": 0, "batch_size": 10}
                if base_intent in {"news", "announcements", "mazaya", "policy_search"}:
                    list_args["search"] = ""
                list_result = await self._execute_tool(base_tool, list_args, context)
                if list_result.get("success") and isinstance(list_result.get("data"), dict):
                    list_data = list_result["data"]
                    rows = _content_rows(list_data)
                    state["list_state"] = {
                        "intent": base_intent,
                        "items": rows[:50],
                        "query": "",
                    }
                    row = _choose_content_item(request, state)
                    print(
                        f"[Detail List Refresh] base={base_intent!r} ordinal={request.ordinal} rows={len(rows)}",
                        file=sys.stderr,
                    )

        if row is None:
            # Keep the exact content-detail request pending so a reply such as
            # "5" can ONLY select from this verified content list.
            state["pending"] = {
                "kind": "content_detail_selection",
                "query": "",
                "candidates": [],
                "request": request.model_dump(),
                "message": "content_detail_selection",
            }
            return None, ("أكيد 😊 أي عنصر تقصد؟" if language == "ar" else "Which item do you mean?")
        item_id = _content_item_id(row)
        if not item_id:
            state["content_focus"] = {
                "detail_intent": request.intent,
                "base_intent": DETAIL_BASE_INTENT.get(request.intent, ""),
                "item_id": "",
                "ordinal": int(request.ordinal or 0),
                "row": row,
                "full_data": row,
            }
            _ls = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
            _ls["selected_detail_intent"] = request.intent
            _ls["selected_index"] = int(request.ordinal or 0)
            _ls["selected_item"] = row
            _ls["selected_full_data"] = row
            state["list_state"] = _ls
            state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}
            return row, ""
        policy = get_policy(request.intent)
        if policy is None:
            return None, ("ما قدرت أحدد أي تفاصيل تقصد. وضّح لي وش تبغى بالضبط 😊" if language == "ar" else "I couldn't determine the detail type.")
        result = await self._execute_tool(policy.tool_name, {"item_id": item_id}, context)
        if not result.get("success"):
            return None, self._tool_error(result, language)
        data = result.get("data")
        resolved_data = data if isinstance(data, dict) else row
        state["content_focus"] = {
            "detail_intent": request.intent,
            "base_intent": DETAIL_BASE_INTENT.get(request.intent, ""),
            "item_id": item_id,
            "ordinal": int(request.ordinal or 0),
            "row": row,
            "full_data": resolved_data if isinstance(resolved_data, dict) else row,
        }
        _ls = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
        _ls["selected_detail_intent"] = request.intent
        _ls["selected_index"] = int(request.ordinal or 0)
        _ls["selected_item"] = row
        _ls["selected_full_data"] = resolved_data if isinstance(resolved_data, dict) else row
        state["list_state"] = _ls
        state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}
        return resolved_data, ""

    @staticmethod
    def _mazaya_row_needs_details(row: dict[str, Any]) -> bool:
        """Return True when a Mazaya list row is too generic to show safely."""
        def clean(value: Any) -> str:
            return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()

        generic = {
            "خصومات خاصة لمنسوبي الهيئة",
            "عروض خاصة لمنسوبي الهيئة",
            "مرفق",
            "مرفق.",
            "نامل الاطلاع على المرفق",
            "نأمل الاطلاع على المرفق",
            "please see attachment",
            "attached",
            "attachment",
            "special offers for rcrc employees",
            "special discounts for rcrc employees",
        }

        subject = clean(row.get("subject") or row.get("title"))
        details = clean(row.get("details") or row.get("description"))
        combined = f"{subject} {details}".strip()

        # Explicit discount or a concrete non-generic description is already useful.
        if re.search(r"(?<!\d)\d{1,3}\s*%", combined):
            return False
        if details and details.casefold() not in {x.casefold() for x in generic} and len(details) > 8:
            return False
        if subject and subject.casefold() not in {x.casefold() for x in generic} and len(subject) > 8:
            return False
        return True

    async def _enrich_mazaya_rows(
        self,
        data: dict[str, Any],
        context: EmployeeContext,
    ) -> dict[str, Any]:
        """Fetch verified details for generic Mazaya list rows before rendering."""
        rows = []
        for key in ("items", "benefits", "mazaya", "results", "list"):
            value = data.get(key)
            if isinstance(value, list):
                rows = [dict(x) for x in value if isinstance(x, dict)]
                if rows:
                    break

        if not rows:
            return data

        enriched: list[dict[str, Any]] = []
        for row in rows[:10]:
            merged = dict(row)
            if self._mazaya_row_needs_details(row):
                item_id = _content_item_id(row)
                if item_id:
                    result = await self._execute_tool(
                        "get_mazaya_details",
                        {"item_id": item_id},
                        context,
                    )
                    if result.get("success"):
                        detail_data = result.get("data")
                        detail_row: dict[str, Any] = {}
                        if isinstance(detail_data, dict):
                            if isinstance(detail_data.get("item"), dict):
                                detail_row = detail_data["item"]
                            else:
                                detail_row = detail_data
                        # Detail fields are authoritative; preserve list fields only
                        # where details did not provide a value.
                        for key, value in detail_row.items():
                            if value not in (None, "", [], {}):
                                merged[key] = value
            enriched.append(merged)

        out = dict(data)
        out["items"] = enriched
        return out

    def _remember_manager_relation(
        self,
        state: dict[str, Any],
        *,
        subject: dict[str, Any],
        manager: dict[str, Any],
        subject_kind: str,
    ) -> None:
        subject_row = _bounded_employee(subject)
        manager_row = _bounded_employee(manager)
        state["relation"] = {"predicate": "manager", "subject": subject_row, "object": manager_row, "subject_kind": subject_kind}

        # Any successful manager lookup advances the conversational subject to
        # the manager that was just shown. This makes natural follow-ups work:
        #   "مين مدير ماجد؟" -> شاهه
        #   "ابي ايميلها"    -> شاهه's email
        # The previous employee is retained automatically by _remember_entity.
        if _uid(manager_row):
            _remember_entity(state, manager_row, set_focus=True)
        elif _uid(subject_row):
            # Defensive fallback only when the manager response has no identity.
            _remember_entity(state, subject_row, set_focus=True)

    async def _execute_manager_chain(
        self,
        request: ParsedRequest,
        target: ResolvedTarget,
        context: EmployeeContext,
        language: str,
        state: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        username = target.username
        initial_subject = deepcopy(target.employee)
        final: dict[str, Any] | None = None
        for hop in range(request.relation_depth):
            embedded_manager = initial_subject.get("manager_details") if hop == 0 and isinstance(initial_subject, dict) else None
            embedded_row: dict[str, Any] = {}
            if isinstance(embedded_manager, list):
                embedded_manager = next((x for x in embedded_manager if isinstance(x, dict)), None)
            if isinstance(embedded_manager, dict):
                embedded_row = {
                    "username": embedded_manager.get("userid") or embedded_manager.get("username"),
                    "emp_code": embedded_manager.get("userCode") or embedded_manager.get("empCode"),
                    "name_ar": embedded_manager.get("namea") or embedded_manager.get("nameAr"),
                    "name_en": embedded_manager.get("namel") or embedded_manager.get("nameEn"),
                    "job_title_ar": embedded_manager.get("jodesca") or embedded_manager.get("jobTitleAr"),
                    "job_title_en": embedded_manager.get("jodesce") or embedded_manager.get("jobTitleEn"),
                    "department_ar": embedded_manager.get("dpDesca") or embedded_manager.get("departmentAr"),
                    "department_en": embedded_manager.get("dpDescl") or embedded_manager.get("departmentEn"),
                    "email": embedded_manager.get("emailAddress") or embedded_manager.get("email"),
                    "mobile": embedded_manager.get("mobile") or embedded_manager.get("mobileNo"),
                    "extension": embedded_manager.get("ext") or embedded_manager.get("extension"),
                }
                embedded_row = {k:v for k,v in embedded_row.items() if v not in (None, "", "-")}
            if hop == 0 and embedded_row and (_uid(embedded_row) or _name(embedded_row, language)):
                final = embedded_row
            else:
                result = await self._execute_tool("get_manager_details", {}, context, target_username=username)
                if not result.get("success"):
                    return self._tool_error(result, language, intent=request.intent), None
                final = result.get("data") if isinstance(result.get("data"), dict) else {}
            username = _uid(final)
            if not username and hop < request.relation_depth - 1:
                return ("آسف 😅 ما قدرت أتحقق من التسلسل الإداري هالمرة، فما راح أعطيك معلومة غير مؤكدة." if language == "ar" else "I couldn't continue the management chain from the system."), None

        final = _bounded_employee(final or {})
        requested_manager_fields = [f for f in request.fields if f not in {"all", "name"}]
        if final and requested_manager_fields and not _verified_row_satisfies(final, request.fields, language):
            manager_username = _uid(final)
            if manager_username:
                enriched = await self._execute_tool(
                    "get_employee_details", {}, context, target_username=manager_username
                )
                if enriched.get("success") and isinstance(enriched.get("data"), dict):
                    merged = dict(final)
                    for k, v in enriched["data"].items():
                        if v not in (None, "", [], {}):
                            merged[k] = v
                    final = _bounded_employee(merged)

        subject_name = _name(initial_subject, language)
        if request.relation_depth > 1 and target.kind == "self":
            subject_name = "مديرك" if language == "ar" else "your manager"
        answer = render_profile(final, request, language, subject_name=subject_name, self_subject=False, manager_result=True)
        self._remember_manager_relation(state, subject=initial_subject, manager=final, subject_kind=target.kind)
        return answer, final

    async def _execute_reporting_employees(
        self,
        target: ResolvedTarget,
        context: EmployeeContext,
        language: str,
        state: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        """RCRC-FIX-v15: direct reports come from EmployeeAPIClient.get_reporting_employees,
        which now reads the list from employeeFullDetails (the old
        ReportingEmployees endpoint was removed)."""
        self_subject = target.kind == "self" or (
            str(target.username or "").casefold() == str(context.username or "").casefold()
        )
        subject_username = context.username if self_subject else target.username
        subject_name = _name(target.employee, language) if target.employee else ""

        # ReportingEmployees accepts username only. For self requests,
        # resolve the authenticated user's real userid from myProfile first.
        # This keeps the flow multi-user and avoids using dev aliases such as
        # "rcuser" or employee codes.
        if self_subject:
            try:
                profile_data = await self.employee_api.get_my_profile_row(
                    username=context.username,
                    access_token=context.access_token,
                )
            except (EmployeeAuthenticationError, EmployeeAuthorizationError, EmployeeAPIError) as exc:
                print(f"[Reporting Self Profile] error: {exc}", file=sys.stderr)
                profile_data = {}

            if isinstance(profile_data, dict):
                verified_username = str(
                    profile_data.get("userid")
                    or profile_data.get("username")
                    or profile_data.get("userName")
                    or ""
                ).strip()
                if verified_username:
                    subject_username = verified_username
                if not subject_name:
                    subject_name = str(
                        profile_data.get("namea")
                        or profile_data.get("namel")
                        or ""
                    ).strip()

            print(
                f"[Reporting Self Identity] "
                f"context={context.username!r} verified={subject_username!r}",
                file=sys.stderr,
            )

        try:
            result = await self.employee_api.get_reporting_employees(
                username=subject_username,
                access_token=context.access_token,
            )
        except EmployeeAuthenticationError:
            return self._tool_error({"error_type": "auth"}, language), None
        except EmployeeAuthorizationError:
            return self._tool_error({"error_type": "authorization"}, language), None
        except EmployeeAPIError as exc:
            print(f"[Reporting] error subject={subject_username!r}: {exc}", file=sys.stderr)
            return (
                "ما قدرت أوصل لقائمة فريقك حاليًا 😅 وأنا حاليًا في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار 😊"
                if language == "ar"
                else "I couldn't reach your team list right now. Please try again later."
            ), None

        raw_rows = [row for row in (result.get("employees") or []) if isinstance(row, dict)]
        if not subject_name:
            subject_name = str(result.get("subject_name") or "").strip()

        subject_uid = str(subject_username or "").casefold()
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_rows:
            row = _normalize_report_row(raw, language)
            key = str(row.get("username") or row.get("name") or "").casefold()
            if not key or key in seen or key == subject_uid:
                continue
            seen.add(key)
            rows.append(row)

        print(
            f"[Reporting] subject={subject_username!r} rows={len(rows)} "
            f"list_key={result.get('list_key')!r}",
            file=sys.stderr,
        )

        answer = _render_reporting_rows(
            rows, language,
            subject_name=subject_name or ("الموظف" if language == "ar" else "the employee"),
            self_subject=self_subject,
        )

        # Keep the list so "2" / "ابي معلومات عدنان" work on the next turn.
        state["list_state"] = {"intent": "reporting_employees", "items": rows[:50], "query": subject_name}
        for row in rows[:50]:
            if _uid(row):
                _remember_entity(state, row, set_focus=False)
        if not self_subject and target.employee:
            _remember_entity(state, target.employee, set_focus=(target.kind != "relation_object"))
        return answer, {"employees": rows}

    async def _execute_subject_request(
        self,
        request: ParsedRequest,
        target: ResolvedTarget,
        context: EmployeeContext,
        language: str,
        state: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        policy = get_policy(request.intent)
        assert policy is not None
        subject_name = _name(target.employee, language)
        self_subject = _self_subject(target)

        # Security is enforced in backend code BEFORE any person-scoped tool call
        # or resolver-row shortcut. The LLM cannot grant access.
        try:
            access = await self.authorization.authorize(
                intent=request.intent,
                requester_username=context.username,
                target_username=target.username,
                access_token=context.access_token,
            )
        except EmployeeAuthenticationError:
            return self._tool_error({"error_type": "auth"}, language), None
        except EmployeeAuthorizationError:
            return self._tool_error({"error_type": "authorization"}, language), None
        if not access.allowed:
            print(
                f"[Authorization] DENY intent={request.intent!r} requester={context.username!r} "
                f"target={target.username!r} reason={access.reason!r}",
                file=sys.stderr,
            )

            if access.reason == "self_only":
                return (
                    (
                        "هذه معلومة خاصة بالموظف نفسه وما أقدر أعرضها لموظف آخر. "
                        "أقدر أساعدك بمعلومات دليل الموظفين مثل الاسم، المسمى الوظيفي، "
                        "الإدارة، البريد الإلكتروني، الجوال أو التحويلة."
                    )
                    if language == "ar"
                    else (
                        "That is private self-service information and cannot be shown "
                        "for another employee. I can still provide directory information "
                        "such as name, job title, department, email, mobile, or extension."
                    )
                ), None

            return (
                "ما عندك صلاحية للوصول لهذه البيانات."
                if language == "ar"
                else "You don't have permission to access this data."
            ), None
        if self.authorization.requires_person_scope(request.intent):
            print(
                f"[Authorization] ALLOW intent={request.intent!r} requester={context.username!r} "
                f"target={target.username!r} reason={access.reason!r}",
                file=sys.stderr,
            )

        # RCRC-FIX-v3: direct reports come from myProfile / employeeFullDetails.
        if request.intent == "reporting_employees":
            return await self._execute_reporting_employees(target, context, language, state)

        if request.intent == "manager":
            return await self._execute_manager_chain(request, target, context, language, state)

        # Manager follow-up shortcut.
        # After a successful "مين مديري؟", the verified manager object is already
        # stored in relation/focus state and get_manager_details already returns
        # public contact fields (email/mobile/extension) when available.
        # For a natural follow-up such as "ابي ايميله ورقم جواله", use that
        # verified manager object directly instead of forcing a new employee
        # directory lookup. Missing fields are rendered as unavailable rather
        # than turning the whole answer into a directory-service error.
        if request.intent == "employee_details" and target.employee:
            relation_state = state.get("relation") if isinstance(state.get("relation"), dict) else {}
            relation_object = (
                relation_state.get("object")
                if isinstance(relation_state.get("object"), dict)
                else {}
            )
            same_as_manager = bool(
                str(relation_state.get("predicate") or "") == "manager"
                and _uid(relation_object)
                and _uid(target.employee)
                and _uid(relation_object).casefold() == _uid(target.employee).casefold()
            )
            public_manager_fields = {
                "name", "name_en", "job_title", "job_title_en",
                "department", "department_en", "sector", "sector_en",
                "email", "mobile", "extension", "username",
            }
            requested = {str(field) for field in (request.fields or []) if str(field) != "all"}
            if same_as_manager and requested and requested.issubset(public_manager_fields):
                data = _merge_employee(relation_object, target.employee)
                manager_name = _name(data, language) or subject_name
                answer = render_profile(
                    data,
                    request,
                    language,
                    subject_name=manager_name,
                    self_subject=False,
                )
                _remember_entity(state, data, set_focus=True)
                _set_employee_more_details_pending(state, request, self_subject=False)
                print(
                    f"[Manager Follow-up] used verified relation object "
                    f"username={_uid(data)!r} fields={request.fields!r}",
                    file=sys.stderr,
                )
                return answer, data

        # A numbered employee choice requires a fresh full-details API call.
        selected_from_department = (
            request.intent == "employee_details"
            and request.query == "employee:selected_from_department_list"
        )
        if selected_from_department and target.username:
            selected = state.get("selected_list_employee") or {}
            selected_code = str(selected.get("emp_code") or "").strip()
            if selected_code:
                self.employee_api.remember_employee_code(target.username, selected_code)

        # If a verified resolver row already contains every requested profile field,
        # do not make a second API call just to re-fetch the same facts.
        if request.intent == "employee_details" and not selected_from_department and target.employee and _verified_row_satisfies(target.employee, request.fields, language):
            data = target.employee
            answer = render_profile(data, request, language, subject_name=subject_name, self_subject=False)
            _remember_entity(state, data, set_focus=True)
            _set_employee_more_details_pending(state, request, self_subject=False)
            return answer, data

        # Meetings are date-range queries in the deployed RCRC API.
        # Always use MeetingList/myMeetings so startDate/endDate return every
        # meeting in the requested period, including historical meetings.
        execution_tool = policy.tool_name
        if request.intent in {"meetings", "meeting_list"} and self_subject:
            execution_tool = "get_meeting_list"
            print(
                "[Meeting API Route] "
                f"intent={request.intent!r} "
                f"tool='get_meeting_list' "
                f"start={request.start_date!r} "
                f"end={request.end_date!r}",
                file=sys.stderr,
            )

        result = await self._execute_tool(
            execution_tool,
            self._tool_args(request, language),
            context,
            target_username=target.username,
        )
        if not result.get("success"):
            # An HTTP 404 from this environment is a real request failure;
            # never treat it as successful verification of no overlap.
            if request.intent == "leave_overlap" and "404" in str(result.get("error") or ""):
                return (
                    "آسف 😅 ما قدرت أتحقق من تداخل الإجازات هالمرة."
                    if language == "ar"
                    else "I can't check overlapping leave right now because the overlap service is unavailable."
                ), None
            # Verified resolver rows are a safe fallback only when they contain
            # every requested field. Never synthesize missing values.
            if request.intent == "employee_details" and target.employee and _verified_row_satisfies(target.employee, request.fields, language):
                data = target.employee
            else:
                return self._tool_error(result, language, intent=request.intent), None
        else:
            data = result.get("data")

        data = data if isinstance(data, dict) else {}

        if request.intent == "leave_overlap":
            return render_leave_overlap(
                data, language,
                start_date=request.start_date,
                end_date=request.end_date,
            ), data

        if request.intent == "employee_image":
            image_data_url = str(data.get("image_data_url") or "").strip()
            if image_data_url.startswith("data:image/"):
                # Angular NavigationService already recognizes this marker and
                # renders a safe <img> card instead of dumping Base64 text.
                return f"[EMPLOYEE_IMAGE_DATA]{image_data_url}", data
            return (
                "الصورة الوظيفية ما ظهرت لي حاليًا."
                if language == "ar"
                else "Your employee profile image is not available right now."
            ), data

        # Target integrity guard. The resolver-selected username is authoritative.
        # If EmployeeDetails returns a different/generic employee identity, do not
        # use that payload for this person. We will fall back to the already
        # verified SearchEmployee row and, when needed, enrich it again using the
        # same verified username. This prevents both wrong answers and focus drift.
        if request.intent == "employee_details" and target.username:
            expected_uid = str(target.username or "").strip().casefold()
            returned_uid = _uid(data).strip().casefold()
            if returned_uid and expected_uid and returned_uid != expected_uid:
                print(
                    f"[Target Integrity Guard] ignored mismatched EmployeeDetails result "
                    f"expected={target.username!r} returned={_uid(data)!r}",
                    file=sys.stderr,
                )
                data = {}

        if target.employee:
            merged_subject = _merge_employee(target.employee, data if request.intent == "employee_details" else {})
        else:
            merged_subject = {}

        if request.intent in {"profile", "employee_details"}:
            render_data = _merge_employee(target.employee, data)

            # EmployeeDetails is not consistent across deployed API versions:
            # some rows omit public directory fields such as email/mobile while
            # SearchEmployee already returned them for the SAME verified userid.
            # If a requested public field is still missing, re-query SearchEmployee
            # by the verified employee name and merge ONLY the row whose username
            # matches the already-resolved target. Never infer contact values.
            if (
                request.intent == "employee_details"
                and request.fields
                and not _verified_row_satisfies(render_data, request.fields, language)
                and subject_name
                and target.username
            ):
                try:
                    search_payload = await self.employee_api.search_employees(
                        access_token=context.access_token,
                        search=subject_name,
                    )
                    search_rows = (
                        search_payload.get("employees")
                        if isinstance(search_payload, dict)
                        else []
                    )
                    verified_username = str(target.username or "").strip().casefold()
                    for search_row in search_rows or []:
                        if not isinstance(search_row, dict):
                            continue
                        row_username = str(
                            search_row.get("username")
                            or search_row.get("userid")
                            or ""
                        ).strip().casefold()
                        if row_username and row_username == verified_username:
                            render_data = _merge_employee(render_data, search_row)
                            print(
                                f"[Directory Contact Enrichment] username={target.username!r} "
                                f"fields={request.fields!r}",
                                file=sys.stderr,
                            )
                            break
                except (EmployeeAuthenticationError, EmployeeAuthorizationError):
                    raise
                except EmployeeAPIError as exc:
                    print(
                        f"[Directory Contact Enrichment Error] username={target.username!r}: {exc}",
                        file=sys.stderr,
                    )

            answer = render_profile(
                render_data,
                request,
                language,
                subject_name=subject_name,
                self_subject=self_subject,
            )
            if not self_subject:
                _remember_entity(state, render_data, set_focus=(target.kind != "relation_object"))
                _set_employee_more_details_pending(state, request, self_subject=False)
            return answer, render_data

        if request.intent == "leave_balance":
            # v239: hypothetical leave follow-ups must calculate from the exact
            # balance that was just shown to the signed-in user.
            #
            # Example:
            #   "كم رصيد إجازاتي؟" -> 3 days
            #   "لو خذت يوم كم يبقى لي" -> 2 days
            #
            # Do not accidentally switch to annual entitlement (e.g. 30 days)
            # and do not mutate the real balance: this is display-only context.
            render_data = data

            leave_ctx = state.get("leave_context") if isinstance(state.get("leave_context"), dict) else {}
            saved_balance = leave_ctx.get("displayed_balance")

            if (
                request.operation == "calculate"
                and self_subject
                and saved_balance is not None
            ):
                render_data = {"balance": saved_balance}
                print(
                    f"[Leave Followup Context] using_displayed_balance={saved_balance!r} "
                    f"amount={request.amount!r}",
                    file=sys.stderr,
                )

            answer = render_leave_balance(
                render_data,
                language,
                subject_name=subject_name,
                self_subject=self_subject,
                amount=request.amount if request.operation == "calculate" else None,
            )

            # Save the balance only after a normal lookup. A hypothetical
            # calculation never changes this stored real/current balance.
            if (
                request.operation != "calculate"
                and self_subject
                and isinstance(data, dict)
            ):
                displayed_balance = data.get("balance")
                if displayed_balance is not None and str(displayed_balance).strip() != "":
                    state["leave_context"] = {
                        "displayed_balance": displayed_balance,
                        "source": "leave_balance_lookup",
                    }
                    print(
                        f"[Leave Balance Context Saved] balance={displayed_balance!r}",
                        file=sys.stderr,
                    )

            if str(request.query or "") == "leave:compare_manager_private":
                if language == "ar":
                    answer = answer.rstrip(".") + ". ما أقدر أقارنه برصيد مديرك لأن رصيد إجازاته من البيانات الخاصة به."
                else:
                    answer = answer.rstrip(".") + ". I can't compare it with your manager's balance because their leave balance is private self-service data."
        elif request.intent == "employee_availability":
            availability_name = (
                ("أنت" if language == "ar" else "you")
                if self_subject
                else (_first_name(subject_name) or subject_name or ("الموظف" if language == "ar" else "the employee"))
            )
            answer = render_availability(
                data,
                language,
                subject_name=availability_name,
                period=request.period,
                start_date=request.start_date,
                end_date=request.end_date,
                time_query=request.time_query,
                mode=str(request.query or "busy"),
            )
            duration_hint = int(state.pop("availability_duration_minutes", 0) or 0)
            if duration_hint and language == "ar":
                hours = duration_hint / 60
                if duration_hint == 30:
                    duration_text = "نص ساعة"
                elif duration_hint == 60:
                    duration_text = "ساعة"
                elif duration_hint == 90:
                    duration_text = "ساعة ونص"
                elif duration_hint == 120:
                    duration_text = "ساعتين"
                elif duration_hint % 60 == 0:
                    duration_text = f"{duration_hint // 60} ساعات"
                else:
                    duration_text = f"{duration_hint} دقيقة"
                answer = f"تمام، مدة الاجتماع {duration_text}. {answer}"
            # Calendar context is separate from generic person focus. A later
            # "بكرة" / "طيب الساعة 3؟" continues with this verified employee,
            # even if another directory lookup changed conversational focus.
            calendar_employee = target.employee if isinstance(target.employee, dict) else {}
            if calendar_employee and _uid(calendar_employee):
                state["meeting_context"] = {
                    "employee": _bounded_employee(calendar_employee),
                    "period": request.period,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "time_query": request.time_query,
                    "mode": str(request.query or "busy"),
                }
        elif request.intent in {"meetings", "meeting_list"}:
            # Historical self-meetings:
            # Calendar/myMeetingBasicInfo may return an empty list for past dates
            # even when MeetingList/myMeetings contains the verified meetings.
            # For yesterday / explicit past dates, prefer the rich date-range
            # MeetingList endpoint as the actual source, not only as link enrichment.
            if self_subject and request.intent == "meetings":
                try:
                    today_local = datetime.now(
                        timezone(timedelta(hours=3))
                    ).date()
                    requested_end = (
                        datetime.fromisoformat(str(request.end_date)).date()
                        if str(request.end_date or "").strip()
                        else None
                    )
                    historical_request = bool(
                        requested_end is not None and requested_end < today_local
                    )

                    current_rows = []
                    if isinstance(data, dict):
                        if isinstance(data.get("meetings"), list):
                            current_rows = data.get("meetings") or []
                        elif isinstance(data.get("records"), list):
                            current_rows = data.get("records") or []

                    if (historical_request or not current_rows) and execution_tool != "get_meeting_list":
                        historical_result = await self._execute_tool(
                            "get_meeting_list",
                            {
                                "start_date": request.start_date,
                                "end_date": request.end_date,
                            },
                            context,
                            target_username=target.username,
                        )

                        if historical_result.get("success"):
                            historical_data = historical_result.get("data")
                            historical_rows = (
                                historical_data.get("records")
                                if isinstance(historical_data, dict)
                                and isinstance(historical_data.get("records"), list)
                                else []
                            )

                            if historical_rows:
                                data = {
                                    "start_date": request.start_date,
                                    "end_date": request.end_date,
                                    "records": historical_rows,
                                }
                                print(
                                    "[Historical Meetings] "
                                    f"source='MeetingList/myMeetings' "
                                    f"period={request.period!r} "
                                    f"start={request.start_date!r} "
                                    f"end={request.end_date!r} "
                                    f"rows={len(historical_rows)}",
                                    file=sys.stderr,
                                )
                            elif historical_request:
                                print(
                                    "[Historical Meetings] "
                                    f"source='MeetingList/myMeetings' "
                                    f"period={request.period!r} rows=0",
                                    file=sys.stderr,
                                )
                except Exception as exc:
                    # Preserve the existing basic-meeting result if the rich
                    # historical lookup itself fails.
                    print(
                        f"[Historical Meetings Error] {exc}",
                        file=sys.stderr,
                    )

            # myMeetingBasicInfo is reliable for title/time but some deployments
            # omit the online join URL. Enrich the same verified meetings from
            # the richer MeetingList endpoint before rendering.
            if self_subject and request.intent == "meetings" and execution_tool != "get_meeting_list":
                basic_rows = data.get("meetings") if isinstance(data.get("meetings"), list) else []
                has_any_link = any(
                    _first_value(
                        row, "teams_link", "meeting_link", "meetingLink",
                        "teamsLink", "TeamsLink", "joinUrl", "JoinUrl",
                        "meetingUrl", "MeetingUrl", "onlineMeetingUrl",
                        "OnlineMeetingUrl", "webLink", "WebLink", "url", "link",
                    )
                    for row in basic_rows if isinstance(row, dict)
                )
                if basic_rows and not has_any_link:
                    try:
                        rich_result = await self._execute_tool(
                            "get_meeting_list",
                            {
                                "start_date": request.start_date,
                                "end_date": request.end_date,
                            },
                            context,
                            target_username=target.username,
                        )
                        rich_data = rich_result.get("data") if rich_result.get("success") else {}
                        rich_rows = (
                            rich_data.get("records")
                            if isinstance(rich_data, dict) and isinstance(rich_data.get("records"), list)
                            else []
                        )

                        def _meeting_subject_key(row: dict[str, Any]) -> str:
                            return " ".join(_first_value(
                                row, "subject", "Subject", "title", "Title",
                                "meetingTitle", "MeetingTitle", "meetingName",
                                "MeetingName", "subjectAr", "subjectEn",
                            ).casefold().split())

                        def _meeting_start_key(row: dict[str, Any]) -> str:
                            raw = _first_value(row, "start_time", "startTime", "StartTime", "start")
                            match = re.search(r"(?:T|\s)(\d{1,2}:\d{2})", raw)
                            return match.group(1) if match else raw[-5:] if re.fullmatch(r"\d{1,2}:\d{2}", raw) else ""

                        for base in basic_rows:
                            if not isinstance(base, dict):
                                continue
                            b_subject = _meeting_subject_key(base)
                            b_start = _meeting_start_key(base)
                            candidates = [
                                row for row in rich_rows
                                if isinstance(row, dict) and _meeting_subject_key(row) == b_subject
                            ]
                            if b_start:
                                exact = [row for row in candidates if _meeting_start_key(row) == b_start]
                                if exact:
                                    candidates = exact
                            if len(candidates) == 1:
                                rich_row = candidates[0]
                                for key, value in rich_row.items():
                                    if value not in (None, "", [], {}) and base.get(key) in (None, "", [], {}):
                                        base[key] = value
                                link = _first_value(
                                    rich_row, "teams_link", "meeting_link", "meetingLink",
                                    "teamsLink", "TeamsLink", "joinUrl", "JoinUrl",
                                    "meetingUrl", "MeetingUrl", "onlineMeetingUrl",
                                    "OnlineMeetingUrl", "webLink", "WebLink", "url", "link",
                                )
                                if link:
                                    base["teams_link"] = link
                                    print(
                                        f"[Meeting Link Enrichment] subject={b_subject!r} link=True",
                                        file=sys.stderr,
                                    )
                    except Exception as exc:
                        print(f"[Meeting Link Enrichment Error] {exc}", file=sys.stderr)

            answer = render_meetings(
                data, language, subject_name=subject_name, self_subject=self_subject,
                period=request.period, start_date=request.start_date, end_date=request.end_date,
                time_query=request.time_query,
                view_mode=str(request.query or "meetings:primary"),
            )
            # An explicit/self meeting request becomes the current calendar
            # context and must replace any older manager/employee availability
            # context. This prevents "بكره؟" from jumping back to a manager.
            if self_subject:
                state["meeting_context"] = {
                    "subject_kind": "self",
                    "employee": {},
                    "period": request.period,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "time_query": request.time_query,
                    "mode": "self_meetings",
                }
                state["task_flow"] = {}

                # Preserve the natural meaning of a short "اي" after the answer.
                # If the current primary meeting response hid Tawasul items, "اي"
                # means "show the Tawasul events". Morning-window continuation keeps
                # its existing priority when no Tawasul offer is present.
                rows_for_offer = [
                    row for row in _rows_from_data(data)
                    if isinstance(row, dict)
                ]
                tawasul_rows = []
                for row in rows_for_offer:
                    organizer_value = str(
                        row.get("organizerName")
                        or row.get("organizer_name")
                        or row.get("OrganizerName")
                        or row.get("organizer")
                        or ""
                    ).strip().casefold()
                    if organizer_value == "tawasul" or organizer_value.startswith("tawasul "):
                        tawasul_rows.append(row)
                is_tawasul_view = str(request.query or "").strip() == "meetings:tawasul"

                if tawasul_rows and not is_tawasul_view:
                    follow_req = request.model_copy(deep=True)
                    follow_req.query = "meetings:tawasul"
                    follow_req.clarification_reason = ""
                    state["pending"] = {
                        "kind": "tawasul_events_offer",
                        "query": "",
                        "candidates": [],
                        "request": follow_req.model_dump(),
                        "message": "after_primary_meetings",
                    }
                    print(
                        f"[Pending Saved] kind='tawasul_events_offer' "
                        f"count={len(tawasul_rows)} period={request.period!r} "
                        f"start={request.start_date!r} end={request.end_date!r}",
                        file=sys.stderr,
                    )
                elif str(request.time_query or "").strip() == "07:00-12:00":
                    follow_req = request.model_copy(deep=True)
                    follow_req.time_query = "12:00-17:00"
                    follow_req.clarification_reason = ""
                    state["pending"] = {
                        "kind": "meeting_followup_window",
                        "query": "",
                        "candidates": [],
                        "request": follow_req.model_dump(),
                        "message": "after_meeting_result",
                    }
                else:
                    _clear_pending(state)
        elif request.intent == "tarasol_count":
            answer = render_tarasol_count(
                data,
                language,
                subject_name=subject_name,
                self_subject=self_subject,
            )

            # If the signed-in user has transactions, persist the exact action
            # offered by the renderer so a short "اي" can execute it.
            if self_subject:
                count_value = data.get("count") if isinstance(data, dict) else None
                try:
                    tarasol_count_value = int(float(str(count_value)))
                except Exception:
                    tarasol_count_value = 0

                if tarasol_count_value > 0:
                    follow_req = request.model_copy(deep=True)
                    follow_req.intent = "tarasol_details"
                    follow_req.operation = "details"
                    follow_req.targets = [TargetRef(kind="self", name="")]
                    follow_req.fields = ["all"]
                    follow_req.query = "tarasol:self_details"
                    follow_req.clarification_reason = ""
                    follow_req.confidence = 1.0
                    state["pending"] = {
                        "kind": "tarasol_details_offer",
                        "query": "",
                        "candidates": [],
                        "request": follow_req.model_dump(),
                        "message": "after_tarasol_count",
                    }
                    print(
                        f"[Pending Saved] kind='tarasol_details_offer' "
                        f"count={tarasol_count_value}",
                        file=sys.stderr,
                    )
                else:
                    _clear_pending(state)

        elif request.intent == "tarasol_details":
            localized_tarasol = await localize_tarasol_data(data, language)
            answer = render_tarasol(localized_tarasol, language, subject_name=subject_name, self_subject=self_subject)
        elif request.intent == "reporting_employees":
            answer = render_reporting_employees(data, language, subject_name=subject_name, self_subject=self_subject)
            report_rows = [_bounded_employee(row) for row in _rows_from_data(data)]
            report_rows = [row for row in report_rows if _uid(row)]
            state["list_state"] = {
                "intent": "reporting_employees",
                "items": report_rows[:25],
                "query": subject_name,
            }
            for row in report_rows[:25]:
                _remember_entity(state, row, set_focus=False)
        elif request.intent in {"attendance_today", "attendance_history", "working_hours", "team_attendance"}:
            answer = render_attendance(
                data,
                language,
                subject_name=subject_name,
                self_subject=self_subject,
                period=request.period,
                intent=request.intent,
                mode=str(request.query or "attendance:summary"),
            )
            if self_subject:
                state["attendance_context"] = {
                    "period": request.period,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "mode": str(request.query or "attendance:summary"),
                }
        elif request.intent == "leave_history":
            # leave_history is target-scoped (normally self), so it reaches this
            # subject execution path. Render the verified VacationData records
            # here instead of falling through to the generic one-field list.
            answer = render_leave_history(data, language, limit=20)
        else:
            answer = render_generic(data, language)

        if not self_subject and target.employee:
            _remember_entity(state, target.employee, set_focus=(target.kind != "relation_object"))
        elif target.kind == "manager_of_self" and target.employee:
            _remember_entity(state, target.employee, set_focus=True)
        return answer, data

    @staticmethod
    def _render_employee_count(data: dict[str, Any], language: str) -> str:
        count = int(data.get("count") or 0)
        if language == "ar":
            return f"إجمالي عدد الموظفين في دليل الموظفين هو {count} موظف."
        return f"The employee directory contains {count} employees."

    @staticmethod
    def _render_employee_directory_list(
        rows: list[dict[str, Any]],
        language: str,
        *,
        limit: int | None = 50,
    ) -> str:
        clean = [directory_employee(row) for row in rows if isinstance(row, dict)]
        clean = [row for row in clean if _uid(row)]
        total = len(clean)
        show_all = limit is None or int(limit) <= 0 or int(limit) >= total
        shown = clean if show_all else clean[:max(1, int(limit))]
        if language == "ar":
            header = (
                f"جميع موظفي الهيئة ({total} موظف):"
                if show_all
                else f"هذي أول {len(shown)} من أصل {total} موظف:"
            )
        else:
            header = (
                f"All organization employees ({total}):"
                if show_all
                else f"Here are the first {len(shown)} of {total} employees:"
            )
        lines = [header]
        for idx, row in enumerate(shown, start=1):
            name = _name(row, language) or _uid(row)
            title_value = (
                (row.get("job_title_ar") or row.get("job_title") or row.get("job_title_en"))
                if language == "ar"
                else (row.get("job_title_en") or row.get("job_title") or row.get("job_title_ar"))
            )
            title = str(title_value or "").strip()
            if title.casefold() in {"none", "null", "n/a", "na", "-"}:
                title = ""
            line = f"{idx}. {name}"
            if title and _norm_role_text(title) != _norm_role_text(name):
                line += f" — {title}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _render_general_department_counts(
        data: dict[str, Any],
        language: str,
        *,
        include_members: bool = False,
    ) -> str:
        rows = data.get("departments") if isinstance(data, dict) else []
        rows = [row for row in (rows or []) if isinstance(row, dict)]
        if not rows:
            return (
                "ما ظهرت لي بيانات كافية عن الإدارات العامة حاليًا."
                if language == "ar"
                else "I couldn't find verified general-department data in the employee directory."
            )

        total = int(data.get("total_unique_employees") or 0)
        if language == "ar":
            lines = [
                f"إجمالي عدد الموظفين في الدليل: {total}",
                "",
                "الموظفون حسب الإدارة العامة:",
            ]
        else:
            lines = [
                f"Total employees in the directory: {total}",
                "",
                "Employees by general department:",
            ]

        for row in rows:
            department_name = str(row.get("general_department") or "").strip()
            count = int(row.get("count") or 0)
            if not department_name:
                continue

            lines.append(f"- {department_name}: {count}")

            if not include_members:
                continue

            employees = [
                employee
                for employee in (row.get("employees") or [])
                if isinstance(employee, dict)
            ]
            for index, employee in enumerate(employees, start=1):
                name = directory_employee_name(employee) or str(employee.get("username") or "").strip()
                title = str(employee.get("job_title") or "").strip()
                employee_line = f"  {index}. {name}"
                if title:
                    employee_line += f" — {title}"
                lines.append(employee_line)

        unmapped = int(data.get("unmapped_employees") or 0)
        if unmapped:
            if language == "ar":
                lines.extend(["", f"موظفون بدون إدارة عامة محددة في الدليل: {unmapped}"])
            else:
                lines.extend(["", f"Employees without a verified general department: {unmapped}"])

        return "\n".join(lines)

    @staticmethod
    def _english_person_name(value: Any) -> str:
        raw = re.sub(r"\s+", " ", str(value or "").strip())
        if not raw:
            return ""
        acronyms = {"RCRC", "IT", "HR", "PMO", "ICT", "AI", "OCI", "API", "CAB"}
        parts = []
        for token in raw.split():
            if token.upper() in acronyms:
                parts.append(token.upper())
            elif token.isupper() or token.islower():
                parts.append(token.capitalize())
            else:
                parts.append(token)
        return " ".join(parts)

    @staticmethod
    def _english_job_title(value: Any) -> str:
        raw = re.sub(r"\s+", " ", str(value or "").strip())
        if not raw or re.search(r"[\u0600-\u06ff]", raw):
            return ""
        small = {"and", "of", "for", "the", "in", "to", "with"}
        acronyms = {"IT", "HR", "PMO", "ICT", "AI", "OCI", "API", "RCRC", "CAB"}
        words = []
        for idx, token in enumerate(raw.split()):
            upper = token.upper()
            lower = token.lower()
            if upper in acronyms:
                words.append(upper)
            elif lower in small and idx > 0:
                words.append(lower)
            else:
                words.append(lower.capitalize())
        return " ".join(words)

    @staticmethod
    def _english_department_name(data: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        for key in ("general_department_en", "department_en"):
            value = str(data.get(key) or "").strip()
            if value and not re.search(r"[\u0600-\u06ff]", value):
                return value

        for row in rows:
            for key in ("general_department_en", "department_en"):
                value = str(row.get(key) or "").strip()
                if value and not re.search(r"[\u0600-\u06ff]", value):
                    return value

        ar_name = str(data.get("general_department") or "").strip()
        known = {
            "الإدارة العامة للتقنية والرقمنة": "General Department of Digital & Technology",
            "الادارة العامة للتقنية والرقمنة": "General Department of Digital & Technology",
        }
        return known.get(ar_name, "the selected department")

    @staticmethod
    def _compact_directory_name(value: Any) -> str:
        raw = re.sub(r"\s+", " ", str(value or "").strip())
        if not raw:
            return ""
        parts = raw.split(" ")
        if len(parts) <= 2:
            return raw
        if parts[0] == "عبد" and len(parts) >= 4:
            return f"{parts[0]} {parts[1]} {parts[-1]}"
        return f"{parts[0]} {parts[-1]}"

    @classmethod
    def _compact_directory_names(cls, names: list[str]) -> list[str]:
        compact = [cls._compact_directory_name(name) for name in names]
        counts: dict[str, int] = {}
        for value in compact:
            key = value.casefold()
            if key:
                counts[key] = counts.get(key, 0) + 1
        return [
            full if short and counts.get(short.casefold(), 0) > 1 else (short or full)
            for full, short in zip(names, compact)
        ]

    @classmethod
    def _render_general_department_members(cls, data: dict[str, Any], language: str) -> str:
        if not data.get("matched"):
            return (
                "ما لقيت إدارة مطابقة بشكل واضح 🔎 جرّب تكتب اسمها بطريقة ثانية."
                if language == "ar"
                else "I couldn't find a verified matching department."
            )

        name = str(data.get("general_department") or "").strip()
        rows = [row for row in (data.get("employees") or []) if isinstance(row, dict)]
        count = len(rows)

        if language == "en":
            name = cls._english_department_name(data, rows)

        raw_names: list[str] = []
        for row in rows:
            if language == "ar":
                raw_names.append(str(
                    row.get("name_ar") or row.get("name")
                    or directory_employee_name(row) or row.get("username") or ""
                ).strip())
            else:
                raw_names.append(cls._english_person_name(
                    row.get("name_en") or row.get("name") or row.get("username") or ""
                ))

        display_names = cls._compact_directory_names(raw_names)

        if language == "ar":
            if count == 0:
                return f"راجعت دليل الموظفين، وما ظهر لي موظفين تحت **{name}** حاليًا."
            lines = [f"أكيد 😊 موظفو {name} ({count}):"]
        else:
            lines = [f"Sure — I found {count} employees in {name}:"]

        for idx, (row, emp_name) in enumerate(zip(rows, display_names), start=1):
            title = str(
                row.get("job_title_ar") or row.get("job_title") or ""
                if language == "ar"
                else row.get("job_title_en") or row.get("job_title") or ""
            ).strip()
            if language == "en":
                title = cls._english_job_title(title)
            lines.append(f"{idx}. {emp_name}" + (f" — {title}" if title else ""))

        if language == "ar":
            lines.extend(["", "إذا تبي بيانات موظف معيّن، اختار رقمه وأنا أكمل لك."])
        return "\n".join(lines)

    async def _execute_directory_query(
        self,
        request: ParsedRequest,
        context: EmployeeContext,
        language: str,
        state: dict[str, Any],
    ) -> str:
        if request.intent == "employee_list":
            employees = await self.directory.all_employees(access_token=context.access_token)
            rows = [directory_employee(row) for row in employees if isinstance(row, dict)]
            rows = [row for row in rows if _uid(row)]

            state["list_state"] = {
                "intent": "employee_list",
                "items": rows,
                "query": request.query,
            }
            state["directory_context"] = {
                "type": "employee_list",
                "count": len(rows),
            }

            show_all = "all" in set(request.fields or [])
            print(
                f"[Employee Directory List] source=search_employee total={len(rows)} "
                f"show_all={show_all}",
                file=sys.stderr,
            )
            return self._render_employee_directory_list(
                rows,
                language,
                limit=None if show_all else 50,
            )

        if request.intent == "employee_count":
            gender_mode = str(request.query or "").strip().lower()
            if gender_mode.startswith("gender:"):
                data = await self.directory.employee_gender_counts(access_token=context.access_token)
                if not bool(data.get("has_verified_gender")):
                    return (
                        "ما أقدر أعطيك عدد دقيق حاليًا لأن البيانات المتاحة ما تكفي للحساب."
                        if language == "ar"
                        else "Verified gender data is not available in the employee directory right now, so I can't count female or male employees without guessing."
                    )
                female = int(data.get("female") or 0)
                male = int(data.get("male") or 0)
                unknown = int(data.get("unknown") or 0)
                state["directory_context"] = {"type": "employee_gender_count", "female": female, "male": male, "unknown": unknown}
                if gender_mode == "gender:female":
                    return f"عدد الموظفات الإناث هو {female}." if language == "ar" else f"There are {female} female employees."
                if gender_mode == "gender:male":
                    return f"عدد الموظفين الذكور هو {male}." if language == "ar" else f"There are {male} male employees."
                if language == "ar":
                    suffix = f"، و{unknown} بدون بيانات جنس محددة في الدليل" if unknown else ""
                    return f"عدد الإناث: {female}، وعدد الذكور: {male}{suffix}."
                suffix = f", with {unknown} employees missing verified gender data" if unknown else ""
                return f"Female employees: {female}. Male employees: {male}{suffix}."

            data = await self.directory.employee_count(access_token=context.access_token)
            count = int(data.get("count") or 0)
            state["directory_context"] = {
                "type": "employee_count",
                "count": count,
            }
            print(f"[Employee Count] total={count}", file=sys.stderr)
            return self._render_employee_count(data, language)

        if request.intent == "general_department_counts":
            mode = str(request.query or "").strip().lower()
            if mode in {"org_units:list", "org_units:count"}:
                data = await self.directory.organizational_unit_summary(access_token=context.access_token)
                units = [x for x in (data.get("units") or []) if isinstance(x, dict)]
                total = len(units)
                state["directory_context"] = {"type": "org_units", "count": total}
                state["list_state"] = {"intent": "general_department_counts", "items": units[:200], "query": mode}
                if mode == "org_units:count":
                    return (
                        f"عدد الإدارات والوحدات التنظيمية الظاهرة في دليل الموظفين هو {total}."
                        if language == "ar"
                        else f"The employee directory currently shows {total} organizational departments/units."
                    )
                if language == "ar":
                    lines = [f"الإدارات والوحدات التنظيمية ({total}):"]
                    for i, row in enumerate(units, 1):
                        lines.append(f"{i}. {row.get('name')} — {int(row.get('count') or 0)} موظف")
                    return "\n".join(lines)
                lines = [f"Organizational departments/units ({total}):"]
                for i, row in enumerate(units, 1):
                    lines.append(f"{i}. {row.get('name')} — {int(row.get('count') or 0)} employees")
                return "\n".join(lines)

            include_members = bool(set(request.fields or []) & {"name", "all"})

            if request.department_query.strip():
                data = await self.directory.general_department_members(
                    access_token=context.access_token,
                    department_query=request.department_query,
                )
                if not data.get("matched"):
                    state.pop("directory_context", None)
                    return self._render_general_department_members(data, language)

                members = [
                    directory_employee(row)
                    for row in data.get("employees", [])
                    if isinstance(row, dict)
                ]
                state["directory_context"] = {
                    "type": "department",
                    "general_department_key": data.get("department_key") or data.get("general_department_key"),
                    "general_department": data.get("department") or data.get("general_department"),
                    "count": len(members),
                }
                print(
                    f"[General Department Query] department={str(data.get('general_department') or '')!r} "
                    f"count={len(members)} include_members={include_members}",
                    file=sys.stderr,
                )

                if include_members:
                    return self._render_general_department_members(data, language)

                name = str(data.get("general_department") or request.department_query).strip()
                count = len(members)

                if str(request.query or "") == "department:summary":
                    follow_req = request.model_copy(deep=True)
                    follow_req.intent = "general_department_members"
                    follow_req.operation = "list"
                    follow_req.fields = ["name", "job_title"]
                    follow_req.query = "department:members"
                    follow_req.clarification_reason = ""
                    follow_req.confidence = 1.0
                    state["pending"] = {
                        "kind": "department_members_offer",
                        "request": follow_req.model_dump(),
                        "query": "",
                        "candidates": [],
                        "message": "after_department_summary",
                    }
                    print(
                        f"[Pending Saved] kind='department_members_offer' "
                        f"department={name!r} count={count}",
                        file=sys.stderr,
                    )
                    return (
                        f"في {name} فيه {count} موظف 😊\n"
                        "إذا تبي، أعرض لك القائمة."
                        if language == "ar"
                        else f"There are {count} employees in {name} ⭐\nSay 'show me' if you'd like the list 👋"
                    )

                return (
                    f"في {name} فيه {count} موظف 😊"
                    if language == "ar"
                    else f"There are {count} employees in {name} ⭐"
                )

            data = await self.directory.general_department_counts(access_token=context.access_token)
            print(
                f"[General Department Counts] groups={len(data.get('departments') or [])} "
                f"total_unique={int(data.get('total_unique_employees') or 0)} "
                f"unmapped={int(data.get('unmapped_employees') or 0)} "
                f"include_members={include_members}",
                file=sys.stderr,
            )
            state["directory_context"] = {"type": "general_department_counts"}
            return self._render_general_department_counts(
                data,
                language,
                include_members=include_members,
            )

        if request.intent == "general_department_members":
            query = request.department_query.strip()
            ctx = state.get("directory_context") if isinstance(state.get("directory_context"), dict) else {}
            if (
                not query
                and ctx.get("type") in {"general_department", "department"}
                and str(ctx.get("general_department") or "").strip()
            ):
                query = str(ctx.get("general_department") or "").strip()

            if not query:
                return "أي إدارة تقصد؟ 😊" if language == "ar" else "Which department do you mean?"

            data = await self.directory.general_department_members(
                access_token=context.access_token,
                department_query=query,
            )
            if data.get("matched"):
                members = [
                    directory_employee(row)
                    for row in data.get("employees", [])
                    if isinstance(row, dict)
                ]
                state["directory_context"] = {
                    "type": "department",
                    "general_department_key": data.get("department_key") or data.get("general_department_key"),
                    "general_department": data.get("department") or data.get("general_department"),
                    "count": len(members),
                }
                print(
                    f"[General Department Members] department={str(data.get('general_department') or '')!r} "
                    f"count={len(members)}",
                    file=sys.stderr,
                )
            rows_for_selection = [

                row for row in (data.get("employees") or [])

                if isinstance(row, dict)

            ]

            state["list_state"] = {

                "kind": "department_employees",
                "intent": "general_department_members",

                "department": str(data.get("general_department") or request.department_query or ""),

                "items": rows_for_selection,

            }

            print(

                f"[List State Saved] kind='department_employees' count={len(rows_for_selection)}",

                file=sys.stderr,

            )

            return self._render_general_department_members(data, language)

        return "أكيد 😊 وضّح لي طلبك أكثر." if language == "ar" else "Could you clarify your request?"

    async def _execute_request(
        self,
        request: ParsedRequest,
        resolved_targets: list[ResolvedTarget],
        context: EmployeeContext,
        language: str,
        state: dict[str, Any],
    ) -> str:
        if request.intent in {"employee_list", "employee_count", "general_department_counts", "general_department_members"}:
            try:
                return await self._execute_directory_query(request, context, language, state)
            except EmployeeAuthenticationError:
                return self._tool_error({"error_type": "auth"}, language)
            except EmployeeAuthorizationError:
                return self._tool_error({"error_type": "authorization"}, language)
            except EmployeeAPIError as exc:
                print(f"[Directory Query Error] {exc}", file=sys.stderr)
                return (
                    "ما قدرت أوصل لدليل الموظفين هالمرة 😊\nجرّب مرة ثانية بعد شوي."
                    if language == "ar"
                    else "I couldn’t reach the employee directory this time 😊\nPlease try again shortly."
                )

        if request.intent in DETAIL_BASE_INTENT:
            data, error = await self._resolve_detail_item(request, state, context, language)
            if error:
                return error

            salutation = ""
            if request.intent == "mazaya_details" and language == "ar":
                profile_result = await self._execute_tool(
                    "get_employee_profile",
                    {},
                    context,
                    target_username=context.username,
                )
                if profile_result.get("success"):
                    profile_data = (
                        profile_result.get("data")
                        if isinstance(profile_result.get("data"), dict)
                        else {}
                    )
                    salutation = _arabic_gender_salutation(profile_data)

                if not salutation:
                    salutation = "أكيد"

            is_full_content = str(request.query or "") == "content:full"
            rendered = render_content_detail(
                request.intent,
                data,
                language,
                salutation=salutation,
                full_details=is_full_content,
            )
            if request.intent in {"news_details", "announcement_details"} and not is_full_content:
                state["pending"] = {
                    "kind": "content_full_details",
                    "query": "content:full",
                    "candidates": [],
                    "request": request.model_dump(),
                    "message": "offered_full_content",
                }
            return rendered

        policy = get_policy(request.intent)
        if policy is None:
            return "أكيد 😊 وضّح لي طلبك أكثر." if language == "ar" else "Could you clarify your request?"

        # Guard against a named employee silently becoming the authenticated user.
        # A missing named target must be clarified/resolved, never mapped to self.
        if request.intent == "employee_details" and any(
            t.kind == "employee" and str(t.name or "").strip() for t in request.targets
        ) and not resolved_targets:
            return ("🔎 ما لقيت تطابقًا مؤكدًا لاسم الموظف. ممكن تكتب الاسم الكامل؟"
                    if language == "ar" else
                    "🔎 I couldn't confirm the employee. Please provide the full name.")

        # Target-scoped execution.
        if policy.target_mode in {"self_default", "required_employee"}:
            targets = resolved_targets
            if not targets:
                targets = [ResolvedTarget(kind="self", username=context.username, employee={}, source_text="self")]
            answers: list[str] = []
            successful_entities: list[dict[str, Any]] = []
            for target in targets:
                answer, entity_or_data = await self._execute_subject_request(request, target, context, language, state)
                answers.append(answer)
                if isinstance(entity_or_data, dict) and _uid(entity_or_data):
                    successful_entities.append(entity_or_data)
            # Multi-person requests must not create an arbitrary pronoun focus.
            if len(targets) > 1:
                state.setdefault("focus", {})["current"] = {}
            return "\n".join(answers)

        # Employee discovery is different from resolving one named employee.
        # Resolver name matching stays strict. Browse-style searches such as
        # "ابحث عن موظفين IT" may match verified name/title/department fields.
        if request.intent == "employee_search":
            if request.targets:
                rows = [target.employee for target in resolved_targets if target.employee]
                if not rows:
                    return "ما لقيت موظف مطابق للاسم هذا حاليًا 🔎😊 تأكد من الاسم وجرب مرة ثانية." if language == "ar" else "I couldn't find matching employees."
                state["list_state"] = {"intent": "employee_search", "items": rows[:20], "query": request.query}
                for row in rows[:20]:
                    _remember_entity(state, row, set_focus=False)
                if len(rows) == 1:
                    _remember_entity(state, rows[0], set_focus=True)
                return render_employee_search_results(rows, language, query=request.query, limit=20)

            search_query = _clean_employee_discovery_query(request.query)
            if not search_query:
                return "اكتب لي اسم الموظف أو المسمى أو الإدارة، وأنا أبحث لك 😊" if language == "ar" else "Enter an employee name, role, or department to search for."

            if re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", search_query):
                try:
                    directory_payload = await self.employee_api.search_directory_rows(
                        access_token=context.access_token, search="", batch_size="5000"
                    )
                    raw_matches = [r for r in (directory_payload.get("employees") or [])
                                   if isinstance(r, dict) and str(r.get("emailAddress") or r.get("email") or "").strip().casefold() == search_query.casefold()]
                except EmployeeAuthenticationError:
                    return self._tool_error({"error_type":"auth"}, language)
                except EmployeeAuthorizationError:
                    return self._tool_error({"error_type":"authorization"}, language)
                except EmployeeAPIError as exc:
                    print(f"[Employee Email Search Error] {exc}", file=sys.stderr)
                    return ("⚠️ تعذر البحث في دليل الموظفين حاليًا." if language == "ar"
                            else "⚠️ Employee directory search is temporarily unavailable.")
                if not raw_matches:
                    return ("🔎 ما لقيت موظفًا مرتبطًا بهذا البريد في دليل الهيئة." if language == "ar"
                            else "🔎 No employee matched that email address.")
                matched = _normalize_directory_discovery_row(raw_matches[0], language)
                _remember_entity(state, matched, set_focus=True)
                detail_request = request.model_copy(update={"intent": "employee_details", "operation": "details",
                                                     "fields": ["name", "job_title", "department", "email"]})
                return render_profile(matched, detail_request, language, self_subject=False)

            try:
                # "IT" is an organisational browse query, not a person's name.
                # Keep its verified-directory path.  For every other employee
                # search, first use the strict PERSON resolver.
                if _norm_role_text(search_query) == "it":
                    raw_rows = await self.directory.all_employees(
                        access_token=context.access_token,
                    )
                    discovery_source = "verified_directory"
                else:
                    person_payload = await self.employee_api.search_employees(
                        access_token=context.access_token,
                        search=search_query,
                    )
                    person_rows = person_payload.get("employees") if isinstance(person_payload, dict) else []
                    if person_rows:
                        raw_rows = person_rows
                        discovery_source = "verified_name_search"
                    else:
                        # If the text was actually a role/department browse query,
                        # fall back to the generic directory search path.
                        raw_payload = await self.employee_api.search_directory_rows(
                            access_token=context.access_token,
                            search=search_query,
                            batch_size="",
                        )
                        raw_rows = raw_payload.get("employees") if isinstance(raw_payload, dict) else []
                        discovery_source = "search_employee"
            except EmployeeAuthenticationError:
                return self._tool_error({"error_type": "auth"}, language)
            except EmployeeAuthorizationError:
                return self._tool_error({"error_type": "authorization"}, language)
            except EmployeeAPIError as exc:
                print(f"[Employee Discovery Error] {exc}", file=sys.stderr)
                return "آسف 😅 ما قدرت أكمل البحث عن الموظف هالمرة، وما راح أخمّن نتيجة." if language == "ar" else "Sorry, I couldn't search for the employee this time. I'm still in my experimental phase and continuing to learn and improve."
            verified_rows: list[dict[str, Any]] = []
            seen: set[str] = set()
            for raw_row in raw_rows or []:
                row = _normalize_directory_discovery_row(raw_row, language)
                uid = _uid(row).strip().casefold()
                if not uid or uid in seen:
                    continue
                if discovery_source != "verified_name_search" and not _directory_discovery_match(row, search_query):
                    continue
                seen.add(uid)
                verified_rows.append(row)

            print(
                f"[Employee Discovery] source={discovery_source!r} query={search_query!r} raw={len(raw_rows or [])} verified={len(verified_rows)}",
                file=sys.stderr,
            )

            state["list_state"] = {
                "intent": "employee_search",
                "items": verified_rows[:20],
                "query": search_query,
            }
            for row in verified_rows[:20]:
                _remember_entity(state, row, set_focus=False)
            if len(verified_rows) == 1:
                _remember_entity(state, verified_rows[0], set_focus=True)

            return render_employee_search_results(
                verified_rows,
                language,
                query=search_query,
                limit=20,
            )

        if request.intent == "employee_role_search":
            try:
                data = await self.directory.find_role_holders(
                    access_token=context.access_token,
                    role_query=request.role_query,
                )
            except EmployeeAuthenticationError:
                return self._tool_error({"error_type": "auth"}, language)
            except EmployeeAuthorizationError:
                return self._tool_error({"error_type": "authorization"}, language)
            except EmployeeAPIError as exc:
                print(f"[Role Directory Query Error] {exc}", file=sys.stderr)
                return "آسف 😅 ما قدرت أتحقق من صاحب المنصب هالمرة، وما راح أخمّن اسم." if language == "ar" else "Sorry, I couldn't search for the role holder this time. I'm still in my experimental phase and continuing to learn and improve."
        else:
            result = await self._execute_tool(policy.tool_name, self._tool_args(request, language), context)
            if not result.get("success"):
                return self._tool_error(result, language)
            data = result.get("data")

        # Role-holder lookup needs special handling because the verified role
        # result becomes the conversational subject for follow-ups such as
        # "ابي ايميله". Never leave an unrelated old focus in place.
        if request.intent == "employee_role_search":
            data_dict = data if isinstance(data, dict) else {}
            rows = _role_rows(data_dict)
            best = _role_best_match(data_dict, request.role_query)

            state["list_state"] = {
                "intent": "employee_role_search",
                "items": rows[:10],
                "query": request.role_query,
            }

            for row in rows[:10]:
                _remember_entity(state, row, set_focus=False)

            print(
                f"[Role Resolution] query={request.role_query!r} "
                f"candidates={len(rows)} best={_uid(best)!r} "
                f"ambiguous={bool(data_dict.get('ambiguous'))}",
                file=sys.stderr,
            )

            role_ambiguous = bool(data_dict.get("ambiguous"))

            # Never bind conversational focus to a role candidate when the
            # deterministic directory query marked the result ambiguous.
            if _uid(best) and not role_ambiguous:
                # A successful independent role lookup starts a fresh person focus.
                state["relation"] = {
                    "predicate": "",
                    "subject": {},
                    "object": {},
                }
                _remember_entity(state, best, set_focus=True)
                print(
                    f"[Role Memory] focus set username={_uid(best)!r} "
                    f"name={_name(best, language)!r}",
                    file=sys.stderr,
                )
                state["role_lookup"] = {
                    "role_query": request.role_query,
                    "employee": best,
                }

                # Support one-turn requests such as:
                # "وش ايميل مدير عام الإدارة العامة للتقنية والرقمنة؟"
                requested_detail_fields = [
                    field for field in request.fields
                    if field not in {"name", "job_title", "department"}
                ]
                if requested_detail_fields:
                    detail_request = request.model_copy(deep=True)
                    detail_request.intent = "employee_details"
                    detail_request.operation = "details"
                    detail_request.fields = requested_detail_fields
                    detail_request.targets = [
                        TargetRef(kind="employee", name=_name(best, language))
                    ]
                    target = ResolvedTarget(
                        kind="employee",
                        username=_uid(best),
                        employee=best,
                        source_text=request.role_query,
                    )
                    answer, _ = await self._execute_subject_request(
                        detail_request,
                        target,
                        context,
                        language,
                        state,
                    )
                    return answer

                # For a unique verified role holder, answer naturally instead
                # of showing a one-row search-results list.
                person_name = _name(best, language)
                if language == "ar":
                    role_name = str(best.get("job_title_ar") or best.get("job_title") or request.role_query or "").strip()
                    return f"{role_name} هو {person_name}." if role_name else f"يشغل هذا المنصب {person_name}."
                role_name = str(best.get("job_title_en") or request.role_query or "").strip()
                if re.search(r"[\u0600-\u06ff]", role_name):
                    role_name = ""
                return f"{person_name} is the {role_name}." if role_name else f"{person_name} holds this role."

            # No safe role winner. Keep the existing verified conversational
            # focus unchanged.

            if bool(data_dict.get("ambiguous")) and rows:
                candidate_text = _render_role_rows(rows[:5], language, limit=5, header=False)
                state["pending"] = {
                    "kind": "role_selection", "query": request.role_query,
                    "candidates": rows[:5], "request": request.model_dump(), "message": "ambiguous_role",
                }
                if language == "ar":
                    return (
                        "لقيت نتائج قريبة، لكن ما أبي أعطيك اسم غير مؤكد.\n"
                        + candidate_text
                        + "\nاكتب رقم الموظف أو اسمه، أو حدّد المنصب بشكل أدق."
                    )
                return (
                    "I found close role candidates, but the match is not verified, so I won't guess.\n"
                    + candidate_text
                    + "\nPlease specify the role or general department more precisely."
                )
            print(
                "[Role Memory] no verified unique role holder; existing focus retained",
                file=sys.stderr,
            )

            if not rows:
                return (
                    f"ما ظهر لي شخص مطابق بشكل مؤكد لهذا المنصب: {request.role_query}."
                    if language == "ar"
                    else f"I couldn't find a verified employee for this role: {request.role_query}."
                )

            state["pending"] = {
                "kind": "role_selection",
                "query": request.role_query,
                "candidates": rows[:5],
                "request": request.model_dump(),
                "message": "ambiguous_role",
            }

            prefix = (
                "لقيت أكثر من موظف قريب من هذا المنصب، تقصد مين؟\n"
                if language == "ar"
                else "I found more than one close match for this role. Which one do you mean?\n"
            )
            suffix = (
                "\nاكتب رقمه أو اسمه 😊" if language == "ar" else "\nReply with the number or the name."
            )
            return prefix + _render_role_rows(rows[:5], language, limit=5, header=False) + suffix

        if request.intent in {"news", "mazaya", "announcements", "policy_search"}:
            data_dict = data if isinstance(data, dict) else {}
            if request.intent == "mazaya":
                data_dict = await self._enrich_mazaya_rows(data_dict, context)
            answer, rows = await render_content_list(request.intent, data_dict, language)
            state["list_state"] = {"intent": request.intent, "items": rows[:25], "query": request.query}
            return answer
        if request.intent in {"leave_types", "vacations_list", "hr_letter_recipients", "hr_letter_types"}:
            data_dict = data if isinstance(data, dict) else {"items": data if isinstance(data, list) else []}
            rows = []
            for value in data_dict.values():
                if isinstance(value, list):
                    rows = [x for x in value if isinstance(x, dict)]
                    if rows:
                        break
            if rows:
                state["list_state"] = {"intent": request.intent, "items": rows[:25], "query": request.query or request.role_query}
            if request.intent == "hr_letter_types":
                return render_hr_letter_types(data_dict, language)
            if request.intent == "hr_letter_recipients":
                return render_hr_letter_recipients(data_dict, language)
            if request.intent == "leave_history":
                return render_leave_history(data_dict, language, limit=20)
            return render_generic_list(data_dict, language, limit=20)
        return render_generic(data, language)

    async def _ask_single(
        self,
        *,
        user_message: str,
        history: list[dict] | None = None,
        employee_context: EmployeeContext | None = None,
        conversation_id: str | None = None,
        record_user: bool = True,
        record_assistant: bool = True,
    ) -> tuple[str, list[dict], list[str]]:
        message = str(user_message or "").strip()
        if not message:
            return "وش حاب تسأل؟", [], []

        username = employee_context.username if employee_context else "anonymous"
        requested_conversation_id = str(conversation_id or "").strip().lower()

        # Each chat owns its own conversation id. The mobile/web client must reuse
        # the id returned by POST /chat for follow-up turns.
        if re.fullmatch(r"c_[a-f0-9]{24}", requested_conversation_id):
            conversation_id = requested_conversation_id
        else:
            conversation_id = "c_" + secrets.token_hex(12)

        memory_key = f"{username}:{conversation_id}"
        state = self.memory.get_state(memory_key)

        # One-time migration from the buggy pre-v64 transient context.
        if str(state.get("context_schema_version") or "") != "v160":
            state["last_request"] = {}
            state["meeting_context"] = {}
            state["attendance_context"] = {}
            state["task_flow"] = {}
            state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}
            state["relation"] = {"predicate": "", "subject": {}, "object": {}}
            state["focus"] = {"current": {}, "previous": {}}
            state["domain_state"] = {}
            state["context_schema_version"] = "v160"
            self.memory.save_state(memory_key, state)
            print(f"[Context Migration v160] reset stale transient context for {memory_key!r}", file=sys.stderr)

        fallback_language = str(state.get("language") or "ar")

        original_message = message
        pending_for_alias = state.get("pending") if isinstance(state.get("pending"), dict) else {}
        pending_kind_for_alias = str(pending_for_alias.get("kind") or "")
        if (
            message.casefold() == "hd"
            and pending_kind_for_alias in {
                "department_members_offer",
                "tarasol_details_offer",
                "tawasul_events_offer",
                "employee_more_details",
            }
        ):
            message = "اي"
            print(
                f"[Keyboard Layout Confirmation] {original_message!r} -> {message!r} "
                f"pending={pending_kind_for_alias!r}",
                file=sys.stderr,
            )

        language = detect_language(message, fallback=fallback_language)
        state["language"] = language
        if record_user:
            self.memory.append_message(memory_key, "user", original_message)

        # Ephemeral language context for the semantic model. This is deliberately
        # not an authority for enterprise facts; verified state/API data remain the
        # grounding source. It lets short follow-ups behave like real conversation.
        state["_recent_dialogue"] = self.memory.recent_messages(memory_key, limit=12)
        print(f"[Conversation Memory] key={memory_key!r}", file=sys.stderr)

        def finish(answer: str, *, sources: list[str] | None = None, save_state: bool = True) -> tuple[str, list[dict], list[str]]:
            clean = polish_response(str(answer or "").strip(), language=language)
            if save_state:
                ephemeral_dialogue = state.pop("_recent_dialogue", None)
                self.memory.save_state(memory_key, state)
                if ephemeral_dialogue is not None:
                    state["_recent_dialogue"] = ephemeral_dialogue
            if record_assistant:
                self.memory.append_message(memory_key, "assistant", clean)
            return clean, [], list(sources or [])

        # Explicit calendar dates lacking a year: ask only for the missing year.
        # Pending context is persisted in the existing sanitized conversation state.
        date_pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
        if date_pending.get("kind") == "date_year_clarification":
            revised, consumed = date_year_turn(message, date_pending.get("message", ""))
            if consumed:
                message = revised
                state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}
            elif not dated_message_needs_year(message):
                # A fresh question cancels obsolete clarification context.
                state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}
        if dated_message_needs_year(message):
            state["pending"] = {"kind": "date_year_clarification", "query": "", "candidates": [], "request": {}, "message": message}
            return finish(
                "📅 لأي سنة تقصد الفترة المذكورة؟" if language == "ar"
                else "📅 Which year should I use for those dates?",
                save_state=True,
            )

        # Leave overlap is enabled: allow verified requests to reach
        # the leave_overlap intent and get_intersected_vacations tool.
        # Do not short-circuit with the old testing-phase disabled message.

        calc = _safe_calculation(message)
        if calc is not None:
            return finish(calc, save_state=True)

        greet = _greeting(message, language)
        if greet is not None:
            return finish(greet, save_state=True)

        assistant_identity = _assistant_identity_reply(message, language)
        if assistant_identity is not None:
            return finish(assistant_identity, save_state=True)

        # Recover naturally from user frustration before any pending task flow.
        frustration = _frustration_reply(message, state, language)
        if frustration is not None:
            return finish(frustration, save_state=True)

        # Generic policy requests must continue through semantic routing and
        # the verified Policy API so Sanad can list the available policies.
        # The renderer adds the in-app navigation hint after the real list.

        # Continue an availability preference question first.
        availability_turn = _availability_preference_turn(message, state, language)
        if availability_turn is not None:
            mode, value = availability_turn
            if mode == "answer":
                return finish(value, save_state=True)
            if mode == "rewrite":
                print(f"[Availability Preference Rewrite] {message!r} -> {value!r}", file=sys.stderr)
                message = value

        # "ابي اسوي اجتماع معه" uses the verified person/calendar context.
        employee_meeting = _employee_meeting_request(message, state, language)
        if employee_meeting is not None:
            return finish(employee_meeting, save_state=True)

        # Continue an employee meeting-planning flow.
        companion_turn = _meeting_companion_turn(message, state, language)
        if companion_turn is not None:
            mode, value = companion_turn
            if mode == "answer":
                return finish(value, save_state=True)
            if mode == "rewrite":
                print(f"[Meeting Companion Rewrite] {message!r} -> {value!r}", file=sys.stderr)
                message = value

        assistant_capabilities = _assistant_capabilities_reply(message, language)
        if assistant_capabilities is not None:
            return finish(assistant_capabilities, save_state=True)

        # Safety/moderation remains ahead of both general and enterprise execution.
        if await self._is_sensitive(message):
            return finish("ما أقدر أساعد في هذا الطلب." if language == "ar" else "I can't help with that request.")

        # Transaction/dependency workflows run before open-ended semantic parsing.
        # They are backend-owned state machines: username comes only from the
        # authenticated context and write endpoints execute only after a pending
        # server-side action receives an explicit user confirmation.
        action_result = await self.actions.handle(
            message=message,
            state=state,
            context=employee_context,
            language=language,
        )
        if action_result.handled:
            return finish(action_result.message, save_state=True)

        # RCRC-FIX-v1 (تعديل): كان هنا فحص _obvious_general_conversation قبل
        # كل شي، فكانت أسئلة البيانات القصيرة تروح للموديل العام. نقلناه تحت
        # بعد الـ preflight.

        # Mixed-reference follow-up: one sentence may ask for a field of the
        # current employee AND a different field of that employee's manager.
        # Example: "ابي تحويلتها وابي ايميل مديرها".
        mix_norm = message.casefold().translate(str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي"}))
        focus_state = state.get("focus") if isinstance(state.get("focus"), dict) else {}
        mixed_focus = focus_state.get("current") if isinstance(focus_state.get("current"), dict) else {}
        mixed_uid = _uid(mixed_focus)
        has_focus = bool(
            mixed_uid
            or str(mixed_focus.get("name") or "").strip()
            or str(mixed_focus.get("name_ar") or "").strip()
            or str(mixed_focus.get("name_en") or "").strip()
        )
        manager_ref = bool(re.search(r"(?:مديرها|مديره|مديرهم|his\s+manager|her\s+manager)", mix_norm, re.I))

        if employee_context is not None and mixed_uid and manager_ref:
            direct_fields: list[str] = []
            manager_fields: list[str] = []

            # Possessive forms point directly to the current focus.
            if re.search(r"(?:ايميلها|ايميله|ايميلة|إيميلها|إيميله|إيميلة|بريدها|بريده|her\s+email|his\s+email)", mix_norm, re.I):
                direct_fields.append("email")
            if re.search(r"(?:جوالها|جواله|موبايلها|موبايله|رقم\s+جوالها|رقم\s+جواله|her\s+(?:mobile|phone)|his\s+(?:mobile|phone))", mix_norm, re.I):
                direct_fields.append("mobile")
            if re.search(r"(?:تحويلتها|تحويلته|her\s+extension|his\s+extension)", mix_norm, re.I):
                direct_fields.append("extension")

            # A field explicitly attached to "مديرها/مديره" belongs to the
            # manager, not to the current focus.
            manager_word = r"(?:مديرها|مديره|مديرهم|his\s+manager|her\s+manager)"
            if re.search(rf"(?:ايميل|بريد|email)\s+{manager_word}|{manager_word}\s+(?:ايميل|بريد|email)", mix_norm, re.I):
                manager_fields.append("email")
            if re.search(rf"(?:رقم|جوال|موبايل|phone|mobile)\s+{manager_word}|{manager_word}\s+(?:رقم|جوال|موبايل|phone|mobile)", mix_norm, re.I):
                manager_fields.append("mobile")
            if re.search(rf"(?:تحويله|تحويلة|extension)\s+{manager_word}|{manager_word}\s+(?:تحويله|تحويلة|extension)", mix_norm, re.I):
                manager_fields.append("extension")

            direct_fields = list(dict.fromkeys(direct_fields))
            manager_fields = list(dict.fromkeys(manager_fields))

            if direct_fields and manager_fields:
                mixed_target = ResolvedTarget(
                    kind="focus",
                    username=mixed_uid,
                    employee=deepcopy(mixed_focus),
                    source_text="focus",
                )
                direct_request = ParsedRequest(
                    intent="employee_details", operation="details",
                    targets=[TargetRef(kind="focus", name="")],
                    fields=direct_fields, query="employee:mixed_focus_fields",
                    confidence=0.99,
                )
                manager_request = ParsedRequest(
                    intent="manager", operation="details",
                    targets=[TargetRef(kind="focus", name="")],
                    fields=manager_fields, relation_depth=1,
                    query="manager:mixed_contact", confidence=0.99,
                )

                first_answer, _ = await self._execute_subject_request(
                    direct_request, mixed_target, employee_context, language, state
                )
                second_answer, _ = await self._execute_subject_request(
                    manager_request, mixed_target, employee_context, language, state
                )
                _clear_pending(state)
                _record_request(state, manager_request)
                print(
                    f"[Mixed Reference Follow-up] focus={mixed_uid!r} "
                    f"direct_fields={direct_fields!r} manager_fields={manager_fields!r}",
                    file=sys.stderr,
                )
                return finish(f"{first_answer}\n{second_answer}")

        # v173 EXACT RESTORE: the original self-meetings wizard is a backend
        # state machine, not an LLM conversation.  Only the flow that *asked*
        # the user to choose a day proceeds to the second question about the
        # part of day.  Direct requests such as "اجتماعاتي اليوم" continue to
        # execute immediately, as before.
        pending_exact = state.get("pending") if isinstance(state.get("pending"), dict) else {}
        pending_exact_kind = str(pending_exact.get("kind") or "")
        pending_exact_req = pending_exact.get("request") if isinstance(pending_exact.get("request"), dict) else {}
        inline_meeting_request = None

        # Step 2 of: "ابي جدول اجتماعاتي" -> choose day -> choose time window.
        if (
            pending_exact_kind == "missing_date_range"
            and str(pending_exact_req.get("intent") or "") in {"meetings", "meeting_list"}
            and _has_explicit_period_cue(message)
        ):
            period, start_date, end_date = _availability_period(message)
            if period not in {"", "none"}:
                try:
                    restored = ParsedRequest.model_validate(pending_exact_req)
                except Exception:
                    restored = ParsedRequest(
                        intent="meetings", operation="list",
                        targets=[TargetRef(kind="self", name="")],
                        fields=["subject", "start_time", "end_time"],
                    )
                restored.intent = "meetings"
                restored.operation = "list"
                restored.targets = [TargetRef(kind="self", name="")]
                restored.fields = ["subject", "start_time", "end_time"]
                restored.period = period
                restored.start_date = start_date
                restored.end_date = end_date
                # If the same turn already contains a time window (e.g.
                # "ابي يوم الثلاثاء الصباح"), execute it immediately.
                # Only ask the second wizard question when no window was supplied.
                inline_time_query = _extract_time_query(message)
                restored.time_query = inline_time_query or ""
                restored.clarification_reason = ""
                restored.confidence = 1.0

                state["meeting_context"] = {
                    "subject_kind": "self",
                    "employee": {},
                    "period": period,
                    "start_date": start_date,
                    "end_date": end_date,
                    "time_query": restored.time_query,
                    "mode": "self_meetings" if inline_time_query else "self_meetings_wizard",
                }
                _record_request(state, restored)

                # The user has now answered the only required follow-up:
                # which day/period to check. Execute all meetings in that
                # selected period unless a time window was explicitly supplied.
                _clear_pending(state)
                inline_meeting_request = restored
                print(
                    f"[Exact Meeting Period Selected] {restored.model_dump()}",
                    file=sys.stderr,
                )

        # Step 3: a short answer such as "الصباح" inherits the selected day.
        if inline_meeting_request is not None:
            pending_request = inline_meeting_request
        elif pending_exact_kind == "meeting_time_window":
            time_query = _extract_time_query(message)
            if time_query:
                try:
                    restored = ParsedRequest.model_validate(pending_exact_req)
                except Exception:
                    restored = ParsedRequest(
                        intent="meetings", operation="list",
                        targets=[TargetRef(kind="self", name="")],
                        fields=["subject", "start_time", "end_time"],
                    )
                restored.intent = "meetings"
                restored.operation = "list"
                restored.targets = [TargetRef(kind="self", name="")]
                restored.fields = ["subject", "start_time", "end_time"]
                restored.time_query = time_query
                restored.clarification_reason = ""
                restored.confidence = 1.0
                # Do not clear meeting_context here; the renderer/execution path
                # uses the same verified selected day and filters by this window.
                state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}
                request = restored
                print(f"[Exact Meeting Window] {request.model_dump()}", file=sys.stderr)
                pending_request = request
            else:
                pending_request = None
        else:
            pending_request = None

        # v196: writing follow-ups stay attached to the last email draft.
        # This must run before semantic/employee routing so short turns such as
        # "ترجمه" or "خله رسمي أكثر" cannot become a directory lookup.
        if _is_writing_followup_request(message, state):
            writing_ctx = _writing_context(state)
            previous_draft = str(writing_ctx.get("draft") or "").strip()
            target_language = _writing_followup_target_language(message, state)
            answer = await self._email_followup_answer(
                message, previous_draft, target_language
            )
            state["writing_context"] = {
                "kind": "email",
                "draft": answer,
                "language": target_language,
                "recipient_name": str(writing_ctx.get("recipient_name") or ""),
                "source_message": str(writing_ctx.get("source_message") or ""),
            }
            print(
                f"[Email Followup Route] target_language={target_language!r} instruction={message[:80]!r}",
                file=sys.stderr,
            )
            return finish(answer, save_state=True)

        # v195: HARD email-drafting route. Writing an email is ordinary LLM work
        # and must never be interpreted as an employee-directory lookup just
        # because the request mentions "مديري" or contains the word "ايميل".
        # Run this before pending/semantic/employee routing.
        if _is_email_drafting_request(message):
            facts = await self._writing_facts(message, employee_context, language, state)
            answer = await self._email_draft_answer(message, language, facts)
            if not _email_request_is_vague(message):
                state["writing_context"] = {
                    "kind": "email",
                    "draft": answer,
                    "language": language,
                    "recipient_name": str(facts.get("recipient_name") or ""),
                    "source_message": message,
                }
            print("[Email Draft Hard Route] direct_writing", file=sys.stderr)
            return finish(answer, save_state=True)

        # Conversation-first routing:
        # 1) pending explicit list selection
        # 2) semantic model with recent dialogue
        # 3) deterministic critical preflight only as fallback
        if pending_request is None:
            # Department employee numbered-list selection has highest priority.
            # The generic reference_resolver also understands numbers, so if it
            # runs first it can steal "22" and reinterpret it in another domain.
            list_state_now = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
            if (
                (str(list_state_now.get("kind") or "") == "department_employees" or str(list_state_now.get("intent") or "") == "general_department_members")
                and _list_selection_index(message) is not None
            ):
                pending_request = _critical_orchestrator_preflight(
                    message,
                    state,
                    language,
                )
                if pending_request is not None:
                    print(
                        f"[Department List Selection Priority] text={message!r} "
                        f"query={pending_request.query!r}",
                        file=sys.stderr,
                    )

        if pending_request is None:
            # Content selection/full-detail follow-ups are resolved from verified
            # state BEFORE generic employee selection or the LLM.
            pending_request = _restore_full_content_followup(message, state)
        if pending_request is None:
            # A second number after opening benefit #1 still refers to the
            # last verified benefits list; never reinterpret it as hours.
            _ls = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
            _base_to_detail = {
                "mazaya": "mazaya_details", "news": "news_details",
                "announcements": "announcement_details", "policy_search": "policy_details",
            }
            _selected_number = _content_selection_number(message)
            _items = _ls.get("items") if isinstance(_ls.get("items"), list) else []
            if (_selected_number > 0 and _selected_number <= len(_items)
                    and str(_ls.get("intent") or "") in _base_to_detail):
                pending_request = ParsedRequest(
                    intent=_base_to_detail[str(_ls["intent"])],
                    operation="details", targets=[], fields=["all"],
                    query="", ordinal=_selected_number, confidence=1.0,
                )
        if pending_request is None:
            pending_request = _restore_content_detail_selection(message, state)
        if pending_request is None:
            pending_request = restore_pending_selection(message, state)
        if pending_request is not None:
            request = pending_request
            print(f"[Pending Selection] intent={request.intent}", file=sys.stderr)
        else:
            semantic_text = _semantic_message(message)

            # v171: CLOSED FOLLOW-UPS BYPASS THE LLM.
            #
            # These turns are structurally determined by verified conversation
            # state and must never be reinterpreted as a new employee/search:
            #   "ابي ايميله ورقم جواله" after a verified person -> focus details
            #   "الصباح" after selecting Tuesday meetings -> same day + morning
            #
            # Running these through Qwen first caused valid focus references to
            # occasionally become invented intents or fake employee names.
            closed_request = _critical_orchestrator_preflight(semantic_text, state, language)
            pending_now = state.get("pending") if isinstance(state.get("pending"), dict) else {}
            closed_query = str(closed_request.query or "") if closed_request is not None else ""
            # A verified leave what-if is closed grammar, not an open-ended LLM guess.
            is_leave_calculation = bool(
                closed_request is not None
                and closed_request.intent == "leave_balance"
                and closed_request.operation == "calculate"
                and str(closed_request.query or "") == "leave:self_hypothetical"
            )
            is_focus_contact_followup = bool(
                closed_request is not None
                and closed_request.intent == "employee_details"
                and any(t.kind == "focus" for t in (closed_request.targets or []))
                and closed_query.startswith("employee:focus_")
            )
            is_meeting_window_followup = bool(
                closed_request is not None
                and closed_request.intent in {"meetings", "meeting_list"}
                and str(pending_now.get("kind") or "") == "meeting_time_window"
                and bool(str(closed_request.time_query or "").strip())
            )

            # Target-priority fix: an explicitly named employee ALWAYS wins over
            # LLM guesses and cached self-profile context.
            from core.direct_employee_intent import direct_employee_lookup as _direct_employee_lookup
            _explicit_employee = _direct_employee_lookup(semantic_text)
            _explicit_employee_request = None
            if _explicit_employee is not None:
                _mode, _person, _fields = _explicit_employee
                if _mode == "email_lookup":
                    _explicit_employee_request = ParsedRequest(
                        intent="employee_search", operation="search", targets=[],
                        fields=_fields, query=_person, confidence=1.0,
                    )
                elif re.match(r"^(?:ابحث|أبحث|دور|دوّر|دورلي|دور\s+لي|search|find)\b", semantic_text.strip(), re.I):
                    _explicit_employee_request = ParsedRequest(
                        intent="employee_search", operation="search", targets=[],
                        fields=["name", "job_title", "department"], query=_person, confidence=1.0,
                    )
                else:
                    _explicit_employee_request = ParsedRequest(
                        intent="employee_details", operation="details",
                        targets=[TargetRef(kind="employee", name=_person)],
                        fields=_fields, query="employee:direct_lookup", confidence=1.0,
                    )

            # Closed-context priority for anaphoric contact requests.
            # Do not let named-person parsing treat Arabic possessive suffix
            # ("ه" from "إيميله") as an employee's actual name.
            _focus_pronoun_request = None
            if has_focus:
                _ptext = " ".join(str(semantic_text or "").casefold().split())
                _pfields = []
                for _field, _pattern in (
                    ("email", r"(?:[اإ]يميل(?:ه|ها)|بريد(?:ه|ها)|(?:his|her|their)\s+email)"),
                    ("mobile", r"(?:جوال(?:ه|ها)|رقم\s+جوال(?:ه|ها)|موبايل(?:ه|ها)|(?:his|her|their)\s+(?:phone|mobile))"),
                    ("extension", r"(?:تحويلت(?:ه|ها)|تحويل(?:ه|ها)|(?:his|her|their)\s+extension)"),
                ):
                    if re.search(_pattern, _ptext, re.I):
                        _pfields.append(_field)
                _refers_to_self = bool(re.search(
                    r"(?:ايميلي|إيميلي|جوالي|بريدي|تحويلتي|my\s+(?:email|phone|mobile|extension))",
                    _ptext, re.I,
                ))
                if _pfields and not _refers_to_self:
                    _focus_pronoun_request = ParsedRequest(
                        intent="employee_details", operation="details",
                        targets=[TargetRef(kind="focus", name="")],
                        fields=_pfields, query="employee:focus_fields",
                        confidence=1.0,
                    )

            is_self_profile_followup = bool(
                closed_request is not None
                and closed_request.intent == "profile"
                and any(t.kind == "self" for t in (closed_request.targets or []))
                and str(closed_request.query or "").startswith("profile:self_")
            )

            # v181: direct enterprise routing for obvious News list requests.
            # Do this BEFORE Qwen so phrases like "ابي ابرز اخبار الهيئة" cannot
            # turn into a generic conversational clarification.
            news_norm = re.sub(r"[؟?!.،,]+", " ", str(semantic_text or "").casefold())
            news_norm = " ".join(news_norm.split())
            direct_news = bool(
                re.search(r"(?:اخبار|أخبار|news)", news_norm, re.I)
                and (
                    re.search(r"(?:ابرز|أبرز|اهم|أهم|اخر|آخر|احدث|أحدث|جديد|latest|recent|top|highlights?)", news_norm, re.I)
                    or re.fullmatch(r"(?:ابي\s+)?(?:وش\s+)?(?:ال)?(?:اخبار|أخبار)(?:\s+(?:ال)?(?:هيئه|هيئة|rcrc))?", news_norm, re.I)
                )
            )

            # Verified employee relations must win over stale semantic queries.
            # "طيب مين مديره؟" refers to the active verified employee, not
            # a previous question about an unrelated department manager.
            _manager_relation_text = re.sub(r"[؟?!.،,]+", " ", str(semantic_text or "").casefold())
            _manager_relation_text = " ".join(_manager_relation_text.split())
            _manager_followup = bool(
                has_focus
                and re.search(
                    r"(?:مديره|مديرها|مديرهم|مدير(?:ة)?\s+مديره|مدير(?:ة)?\s+مديرها|"
                    r"(?:his|her|their)\s+manager|manager\s+of\s+(?:him|her))",
                    _manager_relation_text, re.I,
                )
                and not re.search(r"(?:مديري|مديرتي|my\s+manager)", _manager_relation_text, re.I)
            )
            _manager_fields = []
            if _manager_followup:
                if re.search(r"(?:ايميل|إيميل|بريد|email)", _manager_relation_text, re.I):
                    _manager_fields.append("email")
                if re.search(r"(?:جوال|موبايل|رقم|mobile|phone)", _manager_relation_text, re.I):
                    _manager_fields.append("mobile")
                if re.search(r"(?:تحويل|extension)", _manager_relation_text, re.I):
                    _manager_fields.append("extension")
                # Follow-up can request only the manager name (default), or
                # several contact fields in one question.
                request = ParsedRequest(
                    intent="manager", operation="details",
                    targets=[TargetRef(kind="focus", name="")],
                    fields=_manager_fields or ["name"],
                    relation_depth=2 if re.search(r"مدير\s+مدير(?:ه|ها)|manager.s\s+manager", _manager_relation_text, re.I) else 1,
                    query="manager:verified_focus_followup", confidence=1.0,
                )
                print(f"[Manager Followup Priority] {request.model_dump()}", file=sys.stderr)

            if _manager_followup:
                pass  # request is already routed to the verified focus manager.
            elif direct_news:
                request = ParsedRequest(
                    intent="news",
                    operation="list",
                    targets=[],
                    fields=["all"],
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
                    confidence=1.0,
                    clarification_reason="",
                )
                print(f"[Direct News Route] text={semantic_text!r} {request.model_dump()}", file=sys.stderr)
            elif _focus_pronoun_request is not None:
                request = _focus_pronoun_request
                print(f"[Focus Pronoun Priority] {request.model_dump()}", file=sys.stderr)
            elif _explicit_employee_request is not None:
                request = _explicit_employee_request
                print(f"[Named Employee Priority] {request.model_dump()}", file=sys.stderr)
            elif (
                is_leave_calculation
                or is_focus_contact_followup
                or is_meeting_window_followup
                or is_self_profile_followup
                or (
                    closed_request is not None
                    and str(closed_request.query or "").startswith((
                        "department:",
                        "employee:list_selection",
                        "general:employee_list_selected",
                        "tarasol:",
                    ))
                )
            ):
                request = closed_request
                print(
                    f"[Closed Enterprise Override] text={semantic_text!r} {request.model_dump()}",
                    file=sys.stderr,
                )
            else:
                request = await self.parser.parse(semantic_text, state, language=language)
                print(f"[Semantic Request] text={semantic_text!r} {request.model_dump()}", file=sys.stderr)

                # Qwen handles open-ended meaning, while closed enterprise grammar
                # protects exact relations, requested fields, and pending date flows.
                critical_request = closed_request
                if critical_request is not None:
                    cq = str(critical_request.query or "")
                    strong_closed_grammar = bool(
                        cq.startswith((
                            "manager:",
                            "leave:self_",
                            "employee:focus_",
                            "employee:explicit_",
                            "employee:summary",
                            "employee:list_selection",
                            "role:explicit",
                            "profile:self_",
                            "profile:summary",
                            "meetings:",
                            "attendance:",
                            "department:",
                            "tarasol:",
                        ))
                        or critical_request.clarification_reason == "missing_date_range"
                        or (
                            critical_request.intent == request.intent
                            and set(critical_request.fields or []).issuperset(set(request.fields or []))
                            and len(critical_request.fields or []) > len(request.fields or [])
                        )
                    )
                    semantic_focus_reference = bool(
                        request.intent == "employee_details"
                        and any(getattr(t, "kind", "") == "focus" for t in (request.targets or []))
                        and has_focus
                    )
                    if (
                        request.intent == "general"
                        or (
                            critical_request.intent == request.intent
                            and (
                                request.confidence < 0.60
                                or bool(request.clarification_reason)
                                or (strong_closed_grammar and not semantic_focus_reference)
                            )
                        )
                    ):
                        request = critical_request
                        print(f"[Critical Fallback] {request.model_dump()}", file=sys.stderr)
                    elif semantic_focus_reference and strong_closed_grammar:
                        print(
                            f"[Focus Reference Guard] kept semantic focus request over fallback={critical_request.model_dump()}",
                            file=sys.stderr,
                        )

            # General questions should behave like a normal conversational assistant.
            #
            # Older builds required a narrow keyword shortcut here, which meant a valid
            # semantic `general` classification (for example "وش MCP؟" or "اشرح لي API")
            # could still fall through into enterprise validation and produce an awkward
            # fallback. Enterprise requests are already protected by deterministic
            # preflight + semantic evidence guards, so a genuine `general` request can
            # safely use the no-tools conversational model.
            #
            # The only exception is a tiny reply while an enterprise clarification/list
            # selection is pending ("2", "الثاني", "أنا"). Those must stay in the
            # structured flow instead of being consumed as general chat.
            if (
                request.intent == "general"
                and str(request.query or "") == "general:employee_list_selected"
            ):
                selected = (
                    state.get("selected_list_employee")
                    if isinstance(state.get("selected_list_employee"), dict)
                    else {}
                )
                selected_name = str(selected.get("name") or "").strip()
                focus = (state.get("focus") or {}).get("current") or {}
                title = str(focus.get("job_title") or focus.get("job_title_ar") or "").strip()
                if language == "ar":
                    answer = f"اخترت {selected_name or 'الموظف'}"
                    if title:
                        answer += f" — {title}"
                    answer += ". أقدر أطلع لك بياناته المتاحة، مثل الإيميل أو المنصب، أو أبحث عن مديره."
                else:
                    answer = f"You selected {selected_name or 'the employee'}"
                    if title:
                        answer += f" — {title}"
                    answer += ". I can help with their available contact details, role, or manager."
                print(
                    f"[List Selection Ack] name={selected_name!r}",
                    file=sys.stderr,
                )
                return finish(answer, save_state=True)

            if request.intent == "general" and request.query == "general:org_hierarchy_unverified":
                answer = (
                    "🏢 ما عندي هيكل تنظيمي موثّق يوضح الإدارات التابعة لهذه الإدارة حاليًا، "
                    "فما راح أعرض قائمة موظفين على أنها إدارات."
                    if language == "ar" else
                    "🏢 I don't have a verified organizational hierarchy for this department, "
                    "so I won't present an employee list as a list of sub-departments."
                )
                return finish(answer, save_state=True)

            if request.intent == "general":
                # Final enterprise safety net: natural self-service wording must
                # never fall into the open-ended LLM merely because the semantic
                # model missed the phrasing.
                enterprise_retry = _critical_orchestrator_preflight(
                    message,
                    state,
                    language,
                )
                if enterprise_retry is not None and enterprise_retry.intent != "general":
                    request = enterprise_retry
                    print(
                        f"[Enterprise Safety Route] {request.model_dump()}",
                        file=sys.stderr,
                    )

            if request.intent == "general":
                pending_general = state.get("pending") if isinstance(state.get("pending"), dict) else {}

                # Conversation continuity guard: a short reply to an explicit
                # pending offer belongs to that offer even when the semantic
                # model labels the turn as generic chat.
                if str(pending_general.get("kind") or "") == "tawasul_events_offer":
                    continuation = _critical_orchestrator_preflight(message, state, language)
                    if continuation is not None and continuation.intent != "general":
                        request = continuation
                        print(
                            f"[Pending Context Override] {request.model_dump()}",
                            file=sys.stderr,
                        )

            if request.intent == "general":
                pending_general = state.get("pending") if isinstance(state.get("pending"), dict) else {}
                pending_kind = str(pending_general.get("kind") or "").strip()
                selection_like = bool(
                    re.fullmatch(
                        r"(?:\d{1,2}|الاول|الأول|اول|الثاني|ثاني|الثالث|ثالث|الرابع|رابع|"
                        r"الخامس|خامس|انا|أنا|نفسي|me|myself|first|second|third|fourth|fifth)",
                        _norm_role_text(message),
                        re.I,
                    )
                )
                if not (pending_kind and selection_like):
                    print(f"[Route] general reason=semantic_general msg={message[:80]!r}", file=sys.stderr)
                    answer = await self._general_answer(message, state, language)
                    return finish(answer, save_state=True)

            restored_target = restore_pending_target(request, state)
            has_self_target = any(t.kind == "self" for t in request.targets)
            if restored_target is not None and not has_self_target:
                request = restored_target
                print(f"[Pending Target Restored] {request.model_dump()}", file=sys.stderr)

            # A new complete turn supersedes stale resolver state.
            pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
            target_kinds = {t.kind for t in request.targets}
            complete_reference = bool(
                target_kinds.intersection({
                    "self", "focus", "previous_focus", "manager_of_self",
                    "relation_subject", "relation_object",
                })
            )
            if pending.get("kind") == "employee_disambiguation":
                _clear_pending(state)
            elif pending.get("kind") in {"resolution_retry", "missing_employee_target"} and complete_reference:
                print(
                    f"[Pending Guard] cleared stale {pending.get('kind')!r} before intent={request.intent!r}",
                    file=sys.stderr,
                )
                _clear_pending(state)

        # RCRC-FIX-v137: target consistency guard.
        # A third-person possessive can never become authenticated self.
        focus_guard = state.get("focus") if isinstance(state.get("focus"), dict) else {}
        focus_employee_guard = focus_guard.get("current") if isinstance(focus_guard.get("current"), dict) else {}
        if _uid(focus_employee_guard):
            third_person_reference = bool(re.search(
                r"(?:اجازاته|إجازاته|اجازتها|إجازتها|حضوره|حضورها|دوامه|دوامها|"
                r"ايميله|إيميله|ايميلها|إيميلها|جواله|جوالها|تحويلته|تحويلتها|"
                r"منصبه|منصبها|ادارته|إدارته|ادارتها|إدارتها|"
                r"\bhis\b|\bher\b|\bhe\b|\bshe\b)",
                _norm_role_text(message),
                re.I,
            ))
            if third_person_reference and request.targets and any(t.kind == "self" for t in request.targets):
                if request.intent in {
                    "leave_balance", "leave_history", "attendance_today", "attendance_history",
                    "employee_details", "manager", "employee_availability", "reporting_employees",
                }:
                    request.targets = [
                        TargetRef(kind="focus", name="") if t.kind == "self" else t
                        for t in request.targets
                    ]
                    print(
                        f"[Target Consistency Guard] self->focus intent={request.intent!r}",
                        file=sys.stderr,
                    )

        # Bare self calendar/attendance requests ask for the day instead of
        # silently assuming today. Other-employee calendar data remains self-only.
        temporal_norm = _norm_role_text(message)
        has_explicit_period = bool(re.search(
            r"(?:اليوم|بكره|بكرة|غدا|غداً|امس|أمس|هذا\s+الاسبوع|الاسبوع\s+هذا|"
            r"الاسبوع\s+الماضي|هذا\s+الشهر|الشهر\s+الماضي|today|tomorrow|yesterday|"
            r"this\s+week|last\s+week|this\s+month|last\s+month|"
            r"الاحد|الأحد|احد|أحد|الاثنين|الإثنين|اثنين|الثلاثاء|ثلاثاء|"
            r"الاربعاء|الأربعاء|اربعاء|أربعاء|الخميس|خميس|الجمعه|الجمعة|جمعه|جمعة|السبت|سبت|"
            r"sunday|monday|tuesday|wednesday|thursday|friday|saturday|"
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b)",
            temporal_norm, re.I,
        ))
        self_temporal = bool(
            request.targets
            and all(t.kind in {"self", "none"} for t in request.targets)
            and request.intent in {"meetings", "meeting_list", "employee_availability", "attendance_today", "attendance_history", "working_hours"}
        )

        # Context is allowed to carry a date across a natural follow-up.
        # Example: "وش اجتماعاتي اليوم؟" -> "الصباح" must keep TODAY instead
        # of being reset to missing_date_range simply because the second message
        # does not repeat the word "اليوم". A model-resolved custom date (e.g.
        # "الثلاثاء") is also authoritative structurally because it already has
        # a concrete start_date/end_date.
        pending_for_temporal = state.get("pending") if isinstance(state.get("pending"), dict) else {}
        pending_kind_for_temporal = str(pending_for_temporal.get("kind") or "")
        carries_pending_meeting_date = bool(
            pending_kind_for_temporal == "meeting_time_window"
            and request.intent in {"meetings", "meeting_list"}
            and str(request.period or "none") not in {"", "none"}
        )
        has_structured_custom_date = bool(
            str(request.period or "none") == "custom"
            and (str(request.start_date or "").strip() or str(request.end_date or "").strip())
        )
        # Meeting wizard continuation: once the user selected a day and then a
        # time window (e.g. today -> morning), the second turn must inherit the
        # saved day. Do not reset it just because the word "today" is not
        # repeated in the short follow-up.
        has_meeting_window_with_date = bool(
            request.intent in {"meetings", "meeting_list"}
            and str(getattr(request, "time_query", "") or "").strip()
            and str(request.period or "none") not in {"", "none"}
        )

        if (
            self_temporal
            and not has_explicit_period
            and not carries_pending_meeting_date
            and not has_structured_custom_date
            and not has_meeting_window_with_date
            and str(request.period or "none") in {"", "none", "today"}
        ):
            request.period = "none"
            request.start_date = ""
            request.end_date = ""
            request.clarification_reason = "missing_date_range"

        focus_now = state.get("focus") if isinstance(state.get("focus"), dict) else {}
        current_focus = focus_now.get("current") if isinstance(focus_now.get("current"), dict) else {}
        print(
            f"[Focus Before Resolve] username={_uid(current_focus)!r} name={_name(current_focus, language)!r}",
            file=sys.stderr,
        )

        # RCRC-FIX-v2: friendly, domain-aware clarification instead of a robotic one.
        if request.intent == "general" and request.clarification_reason in {
            "unverified_enterprise_intent", "semantic_parser_failed",
        }:
            _log_nlu_miss(message, request.clarification_reason, state)
            # Keep the user's goal alive; do not ask them to start over.
            focus_state = state.get("focus") if isinstance(state.get("focus"), dict) else {}
            focus_row = focus_state.get("current") if isinstance(focus_state.get("current"), dict) else {}
            focus_name = _name(focus_row, language) if _uid(focus_row) else ""
            if language == "ar":
                if focus_name:
                    return finish(
                        f"أفهم إن سؤالك مرتبط بـ {focus_name}. بس أبي أتأكد من المطلوب: "
                        "تقصد بياناته، مديره، حضوره، إجازاته، أو توفره للاجتماع؟"
                    )
                return finish(
                    "فهمت إنك تطلب معلومة من أنظمة الهيئة، لكن احتاج أحدد النوع بدقة. "
                    "تقصد موظف، إدارة، حضور، إجازات، اجتماعات، أخبار، إعلانات، سياسات أو مزايا؟"
                )
            if focus_name:
                return finish(
                    f"I understand your question is about {focus_name}. "
                    "Do you mean their profile, manager, attendance, leave, or availability?"
                )
            return finish(
                "I understand this is an RCRC system request, but I need to identify the exact service: "
                "employee, department, attendance, leave, meetings, news, announcements, policies, or benefits?"
            )

        if request.clarification_reason == "missing_date_range":
            state["pending"] = {
                "kind": "missing_date_range",
                "query": "",
                "candidates": [],
                "request": request.model_dump(),
                "message": message,
            }
            if language == "ar":
                if request.intent == "employee_availability":
                    prompt = "أكيد، أي يوم أو فترة تبيني أشيك لك؟ اليوم، بكرة، أو تاريخ محدد؟"
                elif request.intent in {"meetings", "meeting_list"}:
                    prompt = "أكيد 😊 متى تبي أشيك اجتماعاتك؟ اليوم، بكرة، أو يوم معيّن؟"
                elif request.intent in {"attendance_today", "attendance_history"}:
                    prompt = "أكيد، أي يوم تبيني أشيك حضورك فيه؟ اليوم، أمس، أو تاريخ محدد؟"
                elif request.intent == "working_hours":
                    prompt = "أكيد، أي فترة تبيني أشيك ساعات دوامك فيها؟ اليوم، أمس، أو فترة محددة؟"
                else:
                    prompt = "أكيد، وش الفترة المطلوبة؟ مثل: هذا الشهر، الشهر الماضي، أو تاريخ بداية ونهاية."
            else:
                if request.intent == "employee_availability":
                    prompt = "Sure — what day or time period should I check: today, tomorrow, or a specific date?"
                elif request.intent in {"meetings", "meeting_list"}:
                    prompt = "Sure 😊 What day would you like me to check your meetings — today, tomorrow, or a specific date?"
                elif request.intent in {"attendance_today", "attendance_history", "working_hours"}:
                    prompt = "Sure — which day or period should I check?"
                else:
                    prompt = "Sure — which period do you want?"
            return finish(prompt)

        # For own meetings, ask for the desired part of the selected day before
        # calling the calendar API. Persist that selected day in BOTH pending
        # and meeting_context so the next short reply ("الصباح") cannot fall
        # back to today.
        if (
            request.intent in {"meetings", "meeting_list"}
            and request.clarification_reason == ""
            and request.period != "none"
            and not str(getattr(request, "time_query", "") or "").strip()
            and str((state.get("meeting_context") or {}).get("mode") or "") == "self_meetings_wizard"
            and str((state.get("pending") or {}).get("kind") or "") == "meeting_time_window"
        ):
            state["meeting_context"] = {
                "subject_kind": "self",
                "employee": {},
                "period": request.period,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "time_query": "",
                "mode": "self_meetings",
            }
            _record_request(state, request)
            state["pending"] = {
                "kind": "meeting_time_window",
                "query": "",
                "candidates": [],
                "request": request.model_dump(),
                "message": message,
            }

            if language == "ar":
                selected_day = "اليوم"
                if request.period == "tomorrow":
                    selected_day = "بكرة"
                elif request.period == "custom" and request.start_date:
                    try:
                        from datetime import datetime as _dt
                        day_idx = _dt.fromisoformat(str(request.start_date)).weekday()
                        selected_day = {
                            0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء",
                            3: "الخميس", 4: "الجمعة", 5: "السبت", 6: "الأحد",
                        }.get(day_idx, "اليوم المحدد")
                    except Exception:
                        selected_day = "اليوم المحدد"
                elif request.period not in {"today", "tomorrow"}:
                    selected_day = "الفترة المحددة"
                return finish(f"تمام، تبيني أشيك لك {selected_day} الصباح، بعد الظهر، أو المساء؟")

            return finish("Sure — should I check the morning, afternoon, or evening?")

        if request.clarification_reason == "missing_contact_subject":
            return finish(
                "تقصد بيانات مين؟ اكتب اسم الموظف."
                if language == "ar"
                else "Whose contact details do you mean? Please enter the employee name."
            )

        # Closed clarification for "a specific department" without its name.
        if (
            request.intent == "general_department_counts"
            and request.clarification_reason == "missing_general_department"
        ):
            return finish(
                "أكيد، أي إدارة تقصد؟ اكتب اسمها."
                if language == "ar"
                else "Sure — which department do you mean? Please give its name."
            )

        # RCRC-FIX-v151: natural follow-up after one verified employee result.
        # Example:
        #   user: "ابحث لي عن عدنان العقيلي" -> exactly one verified result
        #   user: "ابي ايميله ورقم جواله"
        # The second turn must continue with that employee instead of asking
        # "تقصد أي موظف؟". This is backend-owned and does not rely on Qwen.
        if request.intent == "employee_details" and not [t for t in request.targets if t.kind != "none"]:
            focus_state = state.get("focus") if isinstance(state.get("focus"), dict) else {}
            current_focus = focus_state.get("current") if isinstance(focus_state.get("current"), dict) else {}
            list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
            list_items = [x for x in (list_state.get("items") or []) if isinstance(x, dict)]

            if _uid(current_focus):
                request.targets = [TargetRef(kind="focus", name="")]
                request.clarification_reason = ""
                request.confidence = max(float(request.confidence or 0.0), 0.99)
                print(
                    f"[Follow-up Target] source='focus' username={_uid(current_focus)!r} fields={request.fields!r}",
                    file=sys.stderr,
                )
            elif str(list_state.get("intent") or "") in {"employee_search", "employee_role_search"} and len(list_items) == 1 and _uid(list_items[0]):
                # Repair focus from the verified single-result list if older state
                # missed it for any reason, then resolve through the normal focus path.
                _remember_entity(state, list_items[0], set_focus=True)
                request.targets = [TargetRef(kind="focus", name="")]
                request.clarification_reason = ""
                request.confidence = max(float(request.confidence or 0.0), 0.99)
                print(
                    f"[Follow-up Target] source='single_list_result' username={_uid(list_items[0])!r} fields={request.fields!r}",
                    file=sys.stderr,
                )

        # v203: "مديري / مين مديري / my manager" always means the
        # authenticated user's manager when no explicit employee target exists.
        # Do this before generic person-scoped validation so we never ask
        # "تقصد لك أنت ولا لموظف معين؟" for an unambiguous self-manager request.
        if request.intent == "manager" and not [t for t in request.targets if t.kind != "none"]:
            request.targets = [TargetRef(kind="self", name="")]
            request.clarification_reason = ""
            request.confidence = max(float(request.confidence or 0.0), 0.99)
            print(
                f"[Manager Self Route] fields={request.fields!r} depth={request.relation_depth}",
                file=sys.stderr,
            )

        validation = validate_request(request, state, language=language)
        if not validation.ok:
            # RCRC-FIX-v2: keep the question so "انا" / a name can complete it.
            pending_now = state.get("pending") if isinstance(state.get("pending"), dict) else {}
            if (
                not str(pending_now.get("kind") or "")
                and request.intent in _PERSON_SCOPED_INTENTS
                and not [t for t in request.targets if t.kind != "none"]
            ):
                state["pending"] = {
                    "kind": "missing_employee_target", "query": "", "candidates": [],
                    "request": request.model_dump(), "message": message,
                }
            return finish(validation.clarification)
        request = validation.request
        _prepare_memory_for_request(state, request)

        if request.intent == "general":
            reason = "writing" if str(request.query or "") == "general:writing" else "semantic_parser"
            print(f"[Route] general reason={reason} msg={message[:80]!r}", file=sys.stderr)
            facts = await self._writing_facts(message, employee_context, language, state) if reason == "writing" else {}
            if reason == "writing" and _is_email_drafting_request(message):
                answer = await self._email_draft_answer(message, language, facts)
                print("[Email Draft Route] direct_writing", file=sys.stderr)
            else:
                answer = await self._general_answer(message, state, language, facts)
            _record_request(state, request)
            return finish(answer)

        if request.intent == "web_search":
            answer, sources = await self._web_answer(message, language)
            _record_request(state, request)
            return finish(answer, sources=sources)

        if request.intent == "summarize_previous":
            list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}
            answer = await summarize_verified(
                [x for x in (list_state.get("items") or []) if isinstance(x, dict)],
                str(list_state.get("intent") or ""),
                language,
            )
            _record_request(state, request)
            return finish(answer)

        if employee_context is None:
            return finish("سجّل الدخول أول عشان أقدر أوصل لبيانات الهيئة." if language == "ar" else "Please sign in first so I can access RCRC data.")

        # Current production restriction: meeting/availability/attendance/working
        # hours are bound to the authenticated token owner. Keep directory focus
        # and manager features intact, but never call these private endpoints for
        # another employee until the upstream API supports it again.
        self_only_runtime_intents = {
            "meetings", "meeting_list", "employee_availability",
            "attendance_today", "attendance_history", "working_hours", "team_attendance",
        }
        if request.intent in self_only_runtime_intents:
            non_self_target = any(t.kind not in {"self", "none"} for t in request.targets)
            if non_self_target:
                if language == "ar":
                    if request.intent in {"meetings", "meeting_list", "employee_availability"}:
                        return finish(
                            "حاليًا خدمة الاجتماعات والتوفر مرتبطة بحساب الشخص المسجّل دخوله، "
                            "فأقدر أشيك جدولك أنت فقط. أول ما يتوفر لنا الوصول لباقي الموظفين نرجع نفعلها."
                        )
                    return finish(
                        "حاليًا بيانات الحضور وساعات الدوام مرتبطة بحساب الشخص المسجّل دخوله، "
                        "فأقدر أشيك بياناتك أنت فقط."
                    )
                return finish(
                    "This service is currently limited to the signed-in employee because the upstream API is token-owner only."
                )

        # Resolve employee references before tool execution. No model-generated
        # username is ever trusted.
        resolved_targets: list[ResolvedTarget] = []
        # RCRC-FIX-v10: "ماجد السلطان" -> a person already verified in this chat
        # ("ماجد محمد ابراهيم بن سلطان") before any directory search.
        for target in request.targets:
            if target.kind == "employee" and target.name.strip():
                known = _match_known_person(target.name, state)
                known_name = str(known.get("name_ar") or known.get("name") or known.get("name_en") or "").strip()
                if known_name and known_name != target.name:
                    print(f"[Name Match] {target.name!r} -> known {known_name!r}", file=sys.stderr)
                    target.name = known_name
        if request.targets:
            try:
                resolution = await self.resolver.resolve(request, state, employee_context, language=language)
                # RCRC-FIX-v10: not found -> retry spelling variants ("السلطان" -> "بن سلطان"/"سلطان").
                pending_after = state.get("pending") if isinstance(state.get("pending"), dict) else {}
                employee_targets = [t for t in request.targets if t.kind == "employee"]
                if (
                    resolution.clarification
                    and len(employee_targets) == 1
                    and str(pending_after.get("kind") or "") != "employee_disambiguation"
                ):
                    original_name = employee_targets[0].name
                    for variant in _name_search_variants(original_name):
                        retry = request.model_copy(deep=True)
                        for t in retry.targets:
                            if t.kind == "employee" and t.name == original_name:
                                t.name = variant
                        retry_resolution = await self.resolver.resolve(retry, state, employee_context, language=language)
                        print(f"[Name Variant] {original_name!r} -> {variant!r} ok={not retry_resolution.clarification}", file=sys.stderr)
                        retry_pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
                        if (
                            not retry_resolution.clarification
                            or retry_resolution.targets
                            or str(retry_pending.get("kind") or "") == "employee_disambiguation"
                        ):
                            # found, or found a few close people to choose from
                            resolution = retry_resolution
                            request = retry
                            break
            except EmployeeAuthenticationError:
                return finish("انتهت صلاحية الدخول. سجّل الدخول من جديد." if language == "ar" else "Your sign-in session has expired. Please sign in again.")
            except EmployeeAuthorizationError:
                return finish("ما عندك صلاحية للوصول لهذي المعلومة." if language == "ar" else "You don't have permission to access that information.")
            except EmployeeAPIError as exc:
                print(f"[Resolver API error] {exc}", file=sys.stderr)
                return finish("آسف 😅 أنا لسه بمرحلتي التجريبية وما قدرت أتحقق من بيانات الموظفين هالمرة، فما راح أعرض لك بيانات غير مؤكدة." if language == "ar" else "Sorry, I couldn't reach employee information this time. I'm still in my experimental phase and continuing to learn and improve.")

            if resolution.clarification:
                return finish(resolution.clarification)
            resolved_targets = resolution.targets
            if resolved_targets:
                _clear_pending(state)

        # Natural availability flow:
        # "متى فاضي بكرة؟" -> first ask which part of the day, instead of dumping
        # a long calendar list. Exact-time checks still execute immediately.
        if (
            request.intent == "employee_availability"
            and str(request.query or "free").startswith("free")
            and not str(request.time_query or "").strip()
            and not str(state.get("availability_preference_window") or "").strip()
            and resolved_targets
        ):
            verified_target = resolved_targets[0]
            employee = verified_target.employee if isinstance(verified_target.employee, dict) else {}
            if employee and _uid(employee):
                display_name = _first_name(_name(employee, language) or _name(employee, "ar"))
                state["task_flow"] = {
                    "kind": "availability_preference",
                    "stage": "window",
                    "period": request.period if request.period in {"today", "tomorrow"} else "today",
                    "target": _bounded_employee(employee),
                    "original_message": str(message or "")[:1000],
                }
                if language == "ar":
                    who = display_name or "هو"
                    day = "اليوم" if request.period == "today" else "بكرة" if request.period == "tomorrow" else "هالفترة"
                    return finish(
                        f"أكيد، أبغى أحددها لك بشكل أدق. تبي أشوف متى {who} فاضي {day} الصباح، الظهر/العصر، ولا المساء؟",
                        save_state=True,
                    )
                who = display_name or "they"
                day = "today" if request.period == "today" else "tomorrow" if request.period == "tomorrow" else "then"
                return finish(
                    f"Sure — do you want me to check when {who} is free {day} in the morning, afternoon, or evening?",
                    save_state=True,
                )

        # If the user already chose a time window, carry it into the renderer.
        availability_window = str(state.pop("availability_preference_window", "") or "").strip()
        if request.intent == "employee_availability" and availability_window:
            base_mode = "free" if str(request.query or "free").startswith("free") else str(request.query or "busy")
            request.query = f"{base_mode}:{availability_window}"

        if str(request.query or "") == "general:list_selection_out_of_range":
            items = (
                state.get("list_state", {}).get("items", [])
                if isinstance(state.get("list_state"), dict)
                else []
            )
            count = len(items) if isinstance(items, list) else 0
            answer = (
                f"الرقم اللي اخترته مو موجود في القائمة 😊 اختر رقم من 1 إلى {count}."
                if language == "ar"
                else f"That number isn't in the list. Choose a number from 1 to {count}."
            )
        else:
            answer = await self._execute_request(request, resolved_targets, employee_context, language, state)
        if request.intent != "general":
            answer = polish_response(answer, language=language)

        # v235 Displayed-Offer Guarantee:
        # If سند visibly offers to show hidden Tawasul items, persist that exact
        # action here, at the final answer boundary. This keeps UI text and
        # conversation state synchronized even if an inner renderer/execution
        # branch did not save pending state for any reason.
        if (
            request.intent in {"meetings", "meeting_list"}
            and str(request.query or "").strip() != "meetings:tawasul"
            and isinstance(answer, str)
            and (
                "إذا تحب أعرضها لك" in answer
                or "إذا تحب، أعرضها لك" in answer
                or "I can show them if you'd like" in answer
            )
        ):
            follow_req = request.model_copy(deep=True)
            follow_req.query = "meetings:tawasul"
            follow_req.clarification_reason = ""
            follow_req.confidence = 1.0
            state["pending"] = {
                "kind": "tawasul_events_offer",
                "query": "",
                "candidates": [],
                "request": follow_req.model_dump(),
                "message": "displayed_tawasul_offer",
            }
            print(
                f"[Displayed Offer Saved] kind='tawasul_events_offer' "
                f"period={request.period!r} start={request.start_date!r} "
                f"end={request.end_date!r}",
                file=sys.stderr,
            )

        # v241 Tarasol displayed-offer guarantee:
        # Keep UI text and conversation state synchronized. If the answer that
        # actually reached the user asks whether to show Tarasol details, the
        # next short confirmation must resolve to tarasol_details.
        if (
            request.intent == "tarasol_count"
            and isinstance(answer, str)
            and (
                "تحب أعرض لك تفاصيلها؟" in answer
                or "تحب أعرض لك التفاصيل؟" in answer
                or "Would you like me to show the details?" in answer
            )
        ):
            follow_req = request.model_copy(deep=True)
            follow_req.intent = "tarasol_details"
            follow_req.operation = "details"
            follow_req.targets = [TargetRef(kind="self", name="")]
            follow_req.fields = ["all"]
            follow_req.query = "tarasol:self_details"
            follow_req.clarification_reason = ""
            follow_req.confidence = 1.0
            state["pending"] = {
                "kind": "tarasol_details_offer",
                "query": "",
                "candidates": [],
                "request": follow_req.model_dump(),
                "message": "displayed_tarasol_offer",
            }
            print(
                "[Displayed Offer Saved] kind='tarasol_details_offer'",
                file=sys.stderr,
            )

        if isinstance(answer, str):
            answer = answer.replace("**", "")

        focus_after = state.get("focus") if isinstance(state.get("focus"), dict) else {}
        current_after = focus_after.get("current") if isinstance(focus_after.get("current"), dict) else {}
        print(
            f"[Focus After Execute] username={_uid(current_after)!r} name={_name(current_after, language)!r}",
            file=sys.stderr,
        )
        _record_request(state, request)
        return finish(answer)

    async def ask(
        self,
        *,
        user_message: str,
        history: list[dict] | None = None,
        employee_context: EmployeeContext | None = None,
        conversation_id: str | None = None,
    ) -> tuple[str, list[dict], list[str]]:
        """Controlled agentic entry point.

        A normal turn keeps the proven single-request path unchanged. An
        explicitly compound turn is planned into a small bounded sequence and
        executed left-to-right against the SAME verified conversation state.
        This is what lets "ابحث عن عدنان وعطني ايميله وجواله ومديره" work
        without giving the model direct control of API identifiers.
        """
        original = str(user_message or "").strip()
        if not original:
            return "وش حاب تسأل؟", [], []

        requested_id = str(conversation_id or "").strip().lower()
        if re.fullmatch(r"c_[a-f0-9]{24}", requested_id):
            conv_id = requested_id
        else:
            conv_id = "c_" + secrets.token_hex(12)

        plan = _split_compound_message(original)
        if len(plan) <= 1:
            return await self._ask_single(
                user_message=original,
                history=history,
                employee_context=employee_context,
                conversation_id=conv_id,
            )

        username = employee_context.username if employee_context else "anonymous"
        memory_key = f"{username}:{conv_id}"
        self.memory.append_message(memory_key, "user", original)
        print(f"[Agent Plan] steps={plan!r}", file=sys.stderr)

        answers: list[str] = []
        sources: list[str] = []
        for index, step in enumerate(plan, start=1):
            answer, _, step_sources = await self._ask_single(
                user_message=step,
                history=history,
                employee_context=employee_context,
                conversation_id=conv_id,
                record_user=False,
                record_assistant=False,
            )
            if answer:
                answers.append(answer)
            sources.extend(step_sources or [])
            print(f"[Agent Step] {index}/{len(plan)} done", file=sys.stderr)

            # If a step opens a clarification, do not execute later dependent
            # steps with incomplete context. The pending state will resume on the
            # user's next message.
            state = self.memory.get_state(memory_key)
            pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
            if str(pending.get("kind") or "").strip():
                break

        language = detect_language(original, fallback="ar")
        combined = polish_response(_combine_compound_answers(answers, language), language=language)
        self.memory.append_message(memory_key, "assistant", combined)
        return combined, [], list(dict.fromkeys(sources))

