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
from copy import deepcopy
from typing import Any

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
from core.semantic_parser import SEMANTIC_PARSER_BUILD, SemanticParser, detect_language, _extract_leave_amount, _person_slot_target
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


ORCHESTRATOR_BUILD = "rcrc-v163-context-api-routing-fixes-2026-10-03"


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
                "أنا RCRC.AI، مساعدك الذكي في الهيئة 😊 "
                "أساعدك في معلومات الموظفين، الإجازات، الحضور، الاجتماعات، تراسل، "
                "المزايا، الأخبار والسياسات. وأنا حاليًا في مرحلتي التجريبية وقاعد أتطور باستمرار."
            )
        return (
            "I'm RCRC.AI, your AI assistant at RCRC 😊 I can help with employee information, "
            "leave, attendance, meetings, Tarasol, benefits, news, and policies. "
            "I'm currently in my experimental phase and continuously improving."
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
    """Shortcut to the no-tools general model ONLY for clear small talk or
    explicit writing/advice requests. Everything else goes through the
    enterprise router/parser. A short message is NOT a reason to skip the API."""
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
                "ما يظهر في النظام موظفين تابعين لك مباشرة."
                if self_subject
                else f"ما يظهر في النظام موظفين تابعين لـ {subject_name} مباشرة."
            )
        return (
            "The system doesn't show any employees reporting directly to you."
            if self_subject
            else f"The system doesn't show any employees reporting directly to {subject_name}."
        )
    if language == "ar":
        header = (
            f"الموظفين اللي تحتك مباشرة ({len(rows)}):"
            if self_subject
            else f"الموظفين اللي تحت {subject_name} مباشرة ({len(rows)}):"
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


def _critical_orchestrator_preflight(message: str, state: dict[str, Any], language: str) -> ParsedRequest | None:
    """Second safety boundary for critical follow-ups before Qwen.

    This intentionally covers only closed, high-value grammar: authenticated
    self references, gender-count mode, unique selection from the last verified
    search list, and focus-only availability/date follow-ups.
    """
    raw = " ".join(str(message or "").strip().split())
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

    def make(**kwargs: Any) -> ParsedRequest:
        base = dict(
            intent="general", operation="chat", targets=[], fields=[], query="",
            role_query="", department_query="", period="none", start_date="",
            end_date="", ordinal=0, relation_depth=1, amount=None, time_query="",
            confidence=0.99, clarification_reason="",
        )
        base.update(kwargs)
        return ParsedRequest(**base)

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

    # RCRC-FIX-v121: finish a previous date-range clarification using a short
    # period-only reply such as "الشهر الماضي" without forcing the user to
    # repeat the original leave/vacation request.
    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}
    if str(pending.get("kind") or "") == "missing_date_range":
        raw_req = pending.get("request") if isinstance(pending.get("request"), dict) else {}
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

    if re.search(r"(?:رصيد\s+(?:اجازتي|إجازتي)|كم\s+رصيد\s+(?:اجازتي|إجازتي)|my\s+(?:leave|vacation)\s+balance)", norm, re.I):
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
                r"(?:اجتماعاتي|اجتماعي|مواعيدي|جدولي|"
                r"جدول\s+اجتماعاتي|قائمه\s+اجتماعاتي|قائمة\s+اجتماعاتي|"
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
            period = ""
            if re.search(r"(?:بكره|بكرة|غدا|غداً|tomorrow)", norm, re.I):
                period = "tomorrow"
            elif re.search(r"(?:اليوم|today)", norm, re.I):
                period = "today"
            elif re.search(r"(?:الاسبوع\s+(?:هذا|الحالي)|this\s+week)", norm, re.I):
                period = "this_week"
            return make(
                intent="meetings", operation="list",
                targets=[TargetRef(kind="self", name="")],
                fields=["subject", "start_time", "end_time"],
                period=period or "none", query="meetings:self",
                clarification_reason="" if period else "missing_date_range",
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

        # Explicit overlap/intersection request also requires a user-selected
        # date range. The deployed endpoint currently returns 404, but we still
        # collect the correct request before reporting service unavailability.
        if re.search(
            r"(?:متداخل|متداخله|متقاط|تعارض|تقاطع|overlap|intersect|conflict)",
            norm, re.I,
        ):
            return make(
                intent="leave_overlap", operation="lookup",
                targets=[TargetRef(kind="self", name="")],
                fields=["all"], period="none", query="leave:self_overlap",
                clarification_reason="missing_date_range",
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
        re.search(r"(?:حضور|دوام|attendance)", norm, re.I)
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
        return make(
            intent="team_attendance", operation="list",
            targets=[TargetRef(kind="self", name="")],
            fields=["all"], period=period or "none", query="attendance:team",
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
    if not dept_query:
        m_dep = re.search(r"((?:ال)?اداره(?:\s+(?:ال)?عامه)?\s+[^؟?!.،,]+)", norm, re.I)
        if m_dep:
            dept_query = str(m_dep.group(1) or "").strip()
    # RCRC-FIX-v4: "كم عدد الموظفين في إدارة X" is a count, not a list.
    if re.search(r"(?:موظف|موظفين|موظفي|موظفو|employees?|staff)", norm, re.I) and dept_query and re.search(r"(?:كم|عدد|اجمالي|how\s+many|count)", norm, re.I):
        names_too = bool(re.search(r"(?:اسماء|اسامي|اسمائهم|names)", norm, re.I))
        return make(
            intent="general_department_counts", operation="calculate", targets=[],
            fields=["count"] + (["name"] if names_too else []), department_query=dept_query,
            query="department:count",
        )
    if re.search(r"(?:موظف|موظفين|موظفي|موظفو|employees?|staff)", norm, re.I) and dept_query:
        return make(
            intent="general_department_members", operation="list", targets=[],
            fields=["name", "job_title"], department_query=dept_query,
            query="department:members",
        )
    if str(last_req_for_units.get("intent") or "") == "general_department_members" and dept_query and len(norm.split()) <= 8:
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

    # Policies/procedures are their own enterprise domain. Never let a phrase
    # such as "ابحث عن سياسة العمل عن بعد" fall through to employee search.
    policy_cue = bool(re.search(
        r"(?:سياس(?:ه|ة|ات)|اجراء(?:ات)?|إجراء(?:ات)?|policy|policies|procedure|procedures)",
        norm, re.I,
    ))
    if policy_cue:
        policy_query = raw
        policy_query = re.sub(
            r"^(?:(?:ابحث(?:\s+لي)?\s+عن|دور\s+على|اعرض(?:\s+لي)?|ابي|ابغى|وش|ايش|"
            r"search\s+(?:for\s+)?|find\s+|show\s+(?:me\s+)?|what\s+(?:are|is)\s+the\s+))",
            "", policy_query, flags=re.I,
        ).strip()
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
        recent_mentioned = recent_employee_mentioned(norm)
        if recent_mentioned:
            explicit_name = str(
                recent_mentioned.get("name_ar")
                or recent_mentioned.get("name")
                or recent_mentioned.get("name_en")
                or ""
            ).strip()
            if explicit_name:
                return make(
                    intent="employee_details", operation="details",
                    targets=[TargetRef(kind="employee", name=explicit_name)],
                    fields=list(dict.fromkeys(named_detail_fields)),
                    query="employee:explicit_fields_recent",
                )

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
        r"(?:حضوري|الحضور|جدول\s+حضوري|متي\s+(?:حضرت|دخلت|جيت|وصلت|طلعت|خرجت|انصرفت)|"
        r"وقت\s+(?:دخولي|خروجي)|ساعات\s+(?:دوامي|عملي)|check[- ]?in|check[- ]?out|attendance|working\s+hours)",
        norm, re.I,
    ))
    if attendance_followup:
        explicit_period = None
        if re.search(r"(?:امس|yesterday)", norm, re.I): explicit_period = "yesterday"
        elif re.search(r"(?:الاسبوع\s+(?:الي|اللي)?\s*راح|الاسبوع\s+الماضي|last\s+week)", norm, re.I): explicit_period = "last_week"
        elif re.search(r"(?:اليوم|today)", norm, re.I): explicit_period = "today"
        # A full new attendance request defaults to today. Only a short field
        # follow-up such as "متى خرجت؟" inherits yesterday/last week.
        full_attendance_request = bool(re.search(r"(?:حضوري|الحضور|جدول\s+حضوري|ساعات\s+(?:دوامي|عملي)|attendance|working\s+hours)", norm, re.I))
        inherited = str(attendance_context.get("period") or "today") if not full_attendance_request else ""
        period = explicit_period or inherited
        checkout = bool(re.search(r"(?:طلعت|خرجت|انصرفت|خروجي|check[- ]?out|leave)", norm, re.I))
        checkin = bool(re.search(r"(?:حضرت|دخلت|جيت|وصلت|دخولي|check[- ]?in|arrive)", norm, re.I))
        working = bool(re.search(r"(?:ساعات\s+(?:دوامي|عملي)|working\s+hours)", norm, re.I))
        intent = "working_hours" if working else ("attendance_today" if period in {"", "today"} else "attendance_history")
        query = "attendance:working_hours" if working else ("attendance:check_out" if checkout else ("attendance:check_in" if checkin else "attendance:summary"))
        return make(
            intent=intent, operation="lookup", targets=[TargetRef(kind="self", name="")],
            fields=["all"], period=period or "none", query=query,
            clarification_reason="missing_date_range" if full_attendance_request and not explicit_period else "",
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
    """RCRC-FIX-v2: short, natural clarification with an example close to what was asked."""
    norm = _norm_role_text(message)
    if language != "ar":
        return "Sorry, I didn't quite get that 😅 Could you rephrase? For example: \"Who is Shahah's manager?\" or \"What's my leave balance?\""
    examples = (
        (r"(?:اجاز|رصيد|يبقى|يبقي|خذت)", "«كم رصيد إجازاتي؟» أو «إذا أخذت يومين إجازة كم يبقى لي؟»"),
        (r"(?:مدير)", "«مين مديري؟» أو «مين مدير شاهه العقيل؟»"),
        (r"(?:اجتماع|فاضي|متاح|مشغول)", "«اجتماعاتي بكرة» أو «متى عماد فاضي بكرة؟»"),
        (r"(?:ايميل|بريد|جوال|رقم|تحويل)", "«ايميل عماد الهاجري» أو «رقم جوال شاهه»"),
        (r"(?:معامل|تراسل)", "«كم معاملة عندي؟» أو «تفاصيل معاملاتي»"),
    )
    for pattern, example in examples:
        if re.search(pattern, norm):
            return f"آسف، ما فهمت قصدك بالضبط 😅 ممكن توضح أكثر؟ مثلًا: {example}"
    return "آسف، ما فهمت قصدك بالضبط 😅 ممكن توضح أكثر؟ مثلًا: «مين مديري؟» أو «ايميل عماد الهاجري» أو «آخر الأخبار»."


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
            return "هلا فيك! كيف أقدر أساعدك؟"

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
            return "أكيد، اسألني."

        if norm in greeting_words:
            return "هلا فيك! كيف أقدر أساعدك؟"

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
            return "هلا فيك! بخير الحمدلله 😊 كيف أقدر أساعدك؟"

        # Do not swallow a real request just because it starts with a greeting.
        # Examples that must continue to NLU:
        #   "اهلا تعرفني؟"
        #   "هلا مين مديري؟"
        # Pure greetings were already handled by `norm in greeting_words` above.
        if has_small_talk and len(norm.split()) <= 5:
            return "بخير الحمدلله 😊 كيف أقدر أساعدك؟"

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
        return "ما لقيت موظفين مطابقين." if language == "ar" else "I couldn't find matching employees."

    lines = (["النتائج المطابقة:" if language == "ar" else "Matching results:"] if header else [])
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
            return "السياسات والإجراءات تلقينها داخل التطبيق من **المزيد → السياسات والإجراءات**. وإذا كنتِ تقصدين سياسة معيّنة، اكتبي اسمها وأنا أساعدك فيها."
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
        return "هههه حقك علي 😅 حوّستها عليك. خلينا نمشيها طبيعي—قولي طلبك بطريقتك وأنا بكمل معك من السياق بدون أسئلة محفوظة."
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
        key = re.sub(r"\s+", " ", text).casefold()
        if key and key not in seen:
            seen.add(key)
            clean.append(text)
    return "\n\n".join(clean)


def _friendly_surface(answer: str, language: str) -> str:
    """Keep verified replies natural without mechanically adding prefixes/emojis."""
    text = str(answer or "").strip()
    if not text:
        return text

    # Do not mutate English or factual content. The renderers/prompts own tone.
    if language != "ar":
        return text

    # Remove accidental duplicate friendly openers produced by layered renderers.
    text = re.sub(r"^(?:أكيد\s*[🌷😊👌]?\s*){2,}", "أكيد، ", text).strip()

    # Turn the common three-line employee card into a natural sentence while
    # preserving the exact verified values returned by the renderer. Lists and
    # large structured answers stay structured because that is easier to scan.
    profile_card = re.fullmatch(
        r"أكيد،?\s*عن\s+(.+?):\s*\n-\s*الاسم:\s*(.+?)\s*\n-\s*المسمى\s+الوظيفي:\s*(.+?)\s*\n-\s*الإدارة:\s*(.+?)\s*",
        text,
        re.S,
    )
    if profile_card:
        name = profile_card.group(2).strip() or profile_card.group(1).strip()
        title = profile_card.group(3).strip()
        department = profile_card.group(4).strip()
        return f"أكيد، {name} يعمل كـ {title} ضمن {department}."

    text = text.replace("تقصد أي موظف؟", "مين تقصد بالضبط؟ اكتب لي اسم الموظف.")

    # Never prepend a generic word/emoji to every API answer. A friendly assistant
    # can also be direct: "مديرك المباشر..." / "عندك اجتماعين..." etc.
    return text


def _clear_pending(state: dict[str, Any]) -> None:
    state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}


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
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("employees", "items", "results", "data", "rows", "meetings", "transactions"):
        value = data.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


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
        print(f"[Orchestrator] Enterprise tools: {self.tool_names}", file=sys.stderr)
        print(f"[Runtime] orchestrator_file={__file__}", file=sys.stderr)
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
                "أهلًا 👋\nأنا مساعدك الذكي في الهيئة، أساعدك في الوصول للمعلومات والخدمات اليومية بسهولة، مثل الإجازات والاجتماعات وغيرها.\nحاليًا أنا في المرحلة التجريبية 😊 كيف أقدر أساعدك اليوم؟"
                if language == "ar"
                else "Hi 👋 I'm your AI assistant at RCRC. I can help you access everyday information and services, including leave and meetings. I'm currently in the experimental phase 😊 How can I help today?"
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
                "أهلًا 👋\nأنا مساعدك الذكي في الهيئة، أساعدك في الوصول للمعلومات والخدمات اليومية بسهولة، مثل الإجازات والاجتماعات وغيرها.\nحاليًا أنا في المرحلة التجريبية 😊 كيف أقدر أساعدك اليوم؟"
                if language == "ar"
                else "Hi 👋 I'm your AI assistant at RCRC. I can help you access everyday information and services, including leave and meetings. I'm currently in the experimental phase 😊 How can I help today?"
            )
            return {"message": message, "name": ""}

        full_name = _name(data, language) or _name(data, "ar") or context.username
        first_name = str(full_name).strip().split()[0] if str(full_name).strip() else context.username
        message = (
            f"أهلًا {first_name} 👋\nأنا مساعدك الذكي في الهيئة، أساعدك في الوصول للمعلومات والخدمات اليومية بسهولة، مثل الإجازات والاجتماعات وغيرها.\nحاليًا أنا في المرحلة التجريبية 😊 كيف أقدر أساعدك اليوم؟"
            if language == "ar"
            else f"Hi {first_name} 👋 I'm your AI assistant at RCRC. I can help you access everyday information and services, including leave and meetings. I'm currently in the experimental phase 😊 How can I help today?"
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
            answer = enterprise_clarification(language)
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
                "حقك علي 😅 شكلي ما التقطت قصدك صح. خلني أكمل معك من نفس السياق—قولها بطريقتك وأنا أضبطها."
                if language == "ar"
                else "I'm currently in my experimental phase 😊 I couldn't prepare a good response to that request just now. Try phrasing it another way and I'll try again."
            )

        history = list(history[-8:])
        history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        state["general_history"] = history[-10:]
        return answer

    async def _web_answer(self, message: str, language: str) -> tuple[str, list[str]]:
        if not getattr(settings, "enable_web_search", False) or "web_search" not in self._mcp_tool_names:
            return ("البحث في الويب غير متاح حاليًا." if language == "ar" else "Web search is currently unavailable."), []
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
            return ("ما قدرت أوصل لنتائج الويب حاليًا." if language == "ar" else "I couldn't retrieve web results right now."), []
        results = [x for x in (data.get("results") or []) if isinstance(x, dict)][:8]
        if not results:
            return ("ما لقيت نتائج مناسبة في الويب." if language == "ar" else "I couldn't find relevant web results."), []
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

    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any], context: EmployeeContext, target_username: str | None = None) -> dict[str, Any]:
        target = str(target_username or context.username or "").strip()
        ttl = self.cache.ttl_for(tool_name)
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
                print(f"[Tool Cache HIT] tool={tool_name!r} target={target!r}", file=sys.stderr)
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
                print(f"[Tool Cache SET] tool={tool_name!r} target={target!r} ttl={ttl}", file=sys.stderr)
            return {"success": True, "data": data}
        except EmployeeAuthenticationError:
            return {"success": False, "error_type": "auth", "error": "authentication"}
        except EmployeeAuthorizationError:
            return {"success": False, "error_type": "authorization", "error": "authorization"}
        except EmployeeAPIError as exc:
            return {"success": False, "error_type": "api", "error": str(exc)}
        except Exception as exc:
            print(f"[Tool unexpected error] {tool_name}: {exc}", file=sys.stderr)
            return {"success": False, "error_type": "unexpected", "error": str(exc)}

    @staticmethod
    def _tool_error(result: dict[str, Any], language: str) -> str:
        kind = result.get("error_type")
        if kind == "auth":
            return "انتهت صلاحية الدخول. سجّل الدخول من جديد." if language == "ar" else "Your sign-in session has expired. Please sign in again."
        if kind == "authorization":
            return "ما عندك صلاحية للوصول لهذي المعلومة." if language == "ar" else "You don't have permission to access that information."
        return ("ما قدرت أوصل للمعلومة الحين. جرّب بعد شوي، وإذا استمرت المشكلة نقدر نحدد أي خدمة اللي وقفت." if language == "ar" else "I couldn't reach that information right now. Try again shortly; if it continues, we can narrow down which service is failing.")

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
            args.update({"search": request.query, "start_index": 0, "batch_size": 10})
        if request.intent == "news":
            args.update({"search": request.query, "start_index": 0, "batch_size": 5})
        if request.intent == "announcements":
            args.update({"search": request.query, "start_index": 0, "batch_size": 10})
        if request.intent == "mazaya":
            args.update({"search": request.query, "start_index": 0, "batch_size": 10})
        return args

    async def _resolve_detail_item(self, request: ParsedRequest, state: dict[str, Any], context: EmployeeContext, language: str) -> tuple[dict[str, Any] | None, str]:
        row = _choose_content_item(request, state)
        if row is None:
            return None, ("تقصد أي عنصر؟" if language == "ar" else "Which item do you mean?")
        item_id = _content_item_id(row)
        if not item_id:
            return row, ""
        policy = get_policy(request.intent)
        if policy is None:
            return None, ("تعذر تحديد نوع التفاصيل." if language == "ar" else "I couldn't determine the detail type.")
        result = await self._execute_tool(policy.tool_name, {"item_id": item_id}, context)
        if not result.get("success"):
            return None, self._tool_error(result, language)
        data = result.get("data")
        return (data if isinstance(data, dict) else row), ""

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
                    return self._tool_error(result, language), None
                final = result.get("data") if isinstance(result.get("data"), dict) else {}
            username = _uid(final)
            if not username and hop < request.relation_depth - 1:
                return ("تعذر متابعة التسلسل الإداري من النظام." if language == "ar" else "I couldn't continue the management chain from the system."), None

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
                "آسف 😅 ما قدرت أقرأ قائمة الموظفين التابعين من النظام هالمرة. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار."
                if language == "ar"
                else "Sorry, I couldn't read the direct-reports list from the system this time."
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

        # If a verified resolver row already contains every requested profile field,
        # do not make a second API call just to re-fetch the same facts.
        if request.intent == "employee_details" and target.employee and _verified_row_satisfies(target.employee, request.fields, language):
            data = target.employee
            answer = render_profile(data, request, language, subject_name=subject_name, self_subject=False)
            _remember_entity(state, data, set_focus=True)
            return answer, data

        result = await self._execute_tool(policy.tool_name, self._tool_args(request, language), context, target_username=target.username)
        if not result.get("success"):
            # IntersectedVacations is currently absent in the deployed API and
            # returns HTTP 404 even from Postman. Report that accurately.
            if request.intent == "leave_overlap" and "404" in str(result.get("error") or ""):
                return (
                    "تعذر التحقق من تداخل الإجازات حاليًا لأن خدمة التحقق غير متاحة في النظام."
                    if language == "ar"
                    else "I can't check overlapping leave right now because the overlap service is unavailable."
                ), None
            # Verified resolver rows are a safe fallback only when they contain
            # every requested field. Never synthesize missing values.
            if request.intent == "employee_details" and target.employee and _verified_row_satisfies(target.employee, request.fields, language):
                data = target.employee
            else:
                return self._tool_error(result, language), None
        else:
            data = result.get("data")

        data = data if isinstance(data, dict) else {}

        if request.intent == "employee_image":
            image_data_url = str(data.get("image_data_url") or "").strip()
            if image_data_url.startswith("data:image/"):
                # Angular NavigationService already recognizes this marker and
                # renders a safe <img> card instead of dumping Base64 text.
                return f"[EMPLOYEE_IMAGE_DATA]{image_data_url}", data
            return (
                "الصورة الوظيفية غير متاحة حاليًا."
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
            return answer, render_data

        if request.intent == "leave_balance":
            answer = render_leave_balance(
                data,
                language,
                subject_name=subject_name,
                self_subject=self_subject,
                amount=request.amount if request.operation == "calculate" else None,
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
            answer = render_meetings(
                data, language, subject_name=subject_name, self_subject=self_subject,
                period=request.period, start_date=request.start_date, end_date=request.end_date,
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
        elif request.intent == "tarasol_count":
            answer = render_tarasol_count(data, language, subject_name=subject_name, self_subject=self_subject)
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
                "ما لقيت بيانات إدارات عامة مؤكدة في دليل الموظفين."
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
    def _render_general_department_members(data: dict[str, Any], language: str) -> str:
        if not data.get("matched"):
            return (
                "ما لقيت إدارة مطابقة بشكل مؤكد."
                if language == "ar"
                else "I couldn't find a verified matching department."
            )

        name = str(data.get("general_department") or "").strip()
        rows = [row for row in (data.get("employees") or []) if isinstance(row, dict)]
        count = len(rows)
        if language == "en":
            for row in rows:
                candidate = str(row.get("general_department_en") or row.get("department_en") or "").strip()
                if candidate:
                    name = candidate
                    break
            if re.search(r"[\u0600-\u06ff]", name): name = "the selected department"
        header = f"موظفو {name} ({count}):" if language == "ar" else f"Employees in {name} ({count}):"
        lines = [header]
        for idx, row in enumerate(rows, start=1):
            if language == "ar":
                emp_name = str(row.get("name_ar") or row.get("name") or directory_employee_name(row) or row.get("username") or "").strip()
                title = str(row.get("job_title_ar") or row.get("job_title") or "").strip()
            else:
                emp_name = str(row.get("name_en") or row.get("name") or row.get("username") or "").strip()
                title = str(row.get("job_title_en") or row.get("job_title") or "").strip()
                if re.search(r"[\u0600-\u06ff]", emp_name): emp_name = str(row.get("username") or "Employee").strip()
                if re.search(r"[\u0600-\u06ff]", title): title = ""
            lines.append(f"{idx}. {emp_name}" + (f" — {title}" if title else ""))
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
                        "بيانات الجنس غير متوفرة بشكل موثوق في دليل الموظفين حاليًا، لذلك ما أقدر أحسب عدد الإناث أو الذكور بدون تخمين."
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
                return (
                    f"عدد الموظفين في {name} هو {count} موظف."
                    if language == "ar"
                    else f"{name} has {count} employees."
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
                return "أي إدارة تقصد؟" if language == "ar" else "Which department do you mean?"

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
            return self._render_general_department_members(data, language)

        return "ممكن توضح طلبك؟" if language == "ar" else "Could you clarify your request?"

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
                return "آسف 😅 ما قدرت أوصل لبيانات الموظفين هالمرة. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار." if language == "ar" else "Sorry, I couldn\'t retrieve employee information this time. I\'m still in my experimental phase and continuing to learn and improve."

        if request.intent in DETAIL_BASE_INTENT:
            data, error = await self._resolve_detail_item(request, state, context, language)
            if error:
                return error
            return render_content_detail(request.intent, data, language)

        policy = get_policy(request.intent)
        if policy is None:
            return "ممكن توضح طلبك؟" if language == "ar" else "Could you clarify your request?"

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
                    return "ما لقيت موظفين مطابقين." if language == "ar" else "I couldn't find matching employees."
                state["list_state"] = {"intent": "employee_search", "items": rows[:20], "query": request.query}
                for row in rows[:20]:
                    _remember_entity(state, row, set_focus=False)
                if len(rows) == 1:
                    _remember_entity(state, rows[0], set_focus=True)
                return render_employee_search_results(rows, language, query=request.query, limit=20)

            search_query = _clean_employee_discovery_query(request.query)
            if not search_query:
                return "اكتب اسم الموظف أو المسمى أو الإدارة للبحث عنها." if language == "ar" else "Enter an employee name, role, or department to search for."

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
                return "آسف 😅 ما قدرت أبحث عن الموظف هالمرة. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار." if language == "ar" else "Sorry, I couldn't search for the employee this time. I'm still in my experimental phase and continuing to learn and improve."
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
                return "آسف 😅 ما قدرت أبحث عن صاحب المنصب هالمرة. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار." if language == "ar" else "Sorry, I couldn't search for the role holder this time. I'm still in my experimental phase and continuing to learn and improve."
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
                        "لقيت نتيجة/نتائج قريبة للمنصب، لكن المطابقة غير مؤكدة لذلك ما راح أخمّن.\n"
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
                    f"ما لقيت شخصًا مؤكّدًا لهذا المنصب: {request.role_query}."
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
        language = detect_language(message, fallback=fallback_language)
        state["language"] = language
        if record_user:
            self.memory.append_message(memory_key, "user", message)

        # Ephemeral language context for the semantic model. This is deliberately
        # not an authority for enterprise facts; verified state/API data remain the
        # grounding source. It lets short follow-ups behave like real conversation.
        state["_recent_dialogue"] = self.memory.recent_messages(memory_key, limit=12)
        print(f"[Conversation Memory] key={memory_key!r}", file=sys.stderr)

        def finish(answer: str, *, sources: list[str] | None = None, save_state: bool = True) -> tuple[str, list[dict], list[str]]:
            clean = _friendly_surface(str(answer or "").strip(), language)
            if save_state:
                ephemeral_dialogue = state.pop("_recent_dialogue", None)
                self.memory.save_state(memory_key, state)
                if ephemeral_dialogue is not None:
                    state["_recent_dialogue"] = ephemeral_dialogue
            if record_assistant:
                self.memory.append_message(memory_key, "assistant", clean)
            return clean, [], list(sources or [])

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

        # Friendly in-app navigation: generic policy requests point to the app
        # location instead of unnecessarily querying the policy API.
        policy_nav = _generic_policy_navigation(message, language)
        if policy_nav is not None:
            state["task_flow"] = {}
            return finish(policy_nav, save_state=True)

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

        # Conversation-first routing:
        # 1) pending explicit list selection
        # 2) semantic model with recent dialogue
        # 3) deterministic critical preflight only as fallback
        pending_request = restore_pending_selection(message, state)
        if pending_request is not None:
            request = pending_request
            print(f"[Pending Selection] intent={request.intent}", file=sys.stderr)
        else:
            semantic_text = _semantic_message(message)
            request = await self.parser.parse(semantic_text, state, language=language)
            print(f"[Semantic Request] text={semantic_text!r} {request.model_dump()}", file=sys.stderr)

            # Qwen handles open-ended meaning, while closed enterprise grammar
            # protects exact relations, requested fields, and pending date flows.
            critical_request = _critical_orchestrator_preflight(semantic_text, state, language)
            if critical_request is not None:
                cq = str(critical_request.query or "")
                strong_closed_grammar = bool(
                    cq.startswith(("manager:", "employee:focus_", "profile:self_", "profile:summary", "meetings:self", "attendance:"))
                    or critical_request.clarification_reason == "missing_date_range"
                    or (
                        critical_request.intent == request.intent
                        and set(critical_request.fields or []).issuperset(set(request.fields or []))
                        and len(critical_request.fields or []) > len(request.fields or [])
                    )
                )
                if request.intent == "general" or request.confidence < 0.60 or bool(request.clarification_reason) or strong_closed_grammar:
                    request = critical_request
                    print(f"[Critical Fallback] {request.model_dump()}", file=sys.stderr)

            # Only true general conversation goes to the no-tool assistant.
            if request.intent == "general" and _obvious_general_conversation(message, state):
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
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b)",
            temporal_norm, re.I,
        ))
        self_temporal = bool(
            request.targets
            and all(t.kind in {"self", "none"} for t in request.targets)
            and request.intent in {"meetings", "meeting_list", "employee_availability", "attendance_today", "attendance_history", "working_hours"}
        )
        if self_temporal and not has_explicit_period and str(request.period or "none") in {"", "none", "today"}:
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
                    prompt = "أكيد، تبيني أشوف اجتماعاتك متى؟ اليوم، بكرة، أو فترة محددة؟"
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
                    prompt = "Sure — when should I check your meetings: today, tomorrow, or a specific period?"
                elif request.intent in {"attendance_today", "attendance_history", "working_hours"}:
                    prompt = "Sure — which day or period should I check?"
                else:
                    prompt = "Sure — which period do you want?"
            return finish(prompt)

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
                return finish("آسف 😅 ما قدرت أوصل لبيانات الموظفين هالمرة. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار." if language == "ar" else "Sorry, I couldn't reach employee information this time. I'm still in my experimental phase and continuing to learn and improve.")

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
                        f"أكيد، أبغى أحددها لك بشكل أدق. تبي أشوف متى {who} فاضي {day} **الصباح، الظهر/العصر، ولا المساء**؟",
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

        answer = await self._execute_request(request, resolved_targets, employee_context, language, state)
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
        combined = _friendly_surface(_combine_compound_answers(answers, language), language)
        self.memory.append_message(memory_key, "assistant", combined)
        return combined, [], list(dict.fromkeys(sources))

