from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from core.employee_api import (
    EmployeeAPIClient,
    EmployeeAPIError,
    EmployeeAuthenticationError,
    EmployeeAuthorizationError,
)
from core.employee_tools import EmployeeContext


RIYADH_TIMEZONE = timezone(timedelta(hours=3))
ACTION_PENDING_KIND = "enterprise_action"

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_NORMALIZE_ARABIC = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }
)


@dataclass
class ActionWorkflowResult:
    handled: bool
    message: str = ""


def _norm(value: Any) -> str:
    text = str(value or "").translate(_ARABIC_DIGITS).casefold()
    text = text.translate(_NORMALIZE_ARABIC)
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text)
    text = re.sub(r"[^\w\s%:/-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _pending(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("pending")
    if isinstance(value, dict) and value.get("kind") == ACTION_PENDING_KIND:
        return value
    return {}


def _clear_action(state: dict[str, Any]) -> None:
    state["pending"] = {
        "kind": "",
        "query": "",
        "candidates": [],
        "request": {},
        "message": "",
    }


def _set_action(
    state: dict[str, Any],
    *,
    action: str,
    stage: str,
    data: dict[str, Any] | None = None,
    options: list[dict[str, Any]] | None = None,
    message: str = "",
) -> None:
    state["pending"] = {
        "kind": ACTION_PENDING_KIND,
        "action": action,
        "stage": stage,
        "data": dict(data or {}),
        "options": [dict(x) for x in (options or []) if isinstance(x, dict)][:100],
        "message": str(message or ""),
        "query": "",
        "candidates": [],
        "request": {},
    }


def _is_cancel(message: str) -> bool:
    value = _norm(message)
    return bool(
        re.fullmatch(
            r"(?:لا|الغاء|الغي|الغيه|الغ الطلب|كنسل|cancel|stop|nevermind|never mind)",
            value,
            flags=re.I,
        )
    )


def _is_confirm(message: str) -> bool:
    value = _norm(message)
    return bool(
        re.fullmatch(
            r"(?:اي|ايوه|ايوة|نعم|اكيد|اكد|تاكيد|تم|موافق|موافقه|yes|y|confirm|confirmed|proceed|submit)",
            value,
            flags=re.I,
        )
    )


def _looks_like_create_vacation(message: str) -> bool:
    value = _norm(message)
    vacation = bool(
        re.search(r"\b(?:اجازه|اجازة|leave|vacation)\b", value, re.I)
    )
    action = bool(
        re.search(
            r"(?:ابي|ابغى|اريد|اقدم|قدم|تقديم|اطلب|طلب|سوي|سو|انشئ|create|submit|request|apply)",
            value,
            re.I,
        )
    )
    # Do not intercept read-only leave questions.
    readonly = bool(
        re.search(
            r"(?:رصيد|كم باقي|انواع|الانواع|تاريخ|سجل|history|balance|types|رصيدي)",
            value,
            re.I,
        )
    )
    return vacation and action and not readonly


def _looks_like_create_hr_letter(message: str) -> bool:
    value = _norm(message)
    letter = bool(
        re.search(
            r"(?:خطاب|تعريف راتب|تعريف بالراتب|hr letter|salary letter|employment letter|letter)",
            value,
            re.I,
        )
    )
    action = bool(
        re.search(
            r"(?:ابي|ابغى|اريد|طلع|اطلع|اصدر|اصدار|سوي|سو|انشئ|create|generate|issue|request)",
            value,
            re.I,
        )
    )
    readonly = bool(
        re.search(
            r"(?:انواع|الانواع|نوع\s+الخطاب|انواع\s+(?:خطاب|خطابات)|"
            r"لمن|لمين|اوجه|موجه|جهه|جهات|الجهه|الجهات|"
            r"letter\s+types?|recipients?|to\s+whom|addressed\s+to)",
            value, re.I,
        )
    )
    return letter and action and not readonly


def _parse_date_token(token: str) -> str | None:
    raw = str(token or "").translate(_ARABIC_DIGITS).strip()
    if not raw:
        return None
    raw = raw.split("T", 1)[0]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _extract_dates(message: str) -> list[str]:
    value = str(message or "").translate(_ARABIC_DIGITS)
    candidates = re.findall(
        r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
        value,
    )
    out: list[str] = []
    for candidate in candidates:
        parsed = _parse_date_token(candidate)
        if parsed and parsed not in out:
            out.append(parsed)

    norm = _norm(message)
    today = datetime.now(RIYADH_TIMEZONE).date()
    if not out:
        if re.search(r"\b(?:بكره|بكرة|غدا|غداً|tomorrow)\b", norm, re.I):
            out.append((today + timedelta(days=1)).isoformat())
        elif re.search(r"\b(?:اليوم|today)\b", norm, re.I):
            out.append(today.isoformat())
    return out[:2]


def _extract_boolean_preferences(message: str) -> tuple[bool, bool]:
    value = _norm(message)
    require_tickets = bool(
        re.search(r"(?:تذاكر|تذكره|ticket|tickets)", value, re.I)
        and not re.search(r"(?:بدون|لا ابي|ما ابي|no tickets|without tickets)", value, re.I)
    )
    require_exit_return = bool(
        re.search(r"(?:خروج وعوده|خروج و عوده|exit.?return|exit return)", value, re.I)
        and not re.search(r"(?:بدون|لا ابي|ما ابي|without)", value, re.I)
    )
    return require_exit_return, require_tickets


def _row_text(row: dict[str, Any]) -> str:
    values: list[str] = []
    for key, value in row.items():
        if isinstance(value, (str, int, float)) and value not in ("", None):
            values.append(str(value))
    return _norm(" ".join(values))


def _pick_value(
    row: dict[str, Any],
    *,
    preferred_keys: tuple[str, ...],
) -> str:
    for key in preferred_keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    # Fallback: preserve a verified API value rather than inventing one.
    for value in row.values():
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return ""


def _pick_label(row: dict[str, Any], language: str) -> str:
    if language == "ar":
        keys = (
            "DescAr", "descAr", "descriptionAr", "description_ar",
            "nameAr", "name_ar", "namea", "labelAr", "label_ar",
            "name", "description", "label", "title",
        )
    else:
        keys = (
            "DescEn", "descEn", "descriptionEn", "description_en",
            "nameEn", "name_en", "namel", "labelEn", "label_en",
            "name", "description", "label", "title",
        )
    return _pick_value(row, preferred_keys=keys)


def _select_option(
    message: str,
    options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not options:
        return None
    value = _norm(message)
    if value.isdigit():
        index = int(value) - 1
        if 0 <= index < len(options):
            return options[index]

    # Exact/contains match against every verified scalar field.
    best: tuple[int, dict[str, Any] | None] = (0, None)
    user_tokens = {x for x in value.split() if len(x) > 1}
    for row in options:
        row_text = _row_text(row)
        if value and (value == row_text or value in row_text):
            return row
        row_tokens = set(row_text.split())
        score = len(user_tokens.intersection(row_tokens))
        if score > best[0]:
            best = (score, row)
    return best[1] if best[0] >= 1 else None


def _render_options(
    options: list[dict[str, Any]],
    language: str,
    *,
    title_ar: str,
    title_en: str,
) -> str:
    lines = [title_ar if language == "ar" else title_en]
    for index, row in enumerate(options[:20], 1):
        label = _pick_label(row, language) or _pick_value(
            row,
            preferred_keys=("name", "description", "label", "code", "id", "value"),
        )
        lines.append(f"{index}. {label}")
    return "\n".join(lines)


class ActionWorkflowService:
    """Backend-owned workflows for write endpoints.

    The LLM never supplies a username and never directly confirms a transaction.
    The authenticated EmployeeContext is the only source of username and a write
    call is made only after the user explicitly confirms a pending server-side
    action.
    """

    def __init__(self, employee_api: EmployeeAPIClient) -> None:
        self.employee_api = employee_api

    async def handle(
        self,
        *,
        message: str,
        state: dict[str, Any],
        context: EmployeeContext | None,
        language: str,
    ) -> ActionWorkflowResult:
        pending = _pending(state)

        if pending:
            if _is_cancel(message):
                _clear_action(state)
                return ActionWorkflowResult(
                    True,
                    "تم إلغاء الطلب." if language == "ar" else "The request has been cancelled.",
                )

            if context is None:
                return ActionWorkflowResult(
                    True,
                    "سجّل الدخول أول عشان أقدر أكمل الطلب."
                    if language == "ar"
                    else "Please sign in first so I can continue the request.",
                )

            action = str(pending.get("action") or "")
            if action == "create_vacation":
                return await self._continue_vacation(
                    message=message,
                    state=state,
                    context=context,
                    language=language,
                    pending=pending,
                )
            if action == "create_hr_letter":
                return await self._continue_hr_letter(
                    message=message,
                    state=state,
                    context=context,
                    language=language,
                    pending=pending,
                )

            _clear_action(state)

        create_vacation = _looks_like_create_vacation(message)
        create_hr_letter = _looks_like_create_hr_letter(message)
        if not create_vacation and not create_hr_letter:
            return ActionWorkflowResult(False, "")

        if context is None:
            return ActionWorkflowResult(
                True,
                "سجّل الدخول أول عشان أقدر أنفذ الطلب."
                if language == "ar"
                else "Please sign in first so I can perform that action.",
            )

        if create_vacation:
            return await self._start_vacation(
                message=message,
                state=state,
                context=context,
                language=language,
            )

        return await self._start_hr_letter(
            message=message,
            state=state,
            context=context,
            language=language,
        )

    async def _start_vacation(
        self,
        *,
        message: str,
        state: dict[str, Any],
        context: EmployeeContext,
        language: str,
    ) -> ActionWorkflowResult:
        dates = _extract_dates(message)
        require_exit_return, require_tickets = _extract_boolean_preferences(message)

        data: dict[str, Any] = {
            "start_date": dates[0] if dates else "",
            "end_date": dates[1] if len(dates) > 1 else (dates[0] if dates else ""),
            "vacation_type": "",
            "vacation_type_label": "",
            "require_exit_return": require_exit_return,
            "require_tickets": require_tickets,
            "comment": "",
            "attachment": "",
            "precheck_note": "",
        }

        try:
            payload = await self.employee_api.get_vacation_types(
                access_token=context.access_token,
            )
        except (EmployeeAPIError, EmployeeAuthenticationError, EmployeeAuthorizationError) as exc:
            return ActionWorkflowResult(True, self._api_error(exc, language))

        options = [
            dict(x)
            for x in (payload.get("types") or [])
            if isinstance(x, dict)
        ]

        selected = _select_option(message, options)
        if selected:
            data["vacation_type"] = _pick_value(
                selected,
                preferred_keys=(
                    "VacationType", "vacationType", "vacation_type",
                    "code", "value", "id",
                    "name", "nameAr", "nameEn",
                ),
            )
            data["vacation_type_label"] = _pick_label(selected, language)

        if not data["start_date"]:
            _set_action(
                state,
                action="create_vacation",
                stage="start_date",
                data=data,
                options=options,
                message=message,
            )
            return ActionWorkflowResult(
                True,
                "أكيد. وش تاريخ بداية الإجازة؟ اكتبه مثل 2026-10-05."
                if language == "ar"
                else "Sure. What is the leave start date? For example: 2026-10-05.",
            )

        if not data["vacation_type"]:
            _set_action(
                state,
                action="create_vacation",
                stage="vacation_type",
                data=data,
                options=options,
                message=message,
            )
            return ActionWorkflowResult(
                True,
                _render_options(
                    options,
                    language,
                    title_ar="اختر نوع الإجازة:",
                    title_en="Choose the leave type:",
                ),
            )

        return await self._prepare_vacation_confirmation(
            state=state,
            context=context,
            language=language,
            data=data,
        )

    async def _continue_vacation(
        self,
        *,
        message: str,
        state: dict[str, Any],
        context: EmployeeContext,
        language: str,
        pending: dict[str, Any],
    ) -> ActionWorkflowResult:
        stage = str(pending.get("stage") or "")
        data = dict(pending.get("data") or {})
        options = [
            dict(x)
            for x in (pending.get("options") or [])
            if isinstance(x, dict)
        ]

        if stage == "start_date":
            dates = _extract_dates(message)
            if not dates:
                return ActionWorkflowResult(
                    True,
                    "اكتب تاريخ البداية مثل 2026-10-05."
                    if language == "ar"
                    else "Please enter the start date, for example 2026-10-05.",
                )
            data["start_date"] = dates[0]
            data["end_date"] = dates[1] if len(dates) > 1 else dates[0]
            if not data.get("vacation_type"):
                _set_action(
                    state,
                    action="create_vacation",
                    stage="vacation_type",
                    data=data,
                    options=options,
                    message=message,
                )
                return ActionWorkflowResult(
                    True,
                    _render_options(
                        options,
                        language,
                        title_ar="تمام. اختر نوع الإجازة:",
                        title_en="Great. Choose the leave type:",
                    ),
                )
            return await self._prepare_vacation_confirmation(
                state=state, context=context, language=language, data=data
            )

        if stage == "vacation_type":
            selected = _select_option(message, options)
            if not selected:
                return ActionWorkflowResult(
                    True,
                    _render_options(
                        options,
                        language,
                        title_ar="ما قدرت أحدد النوع. اختر رقم أو اسم من القائمة:",
                        title_en="I couldn't identify the type. Choose a number or name from the list:",
                    ),
                )
            data["vacation_type"] = _pick_value(
                selected,
                preferred_keys=(
                    "VacationType", "vacationType", "vacation_type",
                    "code", "value", "id",
                    "name", "nameAr", "nameEn",
                ),
            )
            data["vacation_type_label"] = _pick_label(selected, language)
            if not data.get("start_date"):
                _set_action(
                    state,
                    action="create_vacation",
                    stage="start_date",
                    data=data,
                    options=options,
                    message=message,
                )
                return ActionWorkflowResult(
                    True,
                    "وش تاريخ الإجازة؟ اكتبه مثل 2026-10-05."
                    if language == "ar"
                    else "What is the leave date? For example: 2026-10-05.",
                )
            return await self._prepare_vacation_confirmation(
                state=state, context=context, language=language, data=data
            )

        if stage == "confirm":
            if not _is_confirm(message):
                return ActionWorkflowResult(
                    True,
                    "إذا تبي أرسل الطلب اكتب «تأكيد»، أو «إلغاء»."
                    if language == "ar"
                    else "Reply “confirm” to submit the request, or “cancel”.",
                )

            try:
                result = await self.employee_api.create_vacation(
                    username=context.username,
                    access_token=context.access_token,
                    start_date=str(data.get("start_date") or ""),
                    end_date=str(data.get("end_date") or ""),
                    vacation_type=str(data.get("vacation_type") or ""),
                    require_exit_return=bool(data.get("require_exit_return")),
                    require_tickets=bool(data.get("require_tickets")),
                    comment=str(data.get("comment") or ""),
                    attachment=str(data.get("attachment") or ""),
                )
            except (EmployeeAPIError, EmployeeAuthenticationError, EmployeeAuthorizationError) as exc:
                return ActionWorkflowResult(True, self._api_error(exc, language))

            _clear_action(state)
            return ActionWorkflowResult(
                True,
                (
                    "تم إرسال طلب الإجازة للنظام بنجاح."
                    if language == "ar"
                    else "The leave request was submitted successfully."
                ),
            )

        _clear_action(state)
        return ActionWorkflowResult(False, "")

    async def _prepare_vacation_confirmation(
        self,
        *,
        state: dict[str, Any],
        context: EmployeeContext,
        language: str,
        data: dict[str, Any],
    ) -> ActionWorkflowResult:
        start_date = str(data.get("start_date") or "")
        end_date = str(data.get("end_date") or start_date)

        precheck_note = ""
        try:
            overlap = await self.employee_api.get_intersected_vacations(
                username=context.username,
                access_token=context.access_token,
                start_date=start_date,
                end_date=end_date,
            )
            records = [
                x for x in (overlap.get("records") or [])
                if isinstance(x, dict)
            ]
            if records:
                precheck_note = (
                    f"النظام رجّع {len(records)} نتيجة في فحص تداخل الإجازات."
                    if language == "ar"
                    else f"The overlap check returned {len(records)} record(s)."
                )
        except EmployeeAPIError:
            # The supplied Postman collection has shown this deployed route
            # returning 404. Do not silently claim the pre-check succeeded.
            precheck_note = (
                "تعذر التحقق من تداخل الإجازات من النظام، لكن أقدر أرسل الطلب بعد تأكيدك."
                if language == "ar"
                else "The leave-overlap check could not be completed, but I can submit after your confirmation."
            )

        data["precheck_note"] = precheck_note
        _set_action(
            state,
            action="create_vacation",
            stage="confirm",
            data=data,
        )

        label = str(data.get("vacation_type_label") or data.get("vacation_type") or "")
        if language == "ar":
            lines = [
                "تأكيد طلب الإجازة:",
                f"- النوع: {label}",
                f"- من: {start_date}",
                f"- إلى: {end_date}",
                f"- تذاكر: {'نعم' if data.get('require_tickets') else 'لا'}",
                f"- خروج وعودة: {'نعم' if data.get('require_exit_return') else 'لا'}",
            ]
            if precheck_note:
                lines.append(f"- ملاحظة: {precheck_note}")
            lines.append("إذا البيانات صحيحة اكتب «تأكيد»، وإذا لا اكتب «إلغاء».")
        else:
            lines = [
                "Confirm leave request:",
                f"- Type: {label}",
                f"- From: {start_date}",
                f"- To: {end_date}",
                f"- Tickets: {'Yes' if data.get('require_tickets') else 'No'}",
                f"- Exit/return: {'Yes' if data.get('require_exit_return') else 'No'}",
            ]
            if precheck_note:
                lines.append(f"- Note: {precheck_note}")
            lines.append("Reply “confirm” to submit, or “cancel”.")

        return ActionWorkflowResult(True, "\n".join(lines))

    async def _start_hr_letter(
        self,
        *,
        message: str,
        state: dict[str, Any],
        context: EmployeeContext,
        language: str,
    ) -> ActionWorkflowResult:
        try:
            types_payload = await self.employee_api.get_hr_letter_types(
                access_token=context.access_token,
            )
            recipients_payload = await self.employee_api.get_hr_letter_recipients(
                access_token=context.access_token,
            )
        except (EmployeeAPIError, EmployeeAuthenticationError, EmployeeAuthorizationError) as exc:
            return ActionWorkflowResult(True, self._api_error(exc, language))

        types = [
            dict(x)
            for x in (types_payload.get("types") or [])
            if isinstance(x, dict)
        ]
        recipients = [
            dict(x)
            for x in (recipients_payload.get("recipients") or [])
            if isinstance(x, dict)
        ]

        data: dict[str, Any] = {
            "letter_type": "",
            "letter_type_label": "",
            "to_whom": "",
            "to_whom_label": "",
            "letter_options": types,
            "recipient_options": recipients,
        }

        selected_type = _select_option(message, types)
        if selected_type:
            data["letter_type"] = _pick_value(
                selected_type,
                preferred_keys=("LetterType", "letterType", "letter_type", "code", "value", "id"),
            )
            data["letter_type_label"] = _pick_label(selected_type, language)

        selected_recipient = _select_option(message, recipients)
        if selected_recipient:
            data["to_whom"] = _pick_value(
                selected_recipient,
                preferred_keys=("ToWhom", "toWhom", "to_whom", "code", "value", "id"),
            )
            data["to_whom_label"] = _pick_label(selected_recipient, language)

        if not data["letter_type"]:
            _set_action(
                state,
                action="create_hr_letter",
                stage="letter_type",
                data=data,
                options=types,
                message=message,
            )
            return ActionWorkflowResult(
                True,
                _render_options(
                    types,
                    language,
                    title_ar="اختر نوع خطاب الموارد البشرية:",
                    title_en="Choose the HR letter type:",
                ),
            )

        if not data["to_whom"]:
            _set_action(
                state,
                action="create_hr_letter",
                stage="recipient",
                data=data,
                options=recipients,
                message=message,
            )
            return ActionWorkflowResult(
                True,
                _render_options(
                    recipients,
                    language,
                    title_ar="اختر الجهة الموجه لها الخطاب:",
                    title_en="Choose who the letter is addressed to:",
                ),
            )

        return self._prepare_hr_letter_confirmation(state, data, language)

    async def _continue_hr_letter(
        self,
        *,
        message: str,
        state: dict[str, Any],
        context: EmployeeContext,
        language: str,
        pending: dict[str, Any],
    ) -> ActionWorkflowResult:
        stage = str(pending.get("stage") or "")
        data = dict(pending.get("data") or {})
        options = [
            dict(x)
            for x in (pending.get("options") or [])
            if isinstance(x, dict)
        ]

        if stage == "letter_type":
            selected = _select_option(message, options)
            if not selected:
                return ActionWorkflowResult(
                    True,
                    _render_options(
                        options,
                        language,
                        title_ar="اختر رقم أو اسم نوع الخطاب من القائمة:",
                        title_en="Choose a letter type by number or name:",
                    ),
                )
            data["letter_type"] = _pick_value(
                selected,
                preferred_keys=("LetterType", "letterType", "letter_type", "code", "value", "id"),
            )
            data["letter_type_label"] = _pick_label(selected, language)
            recipients = [
                dict(x)
                for x in (data.get("recipient_options") or [])
                if isinstance(x, dict)
            ]
            _set_action(
                state,
                action="create_hr_letter",
                stage="recipient",
                data=data,
                options=recipients,
                message=message,
            )
            return ActionWorkflowResult(
                True,
                _render_options(
                    recipients,
                    language,
                    title_ar="تمام. اختر الجهة الموجه لها الخطاب:",
                    title_en="Great. Choose who the letter is addressed to:",
                ),
            )

        if stage == "recipient":
            selected = _select_option(message, options)
            if not selected:
                return ActionWorkflowResult(
                    True,
                    _render_options(
                        options,
                        language,
                        title_ar="اختر رقم أو اسم الجهة من القائمة:",
                        title_en="Choose the recipient by number or name:",
                    ),
                )
            data["to_whom"] = _pick_value(
                selected,
                preferred_keys=("ToWhom", "toWhom", "to_whom", "code", "value", "id"),
            )
            data["to_whom_label"] = _pick_label(selected, language)
            return self._prepare_hr_letter_confirmation(state, data, language)

        if stage == "confirm":
            if not _is_confirm(message):
                return ActionWorkflowResult(
                    True,
                    "إذا تبي إصدار الخطاب اكتب «تأكيد»، أو «إلغاء»."
                    if language == "ar"
                    else "Reply “confirm” to generate the letter, or “cancel”.",
                )
            try:
                await self.employee_api.create_hr_letter(
                    username=context.username,
                    access_token=context.access_token,
                    letter_type=str(data.get("letter_type") or ""),
                    to_whom=str(data.get("to_whom") or ""),
                )
            except (EmployeeAPIError, EmployeeAuthenticationError, EmployeeAuthorizationError) as exc:
                return ActionWorkflowResult(True, self._api_error(exc, language))

            _clear_action(state)
            return ActionWorkflowResult(
                True,
                "تم إصدار/إرسال طلب الخطاب للنظام بنجاح."
                if language == "ar"
                else "The HR letter request was submitted successfully.",
            )

        _clear_action(state)
        return ActionWorkflowResult(False, "")

    def _prepare_hr_letter_confirmation(
        self,
        state: dict[str, Any],
        data: dict[str, Any],
        language: str,
    ) -> ActionWorkflowResult:
        _set_action(
            state,
            action="create_hr_letter",
            stage="confirm",
            data=data,
        )
        letter_label = str(data.get("letter_type_label") or data.get("letter_type") or "")
        recipient_label = str(data.get("to_whom_label") or data.get("to_whom") or "")

        if language == "ar":
            message = (
                "تأكيد إصدار الخطاب:\n"
                f"- نوع الخطاب: {letter_label}\n"
                f"- موجه إلى: {recipient_label}\n"
                "إذا البيانات صحيحة اكتب «تأكيد»، وإذا لا اكتب «إلغاء»."
            )
        else:
            message = (
                "Confirm HR letter:\n"
                f"- Letter type: {letter_label}\n"
                f"- Addressed to: {recipient_label}\n"
                "Reply “confirm” to submit, or “cancel”."
            )
        return ActionWorkflowResult(True, message)

    @staticmethod
    def _api_error(exc: Exception, language: str) -> str:
        if isinstance(exc, EmployeeAuthenticationError):
            return (
                "انتهت صلاحية الدخول. سجّل الدخول من جديد."
                if language == "ar"
                else "Your sign-in session has expired. Please sign in again."
            )
        if isinstance(exc, EmployeeAuthorizationError):
            return (
                "ما عندك صلاحية لتنفيذ هذا الطلب."
                if language == "ar"
                else "You don't have permission to perform this action."
            )
        return (
            f"تعذر تنفيذ الطلب من النظام: {exc}"
            if language == "ar"
            else f"The system could not complete the request: {exc}"
        )