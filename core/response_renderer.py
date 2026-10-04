from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from copy import deepcopy
from typing import Any

from core.llm_client import chat
from core.semantic_models import ParsedRequest


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")


# User-facing localization for common API enum values. Identifiers such as
# emails/usernames are intentionally preserved, but prose/status values follow
# the language of the current user turn.
_AR_ENUMS = {
    "absence": "غياب",
    "absent": "غائب",
    "present": "حاضر",
    "late": "متأخر",
    "leave": "إجازة",
    "vacation": "إجازة",
    "sick leave": "إجازة مرضية",
    "annual leave": "إجازة سنوية",
    "immediate": "عاجلة",
    "urgent": "عاجلة",
    "high": "عالية",
    "normal": "عادية",
    "medium": "متوسطة",
    "low": "منخفضة",
    "confidential": "سري",
    "secret": "سري",
    "public": "عام",
    "corporate guide": "دليل الشركات",
}
_EN_ENUMS = {
    "غياب": "Absent",
    "غائب": "Absent",
    "حاضر": "Present",
    "متأخر": "Late",
    "إجازة": "Leave",
    "اجازة": "Leave",
    "إجازة مرضية": "Sick leave",
    "إجازة سنوية": "Annual leave",
    "عاجلة": "Urgent",
    "عاليه": "High",
    "عالية": "High",
    "عادية": "Normal",
    "متوسطة": "Medium",
    "منخفضة": "Low",
    "سري": "Confidential",
    "عام": "Public",
}


def _localized_enum(value: Any, lang: str) -> Any:
    if not _valid(value):
        return value
    text = str(value).strip()
    if lang == "ar":
        return _AR_ENUMS.get(text.casefold(), text)
    return _EN_ENUMS.get(text, text)


def _looks_like_identifier(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if re.search(r"https?://|www\.|@", text, re.I):
        return True
    if re.fullmatch(r"[A-Za-z0-9_.-]{2,64}", text) and ("." in text or "-" in text or "_" in text):
        return True
    if re.fullmatch(r"\+?[0-9 ()-]{5,}", text):
        return True
    if re.fullmatch(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", text):
        return True
    return False

FIELD_LABELS = {
    "ar": {
        "name": "الاسم", "name_en": "الاسم بالإنجليزية", "job_title": "المسمى الوظيفي",
        "job_title_en": "المسمى الوظيفي بالإنجليزية", "department": "الإدارة",
        "department_en": "الإدارة بالإنجليزية", "sector": "القطاع", "sector_en": "القطاع بالإنجليزية",
        "email": "البريد الإلكتروني", "mobile": "رقم الجوال", "extension": "التحويلة",
        "username": "اسم المستخدم", "employee_type": "نوع الموظف", "balance": "الرصيد",
        "count": "العدد", "subject": "الموضوع", "creation_date": "تاريخ الإنشاء",
        "creator": "المنشئ", "sender": "المرسل", "recipient": "المستلم",
        "last_forwarded_date": "آخر تحويل", "document_type": "نوع المعاملة", "priority": "الأولوية",
        "importance": "الأهمية", "secrecy": "السرية", "classification": "التصنيف",
        "creation_department": "إدارة الإنشاء", "sender_department": "إدارة المرسل",
        "company": "الجهة", "details": "التفاصيل", "website": "الموقع", "start_date": "تاريخ البداية",
        "end_date": "تاريخ النهاية", "tags": "الوسوم", "start_time": "وقت البداية", "end_time": "وقت النهاية",
        "status": "الحالة", "date": "التاريخ",
    },
    "en": {
        "name": "Name", "name_en": "English name", "job_title": "Job title", "job_title_en": "English job title",
        "department": "Department", "department_en": "English department", "sector": "Sector", "sector_en": "English sector",
        "email": "Email", "mobile": "Mobile number", "extension": "Extension", "username": "Username",
        "employee_type": "Employee type", "balance": "Balance", "count": "Count", "subject": "Subject",
        "creation_date": "Created", "creator": "Creator", "sender": "Sender", "recipient": "Recipient",
        "last_forwarded_date": "Last forwarded", "document_type": "Document type", "priority": "Priority",
        "importance": "Importance", "secrecy": "Secrecy", "classification": "Classification",
        "creation_department": "Creation department", "sender_department": "Sender department",
        "company": "Company", "details": "Details", "website": "Website", "start_date": "Start date",
        "end_date": "End date", "tags": "Tags", "start_time": "Start time", "end_time": "End time",
        "status": "Status", "date": "Date",
    },
}


RAW_FIELD_KEYS = {
    "email": ("email", "emailAddress", "mail"),
    "mobile": ("mobile", "mobileNo", "mobileNumber", "mobilePhone", "cellPhone", "phoneMobile"),
    "extension": ("extension", "ext"),
    "username": ("username", "userid", "userName", "userCode"),
    "employee_type": ("employee_type", "empType"),
    "balance": ("balance",),
    "count": ("count",),
    "subject": ("subject", "title"),
    "creation_date": ("creationDate", "creation_date", "date"),
    "creator": ("creationUsername", "creator"),
    "sender": ("fromUserName", "sender"),
    "recipient": ("toUserName", "recipient"),
    "last_forwarded_date": ("lastForwaredDate", "lastForwardedDate"),
    "document_type": ("docType", "documentType"),
    "priority": ("priority",),
    "importance": ("importance",),
    "secrecy": ("secrecy",),
    "classification": ("classification",),
    "creation_department": ("creationDep",),
    "sender_department": ("fromDep",),
    "company": ("company", "companyName"),
    "details": ("details", "description", "body"),
    "website": ("website", "url"),
    "start_date": ("starDate", "startDate", "start_date"),
    "end_date": ("endDate", "end_date"),
    "tags": ("tags",),
    "start_time": ("start_time", "startTime"),
    "end_time": ("end_time", "endTime"),
    "status": ("status",),
    "date": ("date",),
}


def _valid(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _show(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    # Some upstream legacy strings contain a Unicode replacement character
    # before otherwise valid numeric/contact values (for example "�1714").
    # Remove encoding artifacts only; never alter the actual digits/text.
    text = text.replace("\ufffd", "").replace("\u200b", "").replace("\ufeff", "")
    return text.strip()



def _format_mobile(value: Any) -> Any:
    """Display Saudi mobile numbers in local ``05XXXXXXXX`` form.

    Examples:
        505227641       -> 0505227641
        +966505227641   -> 0505227641
        966505227641    -> 0505227641
        00966505227641  -> 0505227641

    Formatting is display-only. Unrecognised/non-Saudi values are preserved.
    """
    if not _valid(value):
        return value

    original = str(value).strip()
    digits = re.sub(r"[^0-9]", "", original)

    if digits.startswith("00966"):
        digits = digits[5:]
    elif digits.startswith("966"):
        digits = digits[3:]

    if len(digits) == 9 and digits.startswith("5"):
        return "0" + digits
    if len(digits) == 10 and digits.startswith("05"):
        return digits
    return original


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if _valid(row.get(key)):
            return row.get(key)
    return None


def _field_value(row: dict[str, Any], field: str, lang: str) -> Any:
    if field == "name":
        return _first(row, "name_ar", "name", "name_en") if lang == "ar" else _first(row, "name_en", "name", "name_ar")
    if field == "name_en":
        return _first(row, "name_en", "name")
    if field == "job_title":
        return _first(row, "job_title_ar", "job_title", "job_title_en") if lang == "ar" else _first(row, "job_title_en", "job_title", "job_title_ar")
    if field == "job_title_en":
        return _first(row, "job_title_en", "job_title")
    if field == "department":
        return _first(row, "department_ar", "department", "department_en") if lang == "ar" else _first(row, "department_en", "department", "department_ar")
    if field == "department_en":
        return _first(row, "department_en", "department")
    if field == "sector":
        return _first(row, "sector_ar", "sector", "sector_en") if lang == "ar" else _first(row, "sector_en", "sector", "sector_ar")
    if field == "sector_en":
        return _first(row, "sector_en", "sector")
    for key in RAW_FIELD_KEYS.get(field, (field,)):
        if _valid(row.get(key)):
            value = row.get(key)
            if field == "mobile":
                return _format_mobile(value)
            return value
    return None


def _subject_name(row: dict[str, Any], lang: str, *, fallback: str = "") -> str:
    return str(_field_value(row, "name", lang) or fallback or "").strip()


def _rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    for key in (
        "items", "employees", "transactions", "meetings", "records", "news", "announcements",
        "policies", "benefits", "mazaya", "types", "recipients", "vacations", "results", "list",
        "attendance", "employeeAttendance", "attendanceList",
    ):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []



async def localize_tarasol_data(data: Any, lang: str) -> Any:
    """Localize verified Tarasol display strings without changing facts.

    The API sometimes mixes English and Arabic in names/statuses. We keep
    identifiers (email/usernames/IDs) untouched and translate/transliterate only
    human-readable display text. A translation failure falls back to the
    verified original value.
    """
    localized = deepcopy(data)
    rows = _rows(localized)
    if not rows:
        return localized

    fields = (
        "subject", "sender", "document_type", "priority", "importance",
        "secrecy", "classification", "creation_department", "sender_department",
    )
    pending: list[tuple[dict[str, Any], str, str]] = []
    texts: list[str] = []

    for row in rows[:10]:
        for field in fields:
            keys = RAW_FIELD_KEYS.get(field, (field,))
            for key in keys:
                if not _valid(row.get(key)):
                    continue
                original = str(row.get(key)).strip()
                fixed = _localized_enum(original, lang)
                if str(fixed) != original:
                    row[key] = fixed
                    original = str(fixed)
                wrong_language = (
                    lang == "ar" and LATIN_RE.search(original) and not ARABIC_RE.search(original)
                ) or (
                    lang == "en" and ARABIC_RE.search(original) and not LATIN_RE.search(original)
                )
                if wrong_language and not _looks_like_identifier(original):
                    pending.append((row, key, original))
                    texts.append(original)
                break

    if not texts:
        return localized

    target = "Arabic" if lang == "ar" else "English"
    try:
        response = await asyncio.to_thread(
            chat,
            [
                {
                    "role": "system",
                    "content": (
                        f"Translate or faithfully transliterate each VERIFIED_VALUE to {target}. "
                        "Preserve meaning, proper names, dates, numbers, and factual content exactly. "
                        "Do not add explanations. Return ONLY JSON as "
                        '{"translations":["..."]} with the same number and order.'
                    ),
                },
                {"role": "user", "content": json.dumps({"VERIFIED_VALUES": texts}, ensure_ascii=False)},
            ],
        )
        content = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "").strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 and end > start else {}
        translations = parsed.get("translations") if isinstance(parsed, dict) else []
        if isinstance(translations, list) and len(translations) == len(pending):
            for (row, key, original), translated in zip(pending, translations):
                text = str(translated or "").strip()
                if not text:
                    continue
                # Guard against a language-repair call returning another mixed
                # sentence. Identifiers were never sent here.
                if lang == "ar" and LATIN_RE.search(text) and not ARABIC_RE.search(text):
                    continue
                if lang == "en" and ARABIC_RE.search(text) and not LATIN_RE.search(text):
                    continue
                row[key] = text
    except Exception:
        pass
    return localized


def _default_profile_fields(request: ParsedRequest, *, self_subject: bool = False) -> list[str]:
    if request.fields and "all" not in request.fields:
        fields = list(request.fields)
        # Summary/profile requests must never expose the internal username unless
        # the user explicitly asked for username as the only/specific field.
        summary_modes = {
            "profile:summary", "employee:summary", "employee:focus_summary",
            "employee:selected_from_search",
        }
        if str(request.query or "").strip().lower() in summary_modes and len(fields) > 1:
            fields = [field for field in fields if field != "username"]
        return fields
    # Never expose internal usernames in an ordinary profile answer. Contact
    # fields are shown for self only; another employee's bare-name/profile
    # request stays concise unless the user explicitly asks for contact details.
    if self_subject:
        return ["name", "job_title", "department", "email", "mobile", "extension"]
    return ["name", "job_title", "department"]


def render_profile(
    data: dict[str, Any],
    request: ParsedRequest,
    lang: str,
    *,
    subject_name: str = "",
    self_subject: bool = False,
    manager_result: bool = False,
) -> str:
    """Render profile answers in a direct, question-specific style."""
    fields = _default_profile_fields(request, self_subject=self_subject)
    available = [(field, _field_value(data, field, lang)) for field in fields]
    available = [(field, value) for field, value in available if _valid(value)]
    name = _subject_name(data, lang, fallback=subject_name)

    if not available:
        if lang == "ar":
            requested_labels = [FIELD_LABELS["ar"].get(field, field) for field in fields if field != "all"]
            if len(requested_labels) == 1:
                subject = f" لـ {name}" if name and not self_subject else ""
                return f"للأسف، {requested_labels[0]}{subject} غير متوفر حاليًا."
            subject = f" عن {name}" if name and not self_subject else ""
            return f"للأسف، المعلومات المطلوبة{subject} غير متوفرة عندي حاليًا."
        return (
            f"Sorry, that information isn't available for {name} right now."
            if name
            else "Sorry, that information isn't available right now."
        )

    # A one-field question should get a one-line answer, not a generic card.
    if len(fields) == 1 and len(available) == 1:
        field, raw_value = available[0]
        value = _show(raw_value)

        if lang == "ar":
            if field == "name":
                if manager_result:
                    return (
                        f"مديرك المباشر هو {value}."
                        if not subject_name
                        else f"مدير {subject_name} المباشر هو {value}."
                    )
                return f"أنت {value}." if self_subject else f"اسمه {value}."
            if field == "email":
                return (
                    f"إيميلك هو {value}."
                    if self_subject
                    else f"إيميل {name} هو {value}."
                )
            if field == "mobile":
                return (
                    f"رقم جوالك هو {value}."
                    if self_subject
                    else f"رقم جوال {name} هو {value}."
                )
            if field == "extension":
                return (
                    f"تحويلتك هي {value}."
                    if self_subject
                    else f"تحويلة {name} هي {value}."
                )
            if field == "job_title":
                return (
                    f"مسمّاك الوظيفي هو {value}."
                    if self_subject
                    else f"{name} يعمل كـ {value}."
                )
            if field == "department":
                return (
                    f"إدارتك هي {value}."
                    if self_subject
                    else f"{name} يتبع {value}."
                )
            if field == "username":
                return (
                    f"اسم المستخدم الخاص بك: {value}."
                    if self_subject
                    else f"اسم المستخدم لـ {name}: {value}."
                )
            return f"{FIELD_LABELS['ar'].get(field, field)}: {value}."

        if field == "name":
            if manager_result:
                return (
                    f"Your direct manager is {value}."
                    if not subject_name
                    else f"{subject_name}'s direct manager is {value}."
                )
            return f"You're {value}." if self_subject else f"Name: {value}."
        if field == "email":
            return f"Your email: {value}." if self_subject else f"{name}'s email: {value}."
        if field == "mobile":
            return f"Your mobile number: {value}." if self_subject else f"{name}'s mobile number: {value}."
        if field == "extension":
            return f"Your extension: {value}." if self_subject else f"{name}'s extension: {value}."
        if field == "job_title":
            return f"Your job title: {value}." if self_subject else f"{name}'s job title: {value}."
        if field == "department":
            return f"Your department: {value}." if self_subject else f"{name}'s department: {value}."
        if field == "username":
            return f"Your username: {value}." if self_subject else f"{name}'s username: {value}."
        return f"{FIELD_LABELS['en'].get(field, field)}: {value}."

    values = {field: _show(value) for field, value in available}
    profile_mode = str(request.query or "").strip().lower()

    # Identity questions are better as one natural sentence.
    if self_subject and "name" in values and profile_mode in {
        "profile:recognition",
        "profile:identity",
    }:
        if lang == "ar":
            if profile_mode == "profile:recognition":
                sentence = f"إيه أكيد، أعرفك 😊 أنت {values['name']}"
            else:
                sentence = f"أنت {values['name']}"
            if values.get("job_title") and values.get("department"):
                sentence += f"، {values['job_title']} في {values['department']}"
            elif values.get("job_title"):
                sentence += f"، {values['job_title']}"
            elif values.get("department"):
                sentence += f" في {values['department']}"
            return sentence + "."

        sentence = (
            f"Yes — I know you. You're {values['name']}"
            if profile_mode == "profile:recognition"
            else f"You're {values['name']}"
        )
        if values.get("job_title") and values.get("department"):
            sentence += f", {values['job_title']} in {values['department']}"
        elif values.get("job_title"):
            sentence += f", {values['job_title']}"
        elif values.get("department"):
            sentence += f" in {values['department']}"
        return sentence + "."

    # A bare employee/profile request should read naturally and never expose
    # internal usernames. Keep it to verified identity/work facts.
    if "all" in request.fields or not request.fields:
        values = {field: _field_value(data, field, lang) for field in fields}
        if lang == "ar" and not self_subject:
            parts: list[str] = []
            if _valid(values.get("job_title")):
                parts.append(f"يعمل كـ {_show(values['job_title'])}")
            if _valid(values.get("department")):
                parts.append(f"ضمن {_show(values['department'])}")
            if name and parts:
                return f"{name} " + " ".join(parts) + "."
            if name:
                return f"هذا الموظف هو {name}."
        if lang == "en" and not self_subject:
            parts: list[str] = []
            if _valid(values.get("job_title")):
                parts.append(f"works as {_show(values['job_title'])}")
            if _valid(values.get("department")):
                parts.append(f"in {_show(values['department'])}")
            if name and parts:
                return f"{name} " + " ".join(parts) + "."
            if name:
                return f"This employee is {name}."

    # Multi-field answers are structured around the actual subject. When the
    # user explicitly requested several fields, state which requested values are
    # unavailable instead of silently omitting them.
    explicit_fields = bool(request.fields and "all" not in request.fields)

    if lang == "ar":
        heading = "هذي معلوماتك:" if self_subject else (f"بيانات {name}:" if name else "هذي المعلومات المتوفرة:")
        lines = [heading]
        for field in fields:
            value = _field_value(data, field, lang)
            if _valid(value):
                lines.append(f"- {FIELD_LABELS['ar'].get(field, field)}: {_show(value)}")
            elif explicit_fields:
                lines.append(f"- {FIELD_LABELS['ar'].get(field, field)}: غير متوفر حاليًا")
        return "\n".join(lines)

    heading = "Here are your details:" if self_subject else (f"{name}'s details:" if name else "Details:")
    lines = [heading]
    for field in fields:
        value = _field_value(data, field, lang)
        if _valid(value):
            lines.append(f"- {FIELD_LABELS['en'].get(field, field)}: {_show(value)}")
        elif explicit_fields:
            lines.append(f"- {FIELD_LABELS['en'].get(field, field)}: not available in the system")
    return "\n".join(lines)


def _ar_days(value: float | int | str) -> str:
    try:
        number = float(value)
    except Exception:
        return f"{value} يوم"
    text = str(int(number)) if number.is_integer() else str(number)
    if number == 1:
        return "يوم واحد"
    if number == 2:
        return "يومين"
    if 3 <= number <= 10 and number.is_integer():
        return f"{text} أيام"
    return f"{text} يوم"


def render_leave_balance(data: dict[str, Any], lang: str, *, subject_name: str = "", self_subject: bool = False, amount: float | None = None) -> str:
    balance = data.get("balance") if isinstance(data, dict) else None
    if not _valid(balance):
        return "ما قدرت أطلع رصيد الإجازات حاليًا." if lang == "ar" else "Your leave balance is currently unavailable."
    if amount is not None:
        try:
            current = float(balance)
            requested = float(amount)
            remaining = current - requested
            current_text = str(int(current)) if current.is_integer() else str(current)
            requested_text = str(int(requested)) if requested.is_integer() else str(requested)
            if remaining < 0:
                shortage = abs(remaining)
                shortage_text = str(int(shortage)) if shortage.is_integer() else str(shortage)
                if lang == "ar":
                    return f"رصيدك الحالي {_ar_days(current)}، وما يكفي لإجازة {_ar_days(requested)}. تحتاج {_ar_days(shortage)} إضافي."
                return f"Your current balance is {current_text} day(s), which is not enough for {requested_text} day(s) off. You would need {shortage_text} more day(s)."
            remaining_text = str(int(remaining)) if remaining.is_integer() else str(remaining)
            if lang == "ar":
                return f"رصيدك الحالي {_ar_days(current)}. إذا أخذت {_ar_days(requested)}، بيبقى لك {_ar_days(remaining)}."
            return f"Your current balance is {current_text} day(s). If you take {requested_text} day(s) off, you'll have {remaining_text} day(s) left."
        except Exception:
            pass
    if lang == "ar":
        return f"عندك حاليًا {_ar_days(balance)} في رصيد الإجازات." if self_subject else f"رصيد إجازات {subject_name} الحالي {_ar_days(balance)}."
    return f"Your current leave balance is {_show(balance)} day(s)." if self_subject else f"{subject_name}'s current leave balance is {_show(balance)} day(s)."


def _parse_meeting_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_clock_parts(hour: int, minute: int, lang: str) -> str:
    hour = int(hour) % 24
    minute = int(minute) % 60
    if lang == "ar":
        suffix = "ص" if hour < 12 else "م"
        display_hour = hour % 12 or 12
        return f"{display_hour}:{minute:02d} {suffix}"
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


def _clock_label(value: Any, lang: str = "ar") -> str:
    dt = _parse_meeting_dt(value)
    if dt is not None:
        return _format_clock_parts(dt.hour, dt.minute, lang)
    raw = str(value or "").strip()
    match = re.search(r"(?:T|\s|^)(\d{1,2}):(\d{2})", raw)
    if match:
        return _format_clock_parts(int(match.group(1)), int(match.group(2)), lang)
    return raw


def render_meetings(
    data: dict[str, Any],
    lang: str,
    *,
    subject_name: str = "",
    self_subject: bool = False,
    period: str = "none",
    start_date: str = "",
    end_date: str = "",
) -> str:
    meetings = _rows(data)
    label_ar = {
        "today": "اليوم",
        "tomorrow": "بكرة",
        "yesterday": "أمس",
        "this_week": "هذا الأسبوع",
    }.get(period, "للفترة المطلوبة")
    label_en = {
        "today": "today",
        "tomorrow": "tomorrow",
        "yesterday": "yesterday",
        "this_week": "this week",
    }.get(period, "for the requested period")

    if period == "custom" and start_date:
        try:
            start_dt = datetime.fromisoformat(str(start_date))
            if end_date and end_date != start_date:
                end_dt = datetime.fromisoformat(str(end_date))
                label_ar = f"من {start_dt.strftime('%d/%m/%Y')} إلى {end_dt.strftime('%d/%m/%Y')}"
                label_en = f"from {start_dt.strftime('%d/%m/%Y')} to {end_dt.strftime('%d/%m/%Y')}"
            else:
                label_ar = f"بتاريخ {start_dt.strftime('%d/%m/%Y')}"
                label_en = f"on {start_dt.strftime('%d/%m/%Y')}"
        except Exception:
            pass

    if not meetings:
        if lang == "ar":
            return (
                f"ما عندك اجتماعات {label_ar}، يومك فاضي من الاجتماعات 👌"
                if self_subject
                else f"ما فيه اجتماعات مسجلة لـ {subject_name} {label_ar}."
            )
        return (
            f"You have no meetings {label_en}."
            if self_subject
            else f"There are no meetings registered for {subject_name} {label_en}."
        )

    count = len(meetings)
    if lang == "ar":
        if self_subject:
            heading = f"عندك {count} اجتماع" + ("" if count == 1 else "ات") + f" {label_ar}، هذي تفاصيلها:"
        else:
            heading = f"الاجتماعات المسجلة لـ {subject_name} {label_ar}:"
    else:
        heading = (
            f"You have {count} meeting{'s' if count != 1 else ''} {label_en}:"
            if self_subject
            else f"Meetings registered for {subject_name} {label_en}:"
        )

    lines = [heading]
    for row in meetings[:10]:
        if lang == "ar":
            # Meeting titles are source data/proper names. Prefer Arabic when
            # available, but preserve an English-only title instead of hiding
            # it behind the generic word "اجتماع".
            subject = _first(
                row,
                "subject_ar", "subjectAr", "title_ar", "titleAr",
                "subject", "title",
                "subject_en", "subjectEn", "title_en", "titleEn",
            )
            if not _valid(subject):
                subject = "اجتماع"
        else:
            # Same rule in reverse: prefer English, but keep a verified Arabic
            # title when no English title exists.
            subject = _first(
                row,
                "subject_en", "subjectEn", "title_en", "titleEn",
                "subject", "title",
                "subject_ar", "subjectAr", "title_ar", "titleAr",
            )
            if not _valid(subject):
                subject = "Meeting"

        start = _clock_label(_first(row, "start_time", "startTime"), lang)
        end = _clock_label(_first(row, "end_time", "endTime"), lang)

        if start and end:
            time_text = f"{start} - {end}"
        elif start:
            time_text = start
        else:
            time_text = ""

        lines.append(f"- {subject}" + (f" | {time_text}" if time_text else ""))

    if len(meetings) > 10:
        lines.append(
            f"- تم عرض أول 10 اجتماعات من أصل {len(meetings)}."
            if lang == "ar"
            else f"- Showing the first 10 of {len(meetings)} meetings."
        )

    return "\n".join(lines)


def _time_query_minutes(time_query: str) -> tuple[int, int | None] | None:
    raw = str(time_query or "").strip()
    if not raw:
        return None
    parts = raw.split("-", 1)
    def one(value: str) -> int | None:
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value.strip())
        if not match:
            return None
        h, m = int(match.group(1)), int(match.group(2))
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return h * 60 + m
    start = one(parts[0])
    if start is None:
        return None
    end = one(parts[1]) if len(parts) == 2 else None
    return start, end


def _time_query_label(time_query: str, lang: str) -> str:
    parsed = _time_query_minutes(time_query)
    if parsed is None:
        return str(time_query or "").replace("-", "–")
    start, end = parsed
    start_label = _format_clock_parts(start // 60, start % 60, lang)
    if end is None:
        return start_label
    end_label = _format_clock_parts(end // 60, end % 60, lang)
    return f"{start_label}–{end_label}"


def _merge_busy_slots(slots: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Remove zero-length/duplicate rows and merge overlapping/adjacent slots."""
    clean = [(s, e) for s, e in slots if e > s]
    clean.sort(key=lambda x: x[0])
    merged: list[tuple[datetime, datetime]] = []
    for start, end in clean:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        # Merge only within the same local calendar date. Adjacent meetings are
        # one continuous busy period from an availability perspective.
        if start.date() == prev_start.date() and start <= prev_end:
            if end > prev_end:
                merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def render_availability(
    data: dict[str, Any],
    lang: str,
    *,
    subject_name: str,
    period: str = "none",
    start_date: str = "",
    end_date: str = "",
    time_query: str = "",
    mode: str = "busy",
) -> str:
    """Answer availability questions directly from verified calendar busy slots."""
    meetings = _rows(data)
    label_ar = {
        "today": "اليوم",
        "tomorrow": "بكرة",
        "yesterday": "أمس",
        "this_week": "هذا الأسبوع",
    }.get(period, "للفترة المطلوبة")
    label_en = {
        "today": "today",
        "tomorrow": "tomorrow",
        "yesterday": "yesterday",
        "this_week": "this week",
    }.get(period, "for the requested period")

    if period == "custom" and start_date:
        try:
            start_dt = datetime.fromisoformat(str(start_date))
            if end_date and end_date != start_date:
                end_dt = datetime.fromisoformat(str(end_date))
                label_ar = f"من {start_dt.strftime('%d/%m/%Y')} إلى {end_dt.strftime('%d/%m/%Y')}"
                label_en = f"from {start_dt.strftime('%d/%m/%Y')} to {end_dt.strftime('%d/%m/%Y')}"
            else:
                label_ar = f"بتاريخ {start_dt.strftime('%d/%m/%Y')}"
                label_en = f"on {start_dt.strftime('%d/%m/%Y')}"
        except Exception:
            pass

    slots: list[tuple[datetime, datetime]] = []
    for row in meetings:
        start = _parse_meeting_dt(_first(row, "start_time", "startTime"))
        end = _parse_meeting_dt(_first(row, "end_time", "endTime"))
        if start is None or end is None or end <= start:
            continue
        # Ignore date-only/all-day placeholder rows represented as midnight-midnight.
        if start.hour == 0 and start.minute == 0 and end.hour == 0 and end.minute == 0:
            continue
        slots.append((start, end))

    slots = _merge_busy_slots(slots)

    # "هل هو فاضي الساعة 11؟" -> direct yes/no first.
    requested = _time_query_minutes(time_query)
    if requested is not None:
        req_start, req_end = requested
        busy_slot: tuple[datetime, datetime] | None = None

        for start, end in slots:
            s = start.hour * 60 + start.minute
            e = end.hour * 60 + end.minute
            if req_end is None:
                overlaps = s <= req_start < e
            else:
                overlaps = req_start < e and req_end > s
            if overlaps:
                busy_slot = (start, end)
                break

        query_label = _time_query_label(time_query, lang)
        if busy_slot:
            busy_from = _clock_label(busy_slot[0], lang)
            busy_to = _clock_label(busy_slot[1], lang)
            if lang == "ar":
                if subject_name == "أنت":
                    return (
                        f"لا، أنت مشغول {label_ar} عند {query_label}. "
                        f"عندك انشغال مسجل من {busy_from} إلى {busy_to}."
                    )
                return (
                    f"لا، {subject_name} مشغول {label_ar} عند {query_label}. "
                    f"عنده انشغال مسجل من {busy_from} إلى {busy_to}."
                )
            return (
                f"No, {subject_name} isn't available {label_en} at {query_label}. "
                f"There is a busy slot from {busy_from} to {busy_to}."
            )

        if lang == "ar":
            return (
                f"إيه، أنت فاضي {label_ar} عند {query_label} حسب التقويم."
                if subject_name == "أنت"
                else f"إيه، {subject_name} فاضي {label_ar} عند {query_label} حسب التقويم."
            )
        return f"Yes. According to the calendar, {subject_name} is available {label_en} at {query_label}."

    mode_value = str(mode or "busy").strip().casefold()
    preference = ""
    if ":" in mode_value:
        base_mode, preference = mode_value.split(":", 1)
        mode_value = base_mode.strip()
        preference = preference.strip()

    if not slots:
        if lang == "ar":
            pref_ar = {
                "morning": "الصباح",
                "afternoon": "الظهر/العصر",
                "evening": "المساء",
            }.get(preference, "")
            if mode_value == "free" and pref_ar:
                return f"{subject_name} ما عنده أي انشغال مسجل {pref_ar} {label_ar}، فالوقت مفتوح عنده حسب التقويم."
            return f"{subject_name} ما عنده أي انشغال مسجل {label_ar}، فالوقت مفتوح عنده حسب التقويم."
        pref_en = {"morning": "morning", "afternoon": "afternoon", "evening": "evening"}.get(preference, "")
        if mode_value == "free" and pref_en:
            return f"{subject_name} has no busy time recorded in the {pref_en} {label_en}, so that window is open on the calendar."
        return f"{subject_name} has no busy time recorded {label_en}, so the calendar is open."

    # "متى مشغول؟"
    if mode_value != "free":
        if lang == "ar":
            lines = [f"{subject_name} عنده انشغال {label_ar} بهالأوقات:"]
            lines.extend(
                f"- {_clock_label(start, lang)} - {_clock_label(end, lang)}"
                for start, end in slots[:12]
            )
            return "\n".join(lines)

        lines = [f"{subject_name}'s busy times {label_en}:"]
        lines.extend(
            f"- {_clock_label(start, lang)} - {_clock_label(end, lang)}"
            for start, end in slots[:12]
        )
        return "\n".join(lines)

    # "متى فاضي؟" We describe verified gaps around recorded busy slots.
    # A conversational preference (morning/afternoon/evening) filters the result
    # without inventing office hours.
    free_ranges: list[tuple[int | None, int | None, str]] = []
    first_start = slots[0][0]
    first_min = first_start.hour * 60 + first_start.minute
    free_ranges.append((None, first_min, f"قبل {_clock_label(first_start, lang)}" if lang == "ar" else f"before {_clock_label(first_start, lang)}"))

    for (_, previous_end), (next_start, _) in zip(slots, slots[1:]):
        if next_start > previous_end:
            s = previous_end.hour * 60 + previous_end.minute
            e = next_start.hour * 60 + next_start.minute
            label = (
                f"من {_clock_label(previous_end, lang)} إلى {_clock_label(next_start, lang)}"
                if lang == "ar"
                else f"from {_clock_label(previous_end, lang)} to {_clock_label(next_start, lang)}"
            )
            free_ranges.append((s, e, label))

    last_end = slots[-1][1]
    last_min = last_end.hour * 60 + last_end.minute
    free_ranges.append((last_min, None, f"بعد {_clock_label(last_end, lang)}" if lang == "ar" else f"after {_clock_label(last_end, lang)}"))

    def overlaps_preference(start_min: int | None, end_min: int | None, pref: str) -> bool:
        # These are conversational day-parts, not claimed office hours.
        if pref == "morning":
            p_start, p_end = 0, 12 * 60
        elif pref == "afternoon":
            p_start, p_end = 12 * 60, 18 * 60
        elif pref == "evening":
            p_start, p_end = 18 * 60, 24 * 60
        else:
            return True
        s = 0 if start_min is None else start_min
        e = 24 * 60 if end_min is None else end_min
        return s < p_end and e > p_start

    selected = [label for s, e, label in free_ranges if overlaps_preference(s, e, preference)]

    if lang == "ar":
        pref_ar = {
            "morning": "الصباح",
            "afternoon": "الظهر/العصر",
            "evening": "المساء",
        }.get(preference, "")
        if preference:
            if not selected:
                return f"ما بان لي وقت فاضي لـ {subject_name} {label_ar} خلال {pref_ar} حسب التقويم."
            lines = [f"بالـ{pref_ar}، الأوقات الفاضية الظاهرة عند {subject_name} {label_ar}:"]
        else:
            lines = [f"{subject_name} فاضي {label_ar} بهالأوقات:"]
    else:
        pref_en = {"morning": "morning", "afternoon": "afternoon", "evening": "evening"}.get(preference, "")
        if preference:
            if not selected:
                return f"I couldn't find an open time for {subject_name} {label_en} in the {pref_en}, based on the calendar."
            lines = [f"Great — for the {pref_en}, {subject_name} is free {label_en} at:"]
        else:
            lines = [f"{subject_name} is available {label_en} at:"]

    lines.extend(f"- {item}" for item in selected)
    return "\n".join(lines)


def render_tarasol_count(data: dict[str, Any], lang: str, *, subject_name: str = "", self_subject: bool = False) -> str:
    count = data.get("count") if isinstance(data, dict) else None
    if not _valid(count):
        return "عدد المعاملات غير متوفر حاليًا." if lang == "ar" else "The transaction count is currently unavailable."
    if lang == "ar":
        return f"عندك {_show(count)} معاملة في تراسل." if self_subject else f"عند {subject_name} {_show(count)} معاملة في تراسل."
    return f"You have {_show(count)} Tarasol transaction(s)." if self_subject else f"{subject_name} has {_show(count)} Tarasol transaction(s)."


def render_tarasol(data: dict[str, Any], lang: str, *, subject_name: str = "", self_subject: bool = False) -> str:
    rows = _rows(data)
    if not rows:
        return "ما لقيت معاملات في تراسل." if lang == "ar" else "I couldn't find Tarasol transactions."
    if self_subject and lang == "ar":
        heading = "تفاصيل معاملتك في تراسل:" if len(rows) == 1 else "تفاصيل معاملاتك في تراسل:"
    elif lang == "ar":
        heading = f"تفاصيل معاملات {subject_name} في تراسل:"
    elif self_subject:
        heading = "Your Tarasol transaction details:" if len(rows) == 1 else "Your Tarasol transaction details:"
    else:
        heading = f"{subject_name}'s Tarasol transaction details:"
    lines = [heading]
    for index, row in enumerate(rows[:10], 1):
        parts: list[str] = []
        for field in ("subject", "creation_date", "sender", "recipient", "document_type", "priority"):
            value = _field_value(row, field, lang)
            if _valid(value):
                parts.append(f"{FIELD_LABELS[lang].get(field, field)}: {_show(value)}")
        if parts:
            lines.append(f"{index}. " + " — ".join(parts))
    return "\n".join(lines)


def render_reporting_employees(data: dict[str, Any], lang: str, *, subject_name: str = "", self_subject: bool = False) -> str:
    rows = _rows(data)
    visible: list[tuple[str, Any]] = []
    for row in rows[:25]:
        name = _subject_name(row, lang) or _show(_first(row, "employeeName", "namea", "namel") or "")
        title = _field_value(row, "job_title", lang) or _first(row, "jodesca", "jodesce")
        name = str(name or "").strip()
        if not name or name.casefold() in {"none", "null", "n/a", "na", "-"}:
            continue
        visible.append((name, title))

    if not visible:
        if lang == "ar":
            return "ما فيه موظفين مباشرين مسجلين تحتك في النظام حاليًا." if self_subject else f"ما فيه موظفين مباشرين مسجلين تحت {subject_name} حاليًا."
        return "There are no direct reports registered under you right now." if self_subject else f"There are no direct reports registered under {subject_name} right now."

    lines = ["هذي قائمة الموظفين المباشرين:" if lang == "ar" else "Here are the direct reports:"]
    for index, (name, title) in enumerate(visible, 1):
        lines.append(f"{index}. {name}" + (f" — {title}" if _valid(title) else ""))
    return "\n".join(lines)


def _item_title(row: dict[str, Any], lang: str) -> str:
    """Return a useful verified content title across legacy API shapes.

    Field names have changed casing/nesting across RCRC deployments. Search
    known aliases case-insensitively and recurse through small nested wrappers.
    """
    if not isinstance(row, dict):
        return ""

    ar_keys = (
        "_localized_title", "headerAr", "header_ar", "titleAr", "title_ar",
        "subjectAr", "subject_ar", "nameAr", "name_ar", "announcementTitleAr",
        "announcementTitle", "announcementNameAr", "announcementName",
        "descriptionAr", "description_ar", "DescAr", "labelAr", "label_ar",
        "textAr", "text_ar", "header", "title", "subject", "name", "company",
        "description", "details", "label", "text", "announcementTextAr",
        "announcementText", "contentAr", "content",
    )
    en_keys = (
        "_localized_title", "headerEn", "header_en", "titleEn", "title_en",
        "subjectEn", "subject_en", "nameEn", "name_en", "announcementTitleEn",
        "announcementTitle", "announcementNameEn", "announcementName",
        "descriptionEn", "description_en", "DescEn", "labelEn", "label_en",
        "textEn", "text_en", "header", "title", "subject", "name", "company",
        "description", "details", "label", "text", "announcementTextEn",
        "announcementText", "contentEn", "content",
    )
    wanted = ar_keys if lang == "ar" else en_keys

    def clean(value: Any) -> str:
        if not _valid(value) or isinstance(value, (dict, list)):
            return ""
        text = re.sub(r"<[^>]+>", " ", _show(value))
        text = " ".join(text.split())
        if not text or text.isdigit() or text.startswith(("http://", "https://", "data:")):
            return ""
        return text[:220]

    def walk(obj: Any, depth: int = 0) -> str:
        if depth > 3:
            return ""
        if isinstance(obj, dict):
            lower = {str(key).casefold(): (key, value) for key, value in obj.items()}
            for key in wanted:
                pair = lower.get(key.casefold())
                if pair:
                    text = clean(pair[1])
                    if text:
                        return text
            # Unknown legacy key: prefer human-readable text-like fields.
            candidates: list[tuple[int, str]] = []
            for key, value in obj.items():
                key_l = str(key).casefold()
                if isinstance(value, (dict, list)):
                    continue
                if any(token in key_l for token in (
                    "id", "image", "icon", "url", "link", "created", "updated", "date",
                    "start", "end", "status", "disabled", "count", "index", "code",
                )):
                    continue
                text = clean(value)
                if not text:
                    continue
                score = 0
                if any(token in key_l for token in ("title", "header", "subject", "name")):
                    score += 10
                if any(token in key_l for token in ("announcement", "desc", "description", "text", "label", "content")):
                    score += 6
                if lang == "ar" and ARABIC_RE.search(text):
                    score += 3
                if lang == "en" and LATIN_RE.search(text) and not ARABIC_RE.search(text):
                    score += 3
                candidates.append((score, text))
            if candidates:
                candidates.sort(key=lambda item: (-item[0], len(item[1])))
                if candidates[0][0] > 0:
                    return candidates[0][1]
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    text = walk(value, depth + 1)
                    if text:
                        return text
        elif isinstance(obj, list):
            for item in obj[:5]:
                text = walk(item, depth + 1)
                if text:
                    return text
        return ""

    return walk(row)


async def localize_news_rows(rows: list[dict[str, Any]], lang: str, *, limit: int = 5) -> list[dict[str, Any]]:
    visible = [deepcopy(row) for row in rows[:limit]]
    need: list[tuple[int, str]] = []
    for index, row in enumerate(visible):
        title = _item_title(row, lang)
        if not title:
            continue
        if lang == "ar" and LATIN_RE.search(title) and not ARABIC_RE.search(title):
            need.append((index, title))
        elif lang == "en" and ARABIC_RE.search(title) and not LATIN_RE.search(title):
            need.append((index, title))
    if not need:
        return visible
    target = "Arabic" if lang == "ar" else "English"
    try:
        response = await asyncio.to_thread(
            chat,
            [
                {
                    "role": "system",
                    "content": (
                        f"Translate each verified news TITLE to {target}. Return ONLY JSON "
                        '{"translations":["..."]}. Keep the same number and order. '
                        "Do not summarize, add facts, dates, names, or commentary."
                    ),
                },
                {"role": "user", "content": json.dumps({"titles": [title for _, title in need]}, ensure_ascii=False)},
            ],
        )
        content = ((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or ""
        raw = str(content).strip()
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
        translations = parsed.get("translations") if isinstance(parsed, dict) else []
        if isinstance(translations, list) and len(translations) == len(need):
            for (index, _), translated in zip(need, translations):
                text = str(translated or "").strip()
                if text:
                    visible[index]["_localized_title"] = text
    except Exception:
        pass
    return visible


async def render_content_list(intent: str, data: dict[str, Any], lang: str) -> tuple[str, list[dict[str, Any]]]:
    rows = _rows(data)
    if not rows:
        return ("ما لقيت شيء مطابق حاليًا. جرّبي كلمة أو اسم مختلف وأنا أبحث لك." if lang == "ar" else "I couldn't find matching results right now."), []

    if intent == "news":
        rows = await localize_news_rows(rows, lang, limit=5)
        visible = rows[:5]
        heading = "آخر أخبار الهيئة:" if lang == "ar" else "Latest RCRC news:"
        lines = [heading]
        for index, row in enumerate(visible, 1):
            title = _item_title(row, lang) or ("بدون عنوان" if lang == "ar" else "Untitled")
            lines.append(f"{index}. {title}")
        return "\n".join(lines), visible

    if intent == "mazaya":
        # Keep Mazaya concise and user-facing. The API often returns long
        # marketing paragraphs or repeated attachment-only rows; do not dump
        # those raw descriptions into chat.
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if bool(row.get("disabled")):
                continue
            company = str(_first(row, "company", "companyName") or "").strip()
            website = str(_first(row, "website", "url") or "").strip()
            subject = str(_first(row, "subject", "title") or "").strip()
            semantic_identity = "|".join(
                part.casefold() for part in (company, website, subject) if part
            )
            identity = semantic_identity or str(row.get("id") or "").strip()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            unique.append(row)

        heading = (
            "أبرز المزايا والعروض المتاحة لمنسوبي الهيئة:"
            if lang == "ar"
            else "Highlighted employee benefits and offers:"
        )

        generic_details = {
            "مرفق", "مرفق.", "نامل الاطلاع على المرفق", "نأمل الاطلاع على المرفق",
            "please see attachment", "attached", "attachment",
        }
        generic_subjects = {
            "خصومات خاصة لمنسوبي الهيئة",
            "عروض خاصة لمنسوبي الهيئة",
            "special offers for rcrc employees",
            "special discounts for rcrc employees",
        }

        def clean_text(value: str) -> str:
            return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()

        def benefit_summary(subject: str, details: str) -> str:
            subject_clean = clean_text(subject)
            details_clean = clean_text(details)
            combined = " ".join(x for x in (subject_clean, details_clean) if x).strip()

            discount_match = re.search(r"(?<!\d)(\d{1,3})\s*%", combined)
            if discount_match:
                pct = discount_match.group(1)
                return f"خصم {pct}%" if lang == "ar" else f"{pct}% discount"

            if (
                details_clean
                and details_clean.casefold() not in {x.casefold() for x in generic_details}
                and details_clean.casefold() not in {x.casefold() for x in generic_subjects}
                and len(details_clean) <= 140
            ):
                return details_clean

            if (
                subject_clean
                and subject_clean.casefold() not in {x.casefold() for x in generic_subjects}
                and len(subject_clean) <= 140
            ):
                return subject_clean

            return ""

        display_rows: list[dict[str, Any]] = []
        display_lines: list[str] = []

        for row in unique:
            if len(display_rows) >= 10:
                break

            company = clean_text(_first(row, "company", "companyName", "provider", "providerName") or "")
            subject = clean_text(_first(row, "subject", "title") or "")
            details = clean_text(_first(row, "details", "description") or "")
            summary = benefit_summary(subject, details)

            # Never expose a generic placeholder as if it were a real benefit.
            # If the details endpoint still does not tell us what the offer is,
            # omit it rather than showing "خصومات خاصة لمنسوبي الهيئة".
            if not summary:
                continue

            title = company or (subject if subject.casefold() not in {x.casefold() for x in generic_subjects} else "")
            if not title:
                # A verified summary without a provider is still useful.
                title = "عرض" if lang == "ar" else "Offer"

            if summary.casefold() == title.casefold():
                line = title
            else:
                line = f"{title} — {summary}"

            display_rows.append(row)
            display_lines.append(line)

        if not display_rows:
            return (
                "ما لقيت مزايا بتفاصيل واضحة حاليًا."
                if lang == "ar"
                else "I couldn't find benefits with clear offer details right now."
            ), []

        lines = [heading]
        for index, line in enumerate(display_lines, 1):
            lines.append(f"{index}. {line}")

        return "\n".join(lines), display_rows

    if intent == "announcements":
        limit = min(len(rows), 20)
    elif intent == "policy_search":
        limit = 8
    else:
        limit = 10

    headings_ar = {
        "announcements": "لقيت لك الإعلانات التالية:",
        "policy_search": "لقيت لك السياسات والإجراءات التالية:",
    }
    headings_en = {
        "announcements": "Announcements:",
        "policy_search": "Policies and procedures:",
    }
    heading = headings_ar.get(intent, "النتائج:") if lang == "ar" else headings_en.get(intent, "Results:")
    lines = [heading]
    visible_rows: list[dict[str, Any]] = []
    for row in rows[:limit]:
        title = _item_title(row, lang)
        if not title:
            continue
        visible_rows.append(row)
        lines.append(f"{len(visible_rows)}. {title}")
    if not visible_rows:
        if intent == "announcements":
            return (
                "آسف 😅 لقيت الإعلانات، لكن تفاصيلها مو واضحة عندي حاليًا. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار."
                if lang == "ar"
                else "Announcements were retrieved, but no displayable title or text was returned."
            ), []
        if intent == "policy_search":
            return (
                "آسف 😅 لقيت السياسات، لكن تفاصيلها مو واضحة عندي حاليًا. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار."
                if lang == "ar"
                else "Policies were retrieved, but no displayable title was returned."
            ), []
    return "\n".join(lines), visible_rows


def render_employee_search_results(
    rows: list[dict[str, Any]],
    lang: str,
    *,
    query: str = "",
    limit: int = 20,
) -> str:
    """Render employee-directory discovery results as useful person cards in text."""
    clean_rows = [row for row in rows if isinstance(row, dict)]
    if not clean_rows:
        if lang == "ar":
            return f"ما لقيت موظفين مطابقين لـ «{query}»." if query else "ما لقيت موظفين مطابقين."
        return f"I couldn't find employees matching “{query}”." if query else "I couldn't find matching employees."

    shown = clean_rows[:limit]
    if lang == "ar":
        header = f"لقيت {len(clean_rows)} نتيجة لـ «{query}»:" if query else f"لقيت {len(clean_rows)} موظف/موظفة:"
        lines = [header]
        for index, row in enumerate(shown, 1):
            name = _first(row, "name_ar", "name", "name_en")
            title = _first(row, "job_title_ar", "job_title", "job_title_en")
            department = _first(row, "department_ar", "department", "department_en", "general_department")
            pieces = [str(name).strip()] if _valid(name) else ["موظف"]
            if _valid(title):
                pieces.append(str(title).strip())
            if _valid(department):
                pieces.append(str(department).strip())
            lines.append(f"{index}. " + " — ".join(pieces))
        if len(clean_rows) > limit:
            lines.append(f"… وعندي {len(clean_rows) - limit} نتيجة إضافية.")
        return "\n".join(lines)

    header = f"I found {len(clean_rows)} result(s) for “{query}”:" if query else f"I found {len(clean_rows)} employee result(s):"
    lines = [header]
    for index, row in enumerate(shown, 1):
        name = _first(row, "name_en", "name", "name_ar")
        title = _first(row, "job_title_en", "job_title", "job_title_ar")
        department = _first(row, "department_en", "department", "department_ar", "general_department")
        pieces = [str(name).strip()] if _valid(name) else ["Employee"]
        if _valid(title) and not ARABIC_RE.search(str(title)):
            pieces.append(str(title).strip())
        if _valid(department) and not ARABIC_RE.search(str(department)):
            pieces.append(str(department).strip())
        lines.append(f"{index}. " + " — ".join(pieces))
    if len(clean_rows) > limit:
        lines.append(f"… plus {len(clean_rows) - limit} more result(s).")
    return "\n".join(lines)


def render_hr_letter_types(data: Any, lang: str) -> str:
    rows = _rows(data)
    if not rows and isinstance(data, dict):
        rows = [x for x in (data.get("types") or []) if isinstance(x, dict)]
    if not rows:
        return "ما لقيت أنواع خطابات متاحة حاليًا." if lang == "ar" else "I couldn't find available HR letter types."

    lines = [
        "أكيد 😊 أنواع خطابات الموارد البشرية المتاحة:"
        if lang == "ar"
        else "Sure 😊 Available HR letter types:"
    ]
    for index, row in enumerate(rows[:50], 1):
        code = _first(row, "LetterType", "letterType", "letter_type", "code", "value", "id")
        if lang == "ar":
            desc = _first(
                row, "DescAr", "descAr", "descriptionAr", "description_ar",
                "nameAr", "name_ar", "labelAr", "label_ar", "titleAr", "title_ar",
                "description", "name", "label", "title",
            )
        else:
            desc = _first(
                row, "DescEn", "descEn", "descriptionEn", "description_en",
                "nameEn", "name_en", "labelEn", "label_en", "titleEn", "title_en",
                "description", "name", "label", "title",
            )
        if not _valid(desc):
            desc = _item_title(row, lang)
        code_text = _show(code) if _valid(code) else ""
        desc_text = _show(desc) if _valid(desc) else ""
        if code_text and desc_text and code_text.casefold() != desc_text.casefold():
            lines.append(f"{index}. {code_text} — {desc_text}")
        elif desc_text or code_text:
            lines.append(f"{index}. {desc_text or code_text}")
    return "\n".join(lines) if len(lines) > 1 else (
        "تم جلب أنواع الخطابات، لكن الحقول الراجعة غير قابلة للعرض حاليًا."
        if lang == "ar"
        else "The letter types were retrieved, but their returned fields aren't displayable right now."
    )


def render_hr_letter_recipients(data: Any, lang: str) -> str:
    rows = _rows(data)
    if not rows and isinstance(data, dict):
        rows = [x for x in (data.get("recipients") or []) if isinstance(x, dict)]
    if not rows:
        return (
            "ما لقيت جهات متاحة لتوجيه الخطاب حاليًا."
            if lang == "ar"
            else "I couldn't find available HR letter recipients. This part is still experimental."
        )

    lines = [
        "أكيد، تقدر توجه خطاب الموارد البشرية إلى:"
        if lang == "ar"
        else "Sure, the available HR letter recipients are:"
    ]
    for index, row in enumerate(rows[:50], 1):
        if not isinstance(row, dict):
            continue
        lower = {str(key).casefold(): value for key, value in row.items()}
        if lang == "ar":
            description_keys = (
                "descar", "descriptionar", "namear", "labelar", "titlear",
                "recipientnamear", "towhomar", "description", "name", "label", "title",
            )
        else:
            description_keys = (
                "descen", "descriptionen", "nameen", "labelen", "titleen",
                "recipientnameen", "towhomen", "description", "name", "label", "title",
            )
        desc = next((lower.get(key) for key in description_keys if _valid(lower.get(key))), None)
        if not _valid(desc):
            # Some deployments use generic recipientName / recipientText fields.
            desc = next((
                value for key, value in row.items()
                if isinstance(value, str)
                and _valid(value)
                and any(token in str(key).casefold() for token in ("recipient", "desc", "name", "label", "text"))
                and not str(value).strip().isdigit()
            ), None)
        code = _first(row, "Towhom", "ToWhom", "toWhom", "code", "value", "id")
        if _valid(desc):
            lines.append(f"{len(lines)}. {_show(desc)}")
        elif _valid(code):
            lines.append(f"{len(lines)}. {_show(code)}")

    return "\n".join(lines) if len(lines) > 1 else (
        "آسف 😅 لقيت جهات للخطاب، لكن أسماءها مو واضحة عندي حاليًا. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار."
        if lang == "ar"
        else "The recipients were retrieved, but no clear display name was returned. This part is still experimental."
    )



def render_leave_history(data: Any, lang: str, *, limit: int = 20) -> str:
    """Render VacationData records without inventing fields that the API did not return."""
    rows = _rows(data)
    if not rows:
        return (
            "ما لقيت إجازات مسجلة في الفترة المطلوبة."
            if lang == "ar"
            else "I couldn't find any leave records for the requested period."
        )

    def first_value(row: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = row.get(key)
            if _valid(value):
                return value
        return None

    def clean_date(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        # Keep the source value if parsing is uncertain; only remove a trailing
        # midnight/time component when it is clearly ISO-like.
        if "T" in raw:
            raw = raw.split("T", 1)[0]
        elif " " in raw and raw[:10].count("-") == 2:
            raw = raw.split(" ", 1)[0]

        # Some legacy API responses return a date-only value with a trailing
        # UTC/Zulu marker, e.g. "2026-08-17Z". For a date-only UI this marker
        # is not useful, so remove it without changing the calendar date.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}Z", raw, re.I):
            raw = raw[:-1]

        return raw

    if lang == "ar":
        lines = ["سجل الإجازات:"]
    else:
        lines = ["Leave history:"]

    for index, row in enumerate(rows[:limit], 1):
        if lang == "ar":
            leave_type = first_value(
                row, "typeAr", "type_ar", "vacationTypeAr", "leaveTypeAr",
                "vacationType", "leaveType", "type"
            )
            if not _valid(leave_type):
                leave_type = first_value(row, "typeEn", "type_en", "vacationTypeEn", "leaveTypeEn")
        else:
            leave_type = first_value(
                row, "typeEn", "type_en", "vacationTypeEn", "leaveTypeEn",
                "vacationType", "leaveType", "type"
            )
            if not _valid(leave_type):
                leave_type = first_value(row, "typeAr", "type_ar", "vacationTypeAr", "leaveTypeAr")

        status = (
            first_value(row, "statusAr", "status_ar", "statusArabic", "status")
            if lang == "ar"
            else first_value(row, "statusEn", "status_en", "statusEnglish", "status")
        )
        duration = first_value(
            row, "duration", "days", "numberOfDays", "noOfDays",
            "vacationDuration", "leaveDuration"
        )
        start_date = first_value(
            row, "startDate", "start_date", "fromDate", "dateFrom",
            "vacationStartDate", "leaveStartDate", "start"
        )
        end_date = first_value(
            row, "endDate", "end_date", "toDate", "dateTo",
            "vacationEndDate", "leaveEndDate", "end"
        )

        title = _show(_localized_enum(leave_type, lang)) if _valid(leave_type) else (
            "إجازة" if lang == "ar" else "Leave"
        )
        details: list[str] = []

        if _valid(duration):
            try:
                number = float(str(duration).strip())
                if lang == "ar":
                    details.append(_ar_days(number))
                else:
                    number_text = str(int(number)) if number.is_integer() else str(number)
                    details.append(f"{number_text} day(s)")
            except (TypeError, ValueError):
                details.append(
                    f"المدة: {_show(duration)}" if lang == "ar"
                    else f"Duration: {_show(duration)}"
                )

        if _valid(status):
            status_text = _show(_localized_enum(status, lang))
            details.append(
                f"الحالة: {status_text}" if lang == "ar"
                else f"Status: {status_text}"
            )

        start_text = clean_date(start_date)
        end_text = clean_date(end_date)
        if start_text and end_text:
            details.append(
                f"من {start_text} إلى {end_text}"
                if lang == "ar"
                else f"From {start_text} to {end_text}"
            )
        elif start_text:
            details.append(
                f"تبدأ {start_text}" if lang == "ar"
                else f"Starts {start_text}"
            )
        elif end_text:
            details.append(
                f"تنتهي {end_text}" if lang == "ar"
                else f"Ends {end_text}"
            )

        lines.append(
            f"{index}. {title}" + (f" — {' — '.join(details)}" if details else "")
        )

    if len(rows) > limit:
        remaining = len(rows) - limit
        lines.append(
            f"وفيه {remaining} سجل إضافي."
            if lang == "ar"
            else f"There are {remaining} more record(s)."
        )

    return "\n".join(lines)


def render_generic_list(data: Any, lang: str, *, limit: int = 20) -> str:
    rows = _rows(data)
    if not rows:
        return "ما لقيت بيانات متاحة حاليًا. إذا تبين، جرّبي بصياغة ثانية وأنا أدور لك." if lang == "ar" else "I couldn't find available data right now."
    lines = ["لقيت لك هذي النتائج:" if lang == "ar" else "I found these results:"]
    for index, row in enumerate(rows[:limit], 1):
        values: list[str] = []
        looks_employee = any(_valid(row.get(key)) for key in ("username", "userid", "name_ar", "name_en", "job_title_ar", "job_title_en", "department_ar", "department_en"))
        if looks_employee:
            if lang == "ar":
                name = _first(row, "name_ar", "name")
                title = _first(row, "job_title_ar", "job_title")
            else:
                name = _first(row, "name_en", "name")
                title = _first(row, "job_title_en", "job_title")
                if name and ARABIC_RE.search(str(name)): name = None
                if title and ARABIC_RE.search(str(title)): title = None
            if _valid(name): values.append(_show(name))
            if _valid(title) and str(title).casefold() != str(name or "").casefold(): values.append(_show(title))
        else:
            preferred = (("title_ar", "headerAr", "subject_ar", "name_ar", "title", "subject", "description") if lang == "ar" else ("title_en", "headerEn", "subject_en", "name_en", "title", "subject", "description"))
            for key in preferred:
                value = row.get(key)
                if not _valid(value): continue
                if lang == "en" and ARABIC_RE.search(str(value)): continue
                if lang == "ar" and LATIN_RE.search(str(value)) and not ARABIC_RE.search(str(value)): continue
                values.append(_show(value))
                if len(values) >= 2: break
        if not values:
            fallback = _item_title(row, lang)
            if fallback:
                values.append(fallback)
        if not values:
            values.append("بيانات متاحة" if lang == "ar" else "Available data")
        lines.append(f"{index}. " + " — ".join(values[:2]))
    return "\n".join(lines)


async def summarize_verified(items: list[dict[str, Any]], intent: str, lang: str) -> str:
    compact: list[dict[str, Any]] = []
    for row in items[:20]:
        if not isinstance(row, dict):
            continue
        compact.append({
            key: row.get(key)
            for key in ("headerAr", "headerEn", "header", "title", "subject", "name", "name_ar", "name_en", "company", "details", "description")
            if _valid(row.get(key))
        })
    if not compact:
        return "ما عندي نتيجة سابقة أقدر ألخصها." if lang == "ar" else "I don't have a previous result I can summarize."
    language_name = "Arabic" if lang == "ar" else "English"
    try:
        response = await asyncio.to_thread(
            chat,
            [
                {
                    "role": "system",
                    "content": (
                        f"Summarize VERIFIED_ITEMS in {language_name} in at most 5 concise bullets. "
                        "Use only facts present in VERIFIED_ITEMS. Do not add facts, prices, dates, eligibility, or recommendations. "
                        "Return plain text only."
                    ),
                },
                {"role": "user", "content": json.dumps({"intent": intent, "VERIFIED_ITEMS": compact}, ensure_ascii=False)},
            ],
        )
        answer = str(((response.get("message") or {}).get("content") if isinstance(response, dict) else "") or "").strip()
        if answer:
            return answer
    except Exception:
        pass
    # Deterministic fallback: first five verified titles.
    lines = ["باختصار:" if lang == "ar" else "In short:"]
    for row in compact[:5]:
        title = _item_title(row, lang) or next((str(v) for v in row.values() if _valid(v)), "")
        if title:
            lines.append(f"- {title}")
    return "\n".join(lines)


def _attendance_date_label(value: Any, lang: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        date_part = raw[:10]
        dt = datetime.fromisoformat(date_part)
        if lang == "ar":
            return dt.strftime("%d/%m/%Y")
        return dt.strftime("%d %b %Y")
    except Exception:
        return raw


def render_attendance(
    data: Any,
    lang: str,
    *,
    subject_name: str = "",
    self_subject: bool = False,
    period: str = "today",
    intent: str = "attendance_today",
    mode: str = "attendance:summary",
) -> str:
    rows = _rows(data)
    mismatch = bool(data.get("identity_mismatch")) if isinstance(data, dict) else False
    if mismatch:
        return "تعذر التحقق من سجل حضورك بشكل آمن حاليًا." if lang == "ar" else "I couldn't safely verify your attendance record right now."

    period_ar = {
        "today": "اليوم",
        "yesterday": "أمس",
        "this_week": "هذا الأسبوع",
        "last_week": "الأسبوع الماضي",
        "this_month": "هذا الشهر",
        "last_month": "الشهر الماضي",
        "custom": "للفترة المطلوبة",
    }.get(period, "للفترة المطلوبة")
    period_en = {
        "today": "today",
        "yesterday": "yesterday",
        "this_week": "this week",
        "last_week": "last week",
        "this_month": "this month",
        "last_month": "last month",
        "custom": "for the requested period",
    }.get(period, "for the requested period")

    if not rows:
        if lang == "ar":
            return f"ما لقيت سجل حضور لك {period_ar}." if self_subject else f"ما لقيت سجل حضور {period_ar}."
        return f"I couldn't find an attendance record for you {period_en}." if self_subject else f"I couldn't find attendance data {period_en}."

    # Yesterday is a single-day lookup, so keep the same compact shape as
    # today's attendance instead of showing a numbered one-item history list.
    if period == "yesterday" and len(rows) == 1:
        row = rows[0]
        check_in = _first(row, "startStamp", "checkIn", "checkInTime", "inTime")
        check_out = _first(row, "endStamp", "checkOut", "checkOutTime", "outTime")
        working = _first(row, "workingHours", "working_hours", "totalHours")
        status = _first(row, "status", "attendanceStatus")
        vacation = _first(row, "vacationType", "leaveType")
        if mode == "attendance:check_in":
            if _valid(check_in):
                return (f"حضرت أمس الساعة {_clock_label(check_in, 'ar')}." if lang == "ar" else f"You checked in yesterday at {_clock_label(check_in, 'en')}.")
            return ("ما ظهر وقت دخول مسجل لك أمس." if lang == "ar" else "I couldn't find a recorded check-in time for you yesterday.")
        if mode == "attendance:check_out":
            if _valid(check_out):
                return (f"طلعت أمس الساعة {_clock_label(check_out, 'ar')}." if lang == "ar" else f"You checked out yesterday at {_clock_label(check_out, 'en')}.")
            return ("ما ظهر وقت خروج مسجل لك أمس." if lang == "ar" else "I couldn't find a recorded check-out time for you yesterday.")
        if lang == "ar":
            title = "ساعات عملك أمس:" if intent == "working_hours" else "حضورك أمس:"
            lines = [title]
            if intent != "working_hours" and _valid(check_in): lines.append(f"- وقت الدخول: {_clock_label(check_in, 'ar')}")
            if intent != "working_hours" and _valid(check_out): lines.append(f"- وقت الخروج: {_clock_label(check_out, 'ar')}")
            if _valid(working): lines.append(f"- ساعات العمل: {_show(working)}")
            if intent != "working_hours" and _valid(status): lines.append(f"- الحالة: {_show(_localized_enum(status, lang))}")
            if intent != "working_hours" and _valid(vacation): lines.append(f"- الإجازة: {_show(_localized_enum(vacation, lang))}")
            if len(lines) == 1:
                lines.append("- ما ظهر وقت دخول أو خروج أو ساعات عمل في البيانات الراجعة من النظام.")
            return "\n".join(lines)
        title = "Your working hours yesterday:" if intent == "working_hours" else "Your attendance yesterday:"
        lines = [title]
        if intent != "working_hours" and _valid(check_in): lines.append(f"- Check-in: {_clock_label(check_in, 'en')}")
        if intent != "working_hours" and _valid(check_out): lines.append(f"- Check-out: {_clock_label(check_out, 'en')}")
        if _valid(working): lines.append(f"- Working hours: {_show(working)}")
        if intent != "working_hours" and _valid(status) and not ARABIC_RE.search(str(status)): lines.append(f"- Status: {_show(_localized_enum(status, lang))}")
        return "\n".join(lines)

    # History/range requests: show every returned day instead of silently
    # taking rows[0] and labelling it as today's attendance.
    if period != "today" or len(rows) > 1:
        if lang == "ar":
            ar_range = {
                "this_week": "لهذا الأسبوع",
                "last_week": "للأسبوع الماضي",
                "this_month": "لهذا الشهر",
                "last_month": "للشهر الماضي",
                "custom": "للفترة المطلوبة",
            }.get(period, "للفترة المطلوبة")
            title = (f"ساعات عملك {ar_range}:" if intent == "working_hours" else f"جدول حضورك {ar_range}:")
        else:
            title = (f"Your working hours {period_en}:" if intent == "working_hours" else f"Your attendance {period_en}:")
        lines = [title]
        for index, row in enumerate(rows, 1):
            date_value = _first(row, "date", "attendanceDate", "day")
            day_value = _first(row, "dayText", "dayName")
            check_in = _first(row, "startStamp", "checkIn", "checkInTime", "inTime")
            check_out = _first(row, "endStamp", "checkOut", "checkOutTime", "outTime")
            working = _first(row, "workingHours", "working_hours", "totalHours")
            status = _first(row, "status", "attendanceStatus")
            vacation = _first(row, "vacationType", "leaveType")

            date_label = _attendance_date_label(date_value, lang)
            if lang == "ar":
                pieces = []
                if intent != "working_hours" and _valid(check_in): pieces.append(f"الدخول {_clock_label(check_in, 'ar')}")
                if intent != "working_hours" and _valid(check_out): pieces.append(f"الخروج {_clock_label(check_out, 'ar')}")
                if _valid(working): pieces.append(f"ساعات العمل {_show(working)}")
                if intent != "working_hours" and _valid(status): pieces.append(f"الحالة {_show(_localized_enum(status, lang))}")
                if intent != "working_hours" and _valid(vacation): pieces.append(f"الإجازة {_show(_localized_enum(vacation, lang))}")
                prefix_parts = []
                if _valid(day_value) and ARABIC_RE.search(str(day_value)):
                    prefix_parts.append(str(day_value).strip())
                if date_label:
                    prefix_parts.append(date_label)
                prefix = " ".join(prefix_parts) or f"اليوم {index}"
                lines.append(f"{index}. {prefix}" + (f" — {'، '.join(pieces)}" if pieces else ""))
            else:
                pieces = []
                if intent != "working_hours" and _valid(check_in): pieces.append(f"check-in {_clock_label(check_in, 'en')}")
                if intent != "working_hours" and _valid(check_out): pieces.append(f"check-out {_clock_label(check_out, 'en')}")
                if _valid(working): pieces.append(f"working hours {_show(working)}")
                if intent != "working_hours" and _valid(status) and not ARABIC_RE.search(str(status)): pieces.append(f"status {_show(_localized_enum(status, lang))}")
                prefix = date_label or (str(day_value).strip() if _valid(day_value) and not ARABIC_RE.search(str(day_value)) else f"Day {index}")
                lines.append(f"{index}. {prefix}" + (f" — {', '.join(pieces)}" if pieces else ""))
        return "\n".join(lines)

    row = rows[0]
    date_value = _first(row, "date", "attendanceDate", "day")
    day_value = _first(row, "dayText", "dayName")
    check_in = _first(row, "startStamp", "checkIn", "checkInTime", "inTime")
    check_out = _first(row, "endStamp", "checkOut", "checkOutTime", "outTime")
    working = _first(row, "workingHours", "working_hours", "totalHours")
    status = _first(row, "status", "attendanceStatus")
    vacation = _first(row, "vacationType", "leaveType")
    if mode == "attendance:check_in":
        if _valid(check_in):
            return (f"حضرت اليوم الساعة {_clock_label(check_in, 'ar')}." if lang == "ar" else f"You checked in today at {_clock_label(check_in, 'en')}.")
        if lang == "ar":
            suffix = f" والحالة الحالية في النظام: {_show(_localized_enum(status, lang))}." if _valid(status) else "."
            return "ما ظهر وقت دخول مسجل لك اليوم" + suffix
        suffix = f" The current system status is {_show(_localized_enum(status, lang))}." if _valid(status) and not ARABIC_RE.search(str(status)) else ""
        return "I couldn't find a recorded check-in time for you today." + suffix
    if mode == "attendance:check_out":
        if _valid(check_out):
            return (f"طلعت اليوم الساعة {_clock_label(check_out, 'ar')}." if lang == "ar" else f"You checked out today at {_clock_label(check_out, 'en')}.")
        if lang == "ar":
            suffix = f" والحالة الحالية في النظام: {_show(_localized_enum(status, lang))}." if _valid(status) else "."
            return "ما ظهر وقت خروج مسجل لك اليوم" + suffix
        suffix = f" The current system status is {_show(_localized_enum(status, lang))}." if _valid(status) and not ARABIC_RE.search(str(status)) else ""
        return "I couldn't find a recorded check-out time for you today." + suffix
    if lang == "ar":
        lines = ["حضورك اليوم:" if self_subject else f"حضور {subject_name}:"]
        if _valid(check_in): lines.append(f"- وقت الدخول: {_clock_label(check_in, 'ar')}")
        if _valid(check_out): lines.append(f"- وقت الخروج: {_clock_label(check_out, 'ar')}")
        if _valid(working): lines.append(f"- ساعات العمل: {_show(working)}")
        if _valid(status): lines.append(f"- الحالة: {_show(_localized_enum(status, lang))}")
        if _valid(vacation): lines.append(f"- الإجازة: {_show(_localized_enum(vacation, lang))}")
        if len(lines) == 1:
            date_label = _attendance_date_label(date_value or day_value, "ar")
            lines.append("- ما ظهر وقت دخول أو خروج أو ساعات عمل في البيانات الراجعة من النظام.")
            if date_label:
                lines.append(f"- التاريخ: {date_label}")
        return "\n".join(lines)
    lines = ["Your attendance today:" if self_subject else f"{subject_name}'s attendance:"]
    if _valid(check_in): lines.append(f"- Check-in: {_clock_label(check_in, 'en')}")
    if _valid(check_out): lines.append(f"- Check-out: {_clock_label(check_out, 'en')}")
    if _valid(working): lines.append(f"- Working hours: {_show(working)}")
    if _valid(status) and not ARABIC_RE.search(str(status)): lines.append(f"- Status: {_show(_localized_enum(status, lang))}")
    if len(lines) == 1:
        date_label = _attendance_date_label(date_value or day_value, "en")
        lines.append("- No check-in, check-out, or working-hours details were returned by the system.")
        if date_label:
            lines.append(f"- Date: {date_label}")
    return "\n".join(lines)



def _detail_root(data: Any) -> dict[str, Any]:
    """Unwrap common detail-endpoint wrappers without inventing fields."""
    obj = data
    for _ in range(5):
        if not isinstance(obj, dict):
            return {}
        # Explicit wrappers used by the RCRC detail APIs.
        for key in (
            "item", "data", "result", "news", "announcement", "benefit",
            "mazaya", "details", "newsDetails", "announcementDetails",
        ):
            value = obj.get(key)
            if isinstance(value, dict):
                obj = value
                break
            if isinstance(value, list) and value and isinstance(value[0], dict):
                obj = value[0]
                break
        else:
            # If this is only a one-key wrapper, unwrap it.
            nested = [v for v in obj.values() if isinstance(v, dict)]
            scalars = [v for v in obj.values() if not isinstance(v, (dict, list)) and _valid(v)]
            if len(nested) == 1 and not scalars:
                obj = nested[0]
                continue
            return obj
        continue
    return obj if isinstance(obj, dict) else {}


def _detail_text(row: dict[str, Any], lang: str) -> str:
    preferred = (
        ("descriptionAr", "description_ar", "detailsAr", "details_ar", "contentAr", "content_ar",
         "bodyAr", "body_ar", "textAr", "text_ar", "DescAr", "description", "details", "content", "body", "text")
        if lang == "ar"
        else
        ("descriptionEn", "description_en", "detailsEn", "details_en", "contentEn", "content_en",
         "bodyEn", "body_en", "textEn", "text_en", "DescEn", "description", "details", "content", "body", "text")
    )
    lower = {str(k).casefold(): v for k, v in row.items()}
    for key in preferred:
        value = lower.get(key.casefold())
        if not _valid(value) or isinstance(value, (dict, list)):
            continue
        text = re.sub(r"<[^>]+>", " ", _show(value))
        text = " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()
        if text and not text.startswith("data:image/"):
            return text
    return ""


def render_content_detail(intent: str, data: Any, lang: str) -> str:
    """Render one verified policy/news/announcement/Mazaya detail item."""
    row = _detail_root(data)
    if not row:
        return (
            "وصلت للعنصر، لكن تفاصيله مو متاحة للعرض حاليًا."
            if lang == "ar"
            else "The data was retrieved, but there are no displayable details right now."
        )

    title = _item_title(row, lang)
    detail = _detail_text(row, lang)
    company = str(_first(row, "company", "companyName", "provider", "providerName") or "").strip()
    website = str(_first(row, "website", "url", "link") or "").strip()

    if lang == "ar":
        headings = {
            "news_details": "هذا اللي لقيته عن الخبر:",
            "announcement_details": "هذا اللي لقيته عن الإعلان:",
            "policy_details": "هذا اللي لقيته عن السياسة:",
            "mazaya_details": "هذي تفاصيل الميزة اللي طلبتها:",
        }
    else:
        headings = {
            "news_details": "News details:",
            "announcement_details": "Announcement details:",
            "policy_details": "Policy details:",
            "mazaya_details": "Benefit details:",
        }

    lines = [headings.get(intent, "التفاصيل:" if lang == "ar" else "Details:")]
    if title:
        lines.append(title)
    if company and company.casefold() != title.casefold():
        lines.append(("الجهة: " if lang == "ar" else "Provider: ") + company)
    if detail and detail.casefold() != title.casefold():
        lines.append(detail)
    if website.startswith(("http://", "https://")):
        if intent == "mazaya_details":
            lines.append(
                f"الموقع: [زيارة الموقع]({website})"
                if lang == "ar"
                else f"Website: [Open website]({website})"
            )
        else:
            lines.append(("الموقع: " if lang == "ar" else "Website: ") + website)

    if len(lines) == 1:
        # Safe scalar fallback. Exclude IDs, metadata, images and internal codes.
        for key, value in row.items():
            key_l = str(key).casefold()
            if isinstance(value, (dict, list)) or not _valid(value):
                continue
            if any(token in key_l for token in ("id", "metadata", "image", "icon", "code", "token")):
                continue
            text = _show(value)
            if text.startswith("data:image/"):
                continue
            lines.append(text)
            if len(lines) >= 5:
                break

    return "\n".join(lines) if len(lines) > 1 else (
        "وصلت للعنصر، لكن تفاصيله مو متاحة للعرض حاليًا."
        if lang == "ar"
        else "The data was retrieved, but there are no displayable details right now."
    )

def render_generic(data: Any, lang: str) -> str:
    if isinstance(data, dict):
        rows = _rows(data)
        if rows:
            return render_generic_list(data, lang)
        pairs = []
        for k, v in data.items():
            if not isinstance(v, (str, int, float, bool)) or not _valid(v):
                continue
            text = _show(v)
            if text.startswith("data:image/"):
                continue
            pairs.append((k, v))
        if pairs:
            return "\n".join(f"- {k}: {_show(v)}" for k, v in pairs[:8])
    return "وصلت للعنصر، لكن تفاصيله مو متاحة للعرض حاليًا." if lang == "ar" else "The data was retrieved, but there are no displayable details right now."
