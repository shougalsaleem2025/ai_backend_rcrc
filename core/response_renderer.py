from __future__ import annotations

import asyncio
import json
import re
import math
from datetime import datetime, timedelta, timezone
from copy import deepcopy
from typing import Any

from core.llm_client import chat
from core.semantic_models import ParsedRequest


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")


FRIENDLY_RENDERER_BUILD = "rcrc-v249-natural-list-closings-2026-10-15"


def _natural_opening(kind: str, language: str, seed: str = "") -> str:
    """Stable, varied openings without randomness or changing factual content."""
    import hashlib
    choices = {
        "announcements": ("حاضر! 📢 هذي أحدث إعلانات تواصل:", "تمام! 📢 هذي الإعلانات المتاحة:", "هذي آخر إعلانات تواصل 📢:"),
        "mazaya": ("تفضل! 🎁 هذي أبرز المزايا والعروض المتاحة:", "حاضر! 🎁 هذي العروض المتاحة لمنسوبي الهيئة:", "هذي أبرز المزايا والعروض ✨🎁:"),
        "news": ("حاضر! 📰 هذي آخر أخبار الهيئة:", "تمام! 📰 هذي أحدث الأخبار:", "هذي أبرز أخبار الهيئة 📰:"),
        "employee": ("حاضر! 👤 هذي بيانات", "تفضل، هذي معلومات", "هذي تفاصيل"),
    }
    english = {
        "announcements": ("Here are the latest announcements 📢:", "These announcements are currently available 📢:"),
        "mazaya": ("Here are the available employee offers 🎁:", "Take a look at these benefits 🎁:"),
        "news": ("Here are the latest RCRC updates 📰:", "Latest RCRC news 📰:"),
        "employee": ("Here are the details for", "Employee details for"),
    }
    pool = (choices if language == "ar" else english).get(kind)
    if not pool:
        return ""
    digest = hashlib.sha256((kind + ":" + seed).encode("utf-8")).digest()
    return pool[int.from_bytes(digest[:4], "big") % len(pool)]


# User-facing localization for common API enum values. Identifiers such as
# emails/usernames are intentionally preserved, but prose/status values follow
# the language of the current user turn.
_AR_ENUMS = {
    "absence": "غياب",
    "absent": "غائب",
    "present": "حاضر",
    "late": "تأخير",
    "delay": "تأخير",
    "delayed": "تأخير",
    "forget your finger print": "نسيان تسجيل البصمة",
    "forget your fingerprint": "نسيان تسجيل البصمة",
    "forgot your finger print": "نسيان تسجيل البصمة",
    "forgot your fingerprint": "نسيان تسجيل البصمة",
    "missing punch": "نسيان تسجيل البصمة",
    "missing fingerprint": "نسيان تسجيل البصمة",
    "leave": "إجازة",
    "vacation": "إجازة",
    "sick leave": "إجازة مرضية",
    "annual leave": "إجازة سنوية",
    "sick": "إجازة مرضية",
    "emergency leave": "إجازة طارئة",
    "urgent leave": "إجازة اضطرارية",
    "official holiday": "إجازة رسمية",
    "public holiday": "إجازة رسمية",
    "holiday": "إجازة رسمية",
    "off day": "إجازة نهاية الأسبوع",
    "weekend": "إجازة نهاية الأسبوع",
    "rest day": "يوم راحة",
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
    "إجازة مرضية": "Sick leave",
    "إجازة طارئة": "Emergency leave",
    "إجازة اضطرارية": "Urgent leave",
    "إجازة رسمية": "Official holiday",
    "إجازة نهاية الأسبوع": "Weekend / Off day",
    "يوم راحة": "Rest day",
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

def _looks_like_internal_profile_name(value: Any) -> bool:
    """Hide technical account labels from conversational profile answers."""
    text = str(value or "").strip()
    if not text:
        return True
    normalized = re.sub(r"\s+", " ", text).casefold()
    return bool(
        re.fullmatch(r"(?:rc)?user\s*\d*", normalized, re.I)
        or re.fullmatch(r"[a-z]{1,20}[._-][a-z0-9._-]{1,40}", normalized, re.I)
        or "\\" in text
    )


def _pretty_english_work_text(value: Any) -> str:
    """Lightly normalize common enterprise abbreviations without rewriting facts."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return text
    replacements = [
        (r"\bSW\b", "Software"),
        (r"\bSr\.?\b", "Senior"),
        (r"\bJr\.?\b", "Junior"),
        (r"\bDept\.?\b", "Department"),
        (r"\bDepartmen$", "Department"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    # Preserve known acronyms if title-casing was supplied inconsistently.
    for acronym in ("RCRC", "AI", "IT", "ICT", "HR", "PMO"):
        text = re.sub(rf"\b{acronym}\b", acronym, text, flags=re.I)
    return text

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
        text = value.strip()
        if not text:
            return False
        if text.casefold() in {"-", "--", "---", "n/a", "na", "none", "null", "undefined"}:
            return False
        return True
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


def _short_person_name(value: Any) -> str:
    """
    Friendly employee-name display:
      - default: first name + family name
      - full name remains available to callers when ambiguity exists

    Examples:
      عماد يوسف سلمان الهاجري -> عماد الهاجري
      Naser Ali Alsaleh      -> Naser Alsaleh
    """
    raw = re.sub(r"\s+", " ", str(value or "").strip())
    if not raw:
        return ""

    parts = raw.split(" ")
    if len(parts) <= 2:
        return raw

    # Preserve common Arabic compound first names when written separately.
    compound_second = {
        "الرحمن", "الرحيم", "العزيز", "الله", "المجيد", "المحسن",
        "الكريم", "القادر", "الوهاب", "الرزاق", "اللطيف", "الملك",
    }
    if parts[0] == "عبد" and parts[1] in compound_second and len(parts) >= 4:
        return f"{parts[0]} {parts[1]} {parts[-1]}"

    return f"{parts[0]} {parts[-1]}"


def _person_display_names(names: list[str]) -> list[str]:
    """
    Shorten names unless two or more people collapse to the same first+family
    display. Colliding employees keep their full names for disambiguation.
    """
    clean = [re.sub(r"\s+", " ", str(name or "").strip()) for name in names]
    short = [_short_person_name(name) for name in clean]
    counts = Counter(name.casefold() for name in short if name)

    result: list[str] = []
    for full, compact in zip(clean, short):
        if compact and counts.get(compact.casefold(), 0) > 1:
            result.append(full)
        else:
            result.append(compact or full)
    return result


def _subject_name(row: dict[str, Any], lang: str, *, fallback: str = "") -> str:
    full = str(_field_value(row, "name", lang) or fallback or "").strip()
    return _short_person_name(full)


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
            if fields == ["extension"]:
                if self_subject:
                    return "ما ظهرت لي تحويلة مسجلة لك حاليًا 😊"
                if str(request.query or "").strip().lower() == "employee:focus_fields":
                    return "ما ظهرت لي تحويلة مسجلة له حاليًا 😊\nإذا تحتاج، أقدر أطلع لك الإيميل أو رقم الجوال بدلها."
                if name:
                    return f"ما ظهرت لي تحويلة مسجلة لـ {name} حاليًا 😊"
                return "ما ظهرت لي تحويلة مسجلة لهذا الموظف حاليًا 😊"
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
                    if not subject_name:
                        return (
                            f"مديرك المباشر هو **{_short_person_name(value)}** 😊\n"
                            "إذا تحتاج بيانات التواصل أو منصبه، أطلعها لك."
                        )
                    return f"مدير {_short_person_name(subject_name)} المباشر هو **{_short_person_name(value)}**."
                return f"أنت {value}." if self_subject else f"اسمه {value}."
            focus_followup = str(request.query or "").strip().lower() == "employee:focus_fields"

            if field == "email":
                if self_subject:
                    return f"أكيد 😊 إيميلك هو {value}."
                if manager_result:
                    return f"أكيد 😊 مديرك المباشر هو **{_short_person_name(name)}**، وإيميله **{value}** 📧"
                if focus_followup:
                    return f"إيميله هو **{value}** 📧"
                return f"أكيد 😊 إيميل {name} هو {value}."

            if field == "mobile":
                if self_subject:
                    return f"أكيد 😊 رقم جوالك هو {value}."
                if manager_result:
                    return f"أكيد 😊 مديرك المباشر هو **{_short_person_name(name)}**، ورقم جواله **{value}** 📱"
                if focus_followup:
                    return f"رقم جواله هو **{value}** 📱"
                return f"أكيد 😊 رقم جوال {name} هو {value}."

            if field == "extension":
                if self_subject:
                    return f"أكيد 😊 تحويلتك هي {value}."
                if manager_result:
                    return f"أكيد 😊 مديرك المباشر هو **{_short_person_name(name)}**، وتحويلته **{value}** ☎️"
                if focus_followup:
                    return f"تحويلته هي **{value}** ☎️"
                return f"أكيد 😊 تحويلة {name} هي {value}."

            if field == "job_title":
                if self_subject:
                    return f"أكيد 😊 مسمّاك الوظيفي هو {value}."
                if manager_result:
                    return f"أكيد 😊 مديرك المباشر هو **{_short_person_name(name)}**، ومنصبه **{value}**."
                if focus_followup:
                    return f"منصبه **{value}** 😊\nإذا تحتاج إدارته أو بيانات التواصل، أطلعها لك."
                return f"أكيد 😊 منصب {name} هو {value}."

            if field == "department":
                if self_subject:
                    return f"أكيد 😊 إدارتك هي {value}."
                if manager_result:
                    return f"أكيد 😊 مديرك المباشر هو {name}، ويتبع {value}."
                if focus_followup:
                    return f"يتبع {value} 😊"
                return f"أكيد 😊 {name} يتبع {value}."
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
                    f"Sure 😊 Your direct manager is {value}."
                    if not subject_name
                    else f"Sure 😊 {subject_name}'s direct manager is {value}."
                )
            if self_subject and _looks_like_internal_profile_name(value):
                return "Sure 😊 I found your profile."
            return f"Sure 😊 You're {value}." if self_subject else f"Sure 😊 The employee's name is {value}."
        if field == "email":
            if self_subject:
                return f"Sure 😊 Your email is {value}."
            if focus_followup:
                return f"Sure 😊 The email is {value}."
            return f"Sure 😊 {name}'s email is {value}."
        if field == "mobile":
            if self_subject:
                return f"Sure 😊 Your mobile number is {value}."
            if focus_followup:
                return f"Sure 😊 The mobile number is {value}."
            return f"Sure 😊 {name}'s mobile number is {value}."
        if field == "extension":
            if self_subject:
                return f"Sure 😊 Your extension is {value}."
            if focus_followup:
                return f"Sure 😊 The extension is {value}."
            return f"Sure 😊 {name}'s extension is {value}."
        if field == "job_title":
            pretty_value = _pretty_english_work_text(value)
            return f"Sure 😊 Your job title is {pretty_value}." if self_subject else f"Sure 😊 {name}'s job title is {pretty_value}."
        if field == "department":
            pretty_value = _pretty_english_work_text(value)
            return f"Sure 😊 Your department is {pretty_value}." if self_subject else f"Sure 😊 {name}'s department is {pretty_value}."
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

        raw_name = _show(values.get("name")) if _valid(values.get("name")) else ""
        safe_name = "" if _looks_like_internal_profile_name(raw_name) else raw_name
        job_title = _pretty_english_work_text(values.get("job_title")) if values.get("job_title") else ""
        department = _pretty_english_work_text(values.get("department")) if values.get("department") else ""

        if profile_mode == "profile:recognition":
            if safe_name and job_title and department:
                return f"Of course 😊 I found your profile. You're {safe_name}, a {job_title} in the {department}."
            if job_title and department:
                return f"Of course 😊 I found your profile. You're a {job_title} in the {department}."
            if safe_name and job_title:
                return f"Of course 😊 I found your profile. You're {safe_name}, a {job_title}."
            if job_title:
                return f"Of course 😊 I found your profile. You're a {job_title}."
            if safe_name and department:
                return f"Of course 😊 I found your profile. You're {safe_name} and work in the {department}."
            if department:
                return f"Of course 😊 I found your profile. You work in the {department}."
            if safe_name:
                return f"Of course 😊 I found your profile. You're {safe_name}."
            return "Of course 😊 I found your profile."

        if safe_name and job_title and department:
            return f"You're {safe_name}, a {job_title} in the {department}."
        if job_title and department:
            return f"You're a {job_title} in the {department}."
        if safe_name:
            return f"You're {safe_name}."
        return "I found your profile."

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

    # Two or three explicitly requested profile fields should feel like a
    # friendly employee card instead of a compressed sentence.
    if explicit_fields and 2 <= len(fields) <= 3:
        if lang == "ar":
            lines: list[str] = []
            if self_subject:
                lines.append("هذي معلوماتك 👤:")
            elif manager_result and name:
                lines.append(f"{_natural_opening('employee', lang, _short_person_name(name))} **{_short_person_name(name)}**:")
            elif name:
                lines.append(f"{_natural_opening('employee', lang, _short_person_name(name))} **{_short_person_name(name)}**:")
            else:
                lines.append("هذي المعلومات المتوفرة:")

            for field in fields:
                raw_value = _field_value(data, field, lang)
                value = _show(raw_value) if _valid(raw_value) else ""
                if field == "email":
                    lines.append(f"📧 الإيميل: **{value}**" if value else "📧 الإيميل: غير متوفر حاليًا")
                elif field == "mobile":
                    lines.append(f"📱 الجوال: **{value}**" if value else "📱 الجوال: غير متوفر حاليًا")
                elif field == "extension":
                    lines.append(f"☎️ التحويلة: **{value}**" if value else "☎️ التحويلة: غير متوفرة حاليًا")
                elif field == "job_title":
                    lines.append(f"المسمى الوظيفي: {value}" if value else "المسمى الوظيفي: غير متوفر حاليًا")
                elif field == "department":
                    lines.append(f"الإدارة: {value}" if value else "الإدارة: غير متوفرة حاليًا")
                elif field == "name":
                    if self_subject or not name or not value or value.strip().casefold() != name.strip().casefold():
                        lines.append(f"الاسم: {value}" if value else "الاسم: غير متوفر حاليًا")
                else:
                    label = FIELD_LABELS["ar"].get(field, field)
                    lines.append(f"{label}: {value}" if value else f"{label}: غير متوفر حاليًا")

            if not self_subject and any(field in {"email", "mobile", "extension"} for field, _ in available):
                lines.extend(["", "إذا تحتاج منصبه أو إدارته بعد، أطلعها لك."])
            return "\n".join(lines)

        lines: list[str] = []
        if self_subject:
            lines.append("Sure 😊 Here are your details:")
        elif manager_result and name:
            lines.append(f"Sure 😊 Your direct manager is {name}. Here are the details you requested:")
        elif name:
            lines.append(f"Sure 😊 Here are the details you requested for {name}:")
        else:
            lines.append("Sure 😊 Here are the available details:")

        for field in fields:
            raw_value = _field_value(data, field, lang)
            value = _show(raw_value) if _valid(raw_value) else ""
            if field in {"job_title", "department"} and value:
                value = _pretty_english_work_text(value)
            label = FIELD_LABELS["en"].get(field, field)
            lines.append(f"- {label}: {value}" if value else f"- {label}: Not available")
        return "\n".join(lines)

    if lang == "ar":
        heading = "هذي معلوماتك 👤:" if self_subject else (f"{_natural_opening('employee', lang, str(name))} {name}:" if name else "هذي المعلومات المتوفرة 👤:")
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
    """Render a VERIFIED balance. Never treat an entitlement as a balance.

    A what-if question only calculates; no request is submitted and no balance
    is mutated. This endpoint currently provides one balance value, not a
    breakdown of all leave types, so do not invent annual/sick balances.
    """
    balance = data.get("balance") if isinstance(data, dict) else None
    if not _valid(balance):
        return (
            "تعذر الوصول لرصيد الإجازات حاليًا ⚠️ سند لا يزال في مرحلته التجريبية، وبعض الخدمات قد تتعطل مؤقتًا."
            if lang == "ar" else
            "I couldn't retrieve your leave balance right now ⚠️ Sanad is still in its testing phase, and some services may be temporarily unavailable."
        )

    if amount is not None:
        try:
            current, requested = float(balance), float(amount)
            if not (math.isfinite(current) and math.isfinite(requested)) or current < 0 or requested <= 0:
                raise ValueError("invalid balance or requested days")
            remaining = current - requested
            if remaining < 0:
                return (
                    f"رصيدك الحالي {_ar_days(current)}، ولا يكفي لـ{_ar_days(requested)}. ينقصك {_ar_days(-remaining)}. 📅"
                    if lang == "ar" else
                    f"Your current balance is {_show(current)} days, so taking {_show(requested)} days would exceed it by {_show(-remaining)} days. 📅"
                )
            return (
                f"☀️ لو أخذت {_ar_days(requested)}، بيبقى لك **{_ar_days(remaining)}** من رصيدك. خذ لك إجازة وريح، تستاهل! 😎✨"
                if lang == "ar" else
                f"☀️ If you took {_show(requested)} days off, you'd have **{_show(remaining)} days** left. Enjoy some well-deserved time off! 😎✨"
            )
        except (TypeError, ValueError, OverflowError):
            return (
                "ما قدرت أحسب المتبقي من الرصيد الحالي بدقة. 📅"
                if lang == "ar" else "I couldn't calculate the remaining balance reliably. 📅"
            )

    if lang == "ar":
        if self_subject:
            return f"☀️ رصيد إجازاتك الحالي **{_ar_days(balance)}**. خذ لك وقت للراحة، تستاهل! ✨"
        return f"📅 رصيد إجازات {subject_name} الحالي {_ar_days(balance)}."
    if self_subject:
        return f"☀️ Your current leave balance is **{_show(balance)} days**. Some well-deserved rest sounds good! ✨"
    return f"📅 {subject_name}'s current leave balance is {_show(balance)} days."


def _parse_meeting_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.astimezone(timezone(timedelta(hours=3)))
        return dt
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



_AR_WEEKDAYS = {
    0: "الاثنين",
    1: "الثلاثاء",
    2: "الأربعاء",
    3: "الخميس",
    4: "الجمعة",
    5: "السبت",
    6: "الأحد",
}
_EN_WEEKDAYS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

def _meeting_time_window_label(time_query: str, lang: str) -> str:
    value = str(time_query or "").strip()
    mapping_ar = {
        "07:00-12:00": "الصباح",
        "12:00-17:00": "بعد الظهر",
        "17:00-23:59": "المساء",
    }
    mapping_en = {
        "07:00-12:00": "in the morning",
        "12:00-17:00": "in the afternoon",
        "17:00-23:59": "in the evening",
    }
    return (mapping_ar if lang == "ar" else mapping_en).get(value, "")


def _meeting_in_time_window(row: dict[str, Any], time_query: str) -> bool:
    value = str(time_query or "").strip()
    if not value or value == "all-day":
        return True
    start_dt = _parse_meeting_dt(_first(row, "start_time", "startTime"))
    end_dt = _parse_meeting_dt(_first(row, "end_time", "endTime"))
    if start_dt is None:
        return True
    if "-" in value:
        try:
            left, right = value.split("-", 1)
            sh, sm = [int(x) for x in left.split(":", 1)]
            eh, em = [int(x) for x in right.split(":", 1)]
            start_minutes = start_dt.hour * 60 + start_dt.minute
            end_minutes = (end_dt.hour * 60 + end_dt.minute) if end_dt else start_minutes
            q_start = sh * 60 + sm
            q_end = eh * 60 + em
            return start_minutes < q_end and end_minutes > q_start
        except Exception:
            return True
    try:
        qh, qm = [int(x) for x in value.split(":", 1)]
        q = qh * 60 + qm
        start_minutes = start_dt.hour * 60 + start_dt.minute
        end_minutes = (end_dt.hour * 60 + end_dt.minute) if end_dt else start_minutes + 1
        return start_minutes <= q < end_minutes
    except Exception:
        return True


def _clean_meeting_link(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    markdown = re.match(r"^\[[^\]]*\]\((https?://[^)]+)\)$", raw)
    if markdown:
        return markdown.group(1)
    url = re.search(r"https?://[^\s)]+", raw)
    return url.group(0) if url else ""


def _meeting_response_ar(value: Any) -> str:
    norm = str(value or "").strip().casefold()
    if norm in {"accepted", "accept", "yes"}:
        return "أنت موافق على الدعوة ✅"
    if norm in {"declined", "decline", "no"}:
        return "أنت رافض الدعوة"
    if norm in {"tentative", "tentativelyaccepted"}:
        return "ردك على الدعوة مبدئي"
    if norm in {"noresponsereceived", "none", "notresponded", "not responded"}:
        return "لسه ما رديت على الدعوة"
    return ""



def _meeting_organizer(row: dict[str, Any]) -> str:
    return str(_first(
        row,
        "organizer_name", "organizerName", "OrganizerName",
        "organizer", "organizerDisplayName",
    ) or "").strip()


def _is_tawasul_meeting(row: dict[str, Any]) -> bool:
    organizer = _meeting_organizer(row).casefold()
    return organizer == "tawasul" or organizer.startswith("tawasul ")


def _ar_tawasul_count(value: int) -> str:
    if value == 1:
        return "فعالية أو دعوة واحدة"
    if value == 2:
        return "فعاليتين أو دعوتين"
    if 3 <= value <= 10:
        return f"{value} فعاليات ودعوات"
    return f"{value} فعالية ودعوة"


def render_meetings(
    data: dict[str, Any],
    lang: str,
    *,
    subject_name: str = "",
    self_subject: bool = False,
    period: str = "none",
    start_date: str = "",
    end_date: str = "",
    time_query: str = "",
    view_mode: str = "meetings:primary",
) -> str:
    all_meetings = [row for row in _rows(data) if isinstance(row, dict)]
    if time_query:
        all_meetings = [row for row in all_meetings if _meeting_in_time_window(row, time_query)]

    tawasul_events = [row for row in all_meetings if _is_tawasul_meeting(row)]
    primary_meetings = [row for row in all_meetings if not _is_tawasul_meeting(row)]
    show_tawasul = str(view_mode or "").strip() == "meetings:tawasul"
    meetings = tawasul_events if show_tawasul else primary_meetings

    label_ar = {
        "today": "اليوم",
        "tomorrow": "بكرة",
        "yesterday": "أمس",
        "this_week": "هذا الأسبوع",
        "last_week": "الأسبوع الماضي",
        "this_month": "هذا الشهر",
        "last_month": "الشهر الماضي",
    }.get(period, "للفترة المطلوبة")
    label_en = {
        "today": "today",
        "tomorrow": "tomorrow",
        "yesterday": "yesterday",
        "this_week": "this week",
        "last_week": "last week",
        "this_month": "this month",
        "last_month": "last month",
    }.get(period, "for the requested period")

    if period == "custom" and start_date:
        try:
            start_dt = datetime.fromisoformat(str(start_date))
            if end_date and end_date != start_date:
                end_dt = datetime.fromisoformat(str(end_date))
                label_ar = f"من {start_dt.strftime('%d/%m/%Y')} إلى {end_dt.strftime('%d/%m/%Y')}"
                label_en = f"from {start_dt.strftime('%d/%m/%Y')} to {end_dt.strftime('%d/%m/%Y')}"
            else:
                weekday_ar = _AR_WEEKDAYS.get(start_dt.weekday(), "")
                weekday_en = _EN_WEEKDAYS.get(start_dt.weekday(), "")
                label_ar = f"{weekday_ar} {start_dt.strftime('%d/%m/%Y')}".strip()
                label_en = f"{weekday_en} {start_dt.strftime('%d/%m/%Y')}".strip()
        except Exception:
            pass

    window_ar = _meeting_time_window_label(time_query, "ar")
    window_en = _meeting_time_window_label(time_query, "en")
    when_ar = " ".join(x for x in (label_ar, window_ar) if x)
    when_en = " ".join(x for x in (label_en, window_en) if x)

    now_riyadh = datetime.now(timezone(timedelta(hours=3)))
    historical_period = period in {"yesterday", "last_week", "last_month"}
    try:
        if end_date:
            historical_period = datetime.fromisoformat(str(end_date)).date() < now_riyadh.date()
    except Exception:
        pass

    if not meetings:
        if lang == "ar":
            if show_tawasul:
                return f"ما لقيت فعاليات أو دعوات من تواصل {when_ar} 😊"

            if self_subject:
                if tawasul_events:
                    count_label = _ar_tawasul_count(len(tawasul_events))
                    tense = "كان" if historical_period else "عندك"
                    if historical_period:
                        return (
                            f"ما كان عندك اجتماعات مجدولة {when_ar} 😊\n"
                            f"لكن كان عندك **{count_label} من تواصل** خلال نفس الفترة. "
                            "إذا تحب، أعرضها لك."
                        )
                    return (
                        f"ما عندك اجتماعات مجدولة {when_ar} 😊\n"
                        f"لكن عندك **{count_label} من تواصل** خلال نفس الفترة. "
                        "إذا تحب، أعرضها لك."
                    )

                if historical_period:
                    return f"ما كان عندك أي اجتماعات مجدولة {when_ar} 😊"
                if period == "today" and not time_query:
                    return "ما عندك أي اجتماعات مجدولة اليوم 😊\nتحب أساعدك بشي ثاني؟"
                if period == "today" and time_query:
                    return f"ما عندك أي اجتماعات مجدولة {when_ar} 😊\nتحب أساعدك بشي ثاني؟"
                return f"ما عندك أي اجتماعات مجدولة {when_ar} 😊"

            if historical_period:
                return f"ما كانت فيه اجتماعات مسجلة لـ {subject_name} {when_ar}."
            return f"ما فيه اجتماعات مسجلة لـ {subject_name} {when_ar}."

        if show_tawasul:
            return f"I couldn't find any Tawasul events or invitations {when_en} 😊"

        if self_subject:
            if tawasul_events:
                count = len(tawasul_events)
                noun = "event or invitation" if count == 1 else "events and invitations"
                return (
                    f"You don't have any meetings scheduled {when_en} 😊\n"
                    f"You do have **{count} Tawasul {noun}** in the same period. "
                    "I can show them if you'd like."
                )
            if historical_period:
                return f"You didn't have any meetings scheduled {when_en} 😊"
            if period == "today" and not time_query:
                return "You don't have any meetings scheduled for today 😊\nCan I help you with anything else?"
            return f"You don't have any meetings scheduled {when_en} 😊"

        if historical_period:
            return f"There were no meetings scheduled for {subject_name} {when_en}."
        return f"There are no meetings scheduled for {subject_name} {when_en}."

    def _display_date(value: str) -> str:
        try:
            return datetime.fromisoformat(str(value)).strftime("%d/%m/%Y")
        except Exception:
            return str(value or "")

    display_start = _display_date(start_date) if start_date else ""
    display_end = _display_date(end_date) if end_date else ""

    if lang == "ar":
        if self_subject:
            if show_tawasul:
                if period == "today":
                    heading = "أكيد 😊 هذي فعاليات ودعوات تواصل اليوم:"
                elif period == "yesterday":
                    heading = "أكيد 😊 هذي فعاليات ودعوات تواصل أمس:"
                elif period == "last_week":
                    heading = "أكيد 😊 هذي فعاليات ودعوات تواصل خلال أسبوع العمل الماضي:"
                elif start_date and end_date and start_date != end_date:
                    heading = f"أكيد 😊 هذي فعاليات ودعوات تواصل من {display_start} إلى {display_end}:"
                elif start_date:
                    heading = f"أكيد 😊 هذي فعاليات ودعوات تواصل {when_ar}:"
                else:
                    heading = f"أكيد 😊 هذي فعاليات ودعوات تواصل {when_ar}:"
            else:
                if period == "today":
                    heading = "هذي اجتماعاتك اليوم 😊"
                elif period == "yesterday":
                    heading = "هذي اجتماعاتك أمس 😊"
                elif period == "last_week":
                    heading = "هذي اجتماعاتك خلال أسبوع العمل الماضي 😊"
                elif start_date and end_date and start_date != end_date:
                    heading = f"هذي اجتماعاتك خلال الفترة من {display_start} إلى {display_end} 😊"
                elif start_date:
                    heading = f"هذي اجتماعاتك {when_ar} 😊"
                else:
                    heading = f"هذي اجتماعاتك {when_ar} 😊"
        else:
            heading = f"هذي اجتماعات {subject_name} {when_ar} 😊"
    else:
        if self_subject:
            if show_tawasul:
                heading = f"Here are your Tawasul events and invitations {when_en} 😊"
            elif period == "today":
                heading = "Here are your meetings for today 😊"
            elif period == "yesterday":
                heading = "Here are your meetings from yesterday 😊"
            elif period == "last_week":
                heading = "Here are your meetings from the previous work week 😊"
            elif start_date and end_date and start_date != end_date:
                heading = f"Here are your meetings from {display_start} to {display_end} 😊"
            elif start_date:
                heading = f"Here are your meetings {when_en} 😊"
            else:
                heading = f"Here are your meetings {when_en} 😊"
        else:
            heading = f"Here are {subject_name}'s meetings {when_en} 😊"

    def _sort_key(row: dict[str, Any]):
        dt = _parse_meeting_dt(_first(row, "start_time", "startTime"))
        if dt is None:
            return datetime.max.replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone(timedelta(hours=3)))
        return dt

    meetings = sorted(meetings, key=_sort_key)
    visible = meetings[:20]

    grouped: dict[str, list[dict[str, Any]]] = {}
    undated: list[dict[str, Any]] = []
    for row in visible:
        dt = _parse_meeting_dt(_first(row, "start_time", "startTime"))
        if dt is None:
            undated.append(row)
            continue
        grouped.setdefault(dt.date().isoformat(), []).append(row)

    groups = [(k, grouped[k]) for k in sorted(grouped)]
    if undated:
        groups.append(("", undated))

    def _count_ar(n: int) -> str:
        if n == 2:
            return "اجتماعين"
        if 3 <= n <= 10:
            return f"{n} اجتماعات"
        return f"{n} اجتماع"

    lines = [heading]

    for date_key, day_rows in groups:
        if date_key:
            d = datetime.fromisoformat(date_key)
            if lang == "ar":
                group_title = f"{_AR_WEEKDAYS.get(d.weekday(), '')} {d.strftime('%d/%m/%Y')}".strip()
                if len(day_rows) > 1:
                    group_title += f" · {_count_ar(len(day_rows))}"
            else:
                group_title = f"{_EN_WEEKDAYS.get(d.weekday(), '')} {d.strftime('%d/%m/%Y')}".strip()
                if len(day_rows) > 1:
                    group_title += f" · {len(day_rows)} meetings"
            lines.extend(["", f"**{group_title}**"])

        for idx, row in enumerate(day_rows, 1):
            subject = _first(
                row,
                "subject_ar", "subjectAr", "title_ar", "titleAr",
                "subject", "title",
                "subject_en", "subjectEn", "title_en", "titleEn",
            ) or ("اجتماع" if lang == "ar" else "Meeting")

            start_dt = _parse_meeting_dt(_first(row, "start_time", "startTime"))
            end_dt = _parse_meeting_dt(_first(row, "end_time", "endTime"))
            start_label = _clock_label(_first(row, "start_time", "startTime"), lang)
            end_label = _clock_label(_first(row, "end_time", "endTime"), lang)

            location = str(_first(row, "location", "Location") or "").strip()
            if location and "teams" in location.casefold():
                location = "Microsoft Teams"

            response = (
                _meeting_response_ar(_first(row, "my_response", "myResponse", "MyResponse"))
                if lang == "ar"
                else str(_first(row, "my_response", "myResponse", "MyResponse") or "").strip()
            )

            link = _clean_meeting_link(_first(
                row,
                "teams_link", "meeting_link", "meetingLink",
                "teamsLink", "TeamsLink",
                "joinUrl", "JoinUrl", "join_url",
                "meetingUrl", "MeetingUrl",
                "onlineMeetingUrl", "OnlineMeetingUrl",
                "webLink", "WebLink",
                "url", "URL", "link", "Link",
            ))

            meeting_ended = bool(end_dt and end_dt <= now_riyadh)
            if end_dt is None and start_dt is not None:
                meeting_ended = start_dt < now_riyadh

            if show_tawasul:
                lines.append(f"**{subject}**")
            else:
                lines.append(f"{idx}. **{subject}**")

            if start_label and end_label:
                lines.append(f"🕘 {start_label} - {end_label}")
            elif start_label:
                lines.append(f"🕘 {start_label}")

            if location:
                lines.append(f"📍 {location}")

            # Keep the original approved status/link style, but only while the
            # meeting is still current/future.
            if not meeting_ended:
                if response:
                    if lang == "ar":
                        lines.append(f"💬 {response}")
                    else:
                        lines.append(f"💬 {response}")
                if link:
                    lines.append(
                        f"[**انضم للاجتماع**]({link})"
                        if lang == "ar"
                        else f"[**Join meeting**]({link})"
                    )

    if not show_tawasul and tawasul_events:
        lines.extend([
            "",
            (
                f"وعندك بعد **{_ar_tawasul_count(len(tawasul_events))} من تواصل** خلال نفس الفترة، "
                "إذا تحب أعرضها لك."
                if lang == "ar"
                else
                f"You also have **{len(tawasul_events)} Tawasul events or invitations** in the same period. "
                "I can show them if you'd like."
            ),
        ])

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
        return (
            "ما قدرت أتحقق من معاملات تراسل حاليًا 📬"
            if lang == "ar"
            else "I couldn't verify your Tarasol transactions right now."
        )

    try:
        number = int(float(str(count)))
    except Exception:
        number = None

    if lang == "ar":
        if self_subject:
            if number == 0:
                return "تمام 😊 ما عندك أي معاملات في تراسل حاليًا."
            if number == 1:
                return "عندك **معاملة واحدة** في تراسل حاليًا 📬\nتحب أعرض لك تفاصيلها؟"
            if number == 2:
                return "عندك **معاملتين** في تراسل حاليًا 📬\nتحب أعرض لك تفاصيلها؟"
            if number is not None and 3 <= number <= 10:
                return f"عندك **{number} معاملات** في تراسل حاليًا 📬\nتحب أعرض لك تفاصيلها؟"
            return f"عندك **{_show(count)} معاملة** في تراسل حاليًا 📬\nتحب أعرض لك التفاصيل؟"

        display_name = _short_person_name(subject_name) if subject_name else "الموظف"
        if number == 0:
            return f"ما عند **{display_name}** أي معاملات في تراسل حاليًا."
        if number == 1:
            return f"عند **{display_name}** معاملة واحدة في تراسل حاليًا."
        if number == 2:
            return f"عند **{display_name}** معاملتين في تراسل حاليًا."
        if number is not None and 3 <= number <= 10:
            return f"عند **{display_name}** {number} معاملات في تراسل حاليًا."
        return f"عند **{display_name}** {_show(count)} معاملة في تراسل حاليًا."

    if self_subject:
        if number == 0:
            return "You don't have any Tarasol transactions right now 😊"
        if number == 1:
            return "You have **1 Tarasol transaction** right now 📬\nWould you like me to show the details?"
        return f"You have **{_show(count)} Tarasol transactions** right now 📬\nWould you like me to show the details?"

    display_name = _short_person_name(subject_name) if subject_name else "The employee"
    return f"{display_name} currently has **{_show(count)}** Tarasol transaction(s)."


def render_tarasol(data: dict[str, Any], lang: str, *, subject_name: str = "", self_subject: bool = False) -> str:
    rows = _rows(data)

    if not rows:
        if lang == "ar":
            if self_subject:
                return "ما عندك معاملات ظاهرة في تراسل حاليًا 😊"
            display_name = _short_person_name(subject_name) if subject_name else "الموظف"
            return f"ما ظهرت لي معاملات في تراسل لـ **{display_name}** حاليًا."
        return "I couldn't find any Tarasol transactions right now."

    if lang == "ar":
        if self_subject:
            heading = (
                "هذي تفاصيل معاملتك في تراسل 📬"
                if len(rows) == 1
                else f"هذي معاملاتك في تراسل ({len(rows)}) 📬"
            )
        else:
            display_name = _short_person_name(subject_name) if subject_name else "الموظف"
            heading = f"هذي معاملات **{display_name}** في تراسل 📬"
    else:
        if self_subject:
            heading = (
                "Here are the details of your Tarasol transaction:"
                if len(rows) == 1
                else f"Here are your Tarasol transactions ({len(rows)}):"
            )
        else:
            display_name = _short_person_name(subject_name) if subject_name else "The employee"
            heading = f"Here are {display_name}'s Tarasol transactions:"

    lines = [heading]

    for index, row in enumerate(rows[:10], 1):
        parts: list[str] = []
        for field in ("subject", "creation_date", "sender", "recipient", "document_type", "priority"):
            value = _field_value(row, field, lang)
            if _valid(value):
                parts.append(f"{FIELD_LABELS[lang].get(field, field)}: {_show(value)}")

        if parts:
            lines.append(f"{index}. " + " — ".join(parts))

    if len(rows) > 10:
        remaining = len(rows) - 10
        if lang == "ar":
            lines.extend(["", f"وعندك بعد {remaining} معاملة إضافية."])
        else:
            lines.extend(["", f"There are {remaining} more transactions."])

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
            return "حاليًا ما عندك موظفين تابعين لك مباشرة 👥😊" if self_subject else f"ما فيه موظفين مباشرين مسجلين تحت {subject_name} حاليًا."
        return "There are no direct reports registered under you right now." if self_subject else f"There are no direct reports registered under {subject_name} right now."

    lines = ["أكيد 😊 هذولا الموظفين التابعين لك مباشرة 👥:" if lang == "ar" else "Here are the direct reports:"]
    display_names = _person_display_names([name for name, _ in visible])
    for index, ((_, title), display_name) in enumerate(zip(visible, display_names), 1):
        lines.append(f"{index}. {display_name}" + (f" — {title}" if _valid(title) else ""))
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
        return ("ما لقيت نتائج مطابقة حاليًا 🔎 وأنا حاليًا في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار 😊" if lang == "ar" else "I couldn't find matching results right now."), []

    if intent == "news":
        rows = await localize_news_rows(rows, lang, limit=5)
        visible = rows[:5]
        heading = _natural_opening("news", lang, str(_item_title(visible[0], lang)) if visible else "")
        lines = [heading]
        for index, row in enumerate(visible, 1):
            title = _item_title(row, lang) or ("بدون عنوان" if lang == "ar" else "Untitled")
            lines.append(f"{index}. {title}")
        lines.extend(["", "وإذا فيه خبر لفت انتباهك، تفاصيله متاحة بعد ✨" if lang == "ar" else "More details are available for any story that catches your eye ✨"])
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

        heading = _natural_opening("mazaya", lang, str(unique[0].get("company") or unique[0].get("subject") or "") if unique else "")

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

        lines.extend(["", "وإذا لفت انتباهك أي عرض، تفاصيله متاحة بعد، حتى لو أكثر من عرض ✨" if lang == "ar" else "You can explore the details of any offer, including several in a row ✨"])
        return "\n".join(lines), display_rows

    if intent == "announcements":
        limit = min(len(rows), 20)
    elif intent == "policy_search":
        limit = 8
    else:
        limit = 10

    headings_ar = {
        "announcements": _natural_opening("announcements", lang, str(_item_title(rows[0], lang)) if rows else ""),
        "policy_search": "أكيد 😊📄\nهذي السياسات والإجراءات المتاحة لك حاليًا:",
    }
    headings_en = {
        "announcements": _natural_opening("announcements", lang, str(_item_title(rows[0], lang)) if rows else ""),
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
                "ما ظهرت لي أسماء السياسات حاليًا 📄\n\n"
                "أما التفاصيل الكاملة، بتلقاها داخل التطبيق في **المزيد > السياسات والإجراءات** ✨"
                if lang == "ar"
                else "I couldn't display the policy names right now 📄\n\n"
                     "For details, go to More and select Policies & Procedures."
            ), []

    if intent == "announcements" and visible_rows:
        lines.extend(["", "وإذا لفت انتباهك أي إعلان، تفاصيله متاحة بعد ✨" if lang == "ar" else "More details are available for any announcement you're interested in ✨"])

    if intent == "policy_search":
        if lang == "ar":
            lines.extend([
                "",
                "أما التفاصيل الكاملة، بتلقاها داخل التطبيق في **المزيد > السياسات والإجراءات** ✨"
            ])
        else:
            lines.extend([
                "",
                "📄 For details, go to More and select Policies & Procedures."
            ])

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
            return (
                "ما لقيت موظف مطابق للاسم هذا حاليًا 🔎😊 "
                "جرّب تكتب الاسم بشكل أكمل، أو ابحث بالمسمى الوظيفي أو الإدارة."
            )
        return (f"I couldn’t find an employee matching “{query}” 🔎. Try a fuller name, job title, or department."
                if query else "I couldn’t find a matching employee 🔎. Try a name, job title, or department.")

    shown = clean_rows[:limit]
    if lang == "ar":
        if len(clean_rows) == 1:
            header = "لقيت الموظف اللي تبحث عنه 😊"
        else:
            header = "لقيت أكثر من موظف قريب من بحثك 👥 مين تقصد؟"
        lines = [header]
        raw_names = [
            str(_first(row, "name_ar", "name", "name_en") or "").strip()
            for row in shown
        ]
        display_names = _person_display_names(raw_names)
        for index, (row, display_name) in enumerate(zip(shown, display_names), 1):
            title = _first(row, "job_title_ar", "job_title", "job_title_en")
            department = _first(row, "department_ar", "department", "department_en", "general_department")
            pieces = [display_name] if display_name else ["موظف"]
            if _valid(title):
                pieces.append(str(title).strip())
            if _valid(department):
                pieces.append(str(department).strip())
            lines.append(f"{index}. " + " — ".join(pieces))
        if len(clean_rows) > limit:
            lines.append(f"… وعندي {len(clean_rows) - limit} نتيجة إضافية.")
        if len(clean_rows) == 1:
            lines.extend(["", "إذا تحتاج إيميله، جواله، منصبه أو إدارته، قل لي وش تبي منها."])
        return "\n".join(lines)

    header = (
        f"I found {len(clean_rows)} possible match{'es' if len(clean_rows) != 1 else ''} for “{query}” 👥:"
        if query else f"I found {len(clean_rows)} employee match{'es' if len(clean_rows) != 1 else ''} 👥:"
    )
    lines = [header]
    raw_names = [
        str(_first(row, "name_en", "name", "name_ar") or "").strip()
        for row in shown
    ]
    display_names = _person_display_names(raw_names)
    for index, (row, display_name) in enumerate(zip(shown, display_names), 1):
        title = _first(row, "job_title_en", "job_title", "job_title_ar")
        department = _first(row, "department_en", "department", "department_ar", "general_department")
        pieces = [display_name] if display_name else ["Employee"]
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
        return "ما لقيت نتائج مطابقة حاليًا 🔎 وأنا حاليًا في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار 😊" if lang == "ar" else "I couldn't find available data right now."
    lines = ["لقيت لك هذي النتائج 🔎:" if lang == "ar" else "I found these results:"]
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


_AR_WEEKDAYS = {
    0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
    4: "الجمعة", 5: "السبت", 6: "الأحد",
}


def _attendance_date_obj(value: Any):
    raw = str(value or "").strip()
    if not raw:
        return None
    candidates = [raw[:10], raw]
    for item in candidates:
        try:
            return datetime.fromisoformat(item.replace("Z", "+00:00")).date()
        except Exception:
            pass
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except Exception:
            pass
    return None


def _attendance_day_ar(value: Any, fallback: Any = "") -> str:
    d = _attendance_date_obj(value)
    if d is not None:
        return _AR_WEEKDAYS.get(d.weekday(), "")
    text = str(fallback or "").strip().casefold()
    aliases = {
        "monday":"الاثنين", "tuesday":"الثلاثاء", "wednesday":"الأربعاء",
        "thursday":"الخميس", "friday":"الجمعة", "saturday":"السبت", "sunday":"الأحد",
        "الاثنين":"الاثنين", "الإثنين":"الاثنين", "الثلاثاء":"الثلاثاء",
        "الاربعاء":"الأربعاء", "الأربعاء":"الأربعاء", "الخميس":"الخميس",
        "الجمعه":"الجمعة", "الجمعة":"الجمعة", "السبت":"السبت",
        "الاحد":"الأحد", "الأحد":"الأحد",
    }
    return aliases.get(text, str(fallback or "").strip() if ARABIC_RE.search(str(fallback or "")) else "")


def _friendly_duration_ar(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    m = re.match(r"^(\d{1,3}):(\d{1,2})(?::(\d{1,2}))?$", raw)
    if not m:
        return raw
    hours = int(m.group(1)); minutes = int(m.group(2))
    def hour_label(n: int) -> str:
        if n == 1: return "ساعة"
        if n == 2: return "ساعتين"
        if 3 <= n <= 10: return "ساعات"
        return "ساعة"

    def minute_label(n: int) -> str:
        if n == 1: return "دقيقة"
        if n == 2: return "دقيقتين"
        if 3 <= n <= 10: return "دقائق"
        return "دقيقة"

    parts = []
    if hours:
        parts.append(f"{hours} {hour_label(hours)}")
    if minutes:
        parts.append(f"{minutes} {minute_label(minutes)}")
    return " و".join(parts) if parts else "أقل من دقيقة"



def _duration_minutes(value: Any, *, numeric_unit: str = "hours") -> int | None:
    """Best-effort duration parser for verified attendance fields."""
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        if numeric_unit == "minutes":
            return max(0, int(number))
        return max(0, int(number * 60))

    raw = str(value).strip()
    if not raw:
        return None

    m = re.fullmatch(r"(-?\d{1,4}):(\d{1,2})(?::(\d{1,2}))?", raw)
    if m:
        sign = -1 if raw.startswith("-") else 1
        hours = abs(int(m.group(1)))
        minutes = int(m.group(2))
        total = hours * 60 + minutes
        return max(0, sign * total)

    try:
        number = float(raw)
    except Exception:
        return None

    if numeric_unit == "minutes":
        return max(0, int(number))
    return max(0, int(number * 60))


def _minutes_label_ar(total_minutes: int) -> str:
    total_minutes = max(0, int(total_minutes))
    hours, minutes = divmod(total_minutes, 60)

    if hours and minutes:
        return _friendly_duration_ar(f"{hours:02d}:{minutes:02d}:00")
    if hours:
        return _friendly_duration_ar(f"{hours:02d}:00:00")
    if minutes:
        return _friendly_duration_ar(f"00:{minutes:02d}:00")
    return "0 دقيقة"


def _minutes_label_en(total_minutes: int) -> str:
    total_minutes = max(0, int(total_minutes))
    hours, minutes = divmod(total_minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour" + ("" if hours == 1 else "s"))
    if minutes:
        parts.append(f"{minutes} minute" + ("" if minutes == 1 else "s"))
    return " and ".join(parts) if parts else "0 minutes"


def _overtime_minutes(value: Any) -> int | None:
    if isinstance(value, list):
        total = 0
        found = False
        for item in value:
            if isinstance(item, dict):
                minute_value = _first(item, "minutes", "overTimeMinutes", "overtimeMinutes")
                if _valid(minute_value):
                    parsed = _duration_minutes(minute_value, numeric_unit="minutes")
                else:
                    parsed = _duration_minutes(
                        _first(item, "hours", "duration", "overTime", "overtime"),
                        numeric_unit="hours",
                    )
            else:
                parsed = _duration_minutes(item, numeric_unit="hours")
            if parsed is not None:
                found = True
                total += parsed
        return total if found else None

    if isinstance(value, dict):
        minute_value = _first(value, "minutes", "overTimeMinutes", "overtimeMinutes")
        if _valid(minute_value):
            return _duration_minutes(minute_value, numeric_unit="minutes")
        return _duration_minutes(
            _first(value, "hours", "duration", "overTime", "overtime"),
            numeric_unit="hours",
        )

    return _duration_minutes(value, numeric_unit="hours")



def _attendance_employee_name(row: dict[str, Any], lang: str, fallback: str = "") -> str:
    if not isinstance(row, dict):
        return fallback
    if lang == "ar":
        value = _first(
            row,
            "name_ar", "namea", "name", "employeeNameAr",
            "employeeName", "name_en", "namel",
        )
    else:
        value = _first(
            row,
            "name_en", "namel", "name", "employeeNameEn",
            "employeeName", "name_ar", "namea",
        )
    return _short_person_name(str(value or fallback or "").strip())


def _attendance_work_minutes(row: dict[str, Any]) -> int | None:
    """Return verified worked minutes, deriving from punches only when needed."""
    explicit = _first(
        row,
        "workingMinutes", "workMinutes", "workedMinutes",
    )
    if _valid(explicit):
        parsed = _duration_minutes(explicit, numeric_unit="minutes")
        if parsed is not None:
            return parsed

    explicit_hours = _first(
        row,
        "workingHours", "working_hours", "totalHours", "workedHours",
    )
    if _valid(explicit_hours):
        parsed = _duration_minutes(explicit_hours, numeric_unit="hours")
        if parsed is not None:
            return parsed

    start = _first(row, "startStamp", "checkIn", "checkInTime", "inTime")
    end = _first(row, "endStamp", "checkOut", "checkOutTime", "outTime")
    if not (_valid(start) and _valid(end)):
        return None

    try:
        start_dt = datetime.fromisoformat(str(start).strip().replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(str(end).strip().replace("Z", "+00:00"))
        seconds = (end_dt - start_dt).total_seconds()
        if seconds < 0:
            seconds += 24 * 60 * 60
        return max(0, int(round(seconds / 60)))
    except Exception:
        return None


def _attendance_day_classification(row: dict[str, Any], lang: str) -> dict[str, Any]:
    """
    One classification layer for every attendance source:
    self attendance, historical attendance and subemployee/team attendance.

    Priority:
    explicit leave/vacation > official holiday > weekend/off day > normal workday.
    We never infer absence/shortfall merely because punches are missing.
    """
    vacation = _first(
        row,
        "vacationType", "leaveType", "vacationName", "leaveName",
        "vacationTypeName", "leaveTypeName",
    )
    day_type = _first(row, "dayType", "day_type", "dayTypeName")
    day_code = str(_first(row, "dayTypeCode", "day_type_code") or "").strip()
    status = _first(row, "status", "attendanceStatus", "statusText")
    status_code = str(_first(row, "statusCode", "attendanceStatusCode") or "").strip()

    text = " ".join(
        str(x or "").strip()
        for x in (vacation, day_type, day_code, status, status_code)
        if _valid(x)
    ).casefold()

    # Explicit leave types first.
    if _valid(vacation):
        raw_label = str(vacation).strip()
        localized = str(_localized_enum(raw_label, lang) or raw_label).strip()
        return {
            "kind": "leave",
            "label": localized,
            "is_leave": True,
            "is_non_working": True,
        }

    leave_patterns = (
        ("sick leave", "إجازة مرضية", "Sick leave"),
        ("annual leave", "إجازة سنوية", "Annual leave"),
        ("emergency leave", "إجازة طارئة", "Emergency leave"),
        ("urgent leave", "إجازة اضطرارية", "Urgent leave"),
    )
    for token, ar_label, en_label in leave_patterns:
        if token in text:
            return {
                "kind": "leave",
                "label": ar_label if lang == "ar" else en_label,
                "is_leave": True,
                "is_non_working": True,
            }

    # Arabic leave words that can arrive directly from APIs.
    if re.search(r"(?:إجازة|اجازة|أجازة)\s*(?:مرضية|سنوية|طارئة|اضطرارية)?", text, re.I):
        source = status or day_type or "إجازة"
        label = str(_localized_enum(source, lang) or source).strip()
        return {
            "kind": "leave",
            "label": label,
            "is_leave": True,
            "is_non_working": True,
        }

    if any(token in text for token in ("official holiday", "public holiday", "national holiday")) or re.search(
        r"(?:إجازة|اجازة)\s+رسمية|عطلة\s+رسمية", text, re.I
    ):
        return {
            "kind": "official_holiday",
            "label": "إجازة رسمية" if lang == "ar" else "Official holiday",
            "is_leave": True,
            "is_non_working": True,
        }

    off_tokens = ("off day", "weekend", "week end", "rest day")
    if day_code.upper() in {"OFF", "WEEKEND", "REST"} or any(token in text for token in off_tokens):
        return {
            "kind": "off_day",
            "label": "إجازة نهاية الأسبوع" if lang == "ar" else "Weekend / Off day",
            "is_leave": False,
            "is_non_working": True,
        }

    # A zero required-hours record is non-working only when the API explicitly
    # labels the day as non-working/holiday/leave. Never guess solely from zero.
    return {
        "kind": "workday",
        "label": "",
        "is_leave": False,
        "is_non_working": False,
    }


def _attendance_punch_state(row: dict[str, Any]) -> str:
    check_in = _first(row, "startStamp", "checkIn", "checkInTime", "inTime")
    check_out = _first(row, "endStamp", "checkOut", "checkOutTime", "outTime")
    if _valid(check_in) and _valid(check_out):
        return "both"
    if _valid(check_in):
        return "in_only"
    if _valid(check_out):
        return "out_only"
    return "none"


def _attendance_filter_mode(mode: str) -> tuple[str, str]:
    """
    Supports optional team filtering encoded as:
      attendance:lateness|employee:<display name>
    """
    raw = str(mode or "attendance:summary")
    employee_filter = ""
    if "|employee:" in raw:
        raw, employee_filter = raw.split("|employee:", 1)
    return raw.strip(), employee_filter.strip()


def _attendance_metric_minutes(row: dict[str, Any], mode: str) -> int | None:
    if mode == "attendance:lateness":
        minute_value = _first(
            row,
            "latenessMin",
            "latenessMinutes",
            "lateMinutes",
            "delayMinutes",
        )
        if _valid(minute_value):
            return _duration_minutes(minute_value, numeric_unit="minutes")
        return _duration_minutes(
            _first(row, "lateness", "late", "delay"),
            numeric_unit="hours",
        )

    if mode == "attendance:difference":
        minute_value = _first(row, "differenceMin", "differenceMinutes")
        if _valid(minute_value):
            return _duration_minutes(minute_value, numeric_unit="minutes")
        return _duration_minutes(_first(row, "difference"), numeric_unit="hours")

    if mode == "attendance:regular_hours":
        return _duration_minutes(
            _first(row, "regularHours", "requiredHours", "scheduledHours"),
            numeric_unit="hours",
        )

    if mode == "attendance:overtime":
        return _overtime_minutes(
            _first(row, "overTime", "overtime", "overtimeHours", "extraHours")
        )

    return None


def _render_attendance_metric(
    rows: list[dict[str, Any]],
    lang: str,
    *,
    mode: str,
    period: str,
    period_ar: str,
    period_en: str,
    subject_name: str = "",
    self_subject: bool = False,
) -> str | None:
    if mode not in {
        "attendance:lateness",
        "attendance:overtime",
        "attendance:difference",
        "attendance:regular_hours",
    }:
        return None

    values = []
    for row in rows:
        classification = _attendance_day_classification(row, lang)
        if (
            classification.get("is_non_working")
            and mode in {"attendance:lateness", "attendance:difference", "attendance:regular_hours"}
        ):
            continue
        value = _attendance_metric_minutes(row, mode)
        if value is not None:
            values.append(value)

    if not values:
        if lang == "ar":
            labels = {
                "attendance:lateness": "مقدار التأخير",
                "attendance:overtime": "الأوفر تايم",
                "attendance:difference": "فرق ساعات الدوام",
                "attendance:regular_hours": "الساعات المطلوبة",
            }
            return f"ما ظهر لي {labels.get(mode, 'المعلومة المطلوبة')} في بيانات الحضور {period_ar}."
        labels = {
            "attendance:lateness": "lateness",
            "attendance:overtime": "overtime",
            "attendance:difference": "working-hours difference",
            "attendance:regular_hours": "required hours",
        }
        return f"I couldn't find {labels.get(mode, 'that attendance metric')} in the attendance data {period_en}."

    # Required hours are meaningful per day; for a multi-day range show the total
    # scheduled requirement across returned working-day records.
    total = sum(values)

    if lang == "ar":
        amount = _minutes_label_ar(total)
        who = "" if self_subject else (subject_name.strip() or "")
        if mode == "attendance:lateness":
            if total == 0:
                return (
                    f"ما عندك تأخير مسجل {period_ar}."
                    if self_subject
                    else f"ما ظهر على {who or 'الموظف'} أي تأخير مسجل {period_ar}."
                )
            return (
                f"إجمالي تأخيرك {period_ar}: {amount}."
                if self_subject
                else f"إجمالي تأخير {who or 'الموظف'} {period_ar}: {amount}."
            )
        if mode == "attendance:overtime":
            if total == 0:
                return (
                    f"ما عندك أوفر تايم مسجل {period_ar}."
                    if self_subject
                    else f"ما ظهر لـ {who or 'الموظف'} أوفر تايم مسجل {period_ar}."
                )
            return (
                f"إجمالي الأوفر تايم {period_ar}: {amount}."
                if self_subject
                else f"إجمالي الأوفر تايم لـ {who or 'الموظف'} {period_ar}: {amount}."
            )
        if mode == "attendance:difference":
            return (
                f"فرق ساعات دوامك {period_ar}: {amount}."
                if self_subject
                else f"فرق ساعات دوام {who or 'الموظف'} {period_ar}: {amount}."
            )
        if mode == "attendance:regular_hours":
            return (
                f"إجمالي الساعات المطلوبة {period_ar}: {amount}."
                if self_subject
                else f"إجمالي الساعات المطلوبة لـ {who or 'الموظف'} {period_ar}: {amount}."
            )

    amount = _minutes_label_en(total)
    if mode == "attendance:lateness":
        return f"Your total lateness {period_en}: {amount}."
    if mode == "attendance:overtime":
        return f"Your total overtime {period_en}: {amount}."
    if mode == "attendance:difference":
        return f"Your working-hours difference {period_en}: {amount}."
    if mode == "attendance:regular_hours":
        return f"Your total required hours {period_en}: {amount}."

    return None


def _friendly_attendance_status_ar(status: Any, *, check_in: Any, check_out: Any) -> str:
    raw = str(status or "").strip()
    normalized = raw.casefold()
    localized = str(_localized_enum(raw, "ar") or "").strip() if raw else ""
    fingerprint_missing = any(token in normalized for token in (
        "finger", "fingerprint", "missing punch", "forget", "forgot"
    )) or localized == "نسيان تسجيل البصمة"
    if fingerprint_missing:
        if _valid(check_in) and not _valid(check_out):
            return "يبدو إنك نسيت تبصم خروج هاليوم 😅"
        if _valid(check_out) and not _valid(check_in):
            return "يبدو إنك نسيت تبصم دخول هاليوم 😅"
        return "يبدو إنك نسيت تسجل البصمة هاليوم 😅"
    if localized in {"تأخير", "متأخر"}:
        return "كان عندك تأخير بسيط ⏰"
    if localized in {"غياب", "غائب"}:
        return "مسجل غياب"
    if localized in {"حاضر"}:
        return ""
    return localized


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
    mode, employee_filter = _attendance_filter_mode(mode)

    if employee_filter:
        wanted = re.sub(r"\s+", " ", employee_filter).strip().casefold()
        filtered_rows = []
        for row in rows:
            row_name = re.sub(
                r"\s+", " ",
                _attendance_employee_name(row, lang)
            ).strip().casefold()
            if wanted and (wanted in row_name or row_name in wanted):
                filtered_rows.append(row)
        rows = filtered_rows

    mismatch = bool(data.get("identity_mismatch")) if isinstance(data, dict) else False
    if mismatch:
        return "يبدو إن تسجيل حضورك لليوم ما ظهر عندي إلى الآن 😊" if lang == "ar" else "Your attendance for today is not showing yet."

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
            if self_subject and period == "today":
                return "يبدو إن تسجيل حضورك لليوم ما ظهر عندي إلى الآن 😊"
            return f"ما ظهر لي سجل حضور {period_ar}." if self_subject else f"ما ظهر سجل حضور {period_ar}."
        return f"Your attendance record is not showing {period_en}." if self_subject else f"Attendance data is not showing {period_en}."

    metric_response = _render_attendance_metric(
        rows,
        lang,
        mode=mode,
        period=period,
        period_ar=period_ar,
        period_en=period_en,
        subject_name=subject_name,
        self_subject=self_subject,
    )
    if metric_response is not None:
        return metric_response

    # One returned record: answer naturally for self or employee/team data.
    # This understands Off Day / holidays / leave consistently across APIs.
    if len(rows) == 1:
        row = rows[0]
        employee_name = _attendance_employee_name(row, lang, subject_name)
        check_in = _first(row, "startStamp", "checkIn", "checkInTime", "inTime")
        check_out = _first(row, "endStamp", "checkOut", "checkOutTime", "outTime")
        status = _first(row, "status", "attendanceStatus")
        classification = _attendance_day_classification(row, lang)
        work_minutes = _attendance_work_minutes(row)
        punch_state = _attendance_punch_state(row)
        date_value = _first(row, "date", "attendanceDate", "day")
        day_value = _first(row, "dayText", "dayName")
        date_label = _attendance_date_label(date_value, lang)
        day_label = _attendance_day_ar(date_value, day_value) if lang == "ar" else (
            str(day_value).strip() if _valid(day_value) else ""
        )

        if mode == "attendance:day_type":
            if classification.get("kind") == "off_day":
                return (
                    f"هذا اليوم مصنف **إجازة نهاية الأسبوع** في النظام."
                    if lang == "ar"
                    else "This day is classified as a **Weekend / Off day** in the system."
                )
            if classification.get("label"):
                return (
                    f"هذا اليوم مصنف **{classification['label']}** في النظام."
                    if lang == "ar"
                    else f"This day is classified as **{classification['label']}** in the system."
                )
            return (
                "هذا اليوم مصنف يوم عمل عادي في بيانات الحضور."
                if lang == "ar"
                else "This day is classified as a regular working day in the attendance data."
            )

        if mode == "attendance:punch_status":
            if lang == "ar":
                if punch_state == "both":
                    return "إيه، بصمة الدخول والخروج موجودة ومسجلة."
                if punch_state == "in_only":
                    return "بصمة الدخول موجودة، لكن ما ظهر لي بصمة خروج مسجلة."
                if punch_state == "out_only":
                    return "بصمة الخروج موجودة، لكن ما ظهر لي بصمة دخول مسجلة."
                if classification.get("is_non_working"):
                    return f"ما فيه بصمات مسجلة، واليوم مصنف **{classification['label']}**."
                return "ما ظهر لي بصمة دخول أو خروج مسجلة لهذا اليوم."
            if punch_state == "both":
                return "Yes. Both check-in and check-out are recorded."
            if punch_state == "in_only":
                return "The check-in is recorded, but I couldn't find a check-out."
            if punch_state == "out_only":
                return "The check-out is recorded, but I couldn't find a check-in."
            if classification.get("is_non_working"):
                return f"No punches are recorded, and the day is classified as **{classification['label']}**."
            return "I couldn't find a recorded check-in or check-out for this day."

        if mode == "attendance:working_hours":
            if work_minutes is not None:
                amount = _minutes_label_ar(work_minutes) if lang == "ar" else _minutes_label_en(work_minutes)
                if lang == "ar":
                    return (
                        f"إجمالي دوامك كان تقريبًا **{amount}**."
                        if self_subject
                        else f"إجمالي دوام **{employee_name or 'الموظف'}** كان تقريبًا **{amount}**."
                    )
                return (
                    f"You worked for approximately **{amount}**."
                    if self_subject
                    else f"{employee_name or 'The employee'} worked for approximately **{amount}**."
                )
            if classification.get("is_non_working"):
                return (
                    f"اليوم مصنف **{classification['label']}**، لذلك ما كانت فيه ساعات عمل مطلوبة."
                    if lang == "ar"
                    else f"The day is classified as **{classification['label']}**, so there were no required working hours."
                )

        # Full one-day summary, especially useful for subemployee attendance.
        if mode == "attendance:summary":
            if lang == "ar":
                who = "حضورك" if self_subject else f"حضور **{employee_name or subject_name or 'الموظف'}**"
                when = " ".join(x for x in (day_label, date_label) if x).strip()
                lines = [f"هذا {who}" + (f" يوم {when} 😊" if when else " 😊")]

                if _valid(check_in) and _valid(check_out):
                    lines.append(
                        f"دخل الساعة **{_clock_label(check_in, 'ar')}** "
                        f"وطلع الساعة **{_clock_label(check_out, 'ar')}**."
                    )
                elif _valid(check_in):
                    lines.append(f"سجل دخول الساعة **{_clock_label(check_in, 'ar')}**، وما ظهر لي خروج مسجل.")
                elif _valid(check_out):
                    lines.append(f"سجل خروج الساعة **{_clock_label(check_out, 'ar')}**، وما ظهر لي دخول مسجل.")

                if work_minutes is not None:
                    lines.append(f"إجمالي الدوام كان تقريبًا **{_minutes_label_ar(work_minutes)}**.")

                if classification.get("label"):
                    if classification.get("kind") == "off_day":
                        lines.append(
                            "واليوم كان مصنف **إجازة نهاية الأسبوع (Off Day)**، "
                            "لذلك ما كانت عليه ساعات عمل رسمية."
                        )
                    else:
                        lines.append(f"واليوم كان مصنف **{classification['label']}** في النظام.")

                if not classification.get("is_non_working"):
                    late = _attendance_metric_minutes(row, "attendance:lateness")
                    if late is not None and late > 0:
                        lines.append(f"التأخير المسجل: **{_minutes_label_ar(late)}**.")
                    elif late == 0:
                        lines.append("وما ظهر عليه تأخير مسجل.")

                if not any(_valid(v) for v in (check_in, check_out)) and classification.get("label"):
                    # Leave/non-working day without punches is a valid record,
                    # not a missing-attendance error.
                    pass
                elif not any(_valid(v) for v in (check_in, check_out)) and not classification.get("label"):
                    status_line = _friendly_attendance_status_ar(
                        status, check_in=check_in, check_out=check_out
                    )
                    if status_line:
                        lines.append(status_line)

                return "\n".join(lines)

            who = "Your attendance" if self_subject else f"{employee_name or subject_name or 'The employee'}'s attendance"
            when = " ".join(x for x in (day_label, date_label) if x).strip()
            lines = [f"{who}" + (f" on {when}:" if when else ":")]
            if _valid(check_in) and _valid(check_out):
                lines.append(
                    f"Checked in at **{_clock_label(check_in, 'en')}** and "
                    f"checked out at **{_clock_label(check_out, 'en')}**."
                )
            elif _valid(check_in):
                lines.append(f"Checked in at **{_clock_label(check_in, 'en')}**; no check-out was returned.")
            elif _valid(check_out):
                lines.append(f"Checked out at **{_clock_label(check_out, 'en')}**; no check-in was returned.")
            if work_minutes is not None:
                lines.append(f"Worked for approximately **{_minutes_label_en(work_minutes)}**.")
            if classification.get("label"):
                lines.append(f"The day was classified as **{classification['label']}**.")
            return "\n".join(lines)

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
            title = "ساعات عملك أمس ⏰:" if intent == "working_hours" else "حضورك أمس ⏰:"
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
            if intent == "team_attendance":
                title = f"هذا حضور فريقك {ar_range} 😊"
            else:
                title = (f"ساعات عملك {ar_range} ⏰:" if intent == "working_hours" else f"جدول حضورك {ar_range} ⏰:")
        else:
            title = (f"Your working hours {period_en}:" if intent == "working_hours" else f"Your attendance {period_en}:")
        if lang == "ar" and intent != "working_hours":
            friendly_title = title.replace("جدول حضورك", "هذا حضورك").replace(" ⏰:", " ⏰😊")
            lines = [friendly_title]
        else:
            lines = [title]
        for index, row in enumerate(rows, 1):
            date_value = _first(row, "date", "attendanceDate", "day")
            day_value = _first(row, "dayText", "dayName")
            check_in = _first(row, "startStamp", "checkIn", "checkInTime", "inTime")
            check_out = _first(row, "endStamp", "checkOut", "checkOutTime", "outTime")
            working = _first(row, "workingHours", "working_hours", "totalHours")
            status = _first(row, "status", "attendanceStatus")
            vacation = _first(row, "vacationType", "leaveType")
            classification = _attendance_day_classification(row, lang)
            work_minutes = _attendance_work_minutes(row)

            date_label = _attendance_date_label(date_value, lang)
            if lang == "ar":
                day_ar = _attendance_day_ar(date_value, day_value)
                prefix = " ".join(x for x in (day_ar, date_label) if x) or f"اليوم {index}"
                if intent == "team_attendance":
                    emp_name = _attendance_employee_name(row, "ar")
                    if emp_name:
                        prefix = f"{emp_name} — {prefix}"
                lines.append("")
                lines.append(prefix)
                if classification.get("label") and not (_valid(check_in) or _valid(check_out)):
                    lines.append(classification["label"])
                    continue

                if intent == "working_hours":
                    if work_minutes is not None:
                        lines.append(f"إجمالي دوامك {_minutes_label_ar(work_minutes)} 😊")
                    elif classification.get("is_non_working"):
                        lines.append(f"اليوم مصنف {classification['label']}.")
                    else:
                        lines.append("ما ظهر لي إجمالي ساعات العمل لهاليوم.")
                    continue

                if _valid(check_in) and _valid(check_out):
                    lines.append(f"دخلت {_clock_label(check_in, 'ar')} وطلعت {_clock_label(check_out, 'ar')}")
                elif _valid(check_in):
                    lines.append(f"سجلت دخولك {_clock_label(check_in, 'ar')}")
                elif _valid(check_out):
                    lines.append(f"سجلت خروجك {_clock_label(check_out, 'ar')}")
                elif _valid(vacation):
                    lines.append(f"عندك {_show(_localized_enum(vacation, 'ar'))} 🌴")
                else:
                    lines.append("ما ظهر تسجيل حضور لهاليوم.")

                if work_minutes is not None:
                    lines.append(f"إجمالي دوامك {_minutes_label_ar(work_minutes)}")
                status_line = _friendly_attendance_status_ar(status, check_in=check_in, check_out=check_out)
                if status_line:
                    lines.append(status_line)
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
        lines = ["حضورك اليوم ⏰:" if self_subject else f"حضور {subject_name} ⏰:"]
        if _valid(check_in): lines.append(f"- وقت الدخول: {_clock_label(check_in, 'ar')}")
        if _valid(check_out): lines.append(f"- وقت الخروج: {_clock_label(check_out, 'ar')}")
        if _valid(working): lines.append(f"- ساعات العمل: {_show(working)}")
        if _valid(status): lines.append(f"- الحالة: {_show(_localized_enum(status, lang))}")
        if _valid(vacation): lines.append(f"- الإجازة: {_show(_localized_enum(vacation, lang))}")
        if len(lines) == 1:
            if self_subject:
                return "يبدو إن تسجيل حضورك لليوم ما ظهر عندي إلى الآن 😊"
            date_label = _attendance_date_label(date_value or day_value, "ar")
            lines.append("- ما ظهرت تفاصيل حضور حتى الآن.")
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



def _summarize_news_detail_ar(detail: str, title: str = "") -> tuple[str, list[str]]:
    """Create a concise, friendly Arabic summary from verified news text.

    This is intentionally deterministic: it only compresses text returned by the
    enterprise API and never invents facts. The first sentence becomes the short
    summary, then up to four high-signal sentences become bullets.
    """
    cleaned = " ".join(str(detail or "").replace("\r", " ").replace("\n", " ").split()).strip()
    if not cleaned:
        return "", []

    # Split on Arabic/English sentence boundaries while preserving verified text.
    sentences = [
        s.strip(" -–—•\t")
        for s in re.split(r"(?<=[.!؟])\s+|(?<=؛)\s+", cleaned)
        if s.strip(" -–—•\t")
    ]
    if not sentences:
        sentences = [cleaned]

    def compact(sentence: str, limit: int = 230) -> str:
        sentence = " ".join(sentence.split()).strip()
        if len(sentence) <= limit:
            return sentence
        # Prefer ending at a natural Arabic comma before the hard limit.
        cut = max(sentence.rfind("،", 0, limit), sentence.rfind(",", 0, limit))
        if cut >= 90:
            return sentence[:cut].rstrip(" ،,") + "."
        return sentence[: limit - 1].rstrip(" ،,") + "…"

    # Avoid repeating the title verbatim as the summary when possible.
    title_norm = " ".join(str(title or "").split()).casefold()
    summary = ""
    for sentence in sentences:
        if sentence.casefold() != title_norm:
            summary = compact(sentence, 240)
            break
    if not summary:
        summary = compact(sentences[0], 240)

    signal_terms = (
        "يبلغ", "بطول", "بعرض", "مسار", "استغرق", "شهر", "أشهر", "يهدف",
        "يسهم", "يساهم", "يشمل", "ضمن", "بدأ", "افتتاح", "إتاحة", "اتاحة",
        "تنفيذ", "تطوير", "يربط", "يساعد", "كفاءة", "طاقة", "مشروع", "برنامج",
    )

    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        if sentence == summary or sentence.casefold() == title_norm:
            continue
        score = 0
        if re.search(r"\d", sentence):
            score += 3
        score += sum(1 for term in signal_terms if term in sentence)
        if 45 <= len(sentence) <= 260:
            score += 1
        scored.append((score, -idx, sentence))

    # Highest-signal facts first, while keeping the final bullets concise.
    chosen: list[str] = []
    for _score, _neg_idx, sentence in sorted(scored, reverse=True):
        short = compact(sentence, 180)
        if short and short not in chosen and short != summary:
            chosen.append(short)
        if len(chosen) >= 4:
            break

    # If the article is short / low-signal, preserve its original order.
    if len(chosen) < 3:
        for sentence in sentences[1:]:
            short = compact(sentence, 180)
            if short and short not in chosen and short != summary:
                chosen.append(short)
            if len(chosen) >= 4:
                break

    return summary, chosen[:4]

def render_content_detail(
    intent: str,
    data: Any,
    lang: str,
    *,
    salutation: str = "",
    full_details: bool = False,
) -> str:
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
            "news_details": "هذا اللي لقيته عن الخبر 📰:",
            "announcement_details": "هذا اللي لقيته عن الإعلان 📢:",
            "policy_details": "هذا اللي لقيته عن السياسة 📄:",
            "mazaya_details": "هذي تفاصيل الميزة اللي طلبتها 🎁:",
        }
    else:
        headings = {
            "news_details": "News details:",
            "announcement_details": "Announcement details:",
            "policy_details": "Policy details:",
            "mazaya_details": "Benefit details:",
        }

    lines: list[str] = []
    if lang == "ar" and intent == "mazaya_details" and salutation:
        lines.append(f"{salutation} 😊")

    # Friendly news/announcement rendering. The first detail request is a
    # compact summary; an explicit follow-up for full details prints the full
    # VERIFIED API body of the same selected item.
    if lang == "ar" and intent in {"news_details", "announcement_details"}:
        is_news = intent == "news_details"
        if full_details:
            lines.append(
                "أكيد 😊📰 هذي التفاصيل الكاملة للخبر:"
                if is_news
                else "هذي تفاصيل الإعلان 📢:"
            )
            if title:
                lines.append(title)
            if detail and detail.casefold() != title.casefold():
                lines.append("")
                lines.append(detail)
        else:
            lines.append(
                "أكيد 😊📰 هذي خلاصة الخبر بشكل سريع:"
                if is_news
                else "هذي خلاصة الإعلان 📢:"
            )
            if title:
                lines.append(title)
            if detail and detail.casefold() != title.casefold():
                summary, bullets = _summarize_news_detail_ar(detail, title)
                if summary:
                    lines.append("")
                    lines.append(summary)
                if bullets:
                    lines.append("")
                    lines.append("أبرز التفاصيل ✨:")
                    lines.extend(f"• {item}" for item in bullets)
            # The user is already viewing the selected item; don't upsell
            # its details or add a generic follow-up on every response.
    else:
        lines.append(headings.get(intent, "التفاصيل:" if lang == "ar" else "Details:"))
        if title:
            lines.append(title)
        if company and company.casefold() != title.casefold():
            lines.append(("الجهة: " if lang == "ar" else "Provider: ") + company)
        if detail and detail.casefold() != title.casefold():
            lines.append(detail)
    if website.startswith(("http://", "https://")):
        if intent == "mazaya_details":
            if lang == "ar":
                lines.append("شيّك على الموقع للتفاصيل أكثر 👇")
                lines.append(f"[زيارة الموقع]({website})")
            else:
                lines.append("Check the website for more details 👇")
                lines.append(f"[Open website]({website})")
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
