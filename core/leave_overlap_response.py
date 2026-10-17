"""Safe, bilingual rendering for the verified intersectedVacations response."""
from __future__ import annotations
from datetime import date, datetime
from typing import Any


def _date(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    # The API returns date-only values encoded as UTC timestamps. Preserve the
    # calendar date rather than shifting a date-only field between time zones.
    try:
        return date.fromisoformat(raw[:10]).strftime("%d/%m/%Y")
    except ValueError:
        return raw


def render_leave_overlap(payload: dict[str, Any], language: str = "ar", *, start_date: str = "", end_date: str = "") -> str:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        data = {}
    records = data.get("records")
    if records is None:
        records = data.get("vacations")
    if not isinstance(records, list):
        return ("⚠️ وصلت نتيجة التحقق، لكن ما قدرت أقرأ تفاصيل الإجازات. جرّب مرة ثانية. 💙"
                if language == "ar" else
                "⚠️ I received a response, but couldn't read the leave details. Please try again. 💙")
    is_ar = language != "en"
    if not records:
        period = f" من {_date(start_date)} إلى {_date(end_date)}" if start_date and end_date else ""
        if is_ar:
            return f"✅ تمام! ما ظهرت إجازات متداخلة مع الفترة{period}. ✨"
        period_en = f" from {_date(start_date)} to {_date(end_date)}" if start_date and end_date else ""
        return f"✅ Good news! No overlapping leave was found{period_en}. ✨"
    lines = ([f"📅 لقيت {len(records)} إجازة تتداخل مع الفترة المطلوبة:"] if is_ar else
             [f"📅 I found {len(records)} leave record(s) overlapping the requested period:"])
    for i, row in enumerate(records, 1):
        if not isinstance(row, dict):
            continue
        leave_type = str(row.get("typeAr") if is_ar else row.get("typeEn") or "").strip()
        if not leave_type:
            leave_type = str(row.get("typeEn") or row.get("typeAr") or ("إجازة" if is_ar else "Leave"))
        start = _date(row.get("startDate"))
        end = _date(row.get("endDate"))
        duration = row.get("duration")
        status = str(row.get("status") or "").upper()
        if is_ar:
            statuses = {"APPROVED": "معتمدة", "PENDING": "بانتظار الموافقة", "REJECTED": "مرفوضة", "CANCELLED": "ملغاة", "CANCELED": "ملغاة"}
            line = f"{i}. **{leave_type}**"
            if start: line += f"\n   📆 من {start}" + (f" إلى {end}" if end and end != start else "")
            if duration is not None: line += f"\n   ⏳ المدة: {duration} يوم"
            if status: line += f"\n   📌 الحالة: {statuses.get(status, status)}"
        else:
            line = f"{i}. **{leave_type}**"
            if start: line += f"\n   📆 {start}" + (f" – {end}" if end and end != start else "")
            if duration is not None: line += f"\n   ⏳ Duration: {duration} day(s)"
            if status: line += f"\n   📌 Status: {status.title()}"
        lines.append(line)
    lines.append("✨ هذي الإجازات المسجلة في النظام خلال الفترة المحددة." if is_ar else
                 "✨ These are the leave records returned for that period.")
    return "\n\n".join(lines)
