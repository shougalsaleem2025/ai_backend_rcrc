"""Conservative organization-directory routing.

Only routes explicit organizational questions. This is NOT an authorization layer
and does not manufacture department membership. Directory API results remain the
single source of truth.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata

ALIASES = {
    'it':'التقنية والرقمنة','ict':'التقنية والرقمنة',
    'information technology':'التقنية والرقمنة',
    'technology and digitization':'التقنية والرقمنة',
    'hr':'الموارد البشرية','human resources':'الموارد البشرية',
    'ia':'المراجعة الداخلية','internal audit':'المراجعة الداخلية',
    'cybersecurity':'الأمن السيبراني','cyber security':'الأمن السيبراني',
    'enterprise solutions':'الحلول المؤسسية',
    'finance':'المالية','procurement':'المشتريات',
}

@dataclass(frozen=True)
class DirectoryIntent:
    action: str # members | count | manager | units
    department: str


def normalize(value: str) -> str:
    value = unicodedata.normalize('NFKC', str(value or '')).casefold()
    value = re.sub(r'[\u064b-\u065f\u0670ـ]', '', value)
    value = value.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ى':'ي','ة':'ه'}))
    return ' '.join(value.split())


def extract_department(message: str) -> str:
    normalized = normalize(message)
    # Whole-word matching, including Arabic article directly attached to a Latin acronym.
    for alias, department in sorted(ALIASES.items(), key=lambda item: -len(item[0])):
        pattern = rf'(?<![\w])(?:ال|بال|لل)?\s*{re.escape(alias)}(?![\w])'
        if re.search(pattern, normalized):
            return department
    for phrase in ('التقنيه والرقمنه','المراجعه الداخليه','الموارد البشريه',
                   'الحلول الموسسيه','الامن السيبراني'):
        if phrase in normalized:
            return {'التقنيه والرقمنه':'التقنية والرقمنة',
                    'المراجعه الداخليه':'المراجعة الداخلية',
                    'الموارد البشريه':'الموارد البشرية',
                    'الحلول الموسسيه':'الحلول المؤسسية',
                    'الامن السيبراني':'الأمن السيبراني'}[phrase]
    return ''


def route_department_message(message: str) -> DirectoryIntent | None:
    normalized = normalize(message)
    department = extract_department(message)
    if not department:
        return None
    # Never reinterpret unrelated employee-self-service queries as directory queries.
    if re.search(r'بصم|دوام|حضوري|اجاز|رصيد|اجتماع|معاملات|attendance|leave balance|meetings?', normalized):
        return None
    units = bool(re.search(r'(?:الادارات|الاقسام|اقسام|ادارات|الفرق|الوحدات)\s+(?:اللي\s+)?(?:تحت|التابعه|التابعة)', normalized) or
                 re.search(r'(?:what|which|list|show)\s+(?:departments|divisions|subunits|sections)\s+(?:under|within|in)', normalized))
    if units:
        return DirectoryIntent('units', department)
    manager = bool(re.search(r'(?:مين|من|اسم|who\s+is|who\s+manages|who\s+heads)\s+(?:هو\s+)?(?:مدير|رئيس|المدير|مدير\s+عام|head\s+of|director\s+of)', normalized) or
                   re.search(r'\b(?:manager|director|head)\s+of\b', normalized))
    if manager:
        return DirectoryIntent('manager', department)
    people = bool(re.search(r'موظف|موظفين|موظفي|موظفو|اللي\s+يشتغلون|employees?|staff|who works|who work', normalized))
    if people:
        count = bool(re.search(r'كم|عدد|احصائي|how many|count|number of', normalized))
        return DirectoryIntent('count' if count else 'members', department)
    return None
