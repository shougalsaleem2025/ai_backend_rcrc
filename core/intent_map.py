from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPolicy:
    intent: str
    tool_name: str
    target_mode: str  # none | self_default | required_employee | manager_subject
    result_kind: str
    supports_multiple_targets: bool = False


TOOL_REGISTRY: dict[str, ToolPolicy] = {
    "profile": ToolPolicy("profile", "get_employee_profile", "self_default", "profile"),
    "employee_details": ToolPolicy("employee_details", "get_employee_details", "required_employee", "profile", True),
    "employee_search": ToolPolicy("employee_search", "search_employees", "none", "employee_list"),
    "employee_role_search": ToolPolicy("employee_role_search", "find_employee_by_role", "none", "employee_list"),
    "employee_count": ToolPolicy("employee_count", "directory_query", "none", "employee_count"),
    "general_department_counts": ToolPolicy("general_department_counts", "directory_query", "none", "general_department_counts"),
    "general_department_members": ToolPolicy("general_department_members", "directory_query", "none", "general_department_members"),
    "employee_list": ToolPolicy("employee_list", "directory_query", "none", "employee_list"),
    "manager": ToolPolicy("manager", "get_manager_details", "self_default", "manager", True),
    "leave_balance": ToolPolicy("leave_balance", "get_leave_balance", "self_default", "leave_balance", True),
    "leave_types": ToolPolicy("leave_types", "get_vacation_types", "none", "generic_list"),
    "leave_history": ToolPolicy("leave_history", "get_vacation_history", "self_default", "generic_list", True),
    "vacations_list": ToolPolicy("vacations_list", "get_vacations_list", "none", "generic_list"),
    "leave_overlap": ToolPolicy("leave_overlap", "get_intersected_vacations", "self_default", "generic_list", True),
    "meetings": ToolPolicy("meetings", "get_my_meetings", "self_default", "meetings", True),
    "meeting_list": ToolPolicy("meeting_list", "get_meeting_list", "self_default", "meetings", True),
    "employee_availability": ToolPolicy("employee_availability", "get_my_meetings", "required_employee", "availability"),
    "attendance_today": ToolPolicy("attendance_today", "get_today_attendance", "self_default", "attendance", True),
    "attendance_history": ToolPolicy("attendance_history", "get_employee_attendance", "self_default", "attendance", True),
    "working_hours": ToolPolicy("working_hours", "get_working_hours", "self_default", "attendance", True),
    "reporting_employees": ToolPolicy("reporting_employees", "get_reporting_employees", "self_default", "reporting_employees", True),
    "team_attendance": ToolPolicy("team_attendance", "get_subemployees_attendance", "self_default", "attendance", True),
    "tarasol_count": ToolPolicy("tarasol_count", "get_tarasol_count", "self_default", "tarasol_count", True),
    "tarasol_details": ToolPolicy("tarasol_details", "get_tarasol_transactions", "self_default", "tarasol", True),
    "employee_image": ToolPolicy("employee_image", "get_employee_image", "self_default", "generic"),
    "hr_letter_recipients": ToolPolicy("hr_letter_recipients", "get_hr_letter_recipients", "none", "generic_list"),
    "hr_letter_types": ToolPolicy("hr_letter_types", "get_hr_letter_types", "none", "generic_list"),
    "policy_search": ToolPolicy("policy_search", "search_policies", "none", "content_list"),
    "policy_details": ToolPolicy("policy_details", "get_policy_details", "none", "content_detail"),
    "news": ToolPolicy("news", "get_news_list", "none", "content_list"),
    "news_details": ToolPolicy("news_details", "get_news_details", "none", "content_detail"),
    "announcements": ToolPolicy("announcements", "get_announcements_list", "none", "content_list"),
    "announcement_details": ToolPolicy("announcement_details", "get_announcement_details", "none", "content_detail"),
    "mazaya": ToolPolicy("mazaya", "get_mazaya_list", "none", "content_list"),
    "mazaya_details": ToolPolicy("mazaya_details", "get_mazaya_details", "none", "content_detail"),
}


DETAIL_BASE_INTENT = {
    "policy_details": "policy_search",
    "news_details": "news",
    "announcement_details": "announcements",
    "mazaya_details": "mazaya",
}


def get_policy(intent: str) -> ToolPolicy | None:
    return TOOL_REGISTRY.get(str(intent or ""))
