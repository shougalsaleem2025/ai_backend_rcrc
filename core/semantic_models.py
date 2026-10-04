from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Language = Literal["ar", "en"]
TargetKind = Literal[
    "self",
    "employee",
    "manager_of_self",
    "focus",
    "previous_focus",
    "relation_subject",
    "relation_object",
    "none",
]

Intent = Literal[
    "general",
    "web_search",
    "summarize_previous",
    "profile",
    "employee_details",
    "employee_search",
    "employee_role_search",
    "employee_count",
    "general_department_counts",
    "general_department_members",
    "employee_list",
    "manager",
    "leave_balance",
    "leave_types",
    "leave_history",
    "vacations_list",
    "leave_overlap",
    "meetings",
    "meeting_list",
    "employee_availability",
    "attendance_today",
    "attendance_history",
    "working_hours",
    "reporting_employees",
    "team_attendance",
    "tarasol_count",
    "tarasol_details",
    "employee_image",
    "hr_letter_recipients",
    "hr_letter_types",
    "policy_search",
    "policy_details",
    "news",
    "news_details",
    "announcements",
    "announcement_details",
    "mazaya",
    "mazaya_details",
]

Operation = Literal[
    "chat",
    "lookup",
    "list",
    "details",
    "search",
    "calculate",
    "summarize",
]

Period = Literal[
    "none",
    "today",
    "tomorrow",
    "yesterday",
    "this_week",
    "last_week",
    "this_month",
    "last_month",
    "custom",
]

CANONICAL_FIELDS = {
    "name",
    "name_en",
    "job_title",
    "job_title_en",
    "department",
    "department_en",
    "sector",
    "sector_en",
    "email",
    "mobile",
    "extension",
    "username",
    "employee_type",
    "balance",
    "count",
    "subject",
    "creation_date",
    "creator",
    "sender",
    "recipient",
    "last_forwarded_date",
    "document_type",
    "priority",
    "importance",
    "secrecy",
    "classification",
    "creation_department",
    "sender_department",
    "company",
    "details",
    "website",
    "start_date",
    "end_date",
    "tags",
    "start_time",
    "end_time",
    "status",
    "date",
    "all",
}


class TargetRef(BaseModel):
    kind: TargetKind = "none"
    name: str = ""


class ParsedRequest(BaseModel):
    intent: Intent = "general"
    operation: Operation = "chat"
    targets: list[TargetRef] = Field(default_factory=list, max_length=8)
    fields: list[str] = Field(default_factory=list, max_length=24)
    query: str = ""
    role_query: str = ""
    department_query: str = ""
    period: Period = "none"
    start_date: str = ""
    end_date: str = ""
    ordinal: int = Field(default=0, ge=0, le=100)
    relation_depth: int = Field(default=1, ge=1, le=5)
    amount: float | None = None
    time_query: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification_reason: str = ""

    def normalized_fields(self) -> list[str]:
        output: list[str] = []
        for field in self.fields:
            value = str(field or "").strip()
            if value in CANONICAL_FIELDS and value not in output:
                output.append(value)
        return output


class ResolvedTarget(BaseModel):
    kind: TargetKind
    username: str
    employee: dict = Field(default_factory=dict)
    source_text: str = ""


class ValidationResult(BaseModel):
    request: ParsedRequest
    clarification: str = ""

    @property
    def ok(self) -> bool:
        return not bool(self.clarification)
