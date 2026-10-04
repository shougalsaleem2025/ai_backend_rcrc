from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.employee_api import (
    EmployeeAPIClient,
    EmployeeAPIError,
    EmployeeAuthenticationError,
    EmployeeAuthorizationError,
)


# Internal directory facts (name/title/department/contact/hierarchy) are directory
# information. Operational self-service data such as leave, attendance, meetings
# and Tarasol is private to the signed-in employee.
SELF_ONLY_INTENTS = frozenset({
    "leave_balance",
    "leave_history",
    "leave_overlap",
    "meetings",
    "meeting_list",
    "attendance_today",
    "attendance_history",
    "working_hours",
    "tarasol_count",
    "tarasol_details",
})

# Team attendance, where enabled, may be checked only for the signed-in employee
# or a verified direct report. Employee profile images are directory-visible in this
# application and are therefore not restricted to self/direct-report scope.
SELF_OR_DIRECT_REPORT_INTENTS = frozenset({
    "team_attendance",
})


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


def _uid(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("username", "userid", "userName", "userCode"):
        value = str(row.get(key) or "").strip().casefold()
        if value:
            return value
    return ""


class AuthorizationService:
    """Backend-enforced security boundary for person-scoped enterprise data."""

    def __init__(self, api: EmployeeAPIClient) -> None:
        self.api = api

    def requires_person_scope(self, intent: str) -> bool:
        value = str(intent or "")
        return value in SELF_ONLY_INTENTS or value in SELF_OR_DIRECT_REPORT_INTENTS

    async def authorize(
        self,
        *,
        intent: str,
        requester_username: str,
        target_username: str,
        access_token: str,
    ) -> AccessDecision:
        if not self.requires_person_scope(intent):
            return AccessDecision(True, "directory_or_non_person_scope")

        requester = str(requester_username or "").strip().casefold()
        target = str(target_username or "").strip().casefold()
        if not requester or not target:
            return AccessDecision(False, "missing_identity")

        if requester == target:
            return AccessDecision(True, "self")

        if str(intent or "") in SELF_ONLY_INTENTS:
            return AccessDecision(False, "self_only")

        try:
            result = await self.api.get_reporting_employees(
                username=requester,
                access_token=access_token,
            )
        except (EmployeeAuthenticationError, EmployeeAuthorizationError):
            raise
        except EmployeeAPIError:
            # Fail closed if hierarchy verification is unavailable.
            return AccessDecision(False, "hierarchy_unavailable")

        employees = result.get("employees") if isinstance(result, dict) else []
        direct_reports = {_uid(row) for row in (employees or []) if isinstance(row, dict)}
        direct_reports.discard("")
        if target in direct_reports:
            return AccessDecision(True, "direct_report")
        return AccessDecision(False, "not_self_or_direct_report")
