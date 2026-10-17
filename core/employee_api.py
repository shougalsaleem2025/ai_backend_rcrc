"""
core/employee_api.py

RCRC Employee API integration layer.

Responsibilities:
- Call approved RCRC Employee APIs.
- Forward the authenticated employee OAuth token.
- Never allow the LLM to choose the authenticated username.
- Validate HTTP and application-level responses.
- Normalize data before it is returned to the LLM.
"""

import asyncio
import base64
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from core.config import settings


# Riyadh is UTC+3 year-round.
RIYADH_TIMEZONE = timezone(timedelta(hours=3))


def _api_us_date(value: str) -> str:
    """Convert an internal ISO date to the legacy vacation API MM-DD-YYYY format."""
    raw = str(value or "").strip()
    if not raw:
        return raw
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.strftime("%m-%d-%Y")
        except ValueError:
            continue
    return raw


# ======================================================================
# Exceptions
# ======================================================================

class EmployeeAPIError(Exception):
    """Base error for Employee API failures."""


class EmployeeAuthenticationError(EmployeeAPIError):
    """Raised when authentication fails."""


class EmployeeAuthorizationError(EmployeeAPIError):
    """Raised when authorization fails."""


# ======================================================================
# Helpers
# ======================================================================

def _convert_utc_to_riyadh(value: str | None) -> str | None:
    """
    Convert an ISO UTC datetime returned by RCRC APIs into Riyadh local time.

    Example:
        2026-09-10T09:00:00Z -> 2026-09-10T12:00:00+03:00
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Do not guess if the upstream API returns a datetime without timezone.
        if parsed.tzinfo is None:
            return value
        return parsed.astimezone(RIYADH_TIMEZONE).isoformat()
    except ValueError:
        return value


def _require_dict(value: Any, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EmployeeAPIError(message)
    return value


def _require_list(value: Any, message: str) -> list[Any]:
    if not isinstance(value, list):
        raise EmployeeAPIError(message)
    return value


RCRC_EMAIL_DOMAIN = "rcrc.gov.sa"


def _email_from_username(username: Any) -> str | None:
    """Build the corporate email from a verified directory username.

    RCRC directory usernames are the local part of the corporate email.
    Example: hajri.e -> hajri.e@rcrc.gov.sa
    """
    value = str(username or "").strip()
    if not value:
        return None
    if "@" in value:
        return value
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        return None
    return f"{value}@{RCRC_EMAIL_DOMAIN}"


def _directory_contact_value(row: dict[str, Any], kind: str) -> Any:
    """Read contact data defensively from legacy SearchEmployee rows.

    The deployed API has used different field names between versions, so this
    function first checks known names case-insensitively, then safe key-name
    patterns. It never guesses mobile/extension values.
    """
    lowered = {str(key).casefold(): value for key, value in row.items()}

    def first(*keys: str) -> Any:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
            value = lowered.get(key.casefold())
            if value not in (None, ""):
                return value
        return None

    if kind == "email":
        direct = first(
            "emailAddress", "email", "mail", "workEmail", "email_address",
            "emailId", "emailID",
        )
        if direct not in (None, ""):
            return direct
        for key, value in row.items():
            if value in (None, ""):
                continue
            key_norm = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if "email" in key_norm or key_norm == "mail":
                return value
        return None

    if kind == "mobile":
        direct = first(
            "mobile", "mobileNo", "mobileNumber", "mobilePhone",
            "cellPhone", "cellNo", "cellNumber", "phoneMobile",
            "mobile_number", "mobile_no", "workMobile",
        )
        if direct not in (None, ""):
            return direct
        for key, value in row.items():
            if value in (None, ""):
                continue
            key_norm = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if "mobile" in key_norm or "cell" in key_norm:
                return value
            if "phone" in key_norm or "telephone" in key_norm:
                digits = re.sub(r"\D", "", str(value))
                if len(digits) >= 9:
                    return value
        return None

    if kind == "extension":
        direct = first(
            "ext", "extension", "phoneExt", "extensionNumber",
            "telephoneExtension", "workExtension",
        )
        if direct not in (None, ""):
            return direct
        for key, value in row.items():
            if value in (None, ""):
                continue
            key_norm = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if "extension" in key_norm or key_norm in {"ext", "phoneext"}:
                return value
        return None

    return None


# ======================================================================
# API Client
# ======================================================================

class EmployeeAPIClient:

    MERGED_API_BUILD = "rcrc-employee-api-v14-myprofile-empcode-2026-10-02"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.employee_api_base_url.rstrip("/"),
            timeout=settings.employee_api_timeout_seconds,
        )
        # Cache verified employee codes by userid. Populated only from
        # verified API rows; no identifiers are guessed.
        self._employee_code_by_username: dict[str, str] = {}
        self._employee_name_by_username: dict[str, str] = {}

    async def close(self) -> None:
        await self._client.aclose()


    # ------------------------------------------------------------------
    # Employee code (empcode) resolution + myProfile
    # ------------------------------------------------------------------
    # - PersonalInformation/myProfile          -> needs username (self profile)
    # - PersonalInformation/employeeFullDetails -> needs the employee CODE
    # Codes come from verified sources only: the auth middleware, myProfile,
    # SearchEmployee rows (userCode/empCode) or managerDetails.userCode.

    def remember_employee_code(self, username: str, emp_code: str) -> None:
        user = str(username or "").strip().casefold()
        code = str(emp_code or "").strip()
        if user and code:
            self._employee_code_by_username[user] = code

    @staticmethod
    def _row_code(row: dict[str, Any]) -> str:
        for key in ("empCode", "empcode", "EmpCode", "userCode", "employeeCode", "empNo"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    @staticmethod
    def _row_username(row: dict[str, Any]) -> str:
        return str(
            row.get("userid") or row.get("username") or row.get("userName") or ""
        ).strip().casefold()

    @staticmethod
    def _extract_person_row(data: Any) -> dict[str, Any]:
        """Find the employee record inside a myProfile response (shape varies)."""
        person_keys = {"userid", "username", "namea", "namel", "empCode", "userCode", "jodesca", "dpDesca"}
        if isinstance(data, list):
            data = next((x for x in data if isinstance(x, dict)), None)
        if not isinstance(data, dict):
            raise EmployeeAPIError("myProfile response is invalid.")
        for key in ("employeeDetails", "profile", "myProfile", "employee", "personalInformation", "data"):
            value = data.get(key)
            if isinstance(value, list):
                value = next((x for x in value if isinstance(x, dict)), None)
            if isinstance(value, dict) and person_keys.intersection(value.keys()):
                return value
        if person_keys.intersection(data.keys()):
            return data
        raise EmployeeAPIError("myProfile response does not contain an employee record.")

    async def get_my_profile_row(self, *, username: str, access_token: str) -> dict[str, Any]:
        """POST /api/PersonalInformation/myProfile (form-data: username).

        Returns the employee row only when its identity matches the requested
        username (or carries no identity field). Caches the empcode it finds.
        """
        cleaned = str(username or "").strip()
        if not cleaned:
            raise EmployeeAPIError("Employee username is required.")
        data = await self._request(
            method="POST",
            endpoint="/api/PersonalInformation/myProfile",
            access_token=access_token,
            form_fields={"username": cleaned},
        )
        row = self._extract_person_row(data)
        returned = self._row_username(row)
        if returned and returned != cleaned.casefold():
            print(f"[myProfile Identity Guard] expected={cleaned!r} returned={returned!r}")
            raise EmployeeAPIError("myProfile returned a different employee.")
        code = self._row_code(row)
        if code:
            self.remember_employee_code(cleaned, code)
        print(f"[myProfile] username={cleaned!r} fields={sorted(str(k) for k in row.keys())} code={bool(code)}")
        return row

    async def _self_profile_row(self, *, username: str, access_token: str) -> dict[str, Any]:
        """Return ONLY the authenticated user's myProfile row.

        IMPORTANT: self-service APIs are now token-owner scoped.  If myProfile
        fails, do not fall back to SearchEmployee -> employeeFullDetails using a
        POC/dev username.  That fallback can resolve a different employee and is
        unsafe for "my data" requests.  Fail closed instead.
        """
        try:
            return await self.get_my_profile_row(
                username=username,
                access_token=access_token,
            )
        except EmployeeAuthenticationError:
            raise
        except EmployeeAuthorizationError:
            raise
        except EmployeeAPIError as exc:
            print(
                f"[myProfile Self Only Error] username={username!r} reason={exc}"
            )
            raise

    async def resolve_employee_code(self, *, username: str, access_token: str) -> str:
        cleaned = str(username or "").strip()
        if not cleaned:
            raise EmployeeAPIError("Employee username is required.")
        cached = self._employee_code_by_username.get(cleaned.casefold())
        if cached:
            return cached

        # 1) myProfile by username (cheap, exact). It may be self-only, so a
        #    403 here is not fatal: we fall through to the directory.
        try:
            await self.get_my_profile_row(username=cleaned, access_token=access_token)
        except EmployeeAuthenticationError:
            raise
        except EmployeeAPIError:
            pass
        hit = self._employee_code_by_username.get(cleaned.casefold())
        if hit:
            return hit

        # 2) SearchEmployee with the username text, 3) one wide snapshot.
        #    Every verified (userid, code) pair is cached, so the snapshot runs
        #    at most once per process.
        for search_text, batch in ((cleaned, 5000), ("", 5000)):
            try:
                result = await self.search_directory_rows(
                    access_token=access_token,
                    search=search_text,
                    batch_size=batch,
                )
            except EmployeeAuthenticationError:
                raise
            except EmployeeAPIError:
                continue
            for row in result.get("employees") or []:
                if not isinstance(row, dict):
                    continue
                row_user = self._row_username(row)
                row_code = self._row_code(row)
                if row_user and row_code:
                    self._employee_code_by_username.setdefault(row_user, row_code)
            hit = self._employee_code_by_username.get(cleaned.casefold())
            if hit:
                print(f"[EmpCode Resolved] username={cleaned!r} source={'search' if search_text else 'snapshot'}")
                return hit

        raise EmployeeAPIError("Could not resolve the employee code for this user.")

    async def _full_details_by_code(self, *, emp_code: str, access_token: str) -> dict[str, Any]:
        """Call employeeFullDetails with the exact form-data field required by the API."""
        code = str(emp_code or "").strip()
        if not code:
            raise EmployeeAPIError("Employee code is required.")

        data = await self._request(
            method="POST",
            endpoint="/api/PersonalInformation/employeeFullDetails",
            access_token=access_token,
            form_fields={"empcode": code},
        )

        return _require_dict(data, "Employee full details response is invalid.")

    async def _full_details(
        self,
        *,
        username: str,
        access_token: str,
        emp_code: str = "",
    ) -> dict[str, Any]:
        code = str(emp_code or "").strip() or await self.resolve_employee_code(
            username=username,
            access_token=access_token,
        )
        return await self._full_details_by_code(emp_code=code, access_token=access_token)
    # --- end empcode helpers v2 ---

    # ------------------------------------------------------------------
    # Generic RCRC API request
    # ------------------------------------------------------------------

    async def _request(
        self,
        *,
        method: str,
        endpoint: str,
        access_token: str,
        form_fields: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute an authenticated RCRC API request.

        Supports GET, POST, multipart/form-data and query parameters.
        The API access token is supplied by the trusted backend context.
        """
        if not access_token:
            raise EmployeeAuthenticationError("Employee access token is missing.")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        request_kwargs: dict[str, Any] = {"headers": headers}

        if query_params is not None:
            request_kwargs["params"] = {
                key: str(value)
                for key, value in query_params.items()
                if value is not None
            }

        if form_fields is not None:
            # httpx generates multipart/form-data and the boundary automatically.
            request_kwargs["files"] = {
                key: (None, str(value))
                for key, value in form_fields.items()
                if value is not None
            }

        try:
            response = await self._client.request(method=method, url=endpoint, **request_kwargs)
        except httpx.TimeoutException as exc:
            raise EmployeeAPIError("Employee API request timed out.") from exc
        except httpx.RequestError as exc:
            raise EmployeeAPIError("Could not connect to the Employee API.") from exc

        # HTTP authentication / authorization
        if response.status_code == 401:
            raise EmployeeAuthenticationError("Employee API rejected the access token.")
        if response.status_code == 403:
            raise EmployeeAuthorizationError("Employee API denied access.")
        if response.status_code >= 400:
            raise EmployeeAPIError(f"Employee API returned HTTP {response.status_code}.")

        # JSON parsing
        try:
            payload = response.json()
        except ValueError as exc:
            raise EmployeeAPIError("Employee API returned invalid JSON.") from exc

        # Some legacy APIs may return JSON encoded inside another JSON string.
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise EmployeeAPIError("Employee API returned an unexpected string response.") from exc

        # Application-level status / legacy direct payloads.
        # Most RCRC endpoints use the standard wrapper
        #   {"statusCode": 200, "success": true, "data": ...}
        # but several read-only legacy endpoints return a direct JSON list or
        # object instead. Accept both shapes, while still validating the
        # wrapper whenever wrapper status fields are present.
        if isinstance(payload, dict):
            is_wrapped = "success" in payload or "statusCode" in payload
            if is_wrapped:
                if payload.get("success") is not True or payload.get("statusCode") != 200:
                    raise EmployeeAPIError(payload.get("message") or "Employee API request failed.")
                return payload.get("data")
            return payload

        if isinstance(payload, list):
            return payload

        raise EmployeeAPIError("Employee API returned an unexpected response format.")

    # ==================================================================
    # 1. Leave Balance
    # ==================================================================

    async def get_leave_balance(self, *, username: str, access_token: str) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/Vacation/getVacationBalance",
            access_token=access_token,
            form_fields={"username": username},
        )
        data = _require_dict(data, "Vacation balance response is invalid.")
        balance = data.get("balance")
        if balance is None:
            raise EmployeeAPIError("Vacation balance is missing.")
        return {"balance": balance}

    # ==================================================================
    # 1B. Vacation / Leave Types
    # ==================================================================

    async def get_vacation_types(self, *, access_token: str) -> dict[str, Any]:
        """
        Read the approved leave-type list from GET /api/VacationTypes.

        This endpoint is represented as no-auth in the supplied collection,
        but the backend still sends the current bearer token when available.
        The response shape is not documented in the collection, so this
        parser is intentionally defensive and never invents leave names.
        """
        headers = {"Accept": "application/json"}
        # Postman documents this GET as publicly callable. Supply bearer only
        # when there is a verified token (do not force one for this endpoint).
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        try:
            response = await self._client.request(method="GET", url="/api/VacationTypes", headers=headers)
        except httpx.TimeoutException as exc:
            raise EmployeeAPIError("Vacation types request timed out.") from exc
        except httpx.RequestError as exc:
            raise EmployeeAPIError("Could not connect to the VacationTypes API.") from exc

        if response.status_code == 401:
            raise EmployeeAuthenticationError("VacationTypes API rejected the access token.")
        if response.status_code == 403:
            raise EmployeeAuthorizationError("VacationTypes API denied access.")
        if response.status_code >= 400:
            raise EmployeeAPIError(f"VacationTypes API returned HTTP {response.status_code}.")

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise EmployeeAPIError("VacationTypes API returned invalid JSON.") from exc

        # Some legacy endpoints return JSON encoded inside a JSON string.
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise EmployeeAPIError("VacationTypes API returned an unexpected string response.") from exc

        # Support both the standard RCRC wrapper and a direct raw list/dict.
        data: Any = payload
        if isinstance(payload, dict) and ("success" in payload or "statusCode" in payload):
            if payload.get("success") is not True or payload.get("statusCode") != 200:
                raise EmployeeAPIError(payload.get("message") or "VacationTypes API request failed.")
            data = payload.get("data")

        rows: list[Any] | None = None

        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            for key in (
                "vacationTypes", "vacation_types", "leaveTypes", "leave_types",
                "types", "items", "results", "list", "data",
            ):
                value = data.get(key)
                if isinstance(value, list):
                    rows = value
                    break

            # If the API returns one leave-type object directly, preserve it.
            if rows is None:
                type_like_keys = {
                    "name", "label", "description", "desc",
                    "namea", "namel", "nameAr", "nameEn",
                    "vacationType", "vacationTypeName", "typeName",
                    "vacationTypeAr", "vacationTypeEn",
                }
                if any(key in data for key in type_like_keys):
                    rows = [data]

            # Last defensive shape: a mapping of codes -> labels/objects.
            if rows is None:
                candidate_rows: list[Any] = []
                for key, value in data.items():
                    if isinstance(value, dict):
                        candidate_rows.append(value)
                    elif isinstance(value, str) and value.strip():
                        candidate_rows.append({"code": key, "name": value.strip()})
                if candidate_rows:
                    rows = candidate_rows

        if rows is None:
            raise EmployeeAPIError("Vacation types response is invalid.")

        def first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
            for key in keys:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for row in rows[:50]:
            if isinstance(row, str):
                name = row.strip()
                if not name:
                    continue
                item = {"name": name, "name_ar": None, "name_en": None, "code": None}

            elif isinstance(row, dict):
                name_ar = first_text(row, (
                    "name_ar", "nameAr", "namea", "arabicName",
                    "vacationTypeAr", "vacationTypeNameAr",
                    "typeNameAr", "descriptionAr", "descAr",
                    "arabicDescription", "nameArabic",
                ))
                name_en = first_text(row, (
                    "name_en", "nameEn", "namel", "englishName",
                    "vacationTypeEn", "vacationTypeNameEn",
                    "typeNameEn", "descriptionEn", "descEn",
                    "englishDescription", "nameEnglish",
                ))
                name = first_text(row, (
                    "name", "label", "description", "desc", "text",
                    "vacationType", "vacationTypeName", "typeName", "type",
                ))
                code_value = (
                    row.get("id")
                    or row.get("code")
                    or row.get("value")
                    or row.get("vacationTypeId")
                    or row.get("typeId")
                )
                if not (name or name_ar or name_en):
                    continue
                item = {"name": name, "name_ar": name_ar, "name_en": name_en, "code": code_value}

            else:
                continue

            identity = (
                str(item.get("name") or "").casefold(),
                str(item.get("name_ar") or "").casefold(),
                str(item.get("name_en") or "").casefold(),
            )
            if identity in seen:
                continue
            seen.add(identity)
            normalized.append(item)

        return {"types": normalized}

    # ==================================================================
    # 2. Employee Profile
    # ==================================================================

    async def get_employee_profile(self, *, username: str, access_token: str) -> dict[str, Any]:
        """Return the signed-in employee profile.

        Source: PersonalInformation/myProfile (username), with
        employeeFullDetails (empcode) as fallback. Contact extraction is
        defensive but never invents values.
        """

        employee = await self._self_profile_row(username=username, access_token=access_token)

        def first_value(*keys: str) -> Any:
            for key in keys:
                value = employee.get(key)
                if value not in (None, ""):
                    return value
            lowered = {str(key).casefold(): value for key, value in employee.items()}
            for key in keys:
                value = lowered.get(str(key).casefold())
                if value not in (None, ""):
                    return value
            return None

        mobile = first_value(
            "mobile", "mobileNo", "mobileNumber", "mobilePhone",
            "mobilephone", "cellPhone", "cellphone", "cellNo",
            "cellNumber", "phoneMobile", "mobile_number", "mobile_no",
            "workMobile",
        )

        # Defensive fallback for undocumented legacy mobile/cell field names.
        # Generic phone values are accepted only when they look like a full
        # phone number, not a short extension.
        if mobile in (None, ""):
            for key, value in employee.items():
                if value in (None, ""):
                    continue
                key_norm = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if "mobile" in key_norm or "cell" in key_norm:
                    mobile = value
                    break
                if "phone" in key_norm or "telephone" in key_norm:
                    digits = re.sub(r"\D", "", str(value))
                    if len(digits) >= 9:
                        mobile = value
                        break

        profile = {
            "username": first_value("userid", "username", "userName", "loginName"),
            "name_ar": first_value("namea", "nameAr", "arabicName", "name_ar", "name"),
            "name_en": first_value("namel", "name1", "nameEn", "englishName", "name_en"),
            "department_ar": first_value("dpDesca", "departmentAr", "departmentArabic", "department_ar"),
            "department_en": first_value("dpDescl", "dpDesc1", "departmentEn", "departmentEnglish", "department_en"),
            "job_title_ar": first_value("jodesca", "jobTitleAr", "jobTitleArabic", "job_title_ar"),
            "job_title_en": first_value("jodesce", "hodesce", "jobTitleEn", "jobTitleEnglish", "job_title_en"),
            "email": first_value("emailAddress", "email", "mail", "email_address", "workEmail"),
            "mobile": mobile,
            "extension": first_value("ext", "extension", "phoneExt", "extensionNumber"),
            "employee_type": first_value("empType", "employeeType", "employee_type"),
            "gender_ar": first_value("genderAr", "gender_ar", "genderArabic"),
            "gender_en": first_value("genderEn", "gender_en", "genderEnglish", "gender"),
        }

        if profile.get("email") in (None, ""):
            profile["email"] = _email_from_username(profile.get("username"))

        print(
            "[Employee Profile Resolved] "
            f"username={str(profile.get('username') or '')!r} "
            f"email={bool(profile.get('email'))} "
            f"mobile={bool(profile.get('mobile'))} "
            f"extension={bool(profile.get('extension'))}"
        )

        return {key: value for key, value in profile.items() if value not in (None, "")}

    # ==================================================================
    # 3. Direct Manager
    # ==================================================================

    async def get_manager_details(
        self,
        *,
        username: str,
        access_token: str,
        emp_code: str = "",
    ) -> dict[str, Any]:
        """Return the direct manager from the new PersonalInformation APIs.

        Preferred flow for the authenticated/current employee:
            myProfile(username) -> managerDetails

        Fallback / another verified employee:
            username -> verified userCode/empcode
            -> employeeFullDetails(empcode) -> managerDetails

        The legacy DirectManager endpoint is never used.
        """
        cleaned_username = str(username or "").strip()
        if not cleaned_username:
            raise EmployeeAPIError("Employee username is required to resolve the manager.")

        def first_value(row: dict[str, Any], *keys: str) -> Any:
            lowered = {str(k).casefold(): v for k, v in row.items()}
            for key in keys:
                value = row.get(key)
                if value not in (None, "", "-"):
                    return value
                value = lowered.get(str(key).casefold())
                if value not in (None, "", "-"):
                    return value
            return None

        def extract_manager(payload: Any) -> dict[str, Any] | None:
            if not isinstance(payload, dict):
                return None

            # Both deployed shapes are supported:
            # 1) myProfile: managerDetails may be at root
            # 2) employeeFullDetails: managerDetails is normally under data
            candidates: list[Any] = [
                payload.get("managerDetails"),
            ]
            data = payload.get("data")
            if isinstance(data, dict):
                candidates.append(data.get("managerDetails"))

            for candidate in candidates:
                if isinstance(candidate, list):
                    candidate = next(
                        (row for row in candidate if isinstance(row, dict)),
                        None,
                    )
                if not isinstance(candidate, dict):
                    continue

                # Do not accept an empty legacy manager object.
                if any(
                    first_value(
                        candidate,
                        "userid", "username", "userName",
                        "userCode", "empCode",
                        "namea", "namel", "name",
                    )
                    not in (None, "", "-")
                    for _ in [0]
                ):
                    return candidate

            return None

        def normalize_manager(manager: dict[str, Any]) -> dict[str, Any]:
            manager_username = first_value(
                manager,
                "userid", "username", "userName", "loginName",
            )
            manager_code = first_value(
                manager,
                "userCode", "empCode", "empcode", "employeeCode",
            )

            result = {
                "username": manager_username,
                "emp_code": manager_code,
                "name_ar": first_value(
                    manager, "namea", "nameAr", "arabicName", "name_ar", "name"
                ),
                "name_en": first_value(
                    manager, "namel", "nameEn", "englishName", "name_en"
                ),
                "job_title_ar": first_value(
                    manager, "jodesca", "jobTitleAr", "jobTitleArabic", "job_title_ar"
                ),
                "job_title_en": first_value(
                    manager, "jodesce", "hodesce", "jobTitleEn", "jobTitleEnglish", "job_title_en"
                ),
                "department_ar": first_value(
                    manager, "dpDesca", "departmentAr", "departmentArabic", "department_ar"
                ),
                "department_en": first_value(
                    manager, "dpDescl", "departmentEn", "departmentEnglish", "department_en"
                ),
                "sector_ar": first_value(
                    manager, "seDesca", "sectorAr", "sector_ar"
                ),
                "sector_en": first_value(
                    manager, "seDescl", "sectorEn", "sector_en"
                ),
                "email": first_value(
                    manager, "emailAddress", "email", "mail", "workEmail"
                ),
                "mobile": first_value(
                    manager, "mobile", "mobileNo", "mobileNumber", "mobilePhone", "cellPhone"
                ),
                "extension": first_value(
                    manager, "ext", "extension", "phoneExt", "extensionNumber"
                ),
            }

            if manager_username and manager_code:
                self.remember_employee_code(str(manager_username), str(manager_code))

            if result.get("email") in (None, "") and manager_username:
                result["email"] = _email_from_username(manager_username)

            return {
                key: value
                for key, value in result.items()
                if value not in (None, "", "-")
            }

        # --------------------------------------------------------------
        # 1) PRIMARY: myProfile(username) already returns managerDetails.
        # --------------------------------------------------------------
        try:
            my_profile_payload = await self._request(
                method="POST",
                endpoint="/api/PersonalInformation/myProfile",
                access_token=access_token,
                form_fields={"username": cleaned_username},
            )

            manager = extract_manager(my_profile_payload)
            if manager is not None:
                result = normalize_manager(manager)
                if result.get("username") or result.get("name_ar") or result.get("name_en"):
                    print(
                        "[Manager Resolved] "
                        f"source='myProfile' employee={cleaned_username!r} "
                        f"manager_username={str(result.get('username') or '')!r} "
                        f"manager_code={str(result.get('emp_code') or '')!r} "
                        f"name={bool(result.get('name_ar') or result.get('name_en'))}"
                    )
                    return result

            print(
                "[Manager myProfile Fallback] "
                f"username={cleaned_username!r} managerDetails_missing_or_empty=True"
            )

        except (EmployeeAuthenticationError, EmployeeAuthorizationError):
            raise
        except EmployeeAPIError as exc:
            print(
                "[Manager myProfile Fallback] "
                f"username={cleaned_username!r} error={exc}"
            )

        # --------------------------------------------------------------
        # 2) FALLBACK: verified empcode -> employeeFullDetails.
        # --------------------------------------------------------------
        code = str(emp_code or "").strip()

        # Do not trust placeholder POC codes such as 0000 for another lookup
        # when a verified directory code can be resolved.
        if not code or set(code) == {"0"}:
            code = str(
                self._employee_code_by_username.get(cleaned_username.casefold(), "") or ""
            ).strip()

        if not code or set(code) == {"0"}:
            code = await self.resolve_employee_code(
                username=cleaned_username,
                access_token=access_token,
            )

        if not code:
            raise EmployeeAPIError("Could not resolve employee code for manager lookup.")

        print(
            "[Manager Lookup] "
            f"source='employeeFullDetails' employee_username={cleaned_username!r} "
            f"empcode={code!r}"
        )

        full_payload = await self._full_details_by_code(
            emp_code=code,
            access_token=access_token,
        )

        manager = extract_manager(full_payload)
        if manager is None:
            raise EmployeeAPIError(
                "Manager details are missing from PersonalInformation response."
            )

        result = normalize_manager(manager)
        if not (result.get("username") or result.get("name_ar") or result.get("name_en")):
            raise EmployeeAPIError("Manager details are not available.")

        print(
            "[Manager Resolved] "
            f"source='employeeFullDetails' employee={cleaned_username!r} "
            f"employee_empcode={code!r} "
            f"manager_username={str(result.get('username') or '')!r} "
            f"manager_code={str(result.get('emp_code') or '')!r} "
            f"name={bool(result.get('name_ar') or result.get('name_en'))}"
        )

        return result

    # ==================================================================
    # 4. Meetings
    # ==================================================================

    async def get_my_meetings(
        self,
        *,
        username: str,
        access_token: str,
        start_date: str,
        end_date: str,
        lang: str,
    ) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/Calendar/myMeetingBasicInfo",
            access_token=access_token,
            form_fields={
                "username": username,
                "startDate": start_date,
                "endDate": end_date,
                "lang": lang,
            },
        )
        data = _require_dict(data, "Meeting response is invalid.")
        meetings = _require_list(data.get("meetings"), "Meetings data is missing or invalid.")

        normalized_meetings = []
        for meeting in meetings:
            if not isinstance(meeting, dict):
                continue

            # Keep the complete verified API row so no calendar fields are lost.
            normalized = dict(meeting)

            normalized.update({
                "subject": (
                    meeting.get("subject")
                    or meeting.get("Subject")
                    or meeting.get("title")
                    or meeting.get("Title")
                    or meeting.get("meetingTitle")
                    or meeting.get("MeetingTitle")
                    or meeting.get("meetingName")
                    or meeting.get("MeetingName")
                    or meeting.get("subjectAr")
                    or meeting.get("subjectEn")
                    or meeting.get("eventTitle")
                    or meeting.get("EventTitle")
                ),
                "start_time": _convert_utc_to_riyadh(
                    meeting.get("startTime")
                    or meeting.get("StartTime")
                    or meeting.get("start_time")
                ),
                "end_time": _convert_utc_to_riyadh(
                    meeting.get("endTime")
                    or meeting.get("EndTime")
                    or meeting.get("end_time")
                ),
                "location": (
                    meeting.get("location")
                    or meeting.get("Location")
                    or meeting.get("meetingLocation")
                    or meeting.get("MeetingLocation")
                ),
                "organizer_name": (
                    meeting.get("organizerName")
                    or meeting.get("OrganizerName")
                    or meeting.get("organizer")
                    or meeting.get("Organizer")
                ),
                "my_response": (
                    meeting.get("myResponse")
                    or meeting.get("MyResponse")
                    or meeting.get("response")
                    or meeting.get("Response")
                ),
                "teams_link": (
                    meeting.get("teamsLink")
                    or meeting.get("TeamsLink")
                    or meeting.get("joinUrl")
                    or meeting.get("JoinUrl")
                    or meeting.get("meetingUrl")
                    or meeting.get("MeetingUrl")
                    or meeting.get("onlineMeetingUrl")
                    or meeting.get("OnlineMeetingUrl")
                    or meeting.get("webLink")
                    or meeting.get("WebLink")
                ),
            })

            # Debug only booleans; never print the URL/token itself.
            print(
                "[Meeting Fields] "
                f"subject={bool(normalized.get('subject'))} "
                f"location={bool(normalized.get('location'))} "
                f"organizer={bool(normalized.get('organizer_name'))} "
                f"response={bool(normalized.get('my_response'))} "
                f"teams_link={bool(normalized.get('teams_link'))}"
            )

            normalized_meetings.append(normalized)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Asia/Riyadh",
            "meetings": normalized_meetings,
        }

    # ==================================================================
    # 5. Today's Attendance
    # ==================================================================

    async def get_today_attendance(self, *, username: str, access_token: str) -> dict[str, Any]:
        """Retrieve today's attendance using the available employee APIs.

        First try the lightweight todayAttendance endpoint. If that route is
        unavailable in the current deployment, fall back to the documented
        EmployeeAttendance date-range endpoint using today's Riyadh date.
        """
        try:
            data = await self._request(
                method="POST",
                endpoint="/api/AttendenceController/todayAttendance",
                access_token=access_token,
                form_fields={"username": username},
            )
        except EmployeeAPIError as exc:
            if "HTTP 404" not in str(exc):
                raise
            today = datetime.now(RIYADH_TIMEZONE).date().isoformat()
            data = await self._request(
                method="POST",
                endpoint="/api/EmployeeAttendance/getEmployeeAttendanceByUsername",
                access_token=access_token,
                form_fields={
                    "username": username,
                    "startDate": today,
                    "endDate": today,
                },
            )

        # Normalize legacy nesting.
        if isinstance(data, dict):
            raw_attendance = data.get("attendance", data)
        elif isinstance(data, list):
            raw_attendance = data
        else:
            raise EmployeeAPIError("Attendance response is invalid.")

        if isinstance(raw_attendance, dict) and isinstance(raw_attendance.get("attendance"), list):
            raw_attendance = raw_attendance.get("attendance")

        records: list[dict[str, Any]] = []
        if isinstance(raw_attendance, list):
            records = [item for item in raw_attendance if isinstance(item, dict)]
        elif isinstance(raw_attendance, dict) and raw_attendance:
            records = [raw_attendance]

        # Verify today's attendance against the authenticated employee.
        # Prefer stable employee codes because the todayAttendance endpoint may
        # return a shortened/placeholder name even when the record is correct.
        if records:
            record = records[0]

            record_code = str(
                record.get("empCode")
                or record.get("userCode")
                or record.get("employeeCode")
                or ""
            ).strip()

            try:
                raw_profile = await self.get_my_profile_row(
                    username=username,
                    access_token=access_token,
                )
            except EmployeeAPIError:
                raw_profile = {}

            profile_code = str(
                raw_profile.get("userCode")
                or raw_profile.get("empCode")
                or raw_profile.get("employeeCode")
                or ""
            ).strip() if isinstance(raw_profile, dict) else ""

            if record_code and profile_code:
                if record_code != profile_code:
                    print(
                        "[Today Attendance Identity Guard] "
                        f"code mismatch expected={profile_code!r} returned={record_code!r}"
                    )
                    return {"attendance": [], "identity_mismatch": True}

                return {"attendance": records, "identity_mismatch": False}

            # Fallback to username only when codes are unavailable.
            record_username = str(
                record.get("userid")
                or record.get("username")
                or record.get("userName")
                or ""
            ).strip()
            profile_username = str(
                raw_profile.get("userid")
                or raw_profile.get("username")
                or raw_profile.get("userName")
                or ""
            ).strip() if isinstance(raw_profile, dict) else ""

            if record_username and profile_username:
                if record_username.casefold() != profile_username.casefold():
                    print(
                        "[Today Attendance Identity Guard] "
                        f"username mismatch expected={profile_username!r} returned={record_username!r}"
                    )
                    return {"attendance": [], "identity_mismatch": True}

                return {"attendance": records, "identity_mismatch": False}

            # Last resort: compare names only when no stable identifier exists.
            record_name = str(
                record.get("name")
                or record.get("employeeName")
                or record.get("empName")
                or ""
            ).strip()

            if record_name:
                try:
                    profile = await self.get_employee_profile(
                        username=username,
                        access_token=access_token,
                    )
                except EmployeeAPIError:
                    profile = {}

                expected_names = [
                    str(profile.get("name_ar") or "").strip(),
                    str(profile.get("name_en") or "").strip(),
                ]
                expected_names = [name for name in expected_names if name]

                def normalize_name(value: str) -> str:
                    value = value.casefold()
                    value = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", value)
                    return " ".join(value.split())

                normalized_record_name = normalize_name(record_name)
                normalized_expected = {
                    normalize_name(name) for name in expected_names if normalize_name(name)
                }

                if normalized_expected and normalized_record_name not in normalized_expected:
                    print(
                        "[Today Attendance Identity Guard] "
                        f"name mismatch returned={record_name!r}"
                    )
                    return {"attendance": [], "identity_mismatch": True}

        return {"attendance": records, "identity_mismatch": False}

    # ==================================================================
    # 6. Employee Image
    # ==================================================================

    async def get_employee_image(self, *, username: str, access_token: str) -> dict[str, Any]:
        """Return an employee image using the new PersonalInformation flow.

        Self / username that myProfile accepts:
            POST /api/PersonalInformation/myProfile (username) -> employee image

        Other employees:
            verified username -> SearchEmployee userCode ->
            POST /api/PersonalInformation/employeeFullDetails (empcode) -> image

        The protected image URL is downloaded server-side with the OAuth token and
        returned as a data URL, so the access token is never exposed to Angular.
        """
        cleaned_username = str(username or "").strip()
        if not cleaned_username:
            raise EmployeeAPIError("Employee username is required.")

        employee_row: dict[str, Any] | None = None
        source = ""

        # myProfile is the authoritative source for the signed-in employee.
        try:
            employee_row = await self.get_my_profile_row(
                username=cleaned_username,
                access_token=access_token,
            )
            source = "myProfile"
        except EmployeeAuthenticationError:
            raise
        except EmployeeAuthorizationError:
            # Some deployments reject myProfile for non-self users. Resolve the
            # verified empcode below rather than treating that as a final denial.
            employee_row = None
        except EmployeeAPIError as exc:
            print(
                f"[Employee Image myProfile Fallback] username={cleaned_username!r} "
                f"reason={exc}"
            )
            employee_row = None

        # For another employee, use userCode as empcode with employeeFullDetails.
        if employee_row is None:
            empcode = await self.resolve_employee_code(
                username=cleaned_username,
                access_token=access_token,
            )
            full = await self._full_details_by_code(
                emp_code=empcode,
                access_token=access_token,
            )
            body = full.get("data") if isinstance(full.get("data"), dict) else full
            candidate = body.get("employeeDetails") if isinstance(body, dict) else None
            if not isinstance(candidate, dict):
                raise EmployeeAPIError("Employee details are missing.")
            employee_row = candidate
            source = "employeeFullDetails"

        def first_value(*keys: str) -> str:
            lowered = {str(k).casefold(): v for k, v in employee_row.items()}
            for key in keys:
                value = employee_row.get(key)
                if value in (None, ""):
                    value = lowered.get(str(key).casefold())
                if value not in (None, ""):
                    text = str(value).strip()
                    if text and text.casefold() not in {"none", "null", "n/a", "-"}:
                        return text
            return ""

        image_value = first_value(
            "image", "imageUrl", "imageURL", "photo", "photoUrl",
            "profileImage", "profileImageUrl", "employeeImage",
        )
        if not image_value:
            raise EmployeeAPIError("Employee image is not available.")

        print(
            f"[Employee Image Resolved] source={source!r} "
            f"username={cleaned_username!r} image={bool(image_value)}"
        )

        # Some APIs may already return an inline image.
        if image_value.startswith("data:image/"):
            return {"image_data_url": image_value}

        # Support an API-relative appianDocument path as well as the full URL.
        if image_value.startswith("/"):
            image_url = f"https://api1.rcrc.gov.sa{image_value}"
        else:
            image_url = image_value

        parsed = urlparse(image_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api1.rcrc.gov.sa"
            or "/api/appianDocument/" not in parsed.path
        ):
            raise EmployeeAPIError("Employee image URL is not trusted.")

        try:
            response = await self._client.get(
                image_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "image/*,application/octet-stream",
                },
                follow_redirects=True,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        except httpx.TimeoutException as exc:
            raise EmployeeAPIError("Employee image request timed out.") from exc
        except httpx.RequestError as exc:
            raise EmployeeAPIError("Could not retrieve the employee image.") from exc

        if response.status_code == 401:
            raise EmployeeAuthenticationError("Employee image request rejected the access token.")
        if response.status_code == 403:
            raise EmployeeAuthorizationError("Employee image request denied access.")
        if response.status_code >= 400:
            raise EmployeeAPIError(
                f"Employee image request returned HTTP {response.status_code}."
            )

        image_bytes = response.content
        if not image_bytes:
            raise EmployeeAPIError("Employee image response was empty.")
        if len(image_bytes) > 5 * 1024 * 1024:
            raise EmployeeAPIError("Employee image is too large to display.")

        content_type = (
            response.headers.get("content-type") or ""
        ).split(";", 1)[0].strip().lower()
        allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}

        if content_type not in allowed_types:
            if image_bytes.startswith(b"\xff\xd8\xff"):
                content_type = "image/jpeg"
            elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                content_type = "image/png"
            elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
                content_type = "image/webp"
            elif image_bytes.startswith((b"GIF87a", b"GIF89a")):
                content_type = "image/gif"
            else:
                raise EmployeeAPIError("Employee image format is not supported.")

        encoded = base64.b64encode(image_bytes).decode("ascii")
        return {"image_data_url": f"data:{content_type};base64,{encoded}"}

    # ==================================================================
    # 7. Tarasol Count
    # ==================================================================

    async def get_tarasol_count(self, *, username: str, access_token: str) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/tarasolCount",
            access_token=access_token,
            form_fields={"username": username},
        )
        data = _require_dict(data, "Tarasol count response is invalid.")
        count = data.get("count")
        if count is None:
            raise EmployeeAPIError("Tarasol count is missing.")
        return {"count": count}

    # ==================================================================
    # 8. Tarasol Transactions
    # ==================================================================

    async def get_tarasol_transactions(self, *, username: str, access_token: str, lang: str) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/tarasolDetails",
            access_token=access_token,
            form_fields={"username": username, "lang": lang},
        )
        data = _require_dict(data, "Tarasol response is invalid.")
        transactions = _require_list(data.get("transactions"), "Tarasol transactions are missing.")
        # Keep only a bounded number for the LLM.
        return {"transactions": transactions[:10]}

    # ==================================================================
    # 9. Search Employees
    # ==================================================================

    @staticmethod
    def _unwrap_employee_record(row: Any) -> dict[str, Any]:
        """Flatten SearchEmployee row wrappers without inventing data.

        The deployed SearchEmployee endpoint may return rows as either:
            {"userid": ..., "userCode": ..., ...}
        or:
            {"employeeDetails": {"userid": ..., "userCode": ..., ...}, ...}

        Directory/search logic must always operate on the actual employee row.
        Any useful wrapper fields are preserved only when the employee row does
        not already contain that key.
        """
        if not isinstance(row, dict):
            return {}

        details = row.get("employeeDetails")
        if isinstance(details, list):
            details = next((item for item in details if isinstance(item, dict)), None)

        if isinstance(details, dict):
            out = dict(details)
            for key, value in row.items():
                if key == "employeeDetails":
                    continue
                if key not in out and value not in (None, "", [], {}):
                    out[key] = value
            return out

        return dict(row)

    async def search_directory_rows(
        self,
        *,
        access_token: str,
        search: str = "",
        batch_size: str | int = "",
    ) -> dict[str, Any]:
        """Raw SearchEmployee access for directory analytics.

        IMPORTANT: this intentionally does NOT use EmployeeList. Both
        ``search`` and ``batchSize`` are sent as form fields. Empty ``search``
        is allowed here because counts/role analytics need a directory
        snapshot rather than a person-name lookup.
        """
        cleaned_search = " ".join(str(search or "").strip().split())
        # Named employee lookup works with multipart/form-data and an EMPTY
        # batchSize. Forcing a huge batch makes the upstream service fall back
        # to its generic first 20 rows. Only use a non-empty batch when the
        # caller explicitly asks for a directory snapshot.
        batch_value = str(batch_size if batch_size is not None else "").strip()

        data = await self._request(
            method="POST",
            endpoint="/api/SearchEmployee/searchEmployeesList",
            access_token=access_token,
            form_fields={"search": cleaned_search, "batchSize": batch_value},
        )
        rows = self._normalize_collection(
            data, "employeeList", "employees", "items", "results", "data"
        )
        rows = [
            self._unwrap_employee_record(row)
            for row in rows
            if isinstance(row, dict)
        ]
        rows = [row for row in rows if row]
        print(
            "[SearchEmployee Directory] "
            f"search={cleaned_search!r} batchSize={batch_value!r} rows={len(rows)}"
        )
        return {
            "search": cleaned_search,
            "batch_size": batch_value,
            "employees": rows,
            "source": "search_employee",
        }

    async def search_employees(self, *, access_token: str, search: str) -> dict[str, Any]:
        """
        Search the employee directory and keep only names that really match.

        The upstream SearchEmployee endpoint can return broad/unrelated rows or
        can fail to return a multi-word match. We therefore:
        1) query the full text,
        2) verify every meaningful name token locally,
        3) if nothing survives, retry using individual name tokens and merge,
        4) verify the complete requested name again before returning anything.
        """
        cleaned_search = " ".join(str(search or "").strip().split())
        # Do not send conversational punctuation/filler as part of a name.
        cleaned_search = re.sub(r"[؟?،,؛;:!…]+$", "", cleaned_search).strip()
        cleaned_search = re.sub(
            r"^(?:(?:موظف(?:ه|ة)?|الموظف)\s+)?(?:باسم|اسم)\s+",
            "", cleaned_search, flags=re.I,
        ).strip()

        # Be tolerant of repeated conversational/search filler that may reach
        # this layer from semantic parsing. The directory API must receive only
        # the employee name, never phrases such as:
        # "ابحث عن موظف باسم ابي اوس الدعجان ه".
        previous = None
        while cleaned_search and cleaned_search != previous:
            previous = cleaned_search

            cleaned_search = re.sub(
                r"^(?:(?:ابي|أبي|ابغى|أبغى|اريد|أريد)\s+)?"
                r"(?:ابحث|أبحث|إبحث|دور|دوّر|دورلي|دور\s+لي|ابحث\s+لي|أبحث\s+لي|"
                r"find|search(?:\s+for)?)"
                r"\s+(?:لي\s+)?(?:(?:عن|على)\s+)?",
                "",
                cleaned_search,
                flags=re.I,
            ).strip()

            cleaned_search = re.sub(
                r"^(?:(?:عن|على)\s+)?"
                r"(?:(?:موظف(?:ه|ة)?|الموظف|شخص)\s+)?"
                r"(?:(?:باسم|اسم|اسمه|إسمه)\s+)?",
                "",
                cleaned_search,
                flags=re.I,
            ).strip()

            # A request starter can remain after a generated search prefix.
            cleaned_search = re.sub(
                r"^(?:ابي|أبي|ابغى|أبغى|اريد|أريد)\s+",
                "",
                cleaned_search,
                flags=re.I,
            ).strip()

        # Remove a stray one-letter Arabic possessive artifact occasionally
        # produced by a local semantic rewrite. Real name tokens are left intact.
        cleaned_search = re.sub(r"\s+ه$", "", cleaned_search).strip()

        if not cleaned_search:
            raise EmployeeAPIError("Employee search text is required.")

        def normalize_text(value: str) -> str:
            value = value.casefold()
            # Arabic name search should tolerate common spelling variants.
            value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
            value = value.replace("ـ", "")
            value = (
                value.replace("أ", "ا")
                .replace("إ", "ا")
                .replace("آ", "ا")
                .replace("ى", "ي")
                .replace("ة", "ه")
            )
            value = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", value)
            return " ".join(value.split())

        stop_words = {
            "بن", "بنت", "ابن", "bin", "bint", "al",
            "employee", "named", "called",
            "موظف", "موظفه", "موظفة", "الموظف", "باسم", "اسم",
        }

        query_tokens = [
            token
            for token in normalize_text(cleaned_search).split()
            if token not in stop_words and len(token) >= 2
        ]

        if not query_tokens:
            raise EmployeeAPIError("Employee search text is too broad.")

        prefer_arabic = any("\u0600" <= character <= "\u06ff" for character in cleaned_search)

        async def fetch_rows(api_search: str, *, directory_snapshot: bool = False) -> list[Any]:
            """Use the deployed, verified SearchEmployee request shape only.

            The current API returns the full employee directory for an empty search.
            We use that as the preferred source for local name matching. Named
            server-side search remains a fallback for compatibility.
            """
            api_search = re.sub(r"[؟?،,؛;:!…]+$", "", str(api_search or "").strip()).strip()
            batch_value = "5000" if directory_snapshot else ""

            data = await self._request(
                method="POST",
                endpoint="/api/SearchEmployee/searchEmployeesList",
                access_token=access_token,
                form_fields={"search": api_search, "batchSize": batch_value},
            )

            if isinstance(data, dict):
                rows: Any = (
                    data.get("employeeList")
                    or data.get("employees")
                    or data.get("items")
                    or data.get("results")
                    or data.get("data")
                    or []
                )
                if isinstance(rows, dict):
                    rows = (
                        rows.get("employeeList")
                        or rows.get("employees")
                        or rows.get("items")
                        or rows.get("results")
                        or []
                    )
            elif isinstance(data, list):
                rows = data
            else:
                rows = []

            rows = rows if isinstance(rows, list) else []
            rows = [
                self._unwrap_employee_record(row)
                for row in rows
                if isinstance(row, dict)
            ]
            rows = [row for row in rows if row]
            mode = "directory_snapshot" if directory_snapshot else "named_search"
            print(
                f"[SearchEmployee] mode={mode!r} search={api_search!r} "
                f"batchSize={batch_value!r} rows={len(rows)}"
            )
            return rows

        def merge_unique(destination: list[dict[str, Any]], incoming: list[Any], seen: set[str]) -> None:
            for employee in incoming:
                if not isinstance(employee, dict):
                    continue

                identity = str(
                    employee.get("userid")
                    or employee.get("username")
                    or employee.get("userCode")
                    or f"{employee.get('namea')}|{employee.get('namel')}"
                ).strip().casefold()

                if not identity or identity in seen:
                    continue

                verified_username = str(employee.get("userid") or employee.get("username") or "").strip()
                employee_code = str(employee.get("userCode") or employee.get("empCode") or "").strip()
                if verified_username and employee_code:
                    self._employee_code_by_username[verified_username.casefold()] = employee_code
                if verified_username:
                    verified_name = str(employee.get("namea") or employee.get("namel") or "").strip()
                    if verified_name:
                        self._employee_name_by_username[verified_username.casefold()] = verified_name

                seen.add(identity)
                destination.append(employee)

        def normalize_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            matched: list[dict[str, Any]] = []

            for employee in rows:
                name_ar = str(employee.get("namea") or "").strip()
                name_en = str(employee.get("namel") or "").strip()
                searchable_name = normalize_text(f"{name_ar} {name_en}")

                # Every requested token must be present in the employee name.
                if not all(token in searchable_name for token in query_tokens):
                    continue

                if prefer_arabic:
                    normalized = {
                        "name": name_ar or name_en,
                        "job_title": employee.get("jodesca") or employee.get("jodesce") or employee.get("hodesce"),
                        "department": employee.get("dpDesca") or employee.get("dpDescl"),
                        "username": employee.get("userid") or employee.get("username"),
                        "email": _directory_contact_value(employee, "email"),
                        "mobile": _directory_contact_value(employee, "mobile"),
                        "extension": _directory_contact_value(employee, "extension"),
                        "name_ar": name_ar or None,
                        "name_en": name_en or None,
                        "job_title_ar": employee.get("jodesca"),
                        "job_title_en": employee.get("jodesce") or employee.get("hodesce"),
                        "department_ar": employee.get("dpDesca"),
                        "department_en": employee.get("dpDescl"),
                        "image": employee.get("image"),
                        "manager_details": employee.get("managerDetails") if isinstance(employee.get("managerDetails"), (dict, list)) else None,
                        "emp_code": employee.get("userCode") or employee.get("empCode"),
                    }
                else:
                    normalized = {
                        "name": name_en or name_ar,
                        "job_title": employee.get("jodesce") or employee.get("hodesce") or employee.get("jodesca"),
                        "department": employee.get("dpDescl") or employee.get("dpDesca"),
                        "username": employee.get("userid") or employee.get("username"),
                        "email": _directory_contact_value(employee, "email"),
                        "mobile": _directory_contact_value(employee, "mobile"),
                        "extension": _directory_contact_value(employee, "extension"),
                        "name_ar": name_ar or None,
                        "name_en": name_en or None,
                        "job_title_ar": employee.get("jodesca"),
                        "job_title_en": employee.get("jodesce") or employee.get("hodesce"),
                        "department_ar": employee.get("dpDesca"),
                        "department_en": employee.get("dpDescl"),
                        "image": employee.get("image"),
                        "manager_details": employee.get("managerDetails") if isinstance(employee.get("managerDetails"), (dict, list)) else None,
                        "emp_code": employee.get("userCode") or employee.get("empCode"),
                    }

                # A directory result without a stable username cannot be used
                # later for verified employee details/email lookup.
                if not str(normalized.get("username") or "").strip():
                    continue

                # RCRC business rule: the verified directory username is the
                # local part of the corporate email.
                if normalized.get("email") in (None, ""):
                    normalized["email"] = _email_from_username(normalized.get("username"))

                print(
                    "[SearchEmployee Matched Fields] "
                    f"keys={sorted(str(key) for key in employee.keys())} "
                    f"email={bool(normalized.get('email'))} "
                    f"mobile={bool(normalized.get('mobile'))} "
                    f"extension={bool(normalized.get('extension'))}"
                )

                matched.append(normalized)

            # For a single-token Arabic name search, prefer verified rows whose
            # FIRST name token matches.
            if len(query_tokens) == 1:
                requested = query_tokens[0]
                first_name_matches = [
                    employee
                    for employee in matched
                    if (
                        normalize_text(str(employee.get("name") or "")).split()
                        and normalize_text(str(employee.get("name") or "")).split()[0] == requested
                    )
                ]
                if first_name_matches:
                    matched = first_name_matches

            matched.sort(
                key=lambda employee: (
                    0 if normalize_text(cleaned_search) in normalize_text(str(employee.get("name") or "")) else 1,
                    len(normalize_text(str(employee.get("name") or "")).split()),
                    normalize_text(str(employee.get("name") or "")),
                )
            )
            return matched

        all_rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Prefer the verified full-directory response from an empty search, then
        # perform matching locally. This makes employee lookup independent of
        # inconsistent server-side name-search behavior.
        try:
            directory_rows = await fetch_rows("", directory_snapshot=True)
        except (EmployeeAuthenticationError, EmployeeAuthorizationError):
            raise
        except EmployeeAPIError as exc:
            print(f"[SearchEmployee Directory Snapshot] failed: {exc}")
            directory_rows = []

        merge_unique(all_rows, directory_rows, seen)
        matched_employees = normalize_matches(all_rows)

        # Compatibility fallback: if the directory snapshot did not contain a
        # match, try the server-side named search and merge verified rows.
        if not matched_employees:
            full_rows = await fetch_rows(cleaned_search)
            merge_unique(all_rows, full_rows, seen)
            matched_employees = normalize_matches(all_rows)

        # If local verification still found no actual match, retry with individual
        # tokens (family name first, then first name), merge, and verify the
        # COMPLETE requested name again.
        if not matched_employees and len(query_tokens) > 1:
            fallback_queries: list[str] = []
            for token in (query_tokens[-1], query_tokens[0]):
                if token not in fallback_queries:
                    fallback_queries.append(token)

            for fallback_query in fallback_queries:
                fallback_rows = await fetch_rows(fallback_query)
                merge_unique(all_rows, fallback_rows, seen)
                matched_employees = normalize_matches(all_rows)
                if matched_employees:
                    break

        # Final fallback: use the SAME SearchEmployee endpoint only.
        # Do not call EmployeeList; it is not part of the deployed search flow.
        if not matched_employees:
            try:
                snapshot_data = await self._request(
                    method="POST",
                    endpoint="/api/SearchEmployee/searchEmployeesList",
                    access_token=access_token,
                    form_fields={"search": "", "batchSize": "5000"},
                )
                snapshot_rows = self._normalize_collection(
                    snapshot_data,
                    "employeeList", "employees", "items", "results", "data",
                )
                snapshot_rows = [
                    self._unwrap_employee_record(row)
                    for row in snapshot_rows
                    if isinstance(row, dict)
                ]
                snapshot_rows = [row for row in snapshot_rows if row]
                print(
                    "[SearchEmployee Full Snapshot] "
                    f"search='' batchSize='5000' rows={len(snapshot_rows)}"
                )
                matched_employees = normalize_matches(snapshot_rows)
            except (EmployeeAuthenticationError, EmployeeAuthorizationError):
                raise
            except EmployeeAPIError as exc:
                print(f"[SearchEmployee Full Snapshot] failed: {exc}")

        return {"search": cleaned_search, "employees": matched_employees[:10]}

    # ==================================================================
    # 9B. Find Employee by Job Title / Organisational Role
    # ==================================================================

    async def find_employees_by_role(self, *, access_token: str, role_query: str) -> dict[str, Any]:
        """
        Find employees by verified directory job-title / department fields.

        This method never asks the LLM to infer a person's name. It reads the
        directory API and filters verified rows locally. If the directory does
        not confirm a match, it returns no match instead of guessing.
        """
        cleaned_role = " ".join(role_query.strip().split())
        if not cleaned_role:
            raise EmployeeAPIError("Employee role text is required.")

        def normalize_text(value: str) -> str:
            value = value.casefold()
            value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
            value = value.replace("ـ", "")
            for source, target in (
                ("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"),
                ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي"),
            ):
                value = value.replace(source, target)
            value = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", value)
            return " ".join(value.split())

        stop_words = {
            "هو", "هي", "من", "مين", "الموظف", "الموظفه",
            "اللي", "الذي", "التي", "في", "على", "عن",
            "يشغل", "منصب", "المسؤول", "المسؤوله",
            "who", "is", "the", "employee", "holds", "role",
        }

        normalized_role = normalize_text(cleaned_role)
        role_tokens = [
            token for token in normalized_role.split()
            if token not in stop_words and len(token) >= 2
        ]
        if not role_tokens:
            raise EmployeeAPIError("Employee role text is too broad.")

        prefer_arabic = any("\u0600" <= character <= "\u06ff" for character in cleaned_role)

        def extract_rows(data: Any) -> list[Any]:
            if isinstance(data, list):
                return data
            if not isinstance(data, dict):
                return []
            for key in ("employeeList", "employees", "items", "results", "list", "Employees"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            nested = data.get("data")
            if isinstance(nested, (dict, list)):
                return extract_rows(nested)
            for value in data.values():
                if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                    return value
            return []

        def normalize_row(employee: dict[str, Any]) -> dict[str, Any] | None:
            name_ar = str(employee.get("namea") or employee.get("nameAr") or employee.get("name_ar") or "").strip()
            name_en = str(employee.get("namel") or employee.get("nameEn") or employee.get("name_en") or "").strip()
            generic_name = str(employee.get("name") or employee.get("employeeName") or "").strip()

            title_ar = str(employee.get("jodesca") or employee.get("jobTitleAr") or employee.get("job_title_ar") or "").strip()
            title_en = str(
                employee.get("jodesce") or employee.get("hodesce")
                or employee.get("jobTitleEn") or employee.get("job_title_en") or ""
            ).strip()
            generic_title = str(employee.get("job_title") or employee.get("jobTitle") or employee.get("position") or "").strip()

            department_ar = str(employee.get("dpDesca") or employee.get("departmentAr") or employee.get("department_ar") or "").strip()
            department_en = str(employee.get("dpDescl") or employee.get("departmentEn") or employee.get("department_en") or "").strip()
            generic_department = str(employee.get("department") or employee.get("departmentName") or "").strip()

            username = str(
                employee.get("userid") or employee.get("username")
                or employee.get("userCode") or employee.get("userName") or ""
            ).strip()

            if prefer_arabic:
                name = name_ar or generic_name or name_en
                title = title_ar or generic_title or title_en
                department = department_ar or generic_department or department_en
            else:
                name = name_en or generic_name or name_ar
                title = title_en or generic_title or title_ar
                department = department_en or generic_department or department_ar

            if not name or not title or not username:
                return None

            return {
                "name": name,
                "job_title": title,
                "department": department,
                "username": username,
                "image": employee.get("image"),
            }

        all_rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        page_data = await self._request(
            method="POST",
            endpoint="/api/SearchEmployee/searchEmployeesList",
            access_token=access_token,
            form_fields={"search": "", "batchSize": ""},
        )
        page_rows = extract_rows(page_data)
        page_rows = [
            self._unwrap_employee_record(row)
            for row in page_rows
            if isinstance(row, dict)
        ]
        page_rows = [row for row in page_rows if row]
        for raw_employee in page_rows:
            if not isinstance(raw_employee, dict):
                continue
            normalized = normalize_row(raw_employee)
            if normalized is None:
                continue
            identity = str(normalized.get("username") or "").casefold()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            all_rows.append(normalized)
        print(
            "[SearchEmployee Role Snapshot] "
            f"batchSize='5000' rows={len(page_rows)} unique={len(all_rows)}"
        )

        scored: list[tuple[float, int, dict[str, Any]]] = []

        for employee in all_rows:
            title = normalize_text(str(employee.get("job_title") or ""))
            department = normalize_text(str(employee.get("department") or ""))
            haystack = f"{title} {department}".strip()
            if not haystack:
                continue

            matched = sum(1 for token in role_tokens if token in haystack)
            ratio = matched / len(role_tokens)

            if normalized_role == title:
                ratio += 2.0
            elif normalized_role in title:
                ratio += 1.5
            elif normalized_role in haystack:
                ratio += 1.0

            minimum_matches = max(2, int(len(role_tokens) * 0.70 + 0.999))

            if matched >= minimum_matches and ratio >= 0.70:
                scored.append((ratio, matched, employee))

        scored.sort(
            key=lambda item: (item[0], item[1], -len(str(item[2].get("job_title") or ""))),
            reverse=True,
        )

        return {"role": cleaned_role, "employees": [item[2] for item in scored[:10]]}

    # ==================================================================
    # 10. Verified Directory Employee Details
    # ==================================================================

    async def get_employee_directory_details(
        self,
        *,
        username: str,
        access_token: str,
        emp_code: str = "",
    ) -> dict[str, Any]:
        """Return full details for one verified employee using empcode only.

        Correct flow:
            SearchEmployee -> userid + userCode
            userCode -> employeeFullDetails(form-data: empcode=userCode)

        The legacy username-based EmployeeDetails endpoint is intentionally not
        used for full employee information.
        """
        cleaned_username = str(username or "").strip()
        code = str(emp_code or "").strip()

        if not code and cleaned_username:
            code = str(
                self._employee_code_by_username.get(cleaned_username.casefold(), "")
                or ""
            ).strip()

        if not code:
            if not cleaned_username:
                raise EmployeeAPIError("Employee username or empcode is required.")
            code = await self.resolve_employee_code(
                username=cleaned_username,
                access_token=access_token,
            )

        payload = await self._full_details_by_code(
            emp_code=code,
            access_token=access_token,
        )
        payload = _require_dict(payload, "Employee full details response is invalid.")

        employee = payload.get("employeeDetails")
        if isinstance(employee, list):
            employee = next((row for row in employee if isinstance(row, dict)), None)
        if not isinstance(employee, dict):
            employee = self._extract_person_row(payload)
        employee = _require_dict(employee, "Employee details are missing.")

        def first_value(row: dict[str, Any], *keys: str) -> Any:
            lowered = {str(k).casefold(): v for k, v in row.items()}
            for key in keys:
                value = row.get(key)
                if value not in (None, "", "-"):
                    return value
                value = lowered.get(str(key).casefold())
                if value not in (None, "", "-"):
                    return value
            return None

        returned_code = str(
            first_value(
                employee,
                "userCode", "empCode", "empcode",
                "employeeCode", "employeeId", "employeeID",
            )
            or ""
        ).strip()
        if returned_code and returned_code != code:
            print(
                "[EmployeeFullDetails Code Guard] "
                f"requested={code!r} returned={returned_code!r}"
            )
            raise EmployeeAPIError("employeeFullDetails returned a different employee.")

        returned_username = str(
            first_value(employee, "userid", "username", "userName", "loginName")
            or cleaned_username
        ).strip()

        if returned_username:
            self.remember_employee_code(returned_username, returned_code or code)
            name_for_cache = str(
                first_value(
                    employee,
                    "namea", "nameAr", "arabicName",
                    "namel", "nameEn", "englishName", "name",
                )
                or ""
            ).strip()
            if name_for_cache:
                self._employee_name_by_username[returned_username.casefold()] = name_for_cache

        email = _directory_contact_value(employee, "email")
        mobile = _directory_contact_value(employee, "mobile")
        extension = _directory_contact_value(employee, "extension")
        if email in (None, "") and returned_username:
            email = _email_from_username(returned_username)

        result = {
            "username": returned_username or None,
            "emp_code": returned_code or code,
            "name_ar": first_value(employee, "namea", "nameAr", "arabicName", "name_ar"),
            "name_en": first_value(employee, "namel", "nameEn", "englishName", "name_en"),
            "job_title_ar": first_value(employee, "jodesca", "jobTitleAr", "jobTitleArabic", "job_title_ar"),
            "job_title_en": first_value(employee, "jodesce", "hodesce", "jobTitleEn", "jobTitleEnglish", "job_title_en"),
            "department_ar": first_value(employee, "dpDesca", "departmentAr", "departmentArabic", "department_ar"),
            "department_en": first_value(employee, "dpDescl", "departmentEn", "departmentEnglish", "department_en"),
            "sector_ar": first_value(employee, "seDesca", "sectorAr", "sector_ar"),
            "sector_en": first_value(employee, "seDescl", "sectorEn", "sector_en"),
            "email": email,
            "mobile": mobile,
            "extension": extension,
            "employee_type": first_value(employee, "empType", "employeeType", "employee_type"),
            "image": first_value(employee, "image", "imageUrl", "photo"),
        }

        print(
            "[Employee Details Resolved] "
            f"username={returned_username!r} empcode={code!r} "
            f"email={bool(email)} mobile={bool(mobile)} extension={bool(extension)}"
        )
        return {k: v for k, v in result.items() if v not in (None, "", "-")}

    # ==================================================================
    # 11. HR Letter Recipients
    # ==================================================================

    async def get_hr_letter_recipients(self, *, access_token: str) -> dict[str, Any]:
        data = await self._request(method="GET", endpoint="/api/hrLetterToWhom", access_token=access_token)

        if isinstance(data, list):
            recipients = data
        elif isinstance(data, dict):
            recipients = data.get("items") or data.get("recipients") or data.get("results") or [data]
        else:
            raise EmployeeAPIError("HR letter recipients response is invalid.")

        recipients = [row for row in recipients if isinstance(row, dict)]
        print(
            "[HR Letter Recipients] "
            f"rows={len(recipients)} first_keys={sorted(recipients[0].keys()) if recipients else []}"
        )
        return {"recipients": recipients}

    # ==================================================================
    # 12. Generic collection normalization helpers
    # ==================================================================

    @staticmethod
    def _normalize_collection(data: Any, *preferred_keys: str) -> list[dict[str, Any]]:
        """Return a defensive list of dict records without inventing fields."""
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        if isinstance(data, dict):
            keys = list(preferred_keys) + [
                "items", "results", "data", "list", "records",
                "employees", "employeeList", "vacations", "meetings",
                "news", "announcements", "policies", "benefits",
                "mazaya", "attendance",
            ]
            for key in keys:
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
                if isinstance(value, dict):
                    for nested_key in keys:
                        nested = value.get(nested_key)
                        if isinstance(nested, list):
                            return [item for item in nested if isinstance(item, dict)]

            # Preserve a single structured object as one record.
            if data:
                return [data]

        return []

    # ==================================================================
    # 13. Employee attendance history
    # ==================================================================

    async def get_employee_attendance(
        self, *, username: str, access_token: str, start_date: str, end_date: str,
    ) -> dict[str, Any]:
        request_args = {
            "method": "POST",
            "endpoint": "/api/EmployeeAttendance/getEmployeeAttendanceByUsername",
            "access_token": access_token,
            "form_fields": {
                "username": username,
                "startDate": start_date,
                "endDate": end_date,
            },
        }

        # This legacy endpoint can occasionally fail at the transport layer
        # while the rest of the Employee API remains reachable. Retry once
        # only for connection/timeout failures; never retry auth/403/HTTP errors.
        try:
            data = await self._request(**request_args)
        except EmployeeAPIError as exc:
            message = str(exc)
            if message not in {
                "Could not connect to the Employee API.",
                "Employee API request timed out.",
            }:
                raise
            await asyncio.sleep(0.25)
            data = await self._request(**request_args)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "records": self._normalize_collection(
                data, "attendance", "employeeAttendance", "attendanceList"
            ),
        }

    # ==================================================================
    # 14. Working hours
    # ==================================================================

    async def get_working_hours(
        self, *, username: str, access_token: str, start_date: str, end_date: str,
    ) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/WorkingHours/WorkingHours",
            access_token=access_token,
            form_fields={"username": username, "startDate": _api_us_date(start_date), "endDate": _api_us_date(end_date)},
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "records": self._normalize_collection(data, "workingHours", "hours", "workingHoursList"),
        }

    # ==================================================================
    # 15. Vacation history / employee vacation data
    # ==================================================================

    async def get_vacation_history(
        self,
        *,
        username: str,
        access_token: str,
        start_date: str,
        end_date: str,
        start_index: int = 0,
        batch_size: int = 200,
    ) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/VacationData/getvacationData",
            access_token=access_token,
            form_fields={
                "username": username,
                # The deployed VacationData service returns "Internal error
                # occurred." when numeric pagination values are forced. The
                # verified working Postman request sends both fields blank.
                "startIndex": "",
                # VacationData expects MM-DD-YYYY.
                "startDate": _api_us_date(start_date),
                "endDate": _api_us_date(end_date),
                "batchSize": "",
            },
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "start_index": start_index,
            "batch_size": batch_size,
            "records": self._normalize_collection(data, "vacations", "vacationData", "vacationList"),
        }

    # ==================================================================
    # 16. Vacation list (organization-visible list, subject to API auth)
    # ==================================================================

    async def get_vacations_list(
        self,
        *,
        access_token: str,
        start_date: str,
        end_date: str,
        start_index: int = 0,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/VacationsList",
            access_token=access_token,
            form_fields={
                "startDate": start_date,
                "endDate": end_date,
                "startIndex": start_index,
                "batchSize": batch_size,
            },
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "start_index": start_index,
            "batch_size": batch_size,
            "records": self._normalize_collection(data, "vacations", "vacationList"),
        }

    # ==================================================================
    # 17. Intersected vacations
    # ==================================================================

    async def get_intersected_vacations(
        self, *, username: str, access_token: str, start_date: str, end_date: str,
    ) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/intersectedVacations",
            access_token=access_token,
            form_fields={"username": username, "startDate": _api_us_date(start_date), "endDate": _api_us_date(end_date)},
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "records": self._normalize_collection(data, "vacations", "intersections", "intersectedVacations"),
        }

    # ==================================================================
    # 18. Reporting employees
    # ==================================================================
    # The old /api/ReportingEmployees/reportingEmployees endpoint was removed.
    # Direct reports now live inside the employeeFullDetails response next to
    # employeeDetails and managerDetails.
    # ==================================================================

    async def get_reporting_employees(
        self,
        *,
        username: str,
        access_token: str,
    ) -> dict[str, Any]:
        """Return direct reports using the username-only ReportingEmployees API."""
        cleaned_username = str(username or "").strip()
        if not cleaned_username:
            raise EmployeeAPIError("Employee username is required.")

        data = await self._request(
            method="POST",
            endpoint="/api/ReportingEmployees/reportingEmployees",
            access_token=access_token,
            form_fields={
                "username": cleaned_username,
            },
        )

        employees = self._normalize_collection(
            data,
            "employees",
            "reportingEmployees",
            "employeeList",
        )

        print(
            "[ReportingEmployees] "
            f"username={cleaned_username!r} rows={len(employees)}"
        )

        return {
            "employees": employees,
            "requested_username": cleaned_username,
            "source_identifier": cleaned_username,
            "source": "ReportingEmployees",
            "list_key": "reportingEmployees",
        }

    # ==================================================================
    # 19. Sub-employees attendance
    # ==================================================================

    async def get_subemployees_attendance(
        self, *, username: str, access_token: str, start_date: str, end_date: str,
    ) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/SubEmployeesAttendance/getSubEmployeesAttendance",
            access_token=access_token,
            form_fields={"username": username, "startDate": _api_us_date(start_date), "endDate": _api_us_date(end_date)},
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "records": self._normalize_collection(data, "attendance", "subEmployeesAttendance", "employees"),
        }

    # ==================================================================
    # 20. HR letter types
    # ==================================================================

    async def get_hr_letter_types(self, *, access_token: str) -> dict[str, Any]:
        data = await self._request(method="GET", endpoint="/api/hrLetterTypes", access_token=access_token)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = (
                data.get("items")
                or data.get("types")
                or data.get("letterTypes")
                or data.get("results")
                or [data]
            )
        else:
            items = []

        return {"types": items if isinstance(items, list) else []}

    # ==================================================================
    # 21. Policies and procedures
    # ==================================================================

    async def search_policies(
        self,
        *,
        access_token: str,
        search: str = "",
        start_index: int = 0,
        batch_size: int = 200,
    ) -> dict[str, Any]:
        """Search policies defensively across zero/one-based API variants.

        The deployed Appian-backed endpoint can reject page 0 with a generic
        application error. Retry page 1, then (only if necessary) fetch a broad
        verified policy list and filter locally by the user's query.
        """
        cleaned_search = " ".join(str(search or "").strip().split())
        try:
            requested_start = max(0, int(start_index or 0))
        except (TypeError, ValueError):
            requested_start = 0
        try:
            requested_batch = max(1, min(int(batch_size or 20), 200))
        except (TypeError, ValueError):
            requested_batch = 20

        attempts: list[tuple[int, int, str]] = [(requested_start, requested_batch, cleaned_search)]
        if requested_start == 0:
            attempts.append((1, requested_batch, cleaned_search))
        if requested_batch > 20:
            attempts.append((1 if requested_start == 0 else requested_start, 20, cleaned_search))

        def normalize_text(value: Any) -> str:
            text = str(value or "").casefold()
            text = re.sub(r"[\u064b-\u065f\u0670]", "", text).replace("ـ", "")
            for source, target in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
                text = text.replace(source, target)
            text = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", text)
            return " ".join(text.split())

        query_tokens = [
            token for token in normalize_text(cleaned_search).split()
            if token not in {"عن", "في", "من", "على", "الى", "إلى", "the", "of", "for", "and"}
            and len(token) >= 2
        ]

        def local_filter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not query_tokens:
                return rows
            matched: list[dict[str, Any]] = []
            for row in rows:
                hay = normalize_text(" ".join(
                    str(value or "") for key, value in row.items()
                    if isinstance(value, (str, int, float))
                ))
                if all(token in hay for token in query_tokens):
                    matched.append(row)
            return matched

        last_error: EmployeeAPIError | None = None
        seen: set[tuple[int, int, str]] = set()
        for upstream_start, upstream_batch, upstream_search in attempts:
            key = (upstream_start, upstream_batch, upstream_search)
            if key in seen:
                continue
            seen.add(key)
            try:
                data = await self._request(
                    method="POST",
                    endpoint="/api/PolicyAndProceduresList/policyAndProcedures",
                    access_token=access_token,
                    form_fields={
                        "search": upstream_search,
                        "batchSize": upstream_batch,
                        "startIndex": upstream_start,
                    },
                )
            except (EmployeeAuthenticationError, EmployeeAuthorizationError):
                raise
            except EmployeeAPIError as exc:
                last_error = exc
                continue

            items = self._normalize_collection(
                data, "policies", "policyAndProcedures", "policyList", "items", "results", "list",
            )
            items = [row for row in items if isinstance(row, dict)]
            if items:
                # The endpoint may return a broad page even when search is set.
                verified = local_filter(items)
                if verified or not cleaned_search:
                    print(
                        f"[Policy Search] startIndex={upstream_start} batchSize={upstream_batch} "
                        f"search={cleaned_search!r} rows={len(items)} verified={len(verified)}"
                    )
                    return {
                        "search": cleaned_search,
                        "start_index": upstream_start,
                        "batch_size": upstream_batch,
                        "items": verified if cleaned_search else items,
                    }

        # Last safe fallback: fetch the policy list without a search term and
        # filter it locally. Never return unrelated policies as a match.
        try:
            data = await self._request(
                method="POST",
                endpoint="/api/PolicyAndProceduresList/policyAndProcedures",
                access_token=access_token,
                form_fields={"search": "", "batchSize": 200, "startIndex": 1},
            )
            rows = self._normalize_collection(
                data, "policies", "policyAndProcedures", "policyList", "items", "results", "list",
            )
            rows = [row for row in rows if isinstance(row, dict)]
            verified = local_filter(rows)
            print(f"[Policy Search Fallback] startIndex=1 batchSize=200 rows={len(rows)} verified={len(verified)}")
            return {"search": cleaned_search, "start_index": 1, "batch_size": 200, "items": verified}
        except (EmployeeAuthenticationError, EmployeeAuthorizationError):
            raise
        except EmployeeAPIError:
            if last_error is not None:
                raise last_error
            raise

    async def get_policy_details(self, *, access_token: str, item_id: str) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/policyAndProcedure/getPolicyAndProcedureById",
            access_token=access_token,
            form_fields={"id": item_id},
        )
        if isinstance(data, dict):
            return {"item": data}
        items = self._normalize_collection(data)
        return {"item": items[0] if items else {}}

    # ==================================================================
    # 22. News
    # ==================================================================

    async def get_news_list(
        self,
        *,
        access_token: str,
        search: str = "",
        start_index: int = 0,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        cleaned_search = str(search or "").strip()
        try:
            requested_start = int(start_index)
        except (TypeError, ValueError):
            requested_start = 0
        try:
            requested_batch = int(batch_size)
        except (TypeError, ValueError):
            requested_batch = 100

        # The Appian-backed list endpoints use 1-based paging.
        upstream_start_index = max(1, requested_start)
        upstream_batch_size = max(1, min(requested_batch, 200))

        data = await self._request(
            method="POST",
            endpoint="/api/NewsList/newsList",
            access_token=access_token,
            form_fields={
                "startIndex": upstream_start_index,
                "batchSize": upstream_batch_size,
                "search": cleaned_search,
            },
        )
        return {
            "search": cleaned_search,
            "start_index": upstream_start_index,
            "batch_size": upstream_batch_size,
            "items": self._normalize_collection(data, "news", "newsList", "items"),
        }

    async def get_news_details(self, *, access_token: str, item_id: str) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/SelectedNewsDetails/newsByID",
            access_token=access_token,
            form_fields={"id": item_id},
        )
        if isinstance(data, dict):
            return {"item": data}
        items = self._normalize_collection(data)
        return {"item": items[0] if items else {}}

    # ==================================================================
    # 23. Announcements
    # ==================================================================

    async def get_announcements_list(
        self,
        *,
        access_token: str,
        search: str = "",
        start_index: int = 0,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        cleaned_search = str(search or "").strip()
        try:
            requested_start = int(start_index)
        except (TypeError, ValueError):
            requested_start = 0
        try:
            requested_batch = int(batch_size)
        except (TypeError, ValueError):
            requested_batch = 100

        # The Appian-backed list endpoints use 1-based paging.
        upstream_start_index = max(1, requested_start)
        upstream_batch_size = max(1, min(requested_batch, 200))

        data = await self._request(
            method="POST",
            endpoint="/api/AnnouncementList/announcementsLists",
            access_token=access_token,
            form_fields={
                "startIndex": upstream_start_index,
                "batchSize": upstream_batch_size,
                "search": cleaned_search,
            },
        )
        items = self._normalize_collection(
            data, "announcements", "announcementList", "items", "results", "list", "data",
        )
        items = [row for row in items if isinstance(row, dict)]
        print(f"[Announcements] rows={len(items)} first_keys={sorted(items[0].keys()) if items else []}")
        return {
            "search": cleaned_search,
            "start_index": upstream_start_index,
            "batch_size": upstream_batch_size,
            "items": items,
        }

    async def get_announcement_details(self, *, access_token: str, item_id: str) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/SelectedAnnouncementDetails/announcementById",
            access_token=access_token,
            form_fields={"id": item_id},
        )
        if isinstance(data, dict):
            return {"item": data}
        items = self._normalize_collection(data)
        return {"item": items[0] if items else {}}

    # ==================================================================
    # 24. Mazaya / benefits
    # ==================================================================

    async def get_mazaya_list(
        self,
        *,
        access_token: str,
        search: str = "",
        start_index: int = 0,
        batch_size: int = 200,
    ) -> dict[str, Any]:
        """Return Mazaya / employee benefits.

        The upstream Mazaya API uses 1-based pagination; normalize here.
        """
        cleaned_search = str(search or "").strip()
        try:
            requested_start = int(start_index)
        except (TypeError, ValueError):
            requested_start = 0
        try:
            requested_batch = int(batch_size)
        except (TypeError, ValueError):
            requested_batch = 10

        upstream_start_index = max(1, requested_start)
        upstream_batch_size = max(1, min(requested_batch, 200))

        data = await self._request(
            method="POST",
            endpoint="/api/MazayaListWithSearch/MazyaListWithSearch",
            access_token=access_token,
            form_fields={
                "search": cleaned_search,
                "batchSize": upstream_batch_size,
                "startIndex": upstream_start_index,
            },
        )
        return {
            "search": cleaned_search,
            "start_index": upstream_start_index,
            "batch_size": upstream_batch_size,
            "items": self._normalize_collection(data, "benefits", "mazaya", "mazayaList", "items"),
        }

    async def get_mazaya_details(self, *, access_token: str, item_id: str) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/MazayaDetails/mazayaDetailsById",
            access_token=access_token,
            form_fields={"Id": item_id},
        )
        items = self._normalize_collection(
            data, "item", "benefit", "benefits", "mazaya", "mazayaDetails", "details",
        )
        return {"item": items[0] if items else {}}

    # ==================================================================
    # 25. Full employee list (read-only directory list)
    # ==================================================================

    async def get_employee_list(
        self, *, access_token: str, start_index: int = 0, batch_size: int = 50000,
    ) -> dict[str, Any]:
        # The deployed directory source is SearchEmployee. Keep this compatibility
        # method, but back it with the correct endpoint rather than EmployeeList.
        result = await self.search_directory_rows(
            access_token=access_token,
            search="",
            batch_size=batch_size,
        )
        employees = result.get("employees") if isinstance(result, dict) else []
        return {
            "start_index": start_index,
            "batch_size": batch_size,
            "employees": [row for row in (employees or []) if isinstance(row, dict)],
        }

    # ==================================================================
    # 26. Rich meeting list
    # ==================================================================

    async def get_meeting_list(
        self, *, username: str, access_token: str, start_date: str, end_date: str,
    ) -> dict[str, Any]:
        data = await self._request(
            method="POST",
            endpoint="/api/MeetingList/myMeetings",
            access_token=access_token,
            form_fields={"username": username, "startDate": _api_us_date(start_date), "endDate": _api_us_date(end_date)},
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "records": self._normalize_collection(data, "meetings", "meetingList", "items"),
        }

    # ==================================================================
    # 27. Transaction APIs (must be called only after explicit confirmation)
    # ==================================================================

    async def create_vacation(
        self,
        *,
        username: str,
        access_token: str,
        start_date: str,
        end_date: str,
        vacation_type: str,
        require_exit_return: bool = False,
        require_tickets: bool = False,
        comment: str = "",
        attachment: str = "",
    ) -> dict[str, Any]:
        """Submit a vacation request. Orchestrator must confirm with user first."""

        data = await self._request(
            method="POST",
            endpoint="/api/CreateVacation/createVacation",
            access_token=access_token,
            form_fields={
                "username": username,
                "startDate": start_date,
                "endDate": end_date,
                "requireExitReturn": str(require_exit_return).lower(),
                "requireTickets": str(require_tickets).lower(),
                "comment": comment,
                "vacationType": vacation_type,
                "attachment": attachment or None,
            },
        )

        return {"result": data}

    async def create_hr_letter(
        self,
        *,
        username: str,
        access_token: str,
        letter_type: str,
        to_whom: str,
    ) -> dict[str, Any]:
        """Generate/submit an HR letter. Orchestrator must confirm first."""
        data = await self._request(
            method="POST",
            endpoint="/api/hrLetter",
            access_token=access_token,
            form_fields={
                "username": username,
                "letterType": letter_type,
                "toWhom": to_whom,
            },
        )
        return {"result": data}