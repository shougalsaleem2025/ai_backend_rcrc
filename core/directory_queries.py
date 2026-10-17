from __future__ import annotations







import hashlib



import re



import time



from difflib import SequenceMatcher



from typing import Any







from core.employee_api import EmployeeAPIClient, EmployeeAPIError











_AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")







# Fill this only when EmployeeList does not expose the parent/general department.



# Keys may be written in their normal Arabic form; lookup is normalized.



DEPARTMENT_TO_GENERAL_DEPARTMENT: dict[str, str] = {}







# Short/common aliases. Full English department names are matched dynamically



# from the bilingual fields returned by SearchEmployee.



COMMON_GENERAL_DEPARTMENT_ALIASES: dict[str, str] = {



    "it": "التقنية والرقمنة",



    "ict": "التقنية والرقمنة",



    "information technology": "التقنية والرقمنة",



    "technology and digitization": "التقنية والرقمنة",



    "technology and digitalization": "التقنية والرقمنة",



    "hr": "الموارد البشرية",



    "h r": "الموارد البشرية",



    "human resources": "الموارد البشرية",



    "human resource": "الموارد البشرية",
    "ia": "المراجعة الداخلية",
    "i a": "المراجعة الداخلية",
    "internal audit": "المراجعة الداخلية",
    "internal auditing": "المراجعة الداخلية",



    "finance": "المالية",



    "financial affairs": "المالية",



    "legal": "الشؤون القانونية",



    "legal affairs": "الشؤون القانونية",



    "procurement": "المشتريات",



    "purchasing": "المشتريات",



    "strategy": "الاستراتيجية والسياسات",



    "strategy and policies": "الاستراتيجية والسياسات",



    "pmo": "المشاريع",



    "p m o": "المشاريع",



    "project management office": "المشاريع",



    "cybersecurity": "الأمن السيبراني",



    "cyber security": "الأمن السيبراني",



}











def normalize_text(value: Any) -> str:



    text = str(value or "").strip().casefold()



    text = _AR_DIACRITICS_RE.sub("", text).replace("ـ", "")



    text = text.translate(



        str.maketrans({



            "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",



            "ؤ": "و", "ئ": "ي",



        })



    )



    text = text.replace("ادراه", "اداره").replace("ادراة", "اداره")



    text = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", text)



    return " ".join(text.split())











def _first(row: dict[str, Any], *keys: str) -> str:



    for key in keys:



        value = row.get(key)



        if value not in (None, ""):



            value = str(value).strip()



            if value:



                return value



    return ""











def _unwrap_employee_record(row: dict[str, Any]) -> dict[str, Any]:

    """Return the actual SearchEmployee employee row.



    Some SearchEmployee responses wrap each record as

    {"employeeDetails": {...}}. Directory analytics must operate on the

    inner employeeDetails object while preserving useful wrapper metadata.

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





def employee_username(row: dict[str, Any]) -> str:

    row = _unwrap_employee_record(row)

    return _first(row, "username", "userid", "userName", "loginName").casefold()











def employee_number(row: dict[str, Any]) -> str:

    row = _unwrap_employee_record(row)

    return _first(

        row,

        "userCode", "empCode", "empcode", "employeeCode",

        "employeeId", "employeeID", "employeeNo", "employeeNumber",

        "empNo", "empNumber", "personId", "personID", "personNumber",

    )











def employee_name(row: dict[str, Any]) -> str:



    return _first(



        row,



        "name_ar", "namea", "nameAr", "arabicName",



        "name", "employeeName",



        "name_en", "namel", "name1", "nameEn", "englishName",



    )











def employee_title(row: dict[str, Any]) -> str:



    return _first(



        row,



        "job_title_ar", "jodesca", "jobTitleAr", "jobTitleArabic", "positionAr",



        "job_title", "jobTitle", "position",



        "job_title_en", "jodesce", "jodesc1", "hodesce", "jobTitleEn",



        "jobTitleEnglish", "positionEn",



    )











def employee_department(row: dict[str, Any]) -> str:



    return _first(



        row,



        "department_ar", "dpDesca", "departmentAr", "departmentArabic", "deptAr",



        "department", "departmentName", "orgUnit", "org_unit", "unitName",



        "department_en", "dpDescl", "dpDesc1", "departmentEn", "departmentEnglish", "deptEn",



    )











def employee_name_ar(row: dict[str, Any]) -> str:



    return _first(row, "name_ar", "namea", "nameAr", "arabicName")











def employee_name_en(row: dict[str, Any]) -> str:



    return _first(row, "name_en", "namel", "name1", "nameEn", "englishName")











def employee_title_ar(row: dict[str, Any]) -> str:



    return _first(row, "job_title_ar", "jodesca", "jobTitleAr", "jobTitleArabic", "positionAr")











def employee_title_en(row: dict[str, Any]) -> str:



    return _first(row, "job_title_en", "jodesce", "jodesc1", "hodesce", "jobTitleEn", "jobTitleEnglish", "positionEn")











def employee_department_ar(row: dict[str, Any]) -> str:



    return _first(row, "department_ar", "dpDesca", "departmentAr", "departmentArabic", "deptAr")











def employee_department_en(row: dict[str, Any]) -> str:



    return _first(row, "department_en", "dpDescl", "dpDesc1", "departmentEn", "departmentEnglish", "deptEn")











def _title_variants(row: dict[str, Any]) -> list[str]:



    values = [employee_title_ar(row), employee_title_en(row), _first(row, "job_title", "jobTitle", "position")]



    out: list[str] = []



    for value in values:



        text = str(value or "").strip()



        if text and text.casefold() not in {x.casefold() for x in out}:



            out.append(text)



    return out











def _department_variants(row: dict[str, Any]) -> list[str]:



    values = [employee_department_ar(row), employee_department_en(row), _first(row, "department", "departmentName", "orgUnit", "unitName")]



    out: list[str] = []



    for value in values:



        text = str(value or "").strip()



        if text and text.casefold() not in {x.casefold() for x in out}:



            out.append(text)



    return out











def employee_email(row: dict[str, Any]) -> str:



    return _first(row, "email", "emailAddress", "mail", "email_address")











def employee_mobile(row: dict[str, Any]) -> str:



    return _first(



        row,



        "mobile", "mobileNo", "mobileNumber", "mobilePhone", "mobilephone",



        "cellPhone", "cellphone", "cellNo", "cellNumber", "phoneMobile",



        "mobile_number", "mobile_no",



    )











def _looks_like_general_department(value: Any) -> bool:



    raw = str(value or "").strip()



    normalized = normalize_text(raw)



    if not normalized:



        return False



    return (



        normalized.startswith("الاداره العامه")



        or normalized.startswith("general department")



    )











def _clean_department_display(value: Any) -> str:



    """Hide trailing internal org codes from user-facing department names.







    Example:



        الإدارة العامة لمجتمع التواصل الحضري U20



        -> الإدارة العامة لمجتمع التواصل الحضري







    The raw API value is still used during extraction/matching; only the



    canonical display/group label is cleaned.



    """



    raw = str(value or "").strip()



    if not raw:



        return ""



    cleaned = re.sub(



        r"\s*(?:[-–—|:/]\s*)?\(?[A-Za-z]{1,8}\d{1,8}\)?\s*$",



        "",



        raw,



    ).strip()



    return cleaned or raw











def _manual_general_department(department: str) -> str:



    dep_norm = normalize_text(department)



    if not dep_norm:



        return ""



    for source, destination in DEPARTMENT_TO_GENERAL_DEPARTMENT.items():



        if normalize_text(source) == dep_norm:



            return str(destination or "").strip()



    return ""











def extract_general_department(



    row: dict[str, Any],



) -> tuple[str, str]:



    """Return (verified general department, source field).







    Explicit general-department fields are authoritative even when their value



    does not literally contain the words "الإدارة العامة". Parent hierarchy



    fields are accepted only when their value clearly denotes a general



    department. Lower-level departments are never promoted by guesswork.



    """







    explicit_general_keys = (



        "general_department",



        "general_department_ar",



        "generalDepartment",



        "generalDepartmentAr",



        "generalDepartmentName",



        "generalDept",



        "generalDeptAr",



        "generalDeptName",



        "generalDepartmentDescription",



        "generalDepartmentDesc",



        "generalDepName",



        "generalDepAr",



    )







    for key in explicit_general_keys:



        value = row.get(key)



        if value not in (None, ""):



            candidate = str(value).strip()



            if candidate and not (



                normalize_text(candidate).startswith("الاداره العامه")



                and re.search(r"\b[A-Za-z]{1,8}\d+\b", candidate)



            ):



                return candidate, key







    parent_keys = (



        "parent_department",



        "parent_department_ar",



        "parentDepartment",



        "parentDepartmentAr",



        "parentDepartmentName",



        "parentDept",



        "parentDeptAr",



        "parentOrgUnit",



        "parentOrgUnitAr",



        "level1Department",



        "level2Department",



        "level3Department",



        "orgLevel1",



        "orgLevel2",



        "orgLevel3",



    )







    for key in parent_keys:



        value = row.get(key)



        if value in (None, ""):



            continue



        candidate = str(value).strip()



        if _looks_like_general_department(candidate):



            return candidate, key







    for key, raw in row.items():



        if not isinstance(raw, str) or not raw.strip():



            continue







        key_norm = re.sub(r"[^a-z0-9]", "", str(key).casefold())



        candidate = raw.strip()







        if (



            "general" in key_norm



            and ("department" in key_norm or "dept" in key_norm)



        ):



            if not (



                normalize_text(candidate).startswith("الاداره العامه")



                and re.search(r"\b[A-Za-z]{1,8}\d+\b", candidate)



            ):



                return candidate, str(key)







        if (



            "parent" in key_norm



            and any(x in key_norm for x in ("department", "dept", "org", "unit"))



            and _looks_like_general_department(candidate)



        ):



            return candidate, str(key)







    blocked = (



        "name", "title", "job", "position", "email", "mobile", "phone",



        "image", "userid", "username",



    )







    for key, raw in row.items():



        if not isinstance(raw, str) or not raw.strip():



            continue







        key_norm = re.sub(r"[^a-z0-9]", "", str(key).casefold())







        if any(item in key_norm for item in blocked):



            continue







        if _looks_like_general_department(raw):



            return raw.strip(), str(key)







    department = employee_department(row)







    if _looks_like_general_department(department):



        return department, "department"







    mapped = _manual_general_department(department)







    if mapped:



        return mapped, "manual_mapping"







    return "", ""







def general_department(row: dict[str, Any]) -> str:



    value, _ = extract_general_department(row)



    return _clean_department_display(value)











def employee_identity(row: dict[str, Any]) -> str:



    username = employee_username(row)



    if username:



        return "username:" + username







    number = employee_number(row)



    if number:



        return "employee:" + normalize_text(number)







    email = employee_email(row)



    if email:



        return "email:" + email.casefold()







    name = normalize_text(employee_name(row))



    if name:



        return "composite:" + "|".join(



            [



                name,



                normalize_text(employee_title(row)),



                normalize_text(employee_department(row)),



                normalize_text(employee_mobile(row)),



            ]



        )



    return ""











def map_employee(row: dict[str, Any]) -> dict[str, Any]:



    if not isinstance(row, dict):

        return {}



    row = _unwrap_employee_record(row)

    if not row:

        return {}



    out = dict(row)



    username = employee_username(row)



    name_ar = employee_name_ar(row)



    name_en = employee_name_en(row)



    name = employee_name(row)



    title_ar = employee_title_ar(row)



    title_en = employee_title_en(row)



    title = employee_title(row)



    department_ar = employee_department_ar(row)



    department_en = employee_department_en(row)



    department = employee_department(row)



    gdep, gdep_source = extract_general_department(row)



    email = employee_email(row)



    mobile = employee_mobile(row)







    if username:

        out["username"] = username



    emp_code = employee_number(row)

    if emp_code:

        out["emp_code"] = emp_code



    if name_ar:



        out["name_ar"] = name_ar



    if name_en:



        out["name_en"] = name_en



    if name:



        out["name"] = name



    if title_ar:



        out["job_title_ar"] = title_ar



    if title_en:



        out["job_title_en"] = title_en



    if title:



        out["job_title"] = title



    if department_ar:



        out["department_ar"] = department_ar



    if department_en:



        out["department_en"] = department_en



    if department:



        out["department"] = department



    if gdep:



        out["general_department"] = gdep



        out["general_department_source"] = gdep_source



    if email:



        out["email"] = email



    if mobile:



        out["mobile"] = mobile







    return out















def _department_alias_values(row: dict[str, Any]) -> set[str]:



    """Collect Arabic/English organizational labels exposed by SearchEmployee."""



    keys = (



        "general_department", "general_department_ar", "general_department_en",



        "generalDepartment", "generalDepartmentAr", "generalDepartmentEn",



        "generalDepartmentName", "generalDepartmentNameEn",



        "department", "department_ar", "department_en",



        "departmentAr", "departmentEn", "departmentArabic", "departmentEnglish",



        "dpDesca", "dpDescl", "dpDesc1", "deptAr", "deptEn",



        "orgUnit", "org_unit", "unitName",



    )



    values: set[str] = set()



    for key in keys:



        value = row.get(key)



        if value not in (None, ""):



            text = str(value).strip()



            if text:



                values.add(text)



    return values















def _actual_department_alias_values(row: dict[str, Any]) -> set[str]:



    keys = (



        "department", "department_ar", "department_en",



        "departmentAr", "departmentEn", "departmentArabic", "departmentEnglish",



        "dpDesca", "dpDescl", "dpDesc1", "deptAr", "deptEn",



        "orgUnit", "org_unit", "unitName",



    )



    values: set[str] = set()



    for key in keys:



        value = row.get(key)



        if value not in (None, ""):



            text = _clean_department_display(value)



            if text:



                values.add(text)



    return values











def _unit_core(value: Any) -> str:



    text = normalize_text(value)



    # Remove only leading organisational level words for a secondary exact-core



    # comparison. The full exact label is still tried first, so different levels



    # are not merged unless the remaining name is unambiguous.



    text = re.sub(



        r"^(?:الاداره\s+العامه|اداره\s+عامه|الاداره|اداره|القسم|قسم|القطاع|قطاع|المكتب|مكتب|الوحده|وحده)\s*",



        "",



        text,



    ).strip()



    if text.startswith("لل") and len(text) > 2:



        text = "ال" + text[2:]



    elif text.startswith("ل") and len(text) > 1:



        text = text[1:]



    return text.strip()











def _query_alias_candidates(query: str) -> set[str]:



    q = normalize_text(query)



    candidates = {q} if q else set()



    mapped = COMMON_GENERAL_DEPARTMENT_ALIASES.get(q)



    if mapped:



        candidates.add(normalize_text(mapped))



    # Also accept aliases when the query includes surrounding words.



    for alias, target in COMMON_GENERAL_DEPARTMENT_ALIASES.items():



        alias_norm = normalize_text(alias)



        if alias_norm and re.search(rf"(?:^|\s){re.escape(alias_norm)}(?:$|\s)", q):



            candidates.add(normalize_text(target))



    return {value for value in candidates if value}















def _general_department_core(value: Any) -> str:



    text = normalize_text(value)



    text = re.sub(r"^الاداره\s+العامه\s*", "", text).strip()



    # Arabic official names frequently use forms such as



    # "الإدارة العامة للتقنية...". After stripping the prefix this becomes



    # "للتقنية...". Normalize "للـ" back to the definite article "الـ" so



    # aliases such as IT -> "التقنية والرقمنة" match exactly.



    if text.startswith("لل") and len(text) > 2:



        text = "ال" + text[2:]



    elif text.startswith("ل") and len(text) > 1:



        text = text[1:]



    return text.strip()











# Existing orchestrator imports this name.



bounded_employee = map_employee











class DirectoryQueryService:



    DIRECTORY_CACHE_TTL_SECONDS = 300.0



    DIRECTORY_MAX_PAGES = 300







    def __init__(self, api: EmployeeAPIClient) -> None:



        self.api = api



        self._employee_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}







    @staticmethod



    def _token_cache_key(access_token: str) -> str:



        # Never keep/log the access token itself as a cache key.



        return hashlib.sha256(str(access_token or "").encode("utf-8")).hexdigest()[:24]







    @staticmethod



    def _merge_unique_rows(



        unique: dict[str, dict[str, Any]],



        rows: list[dict[str, Any]],



        *,



        anonymous_counter: list[int],



    ) -> int:



        added = 0



        for raw in rows:



            employee = map_employee(raw)



            identity = employee_identity(employee)



            if not identity:



                anonymous_counter[0] += 1



                identity = f"row:{anonymous_counter[0]}"







            if identity not in unique:



                unique[identity] = employee



                added += 1



                continue







            current = unique[identity]



            for key, value in employee.items():



                if current.get(key) in (None, "") and value not in (None, ""):



                    current[key] = value



        return added







    async def _search_snapshot_call(



        self,



        *,



        access_token: str,



        batch_size: str | int,



    ) -> list[dict[str, Any]]:



        result = await self.api.search_directory_rows(



            access_token=access_token,



            search="",



            batch_size=batch_size,



        )



        raw_rows = result.get("employees") if isinstance(result, dict) else []



        rows = [row for row in (raw_rows or []) if isinstance(row, dict)]



        print(



            f"[Search Directory Snapshot] batchSize={str(batch_size)!r} "



            f"returned={len(rows)}"



        )



        return rows







    async def all_employees(self, *, access_token: str) -> list[dict[str, Any]]:



        """Load the directory from SearchEmployee ONLY.







        EmployeeList is retired and deliberately not used. We first call the deployed



        SearchEmployee contract with empty batchSize (the form-data shape that



        worked in this environment), then retry with 5000 and keep the larger



        verified result. The snapshot is deduplicated by userid/employee identity.



        """



        cache_key = self._token_cache_key(access_token)



        cached = self._employee_cache.get(cache_key)



        now = time.monotonic()



        if cached is not None and (now - cached[0]) < self.DIRECTORY_CACHE_TTL_SECONDS:



            employees = cached[1]



            print(f"[Directory Cache] source=search_employee unique_employees={len(employees)}")



            return [dict(row) for row in employees]







        # The deployed SearchEmployee contract now returns the full directory when
        # both search and batchSize are empty. Use that verified shape first and
        # avoid a second large request when it already proves a full snapshot.
        attempts: list[tuple[str, list[dict[str, Any]]]] = []

        try:
            rows = await self._search_snapshot_call(
                access_token=access_token,
                batch_size="",
            )
        except EmployeeAPIError as exc:
            print(f"[Search Directory Snapshot Error] batchSize='': {exc}")
            rows = []
        attempts.append(("", rows))

        # Keep the old 5000 variant only as a compatibility fallback for an
        # environment that still returns the historical 20-row default.
        if len(rows) <= 20:
            try:
                fallback_rows = await self._search_snapshot_call(
                    access_token=access_token,
                    batch_size=5000,
                )
            except EmployeeAPIError as exc:
                print(f"[Search Directory Snapshot Error] batchSize=5000: {exc}")
                fallback_rows = []
            attempts.append(("5000", fallback_rows))

        # Pick the largest verified response, then dedupe. No EmployeeList fallback.
        chosen_batch, chosen_rows = max(attempts, key=lambda item: len(item[1]))



        unique: dict[str, dict[str, Any]] = {}



        anonymous_counter = [0]



        self._merge_unique_rows(



            unique,



            chosen_rows,



            anonymous_counter=anonymous_counter,



        )



        employees = list(unique.values())







        print(



            "[Directory API] source=search_employee "



            f"chosen_batch={chosen_batch!r} raw_rows={len(chosen_rows)} "



            f"unique_employees={len(employees)}"



        )



        if chosen_rows:



            print("[Directory API Fields]", sorted(str(k) for k in chosen_rows[0].keys()))







        if not employees:



            raise EmployeeAPIError(



                "SearchEmployee returned no directory rows, so organization-wide "



                "counts and role lookup cannot be verified."



            )







        # Accuracy guard: exactly 20 has repeatedly been the server default cap.



        # Do not present it as the organization-wide total unless SearchEmployee



        # proves a larger snapshot.



        if len(employees) <= 20:



            raise EmployeeAPIError(



                "SearchEmployee is still returning only "



                f"{len(employees)} rows for an empty directory search. "



                "Only SearchEmployee was used. A complete organization-wide "



                "snapshot could not be verified from SearchEmployee."



            )







        self._employee_cache[cache_key] = (now, [dict(row) for row in employees])



        return [dict(row) for row in employees]







    async def employee_count(self, *, access_token: str) -> dict[str, Any]:



        employees = await self.all_employees(access_token=access_token)



        return {"count": len(employees)}







    @staticmethod



    def _employee_gender(row: dict[str, Any]) -> str:



        """Return female/male only from an explicit SearchEmployee field.







        Gender is never inferred from a person's Arabic/English name, title, or



        salutation. Unknown/numeric-only values stay unknown unless the API also



        supplies a textual gender description.



        """



        keys = (



            "gender", "gender_ar", "gender_en", "genderAr", "genderEn",



            "genderName", "genderNameAr", "genderNameEn", "genderDescription",



            "sex", "sex_ar", "sex_en", "sexAr", "sexEn", "sexDescription",



            "employeeGender", "employeeGenderAr", "employeeGenderEn",



            "genderDesc", "genderDesca", "genderDesce",



        )



        values = [normalize_text(row.get(key)) for key in keys if row.get(key) not in (None, "")]



        for value in values:



            tokens = set(value.split())



            if value in {"انثى", "انثي", "اناث", "female", "woman", "women", "f"} or tokens & {"انثى", "انثي", "اناث", "female", "woman", "women"}:



                return "female"



            if value in {"ذكر", "ذكور", "male", "man", "men", "m"} or tokens & {"ذكر", "ذكور", "male", "man", "men"}:



                return "male"



        return ""







    async def employee_gender_counts(self, *, access_token: str) -> dict[str, Any]:



        employees = await self.all_employees(access_token=access_token)



        female = male = unknown = 0



        for employee in employees:



            gender = self._employee_gender(employee)



            if gender == "female":



                female += 1



            elif gender == "male":



                male += 1



            else:



                unknown += 1



        return {



            "female": female,



            "male": male,



            "unknown": unknown,



            "total": len(employees),



            "has_verified_gender": (female + male) > 0,



        }







    @staticmethod



    def _group_general_departments(



        employees: list[dict[str, Any]],



    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:



        groups: dict[str, dict[str, Any]] = {}



        unmapped: list[dict[str, Any]] = []







        for raw in employees:



            employee = map_employee(raw)



            gdep = general_department(employee)



            if not gdep or normalize_text(gdep) in {"dpdesca", "dpdescl", "department", "departmentname"}:



                unmapped.append(employee)



                continue







            key = normalize_text(gdep)



            if not key:



                unmapped.append(employee)



                continue







            bucket = groups.setdefault(key, {"name": gdep, "employees": {}, "aliases": set()})



            bucket["aliases"].add(gdep)



            for alias_value in _department_alias_values(employee):



                bucket["aliases"].add(alias_value)



            identity = employee_identity(employee)



            if identity:



                bucket["employees"][identity] = employee







        return groups, unmapped







    async def general_department_counts(self, *, access_token: str) -> dict[str, Any]:



        employees = await self.all_employees(access_token=access_token)



        groups, unmapped = self._group_general_departments(employees)







        result: list[dict[str, Any]] = []



        for bucket in groups.values():



            members = list(bucket["employees"].values())



            members.sort(key=lambda row: normalize_text(employee_name(row)))



            result.append(



                {



                    "general_department": bucket["name"],



                    "count": len(members),



                    "employees": members,



                }



            )







        result.sort(key=lambda row: (-int(row["count"]), normalize_text(row["general_department"])))



        unmapped_departments = sorted(



            {employee_department(row) for row in unmapped if employee_department(row)},



            key=normalize_text,



        )







        print(



            "[General Department Mapping] "



            f"mapped_groups={len(result)} "



            f"mapped_employees={sum(int(x['count']) for x in result)} "



            f"unmapped_employees={len(unmapped)}"



        )



        if unmapped_departments:



            print("[General Department Unmapped]", unmapped_departments[:50])







        return {



            "departments": result,



            "total_unique_employees": len(employees),



            "unmapped_employees": len(unmapped),



            "unmapped_departments": unmapped_departments,



        }







    @staticmethod



    def _match_department(



        groups: dict[str, dict[str, Any]],



        query: str,



    ) -> tuple[str, dict[str, Any]] | None:



        query_candidates = _query_alias_candidates(query)



        if not query_candidates:



            return None







        # Canonical-core exact match FIRST. This is important for short aliases



        # such as IT -> التقنية والرقمنة while the API stores the official name



        # as الإدارة العامة للتقنية والرقمنة. It also avoids fuzzy ambiguity



        # when multiple organizational labels share some of the same words.



        candidate_cores = {_general_department_core(value) for value in query_candidates}



        candidate_cores = {value for value in candidate_cores if value}



        core_matches: list[tuple[str, dict[str, Any]]] = []



        for key, bucket in groups.items():



            aliases = {str(bucket.get("name") or "")}



            aliases.update(str(value or "") for value in (bucket.get("aliases") or set()))



            alias_cores = {_general_department_core(value) for value in aliases}



            alias_cores = {value for value in alias_cores if value}



            if candidate_cores & alias_cores:



                core_matches.append((key, bucket))



        if len(core_matches) == 1:



            return core_matches[0]







        # Exact match against Arabic canonical name OR any bilingual API alias.



        exact_matches: list[tuple[str, dict[str, Any]]] = []



        for key, bucket in groups.items():



            aliases = {normalize_text(bucket.get("name"))}



            aliases.update(normalize_text(value) for value in (bucket.get("aliases") or set()))



            aliases = {value for value in aliases if value}



            if any(candidate in aliases for candidate in query_candidates):



                exact_matches.append((key, bucket))



        if len(exact_matches) == 1:



            return exact_matches[0]







        # Containment is useful for queries such as "HR" -> "Human Resources



        # General Department" or "التقنية والرقمنة" -> full Arabic label.



        contained: list[tuple[str, dict[str, Any]]] = []



        for key, bucket in groups.items():



            aliases = {normalize_text(bucket.get("name")), key}



            aliases.update(normalize_text(value) for value in (bucket.get("aliases") or set()))



            aliases = {value for value in aliases if value}



            if any(



                candidate in alias or alias in candidate



                for candidate in query_candidates



                for alias in aliases



            ):



                contained.append((key, bucket))



        # dedupe by key



        unique_contained = {key: bucket for key, bucket in contained}



        if len(unique_contained) == 1:



            key, bucket = next(iter(unique_contained.items()))



            return key, bucket







        scored: list[tuple[float, str, dict[str, Any]]] = []



        for key, bucket in groups.items():



            aliases = {normalize_text(bucket.get("name")), key}



            aliases.update(normalize_text(value) for value in (bucket.get("aliases") or set()))



            aliases = {value for value in aliases if value}



            best_score = 0.0



            for candidate in query_candidates:



                candidate_tokens = set(candidate.split())



                for alias in aliases:



                    alias_tokens = set(alias.split())



                    overlap = len(candidate_tokens & alias_tokens)



                    coverage = overlap / max(1, len(candidate_tokens))



                    reverse = overlap / max(1, len(alias_tokens))



                    similarity = SequenceMatcher(None, candidate, alias).ratio()



                    score = 0.50 * coverage + 0.25 * reverse + 0.25 * similarity



                    best_score = max(best_score, score)



            scored.append((best_score, key, bucket))







        scored.sort(key=lambda item: item[0], reverse=True)



        if not scored or scored[0][0] < 0.72:



            return None



        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.07:



            return None



        _, key, bucket = scored[0]



        return key, bucket







    @staticmethod



    def _group_organizational_units(



        employees: list[dict[str, Any]],



    ) -> dict[str, dict[str, Any]]:



        """Group every verified SearchEmployee department, not only general depts."""



        groups: dict[str, dict[str, Any]] = {}







        def add(unit_name: str, employee: dict[str, Any], unit_type: str, aliases: set[str]) -> None:



            display = _clean_department_display(unit_name)



            key = normalize_text(display)



            identity = employee_identity(employee)



            if not key or not identity:



                return



            bucket = groups.setdefault(



                key,



                {"name": display, "unit_type": unit_type, "employees": {}, "aliases": set()},



            )



            # Prefer the more specific actual department label when the same



            # normalized name was observed through multiple fields.



            if unit_type == "department":



                bucket["unit_type"] = "department"



                bucket["name"] = display



            bucket["aliases"].add(display)



            bucket["aliases"].update(x for x in aliases if x)



            bucket["employees"][identity] = employee







        for raw in employees:



            employee = map_employee(raw)



            actual = _clean_department_display(employee_department(employee))



            actual_aliases = _actual_department_alias_values(employee)



            if actual:



                add(actual, employee, "department", actual_aliases)







            gdep = general_department(employee)



            if gdep:



                # General-department aliases may include bilingual API values.



                general_aliases = {



                    value for value in _department_alias_values(employee)



                    if _looks_like_general_department(value)



                    or normalize_text(value) == normalize_text(gdep)



                }



                add(gdep, employee, "general_department", general_aliases)







        return groups







    @staticmethod



    def _match_organizational_unit(



        groups: dict[str, dict[str, Any]],



        query: str,



    ) -> tuple[str, dict[str, Any]] | None:



        candidates = _query_alias_candidates(query)



        if not candidates:



            return None







        # 1) exact verified label / bilingual alias.



        exact: dict[str, dict[str, Any]] = {}



        for key, bucket in groups.items():



            aliases = {normalize_text(bucket.get("name")), key}



            aliases.update(normalize_text(v) for v in (bucket.get("aliases") or set()))



            aliases = {v for v in aliases if v}



            if candidates & aliases:



                exact[key] = bucket



        if len(exact) == 1:



            key, bucket = next(iter(exact.items()))



            return key, bucket







        # 2) exact core ("المراجعة الداخلية" == "إدارة المراجعة الداخلية").



        candidate_cores = {_unit_core(v) for v in candidates if _unit_core(v)}



        core_matches: dict[str, dict[str, Any]] = {}



        for key, bucket in groups.items():



            aliases = {str(bucket.get("name") or "")}



            aliases.update(str(v or "") for v in (bucket.get("aliases") or set()))



            alias_cores = {_unit_core(v) for v in aliases if _unit_core(v)}



            if candidate_cores & alias_cores:



                core_matches[key] = bucket



        if len(core_matches) == 1:



            key, bucket = next(iter(core_matches.items()))



            return key, bucket



        if len(core_matches) > 1 and candidate_cores:



            matched_cores = {



                _unit_core(bucket.get("name"))



                for bucket in core_matches.values()



                if _unit_core(bucket.get("name"))



            }



            # SearchEmployee can expose the same logical unit at more than one



            # hierarchy label (e.g. إدارة/قسم الأمن السيبراني). If all exact-core



            # matches collapse to the same semantic unit, union the verified



            # employees instead of declaring a false ambiguity.



            if len(matched_cores) == 1 and next(iter(matched_cores)) in candidate_cores:



                merged_employees: dict[str, dict[str, Any]] = {}



                merged_aliases: set[str] = set()



                preferred_name = ""



                for bucket in core_matches.values():



                    if not preferred_name or str(bucket.get("unit_type") or "") == "department":



                        preferred_name = str(bucket.get("name") or preferred_name)



                    merged_aliases.update(str(v) for v in (bucket.get("aliases") or set()) if v)



                    for ident, row in (bucket.get("employees") or {}).items():



                        if ident:



                            merged_employees[str(ident)] = row



                synthetic_key = normalize_text(preferred_name or query)



                return synthetic_key, {



                    "name": preferred_name or query,



                    "unit_type": "department",



                    "employees": merged_employees,



                    "aliases": merged_aliases,



                }







        # 3) unique containment. Compare both full labels and normalized unit



        # cores so a short alias such as HR -> "الموارد البشرية" can match the



        # verified label "الإدارة العامة للموارد البشرية والشؤون الإدارية".



        contained: dict[str, dict[str, Any]] = {}



        candidate_cores_for_containment = {_unit_core(v) for v in candidates if _unit_core(v)}



        for key, bucket in groups.items():



            aliases = {normalize_text(bucket.get("name")), key}



            aliases.update(normalize_text(v) for v in (bucket.get("aliases") or set()))



            aliases = {v for v in aliases if v}



            alias_cores_for_containment = {_unit_core(v) for v in aliases if _unit_core(v)}



            full_match = any(c in a or a in c for c in candidates for a in aliases)



            core_match = any(



                c in a or a in c



                for c in candidate_cores_for_containment



                for a in alias_cores_for_containment



            )



            if full_match or core_match:



                contained[key] = bucket



        if len(contained) == 1:



            key, bucket = next(iter(contained.items()))



            return key, bucket







        # 4) conservative fuzzy fallback only when clearly ahead.



        scored: list[tuple[float, str, dict[str, Any]]] = []



        for key, bucket in groups.items():



            aliases = {normalize_text(bucket.get("name")), key}



            aliases.update(normalize_text(v) for v in (bucket.get("aliases") or set()))



            aliases = {v for v in aliases if v}



            best = 0.0



            for candidate in candidates:



                ct = set(candidate.split())



                for alias in aliases:



                    at = set(alias.split())



                    overlap = len(ct & at)



                    coverage = overlap / max(1, len(ct))



                    reverse = overlap / max(1, len(at))



                    similarity = SequenceMatcher(None, candidate, alias).ratio()



                    best = max(best, 0.55 * coverage + 0.20 * reverse + 0.25 * similarity)



            scored.append((best, key, bucket))



        scored.sort(key=lambda x: x[0], reverse=True)



        if not scored or scored[0][0] < 0.66:



            return None



        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.10:



            return None



        _, key, bucket = scored[0]



        return key, bucket







    async def general_department_members(



        self,



        *,



        access_token: str,



        department_query: str,



    ) -> dict[str, Any]:



        """Return members of any verified department/unit from SearchEmployee.







        The public method name is kept for compatibility with the orchestrator,



        but the implementation now supports lower-level departments as well as



        general departments.



        """



        employees = await self.all_employees(access_token=access_token)



        groups = self._group_organizational_units(employees)



        match = self._match_organizational_unit(groups, department_query)







        if match is None:



            return {



                "matched": False,



                "query": department_query,



                "employees": [],



                "available_departments": sorted(



                    {str(bucket.get("name") or "") for bucket in groups.values() if bucket.get("name")},



                    key=normalize_text,



                ),



            }







        key, bucket = match



        members = list(bucket["employees"].values())



        members.sort(key=lambda row: normalize_text(employee_name(row)))



        name = str(bucket.get("name") or department_query).strip()



        unit_type = str(bucket.get("unit_type") or "department")



        return {



            "matched": True,



            # Compatibility keys used by the current orchestrator.



            "general_department_key": key,



            "general_department": name,



            "department_key": key,



            "department": name,



            "unit_type": unit_type,



            "count": len(members),



            "employees": members,



        }







    async def organizational_unit_summary(self, *, access_token: str) -> dict[str, Any]:



        """Return every verified organizational department/unit with member count.







        This uses the same complete SearchEmployee snapshot as employee counts and



        never fabricates a hierarchy that the directory did not expose.



        """



        employees = await self.all_employees(access_token=access_token)



        groups = self._group_organizational_units(employees)



        units: list[dict[str, Any]] = []



        for key, bucket in groups.items():



            name = str(bucket.get("name") or "").strip()



            if not name:



                continue



            members = bucket.get("employees") if isinstance(bucket.get("employees"), dict) else {}



            units.append({



                "name": name,



                "unit_type": str(bucket.get("unit_type") or "department"),



                "count": len(members),



                "department_key": key,



            })



        units.sort(key=lambda row: normalize_text(row.get("name")))



        return {



            "count": len(units),



            "units": units,



            "total_unique_employees": len(employees),



        }







    async def find_role_holders(



        self,



        *,



        access_token: str,



        role_query: str,



    ) -> dict[str, Any]:



        """Find a verified employee who holds an organisational role.







        Strategy:



        1) Use SearchEmployee as the directory source.



        2) Extract the requested general department separately from the role title.



        3) Match department first using dpDesca/general_department.



        4) Match the role title inside that department.



        5) Never invent a holder. If several verified rows are equally plausible,



           return them as ambiguous candidates.



        """







        query = normalize_text(role_query)







        if not query:



            return {



                "role": role_query,



                "employees": [],



                "best_match": {},



                "ambiguous": False,



            }







        # SearchEmployee is authoritative for role lookup too. First ask it



        # directly with the role text, then verify title/department fields locally.



        direct_result = await self.api.search_directory_rows(



            access_token=access_token,



            search=role_query,



            batch_size="",



        )



        direct_rows = [



            map_employee(row)



            for row in (direct_result.get("employees") or [])



            if isinstance(row, dict)



        ]



        print(



            "[Role SearchEmployee] "



            f"query={role_query!r} raw_matches={len(direct_rows)}"



        )







        # If SearchEmployee found rows, prefer verified exact title matches.



        exact_direct = [



            employee for employee in direct_rows



            if any(normalize_text(title) == query for title in _title_variants(employee))



        ]



        if len(exact_direct) == 1:



            return {



                "role": role_query,



                "employees": exact_direct,



                "best_match": exact_direct[0],



                "ambiguous": False,



            }



        if len(exact_direct) > 1:



            return {



                "role": role_query,



                "employees": exact_direct[:10],



                "best_match": {},



                "ambiguous": True,



            }







        # SearchEmployee sometimes returns the server-default 20 unrelated rows



        # for a role-text query. If no exact title was proven above, always merge



        # those rows with the complete verified directory snapshot before local



        # role scoring. This prevents a generic 20-row response from hiding the



        # real holder of a valid role.



        full_snapshot = await self.all_employees(access_token=access_token)



        merged: dict[str, dict[str, Any]] = {}



        for row in [*direct_rows, *full_snapshot]:



            ident = employee_identity(row)



            if ident:



                merged[ident] = row



        employees = list(merged.values())







        # ------------------------------------------------------------



        # Authoritative exact-title pass FIRST.



        #



        # SearchEmployee rows expose jodesca/job_title_ar when available. If the user's role



        # query exactly equals a verified employee title, do not send it through



        # fuzzy scoring. This fixes cases such as:



        #



        #   query:



        #     مدير عام الإدارة العامة للتقنية والرقمنة



        #



        #   SearchEmployee jodesca:



        #     مدير عام الإدارة العامة للتقنية والرقمنة



        #



        # One exact verified row => definitive result.



        # Multiple exact rows => ambiguous; never guess.



        # ------------------------------------------------------------



        exact_title_matches: list[dict[str, Any]] = []







        for raw in employees:



            employee = map_employee(raw)



            titles = [normalize_text(value) for value in _title_variants(employee) if normalize_text(value)]







            if query in titles:



                exact_title_matches.append(employee)







        if exact_title_matches:



            print(



                "[Directory Role Exact] "



                f"query={role_query!r} matches={len(exact_title_matches)}"



            )







            for index, employee in enumerate(



                exact_title_matches[:5],



                start=1,



            ):



                print(



                    "[Role Exact Candidate] "



                    f"#{index} "



                    f"username={employee_username(employee)!r} "



                    f"name={employee_name(employee)!r} "



                    f"title={employee_title(employee)!r} "



                    f"department={general_department(employee) or employee_department(employee)!r}"



                )







            if len(exact_title_matches) == 1:



                return {



                    "role": role_query,



                    "employees": exact_title_matches,



                    "best_match": exact_title_matches[0],



                    "ambiguous": False,



                }







            return {



                "role": role_query,



                "employees": exact_title_matches[:10],



                "best_match": {},



                "ambiguous": True,



            }







        # ------------------------------------------------------------



        # Split organisational unit from the requested role.



        # Example:



        #   مدير عام الإدارة العامة للتقنية والرقمنة



        # becomes:



        #   role_core = مدير عام



        #   department_core = التقنيه والرقمنه



        # ------------------------------------------------------------



        department_query = ""



        role_core = query







        dept_match = re.search(



            r"(الاداره\s+العامه\s+.+)$",



            query,



            flags=re.I,



        )







        if dept_match:



            department_query = dept_match.group(1).strip()



            role_core = query[:dept_match.start()].strip()







        # User may write 'مدير عام للتقنية والرقمنة' without the literal



        # 'الإدارة العامة'. Keep the title extraction conservative.



        if not role_core:



            role_core = query







        requested_department_core = _general_department_core(department_query)







        stop_words = {



            "مين", "من", "هو", "هي", "اللي", "الذي", "التي",



            "ماسك", "يشغل", "يتولي", "منصب", "دور",



            "who", "is", "holds", "occupies", "role", "position",



        }







        role_tokens = {



            token



            for token in role_core.split()



            if token and token not in stop_words



        }







        # Closed role-family check. This is intentionally small and is used only



        # for ranking already-verified SearchEmployee rows.



        wants_general_manager = bool(



            re.search(r"\bمدير\s+عام(?:\s+اول)?\b", role_core)



            or re.search(r"\bgeneral\s+manager\b", role_core)



        )







        ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []







        for raw in employees:



            employee = map_employee(raw)







            title_raw = employee_title(employee)



            department_raw = employee_department(employee)



            gdep_raw = general_department(employee)







            title = normalize_text(title_raw)



            department = normalize_text(department_raw)



            gdep = normalize_text(gdep_raw)







            department_score = 0.0



            title_score = 0.0







            # --------------------------------------------------------



            # Department match



            # --------------------------------------------------------



            if requested_department_core:



                org_values = [value for value in (gdep, department) if value]







                for org_value in org_values:



                    org_core = _general_department_core(org_value)







                    if not org_core:



                        continue







                    if org_core == requested_department_core:



                        department_score = max(department_score, 1.0)



                        continue







                    if (



                        requested_department_core in org_core



                        or org_core in requested_department_core



                    ):



                        department_score = max(department_score, 0.95)



                        continue







                    ratio = SequenceMatcher(



                        None,



                        requested_department_core,



                        org_core,



                    ).ratio()







                    if ratio >= 0.84:



                        department_score = max(department_score, 0.88 * ratio)







                # A few SearchEmployee rows have blank dpDesca but the full



                # general-department name is embedded in jodesca.



                title_org_core = _general_department_core(title)



                if requested_department_core and requested_department_core in title_org_core:



                    department_score = max(department_score, 0.93)







                if department_score <= 0.0:



                    continue



            else:



                department_score = 0.5







            # --------------------------------------------------------



            # Role/title match



            # --------------------------------------------------------



            if title:



                if title == role_core:



                    title_score = 1.0







                if wants_general_manager:



                    # Accept both 'مدير عام' and 'مدير عام أول'.



                    if re.search(r"^مدير\s+عام(?:\s+اول)?\b", title):



                        title_score = max(title_score, 0.98)



                    elif re.search(r"\bمدير\s+عام(?:\s+اول)?\b", title):



                        title_score = max(title_score, 0.94)







                if role_tokens:



                    title_tokens = set(title.split())



                    overlap = len(role_tokens & title_tokens)



                    coverage = overlap / max(1, len(role_tokens))



                    similarity = SequenceMatcher(None, role_core, title).ratio()



                    title_score = max(



                        title_score,



                        0.72 * coverage + 0.28 * similarity,



                    )







                # Exact full-query match against title is authoritative for rows



                # whose dpDesca is blank but jodesca contains the organisation.



                if title == query:



                    title_score = 1.0



                    department_score = max(department_score, 1.0)







            if title_score < 0.60:



                continue







            total_score = (0.58 * department_score) + (0.42 * title_score)







            # General-manager title + exact department is a very strong match.



            if wants_general_manager and department_score >= 0.95 and title_score >= 0.94:



                total_score = max(total_score, 1.25)







            ranked.append(



                (



                    total_score,



                    employee,



                    {



                        "department_score": round(department_score, 4),



                        "title_score": round(title_score, 4),



                        "title": title_raw,



                        "department": gdep_raw or department_raw,



                    },



                )



            )







        ranked.sort(key=lambda item: item[0], reverse=True)







        candidates = [employee for _, employee, _ in ranked[:10]]



        best_match: dict[str, Any] = {}



        ambiguous = False







        if ranked:



            top_score, top_employee, _ = ranked[0]



            second_score = ranked[1][0] if len(ranked) > 1 else -1.0







            # Strong role + exact department may be selected even when another



            # unrelated title has partial token overlap.



            if top_score >= 1.20:



                if len(ranked) == 1 or second_score < 1.20:



                    best_match = top_employee



                else:



                    ambiguous = True



            elif top_score >= 0.82:



                if len(ranked) == 1 or (top_score - second_score) >= 0.08:



                    best_match = top_employee



                else:



                    ambiguous = True







        print(



            "[Directory Role Search] "



            f"query={role_query!r} role_core={role_core!r} "



            f"department={department_query!r} candidates={len(candidates)} "



            f"best={employee_name(best_match)!r} ambiguous={ambiguous}"



        )







        for index, (score, employee, debug) in enumerate(ranked[:5], start=1):



            print(



                "[Role Candidate] "



                f"#{index} score={score:.4f} "



                f"username={employee_username(employee)!r} "



                f"name={employee_name(employee)!r} "



                f"title={debug['title']!r} "



                f"department={debug['department']!r} "



                f"department_score={debug['department_score']} "



                f"title_score={debug['title_score']}"



            )







        return {



            "role": role_query,



            "employees": candidates,



            "best_match": best_match,



            "ambiguous": ambiguous,



        }




