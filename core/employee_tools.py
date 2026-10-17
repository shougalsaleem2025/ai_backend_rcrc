"""Approved read-only enterprise tools for the RCRC AI Assistant.







Qwen receives these capability descriptions directly. They describe what each



backend capability means, not a list of user phrases. The model can therefore



understand new wording without adding regex branches.



"""







from __future__ import annotations







from dataclasses import dataclass



from datetime import datetime, timedelta, timezone



from typing import Any







from core.config import settings



from core.employee_api import EmployeeAPIClient, EmployeeAPIError, EmployeeAuthorizationError







RIYADH_TIMEZONE = timezone(timedelta(hours=3))



# Current production behavior: calendar/attendance endpoints are bound to the

# authenticated token owner. Keep this guard in the backend even if the model or

# parser asks for another employee.

SELF_ONLY_RUNTIME_TOOLS = {

    "get_my_meetings",

    "get_meeting_list",

    "get_today_attendance",

    "get_employee_attendance",

    "get_working_hours",

    "get_subemployees_attendance",

}











@dataclass(frozen=True)



class EmployeeContext:



    username: str



    access_token: str



    emp_code: str = ""











# Keep descriptions semantic and explicit about scope. In particular, tools



# that say "authenticated employee" MUST NOT be used for a manager or another



# person.



_TOOL_DEFS: list[dict[str, Any]] = [



    {



        "name": "get_employee_profile",



        "description": "Get the signed-in employee's own profile: identity, names, username, job title, department, email, mobile, extension and employee type when those fields are returned by the verified profile API. SELF ONLY; never use this to look up another employee.",



        "args": {},



    },



    {



        "name": "get_manager_details",



        "description": "Get direct manager details for the subject employee. Defaults to the signed-in employee; for an explicitly named employee the backend first resolves the employee through the directory, then uses the verified username.",



        "args": {},



    },



    {



        "name": "get_leave_balance",



        "description": "Get the current vacation/leave balance for the subject employee. Defaults to the signed-in employee; a named employee is used only after backend directory resolution and remains subject to upstream authorization.",



        "args": {},



    },



    {



        "name": "get_vacation_types",



        "description": "List the available vacation/leave types in the employee system.",



        "args": {},



    },



    {



        "name": "get_my_meetings",



        "description": "Get the signed-in employee's own meetings for a requested day or date range. SELF ONLY because the deployed endpoint is bound to the authenticated token owner.",



        "args": {



            "period": "today|tomorrow|yesterday|this_week|last_week|this_month|last_month|custom optional",



            "start_date": "YYYY-MM-DD optional; use with custom",



            "end_date": "YYYY-MM-DD optional; use with custom",



            "language": "ar|en optional",



        },



    },



    {



        "name": "get_today_attendance",



        "description": "Get today's attendance/check-in for the signed-in employee. SELF ONLY because the deployed endpoint is bound to the authenticated token owner.",



        "args": {},



    },



    {



        "name": "get_employee_attendance",



        "description": "Get attendance history for the signed-in employee over a requested period. SELF ONLY because the deployed endpoint is bound to the authenticated token owner.",



        "args": {



            "period": "today|yesterday|this_week|last_week|this_month|last_month|custom optional",



            "start_date": "YYYY-MM-DD optional; use with custom",



            "end_date": "YYYY-MM-DD optional; use with custom",



        },



    },



    {



        "name": "get_working_hours",



        "description": "Get working hours for the signed-in employee over a requested period. SELF ONLY because the deployed endpoint is bound to the authenticated token owner.",



        "args": {



            "period": "today|yesterday|this_week|last_week|this_month|last_month|custom optional",



            "start_date": "YYYY-MM-DD optional; use with custom",



            "end_date": "YYYY-MM-DD optional; use with custom",



        },



    },



    {



        "name": "get_vacation_history",



        "description": "Get leave/vacation history for the subject employee over a requested period. Defaults to the signed-in employee; a named employee is used only after verified directory resolution and authorization.",



        "args": {



            "period": "this_month|last_month|custom optional",



            "start_date": "YYYY-MM-DD optional",



            "end_date": "YYYY-MM-DD optional",



            "start_index": "integer optional",



            "batch_size": "integer optional",



        },



    },



    {



        "name": "get_vacations_list",



        "description": "List organization-visible employees who are on leave for a period, subject to authorization. This is NOT another employee's leave balance.",



        "args": {



            "period": "today|tomorrow|yesterday|this_week|last_week|custom optional",



            "start_date": "YYYY-MM-DD optional",



            "end_date": "YYYY-MM-DD optional",



            "start_index": "integer optional",



            "batch_size": "integer optional",



        },



    },



    {



        "name": "get_intersected_vacations",



        "description": "Get leave overlap/intersection information for the signed-in employee over a requested period.",



        "args": {



            "period": "custom optional",



            "start_date": "YYYY-MM-DD optional",



            "end_date": "YYYY-MM-DD optional",



        },



    },



    {



        "name": "get_reporting_employees",



        "description": "List employees who directly report to the subject employee. Defaults to the signed-in employee; a named manager is used only after verified directory resolution and authorization.",



        "args": {},



    },



    {



        "name": "get_subemployees_attendance",



        "description": "Get attendance for employees reporting to the subject employee for a requested period. Defaults to the signed-in employee; a named manager is used only after verified directory resolution and authorization.",



        "args": {



            "period": "today|yesterday|this_week|custom optional",



            "start_date": "YYYY-MM-DD optional",



            "end_date": "YYYY-MM-DD optional",



        },



    },



    {



        "name": "get_employee_image",



        "description": "Get the official profile image for the subject employee. Defaults to the signed-in employee; a named employee is used only after verified directory resolution and authorization.",



        "args": {},



    },



    {



        "name": "get_tarasol_count",



        "description": "Get the Tarasol transaction count for the subject employee. Defaults to the signed-in employee; a named employee is used only after verified directory resolution and authorization.",



        "args": {},



    },



    {



        "name": "get_tarasol_transactions",



        "description": "Get Tarasol transactions for the subject employee. Defaults to the signed-in employee; a named employee is used only after verified directory resolution and authorization.",



        "args": {"language": "ar|en optional"},



    },



    {



        "name": "search_employees",



        "description": "Search the employee directory by name using SearchEmployee. The backend sends search plus the required empty batchSize field as multipart/form-data and verifies returned employees locally.",



        "args": {



            "search": "required: person's name or concise identifying search text",



        },



    },



    {



        "name": "get_employee_details",



        "description": "Get verified details for one employee after the backend has resolved that employee through the directory search. The LLM never supplies a username directly.",



        "args": {},



    },



    {



        "name": "find_employee_by_role",



        "description": "Find employee(s) by verified job title, organizational role or department role when the user does not know the person's name.",



        "args": {"role_query": "required: concise role/title/organizational role text"},



    },



    {



        "name": "get_hr_letter_recipients",



        "description": "List available recipients/addressees for HR letters.",



        "args": {},



    },



    {



        "name": "get_hr_letter_types",



        "description": "List available HR letter types.",



        "args": {},



    },



    {



        "name": "search_policies",



        "description": "Search or list internal policies and procedures.",



        "args": {



            "search": "optional topic/search text",



            "start_index": "integer optional",



            "batch_size": "integer optional",



        },



    },



    {



        "name": "get_policy_details",



        "description": "Get details of one policy/procedure. The item_id must come from a verified search_policies result in the same turn; never guess an ID.",



        "args": {"item_id": "required verified policy ID"},



    },



    {



        "name": "get_news_list",



        "description": "Search or list internal RCRC news.",



        "args": {



            "search": "optional topic/search text",



            "start_index": "integer optional",



            "batch_size": "integer optional",



        },



    },



    {



        "name": "get_news_details",



        "description": "Get details of one news item. The item_id must come from a verified get_news_list result in the same turn; never guess an ID.",



        "args": {"item_id": "required verified news ID"},



    },



    {



        "name": "get_announcements_list",



        "description": "Search or list internal announcements/circulars.",



        "args": {



            "search": "optional topic/search text",



            "start_index": "integer optional",



            "batch_size": "integer optional",



        },



    },



    {



        "name": "get_announcement_details",



        "description": "Get details of one announcement. The item_id must come from a verified get_announcements_list result in the same turn; never guess an ID.",



        "args": {"item_id": "required verified announcement ID"},



    },



    {



        "name": "get_mazaya_list",



        "description": "Search or list employee benefits, discounts and offers (Mazaya).",



        "args": {



            "search": "optional topic/provider/search text",



            "start_index": "integer optional",



            "batch_size": "integer optional",



        },



    },



    {



        "name": "get_mazaya_details",



        "description": "Get details of one employee benefit/offer. The item_id must come from a verified get_mazaya_list result in the same turn; never guess an ID.",



        "args": {"item_id": "required verified benefit/offer ID"},



    },



    {



        "name": "get_meeting_list",



        "description": "Get a richer enterprise meeting list for the signed-in employee over a date range. SELF ONLY because the deployed endpoint is bound to the authenticated token owner.",



        "args": {



            "period": "today|tomorrow|yesterday|this_week|last_week|custom optional",



            "start_date": "YYYY-MM-DD optional",



            "end_date": "YYYY-MM-DD optional",



        },



    },



]











EMPLOYEE_TOOL_NAMES = {item["name"] for item in _TOOL_DEFS}







# Tools whose API contract requires a username.  The orchestrator may resolve



# a different employee first and pass that verified username separately.



# The LLM never gets to invent or directly provide this value.



USERNAME_TOOL_NAMES = {



    "get_employee_profile",



    "get_employee_details",



    "get_manager_details",



    "get_leave_balance",



    "get_my_meetings",



    "get_today_attendance",



    "get_employee_attendance",



    "get_working_hours",



    "get_vacation_history",



    "get_intersected_vacations",



    "get_reporting_employees",



    "get_subemployees_attendance",



    "get_employee_image",



    "get_tarasol_count",



    "get_tarasol_transactions",



    "get_meeting_list",



}







EMPLOYEE_TOOLS: list[dict[str, Any]] = []



for item in _TOOL_DEFS:



    properties: dict[str, Any] = {}



    required: list[str] = []







    for arg_name, arg_description in item["args"].items():



        arg_type = "integer" if "integer" in arg_description else "string"



        properties[arg_name] = {



            "type": arg_type,



            "description": arg_description,



        }



        if "required" in arg_description:



            required.append(arg_name)







    EMPLOYEE_TOOLS.append({



        "type": "function",



        "function": {



            "name": item["name"],



            "description": item["description"],



            "parameters": {



                "type": "object",



                "properties": properties,



                "required": required,



            },



        },



    })











def _iso(value: Any) -> str:



    """Normalize common user/planner date formats to API-safe YYYY-MM-DD."""



    raw = str(value or "").strip()



    if not raw:



        raise EmployeeAPIError("Date is required.")







    # Accept ISO timestamps by taking the calendar component first.



    if "T" in raw:



        raw = raw.split("T", 1)[0]







    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):



        try:



            return datetime.strptime(raw, fmt).date().isoformat()



        except ValueError:



            continue







    raise EmployeeAPIError("Invalid date. Expected YYYY-MM-DD.")











def _date_range(



    arguments: dict[str, Any],



    *,



    default_period: str = "today",



) -> tuple[str, str]:



    today = datetime.now(RIYADH_TIMEZONE).date()



    period = str(arguments.get("period") or default_period).strip().lower()







    if period == "today":



        return today.isoformat(), today.isoformat()



    if period == "tomorrow":



        day = today + timedelta(days=1)



        return day.isoformat(), day.isoformat()



    if period == "yesterday":



        day = today - timedelta(days=1)



        return day.isoformat(), day.isoformat()



    if period == "this_week":

        # RCRC work week: Sunday -> Thursday. Friday/Saturday are weekend.
        days_since_sunday = (today.weekday() + 1) % 7
        start = today - timedelta(days=days_since_sunday)
        thursday = start + timedelta(days=4)
        end = min(today, thursday)
        return start.isoformat(), end.isoformat()

    if period == "last_week":

        # RCRC work week: previous Sunday -> previous Thursday.
        days_since_sunday = (today.weekday() + 1) % 7
        current_sunday = today - timedelta(days=days_since_sunday)
        start = current_sunday - timedelta(days=7)
        end = start + timedelta(days=4)
        return start.isoformat(), end.isoformat()



    if period == "this_month":



        start = today.replace(day=1)



        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)



        return start.isoformat(), (next_month - timedelta(days=1)).isoformat()



    if period == "last_month":



        this_month = today.replace(day=1)



        end = this_month - timedelta(days=1)



        return end.replace(day=1).isoformat(), end.isoformat()







    start_raw = arguments.get("start_date")



    end_raw = arguments.get("end_date") or start_raw



    if start_raw:



        start = _iso(start_raw)



        end = _iso(end_raw)



        if end < start:



            raise EmployeeAPIError("End date cannot be before start date.")



        return start, end







    return today.isoformat(), today.isoformat()











def _int_arg(



    arguments: dict[str, Any],



    name: str,



    default: int,



    *,



    minimum: int = 0,



    maximum: int = 5000,



) -> int:



    try:



        value = int(arguments.get(name, default))



    except (TypeError, ValueError):



        value = default



    return max(minimum, min(value, maximum))











async def execute_employee_tool(



    *,



    tool_name: str,



    arguments: dict[str, Any],



    context: EmployeeContext,



    api_client: EmployeeAPIClient,



    target_username: str | None = None,



) -> dict[str, Any]:



    if tool_name not in EMPLOYEE_TOOL_NAMES:



        raise EmployeeAPIError(f"Unsupported enterprise tool: {tool_name}")







    token = context.access_token



    # Never take a target username from model-generated arguments.  A non-self



    # username can only arrive through the orchestrator after a verified



    # SearchEmployee resolution.



    username = str(target_username or context.username).strip()



    if tool_name in SELF_ONLY_RUNTIME_TOOLS:

        requester = str(context.username or "").strip().casefold()

        requested = str(username or "").strip().casefold()

        if requested and requester and requested != requester:

            raise EmployeeAuthorizationError(

                "This self-service endpoint can only be called for the authenticated token owner."

            )

        # Force the upstream username to the authenticated identity so a stale

        # conversational focus can never leak into a private self-service call.

        username = str(context.username or "").strip()



    if tool_name == "get_employee_profile":



        return await api_client.get_employee_profile(username=username, access_token=token)



    if tool_name == "get_employee_details":



        return await api_client.get_employee_directory_details(



            username=username,



            access_token=token,



        )



    if tool_name == "get_manager_details":



        return await api_client.get_manager_details(username=username, access_token=token)



    if tool_name == "get_leave_balance":



        return await api_client.get_leave_balance(username=username, access_token=token)



    if tool_name == "get_vacation_types":



        return await api_client.get_vacation_types(access_token=token)



    if tool_name == "get_today_attendance":



        return await api_client.get_today_attendance(username=username, access_token=token)



    if tool_name == "get_employee_image":



        return await api_client.get_employee_image(username=username, access_token=token)



    if tool_name == "get_tarasol_count":



        return await api_client.get_tarasol_count(username=username, access_token=token)



    if tool_name == "get_tarasol_transactions":



        language = str(arguments.get("language") or "ar").lower()



        if language not in {"ar", "en"}:



            language = "ar"



        return await api_client.get_tarasol_transactions(



            username=username,



            access_token=token,



            lang=language,



        )



    if tool_name == "search_employees":



        search = str(arguments.get("search") or "").strip()



        if not search:



            raise EmployeeAPIError("Employee search text is required.")



        return await api_client.search_employees(



            access_token=token,



            search=search,



        )



    if tool_name == "find_employee_by_role":



        role_query = str(arguments.get("role_query") or "").strip()



        if not role_query:



            raise EmployeeAPIError("Employee role text is required.")



        return await api_client.find_employees_by_role(



            access_token=token,



            role_query=role_query,



        )



    if tool_name == "get_hr_letter_recipients":



        return await api_client.get_hr_letter_recipients(access_token=token)



    if tool_name == "get_hr_letter_types":



        return await api_client.get_hr_letter_types(access_token=token)



    if tool_name == "get_reporting_employees":



        return await api_client.get_reporting_employees(



            username=username,



            access_token=token,



        )







    if tool_name == "get_my_meetings":



        start, end = _date_range(arguments)



        language = str(arguments.get("language") or "ar").lower()



        if language not in {"ar", "en"}:



            language = "ar"



        return await api_client.get_my_meetings(



            username=username,



            access_token=token,



            start_date=start,



            end_date=end,



            lang=language,



        )



    if tool_name == "get_employee_attendance":



        start, end = _date_range(arguments)



        return await api_client.get_employee_attendance(



            username=username,



            access_token=token,



            start_date=start,



            end_date=end,



        )



    if tool_name == "get_working_hours":



        start, end = _date_range(arguments)



        return await api_client.get_working_hours(



            username=username,



            access_token=token,



            start_date=start,



            end_date=end,



        )



    if tool_name == "get_vacation_history":



        start, end = _date_range(arguments, default_period="this_month")



        return await api_client.get_vacation_history(



            username=username,



            access_token=token,



            start_date=start,



            end_date=end,



            start_index=_int_arg(arguments, "start_index", 0),



            batch_size=_int_arg(arguments, "batch_size", 50, minimum=1, maximum=1000),



        )



    if tool_name == "get_vacations_list":



        start, end = _date_range(arguments)



        return await api_client.get_vacations_list(



            access_token=token,



            start_date=start,



            end_date=end,



            start_index=_int_arg(arguments, "start_index", 0),



            batch_size=_int_arg(arguments, "batch_size", 30, minimum=1, maximum=1000),



        )



    if tool_name == "get_intersected_vacations":



        start, end = _date_range(arguments)



        return await api_client.get_intersected_vacations(



            username=username,



            access_token=token,



            start_date=start,



            end_date=end,



        )



    if tool_name == "get_subemployees_attendance":



        start, end = _date_range(arguments)



        return await api_client.get_subemployees_attendance(



            username=username,



            access_token=token,



            start_date=start,



            end_date=end,



        )



    if tool_name == "get_meeting_list":



        start, end = _date_range(arguments)



        return await api_client.get_meeting_list(



            username=username,



            access_token=token,



            start_date=start,



            end_date=end,



        )







    if tool_name == "search_policies":



        return await api_client.search_policies(



            access_token=token,



            search=str(arguments.get("search") or "").strip(),



            start_index=_int_arg(arguments, "start_index", 0),



            batch_size=_int_arg(arguments, "batch_size", 20, minimum=1, maximum=500),



        )



    if tool_name == "get_policy_details":



        item_id = str(arguments.get("item_id") or "").strip()



        if not item_id:



            raise EmployeeAPIError("Policy id is required.")



        return await api_client.get_policy_details(access_token=token, item_id=item_id)



    if tool_name == "get_news_list":



        return await api_client.get_news_list(



            access_token=token,



            search=str(arguments.get("search") or "").strip(),



            start_index=_int_arg(arguments, "start_index", 0),



            batch_size=_int_arg(arguments, "batch_size", 20, minimum=1, maximum=500),



        )



    if tool_name == "get_news_details":



        item_id = str(arguments.get("item_id") or "").strip()



        if not item_id:



            raise EmployeeAPIError("News id is required.")



        return await api_client.get_news_details(access_token=token, item_id=item_id)



    if tool_name == "get_announcements_list":



        return await api_client.get_announcements_list(



            access_token=token,



            search=str(arguments.get("search") or "").strip(),



            start_index=_int_arg(arguments, "start_index", 0),



            batch_size=_int_arg(arguments, "batch_size", 20, minimum=1, maximum=500),



        )



    if tool_name == "get_announcement_details":



        item_id = str(arguments.get("item_id") or "").strip()



        if not item_id:



            raise EmployeeAPIError("Announcement id is required.")



        return await api_client.get_announcement_details(access_token=token, item_id=item_id)



    if tool_name == "get_mazaya_list":



        return await api_client.get_mazaya_list(



            access_token=token,



            search=str(arguments.get("search") or "").strip(),



            start_index=_int_arg(arguments, "start_index", 0),



            batch_size=_int_arg(arguments, "batch_size", 20, minimum=1, maximum=200),



        )



    if tool_name == "get_mazaya_details":



        item_id = str(arguments.get("item_id") or "").strip()



        if not item_id:



            raise EmployeeAPIError("Mazaya id is required.")



        return await api_client.get_mazaya_details(access_token=token, item_id=item_id)







    raise EmployeeAPIError(f"Unsupported enterprise tool: {tool_name}")