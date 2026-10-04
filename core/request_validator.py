from __future__ import annotations



from copy import deepcopy

from typing import Any



from core.semantic_models import ParsedRequest, TargetRef, ValidationResult

from core.intent_map import DETAIL_BASE_INTENT, get_policy





PROFILE_FIELDS = {

    "name", "name_en", "job_title", "job_title_en", "department", "department_en",

    "sector", "sector_en", "email", "mobile", "extension", "username", "employee_type", "all",

}



CONTENT_FIELDS = {"company", "details", "website", "start_date", "end_date", "tags", "subject", "date", "all"}





def _clarification(lang: str, ar: str, en: str) -> str:

    return ar if lang == "ar" else en





def _has_focus(state: dict[str, Any]) -> bool:

    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}

    current = focus.get("current") if isinstance(focus.get("current"), dict) else {}

    return bool(current.get("username") or current.get("userid"))







def _has_previous_focus(state: dict[str, Any]) -> bool:

    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}

    previous = focus.get("previous") if isinstance(focus.get("previous"), dict) else {}

    return bool(previous.get("username") or previous.get("userid"))





def _relation_row(state: dict[str, Any], key: str) -> dict[str, Any]:

    relation = state.get("relation") if isinstance(state.get("relation"), dict) else {}

    row = relation.get(key) if isinstance(relation.get(key), dict) else {}

    return row





def _has_relation_target(state: dict[str, Any], key: str) -> bool:

    row = _relation_row(state, key)

    return bool(row.get("username") or row.get("userid"))





def validate_request(request: ParsedRequest, state: dict[str, Any], *, language: str) -> ValidationResult:

    """Validate semantic output before any API call.



    The LLM proposes meaning. This function enforces backend invariants and

    asks for clarification when a safe execution plan cannot be proven.

    """

    req = request.model_copy(deep=True)

    req.fields = req.normalized_fields()



    if req.intent == "general":

        if req.clarification_reason:

            return ValidationResult(

                request=req,

                clarification=_clarification(

                    language,

                    "ما قدرت أفهم الطلب بشكل موثوق. ممكن تعيد صياغته باختصار؟",

                    "I couldn't understand the request reliably. Could you rephrase it briefly?",

                ),

            )

        return ValidationResult(request=req)

    if req.intent == "web_search":

        return ValidationResult(request=req)

    if req.intent == "summarize_previous":

        items = ((state.get("list_state") or {}).get("items") or []) if isinstance(state.get("list_state"), dict) else []

        if not items:

            return ValidationResult(

                request=req,

                clarification=_clarification(

                    language,

                    "تقصد ألخّص أي نتيجة؟ ما عندي قائمة سابقة واضحة في هالمحادثة.",

                    "Which result would you like me to summarize? I don't have a clear previous list in this chat.",

                ),

            )

        return ValidationResult(request=req)



    policy = get_policy(req.intent)

    if policy is None:

        return ValidationResult(

            request=req,

            clarification=_clarification(language, "ممكن توضح طلبك أكثر؟", "Could you clarify your request a little more?"),

        )



    # Confidence is advisory only. Structural invariants below decide whether execution is safe.

    # This avoids rejecting clear colloquial requests just because the model under-estimated confidence.



    # Target binding is semantic, not guessed by the backend. The parser must

    # explicitly say self / employee / manager_of_self / focus. This prevents a

    # missing target from silently falling back to the signed-in user or an old

    # employee in memory. Profile is the only safe exception because by

    # definition it means the signed-in user's own profile.

    if req.intent == "profile":

        req.targets = [TargetRef(kind="self")]

    elif policy.target_mode in {"self_default", "required_employee"} and not req.targets:

        state["pending"] = {

            "kind": "missing_employee_target",

            "query": "",

            "candidates": [],

            "request": req.model_dump(),

            "message": "missing_target",

        }

        return ValidationResult(

            request=req,

            clarification=_clarification(

                language,

                "تقصد لك أنت، ولا لموظف معيّن؟" if policy.target_mode == "self_default" else "تقصد أي موظف؟",

                "Do you mean yourself, or a specific employee?" if policy.target_mode == "self_default" else "Which employee do you mean?",

            ),

        )





    # Anaphoric references are valid only when structured memory has a verified

    # employee focus. Never fall back to a directory search or self.

    if any(target.kind == "focus" for target in req.targets) and not _has_focus(state):

        return ValidationResult(

            request=req,

            clarification=_clarification(

                language,

                "تقصد أي موظف؟ ما عندي شخص سابق واضح في هالمحادثة.",

                "Which employee do you mean? I don't have a clear previous person in this conversation.",

            ),

        )



    if any(target.kind == "previous_focus" for target in req.targets) and not _has_previous_focus(state):

        return ValidationResult(

            request=req,

            clarification=_clarification(

                language,

                "ما عندي موظف سابق ثاني واضح في هالمحادثة. من تقصد؟",

                "I don't have a second previous employee in context. Who do you mean?",

            ),

        )



    if any(target.kind == "relation_subject" for target in req.targets) and not _has_relation_target(state, "subject"):

        return ValidationResult(

            request=req,

            clarification=_clarification(language, "ما عندي الشخص الأساسي للعلاقة السابقة بشكل واضح. من تقصد؟", "I don't have the previous relation subject clearly. Who do you mean?"),

        )



    if any(target.kind == "relation_object" for target in req.targets) and not _has_relation_target(state, "object"):

        return ValidationResult(

            request=req,

            clarification=_clarification(language, "ما عندي الطرف المرتبط من العلاقة السابقة بشكل واضح. من تقصد؟", "I don't have the related person from the previous relation clearly. Who do you mean?"),

        )



    # Explicit employee targets require an actual name. Pronouns belong to focus.

    for target in req.targets:

        if target.kind == "employee" and not target.name.strip():

            return ValidationResult(

                request=req,

                clarification=_clarification(language, "تقصد أي موظف بالضبط؟", "Which employee do you mean exactly?"),

            )





    if policy.target_mode in {"self_default", "required_employee"} and req.targets and all(t.kind == "none" for t in req.targets):

        state["pending"] = {

            "kind": "missing_employee_target",

            "query": "",

            "candidates": [],

            "request": req.model_dump(),

            "message": "missing_target",

        }

        return ValidationResult(

            request=req,

            clarification=_clarification(

                language,

                "تقصد أي موظف؟",

                "Which employee do you mean?",

            ),

        )



    # Multiple targets are allowed only for capabilities that explicitly support them.

    if len(req.targets) > 1 and not policy.supports_multiple_targets:

        return ValidationResult(

            request=req,

            clarification=_clarification(

                language,

                "أقدر أنفذ الطلب، بس حدد لي شخص واحد أول.",

                "I can do that, but please specify one person first.",

            ),

        )



    # Prevent cross-domain field leakage from a parser error.

    if req.intent in {"profile", "employee_details", "manager"}:

        if req.fields:

            req.fields = [field for field in req.fields if field in PROFILE_FIELDS]

        if not req.fields:

            req.fields = ["name"] if req.intent == "manager" else ["all"]



    if req.intent in {"hr_letter_recipients", "hr_letter_types"}:

        # These are metadata/list APIs. Once the semantic intent is known,
        # profile-like parser fields (including the generic "all") are irrelevant
        # and must not force an employee-vs-HR clarification.
        req.targets = []
        req.fields = []
        req.operation = "list"




    if req.intent in {"news", "announcements", "mazaya", "policy_search"}:

        req.targets = []

        req.operation = "list" if req.operation not in {"search"} else req.operation



    if req.intent in DETAIL_BASE_INTENT:

        req.targets = []

        # Detail can be resolved from a prior list item or an explicit query.

        if req.ordinal <= 0 and not req.query.strip():

            list_intent = str((state.get("list_state") or {}).get("intent") or "")

            if list_intent != DETAIL_BASE_INTENT[req.intent]:

                return ValidationResult(

                    request=req,

                    clarification=_clarification(language, "تقصد أي عنصر؟", "Which item do you mean?"),

                )

        if req.fields:

            req.fields = [field for field in req.fields if field in CONTENT_FIELDS]



    if req.intent == "employee_search" and not req.query.strip() and not req.targets:

        return ValidationResult(

            request=req,

            clarification=_clarification(language, "من الموظف اللي تبي أبحث عنه؟", "Which employee would you like me to search for?"),

        )



    if req.intent == "general_department_counts":

        req.targets = []

        req.operation = "lookup"



    if req.intent == "general_department_members":

        req.targets = []

        req.operation = "list"

        directory_context = state.get("directory_context") if isinstance(state.get("directory_context"), dict) else {}

        if not req.department_query.strip() and not directory_context.get("general_department_key"):

            return ValidationResult(

                request=req,

                clarification=_clarification(

                    language,

                    "أي إدارة عامة تقصد؟",

                    "Which general department do you mean?",

                ),

            )



    if req.intent == "employee_role_search" and not req.role_query.strip():

        return ValidationResult(

            request=req,

            clarification=_clarification(language, "وش المسمى أو الدور اللي تبي أبحث عنه؟", "What role or job title should I search for?"),

        )



    if req.period == "custom" and (not req.start_date or not req.end_date):

        return ValidationResult(

            request=req,

            clarification=_clarification(language, "وش التاريخ أو الفترة اللي تقصدها؟", "What date or date range do you mean?"),

        )



    return ValidationResult(request=req)
