from __future__ import annotations







import re



from difflib import SequenceMatcher



from copy import deepcopy



from dataclasses import dataclass



from typing import Any







from core.employee_api import (



    EmployeeAPIClient,



    EmployeeAPIError,



    EmployeeAuthenticationError,



    EmployeeAuthorizationError,



)



from core.employee_tools import EmployeeContext



from core.semantic_models import ParsedRequest, ResolvedTarget, TargetRef











AR_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")



NON_WORD = re.compile(r"[^0-9a-z\u0600-\u06ff]+", re.I)











def _is_placeholder_employee_name(value: str) -> bool:

    text = _norm(value)

    return text in {

        "موظف", "الموظف", "موظف معين", "موظف محدد", "شخص", "شخص معين",

        "احد الموظفين", "احد الموظفات", "employee", "an employee",

        "specific employee", "someone", "somebody",

    }





@dataclass



class ResolutionResult:



    targets: list[ResolvedTarget]



    clarification: str = ""



    retryable_error: bool = False











def _norm(value: Any) -> str:



    text = str(value or "").casefold()



    text = AR_DIACRITICS.sub("", text).replace("ـ", "")



    for source, target in (



        ("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه"),



        ("ؤ", "و"), ("ئ", "ي"),



    ):



        text = text.replace(source, target)



    return " ".join(NON_WORD.sub(" ", text).split())











def _uid(row: dict[str, Any]) -> str:



    return str(row.get("username") or row.get("userid") or "").strip()











def _name(row: dict[str, Any], lang: str) -> str:



    if lang == "en":



        return str(row.get("name_en") or row.get("name") or row.get("name_ar") or _uid(row)).strip()



    return str(row.get("name_ar") or row.get("name") or row.get("name_en") or _uid(row)).strip()











def _bounded(row: Any) -> dict[str, Any]:



    if not isinstance(row, dict):



        return {}



    keys = (



        "username", "userid", "name", "name_ar", "name_en", "job_title", "job_title_ar",



        "job_title_en", "department", "department_ar", "department_en", "sector_ar", "sector_en",



        "email", "mobile", "extension",



    )



    return {k: row.get(k) for k in keys if row.get(k) not in (None, "")}











def _employee_text(row: dict[str, Any]) -> str:



    return _norm(" ".join(



        str(row.get(key) or "")



        for key in ("name", "name_ar", "name_en", "username", "userid")



    ))











def _local_match(query: str, state: dict[str, Any]) -> dict[str, Any] | None:



    q = _norm(query)



    if not q:



        return None



    q_tokens = q.split()







    rows: list[dict[str, Any]] = []



    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}



    for candidate in (focus.get("current"), focus.get("previous")):



        if isinstance(candidate, dict) and _uid(candidate):



            rows.append(candidate)



    relation = state.get("relation") if isinstance(state.get("relation"), dict) else {}



    for candidate in (relation.get("object"), relation.get("subject")):



        if isinstance(candidate, dict) and _uid(candidate):



            rows.append(candidate)



    list_state = state.get("list_state") if isinstance(state.get("list_state"), dict) else {}



    for candidate in list_state.get("items") or []:



        if isinstance(candidate, dict) and _uid(candidate):



            rows.append(candidate)



    for candidate in state.get("entity_history") or []:



        if isinstance(candidate, dict) and _uid(candidate):



            rows.append(candidate)







    scored: list[tuple[int, dict[str, Any]]] = []



    seen: set[str] = set()



    for row in rows:



        uid = _uid(row).casefold()



        if not uid or uid in seen:



            continue



        seen.add(uid)



        hay = _employee_text(row)



        name_ar = _norm(row.get("name_ar"))



        name_en = _norm(row.get("name_en"))



        score = 0



        if q == _norm(uid):



            score = 100



        elif q and q in {name_ar, name_en}:



            score = 95



        elif q_tokens and all(token in hay.split() for token in q_tokens):



            score = 80 + len(q_tokens)



        elif q_tokens:



            # Conservative fuzzy matching is allowed ONLY against already verified



            # local entities/list rows. It never selects a directory employee by



            # fuzzy guess. This handles small typos such as "الهاجريي" and



            # "وس الدعجان" when the verified person is already in context.



            candidate_tokens = hay.split()



            similarities: list[float] = []



            for token in q_tokens:



                if not candidate_tokens:



                    similarities = []



                    break



                similarities.append(max(SequenceMatcher(None, token, c).ratio() for c in candidate_tokens))



            if similarities and min(similarities) >= 0.78:



                score = 45 + int(sum(similarities) / len(similarities) * 20)



        if score:



            scored.append((score, row))







    scored.sort(key=lambda pair: pair[0], reverse=True)



    if not scored:



        return None



    if len(scored) == 1 or scored[0][0] >= scored[1][0] + 5:



        return _bounded(scored[0][1])



    return None











def _candidate_prompt(candidates: list[dict[str, Any]], lang: str) -> str:



    if lang == "ar":



        lines = ["لقيت أكثر من موظف مطابق. مين تقصد؟"]



    else:



        lines = ["I found more than one matching employee. Which one do you mean?"]



    for index, row in enumerate(candidates[:10], 1):



        name = _name(row, lang)



        title = str(row.get("job_title") or row.get("job_title_ar") or row.get("job_title_en") or "").strip()



        dept = str(row.get("department") or row.get("department_ar") or row.get("department_en") or "").strip()



        parts = [p for p in (name, title, dept) if p]



        lines.append(f"{index}. " + " — ".join(parts))



    return "\n".join(lines)











def _selection_index(message: str, candidates: list[dict[str, Any]]) -> int | None:



    text = _norm(message)



    if not text:



        return None



    if text.isdigit():



        index = int(text) - 1



        return index if 0 <= index < len(candidates) else None







    ordinals = {



        "الاول": 0, "الاولي": 0, "اول": 0,



        "الثاني": 1, "الثانيه": 1,



        "الثالث": 2, "الثالثه": 2,



        "الرابع": 3, "الرابعه": 3,



        "الخامس": 4, "الخامسه": 4,



        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,



    }



    if text in ordinals and ordinals[text] < len(candidates):



        return ordinals[text]







    # Closed selection grammar: username, verified name, title or department.



    matches: list[int] = []



    for i, row in enumerate(candidates):



        values = (



            _uid(row), row.get("name"), row.get("name_ar"), row.get("name_en"),



            row.get("job_title"), row.get("job_title_ar"), row.get("job_title_en"),



            row.get("department"), row.get("department_ar"), row.get("department_en"),



        )



        hay = _norm(" ".join(str(v or "") for v in values))



        tokens = text.split()



        if text and (text == _norm(_uid(row)) or all(token in hay.split() for token in tokens)):



            matches.append(i)



    return matches[0] if len(matches) == 1 else None











def is_selection_like(message: str, candidates: list[dict[str, Any]]) -> bool:



    return _selection_index(message, candidates) is not None











def restore_pending_selection(message: str, state: dict[str, Any]) -> ParsedRequest | None:



    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}



    if pending.get("kind") != "employee_disambiguation":



        return None



    candidates = pending.get("candidates") if isinstance(pending.get("candidates"), list) else []



    index = _selection_index(message, candidates)



    if index is None:



        return None



    raw_request = pending.get("request") if isinstance(pending.get("request"), dict) else {}



    try:



        request = ParsedRequest.model_validate(raw_request)



    except Exception:



        return None



    selected = _bounded(candidates[index])



    # Replace the first employee/focus target with a concrete verified employee name.



    selected_name = str(selected.get("name_ar") or selected.get("name_en") or selected.get("name") or _uid(selected))



    request.targets = [TargetRef(kind="employee", name=selected_name)]



    request.confidence = 1.0



    state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}



    # Put selected verified row into focus so local resolution uses it immediately.



    focus = state.setdefault("focus", {"current": {}, "previous": {}})



    focus["previous"] = deepcopy(focus.get("current") or {})



    focus["current"] = selected



    return request











def restore_pending_target(request: ParsedRequest, state: dict[str, Any]) -> ParsedRequest | None:



    """Fill a previously missing employee subject from a bare employee-name reply.







    We only restore when the new turn semantically parsed as an employee search



    with exactly one explicit employee target. A complete new request therefore



    always wins over the pending clarification.



    """



    pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}



    if pending.get("kind") != "missing_employee_target":



        return None



    if request.intent != "employee_search" or len(request.targets) != 1:



        return None



    target = request.targets[0]



    if target.kind != "employee" or not str(target.name or "").strip():



        return None



    raw_request = pending.get("request") if isinstance(pending.get("request"), dict) else {}



    try:



        restored = ParsedRequest.model_validate(raw_request)



    except Exception:



        return None



    restored.targets = [TargetRef(kind="employee", name=str(target.name).strip())]



    restored.confidence = max(restored.confidence, request.confidence, 0.90)



    state["pending"] = {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}



    return restored











class ReferenceResolver:



    def __init__(self, api_client: EmployeeAPIClient) -> None:



        self.api = api_client







    async def _search_employee(



        self,



        query: str,



        context: EmployeeContext,



        state: dict[str, Any],



        request: ParsedRequest,



        lang: str,



    ) -> ResolutionResult:



        local = _local_match(query, state)



        if local:



            return ResolutionResult([



                ResolvedTarget(kind="employee", username=_uid(local), employee=local, source_text=query)



            ])







        try:



            payload = await self.api.search_employees(access_token=context.access_token, search=query)



        except (EmployeeAuthenticationError, EmployeeAuthorizationError):



            raise



        except EmployeeAPIError:



            state["pending"] = {



                "kind": "resolution_retry",



                "query": query,



                "candidates": [],



                "request": request.model_dump(),



                "message": "api_unavailable",



            }



            msg = (



                "آسف 😅 تعذر عليّ الوصول لدليل الموظفين هالمرة. وأنا لسه في مرحلتي التجريبية، وقاعد أتعلم وأتطور باستمرار."



                if lang == "ar"



                else "I couldn't reach the employee directory right now. Please try again shortly; I won't substitute an older employee from context."



            )



            return ResolutionResult([], msg, retryable_error=True)







        rows = [row for row in (payload.get("employees") or []) if isinstance(row, dict) and _uid(row)]



        if not rows:



            state["pending"] = {



                "kind": "resolution_retry",



                "query": query,



                "candidates": [],



                "request": request.model_dump(),



                "message": "no_match",



            }



            msg = f"ما لقيت موظف مطابق لـ «{query}»." if lang == "ar" else f'I could not find an employee matching "{query}".'



            return ResolutionResult([], msg)







        q = _norm(query)



        exact = [row for row in rows if q in {_norm(row.get("name_ar")), _norm(row.get("name_en")), _norm(_uid(row))}]



        chosen: dict[str, Any] | None = None



        if len(exact) == 1:



            chosen = exact[0]



        elif len(rows) == 1:



            chosen = rows[0]







        if chosen is not None:



            chosen = _bounded(chosen)



            return ResolutionResult([



                ResolvedTarget(kind="employee", username=_uid(chosen), employee=chosen, source_text=query)



            ])







        candidates = [_bounded(row) for row in rows[:10]]



        state["pending"] = {



            "kind": "employee_disambiguation",



            "query": query,



            "candidates": candidates,



            "request": request.model_dump(),



            "message": "choose_employee",



        }



        return ResolutionResult([], _candidate_prompt(candidates, lang))







    async def resolve(



        self,



        request: ParsedRequest,



        state: dict[str, Any],



        context: EmployeeContext,



        *,



        language: str,



    ) -> ResolutionResult:



        resolved: list[ResolvedTarget] = []



        targets = list(request.targets)







        for target in targets:



            if target.kind == "none":



                continue



            if target.kind == "self":



                # Self is always the authenticated principal. Conversational

                # focus may point to a manager or another employee, but it must

                # never redefine who "I/me/my" refers to. Reuse only a verified

                # cached self row whose username matches the current principal.

                cached_self = _bounded(

                    state.get("authenticated_user")

                    if isinstance(state.get("authenticated_user"), dict)

                    else {}

                )

                cached_uid = _uid(cached_self).strip().casefold()

                principal_uid = str(context.username or "").strip().casefold()

                if cached_uid != principal_uid:

                    cached_self = {}



                resolved.append(

                    ResolvedTarget(

                        kind="self",

                        username=context.username,

                        employee=cached_self,

                        source_text="self",

                    )

                )



                continue







            if target.kind == "manager_of_self":



                try:



                    manager = await self.api.get_manager_details(



                        username=context.username,



                        access_token=context.access_token,



                    )



                except EmployeeAPIError:



                    msg = "تعذر جلب بيانات مديرك حاليًا." if language == "ar" else "I couldn't retrieve your manager's details right now."



                    return ResolutionResult([], msg, retryable_error=True)



                manager = _bounded(manager)



                username = _uid(manager)



                if not username:



                    msg = "بيانات مديرك في النظام ما فيها اسم مستخدم صالح." if language == "ar" else "Your manager's record does not contain a valid username."



                    return ResolutionResult([], msg)



                resolved.append(ResolvedTarget(kind="manager_of_self", username=username, employee=manager, source_text="manager_of_self"))



                continue







            if target.kind in {"relation_subject", "relation_object"}:



                relation = state.get("relation") if isinstance(state.get("relation"), dict) else {}



                key = "subject" if target.kind == "relation_subject" else "object"



                row = _bounded(relation.get(key))



                if not _uid(row):



                    msg = "ما قدرت أحدد الشخص المقصود من العلاقة السابقة." if language == "ar" else "I couldn't determine the intended person from the previous relation."



                    return ResolutionResult([], msg)



                resolved.append(ResolvedTarget(kind=target.kind, username=_uid(row), employee=row, source_text=target.kind))



                continue







            if target.kind in {"focus", "previous_focus"}:



                pending = state.get("pending") if isinstance(state.get("pending"), dict) else {}



                if target.kind == "focus" and pending.get("kind") == "resolution_retry" and str(pending.get("query") or "").strip():



                    retry = await self._search_employee(str(pending.get("query")), context, state, request, language)



                    if retry.clarification:



                        return retry



                    resolved.extend(retry.targets)



                    continue







                focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}



                row = focus.get("current" if target.kind == "focus" else "previous")



                row = _bounded(row)



                if not _uid(row):



                    msg = "تقصد أي موظف؟" if language == "ar" else "Which employee do you mean?"



                    return ResolutionResult([], msg)



                resolved.append(ResolvedTarget(kind=target.kind, username=_uid(row), employee=row, source_text=target.kind))



                continue







            if target.kind == "employee":



                result = await self._search_employee(target.name, context, state, request, language)



                if result.clarification:



                    return result



                resolved.extend(result.targets)



                continue







        return ResolutionResult(resolved)