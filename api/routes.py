"""HTTP layer for Sanad (سَنَد) — RCRC AI Assistant.

Conversation flow for the app:
- New chat:        POST /chat without conversation_id.
- Continue a chat: POST /chat with the conversation_id from the previous
                   response or from GET /chat/conversations.
- Chat list:       GET  /chat/conversations
- Open a chat:     GET  /chat/conversations/{conversation_id}
- Rename / delete: PATCH / DELETE /chat/conversations/{conversation_id}
"""

import inspect
import os
import secrets
import time

from fastapi import APIRouter, HTTPException, Path, Query, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.schemas import (
    CONVERSATION_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ConversationActionResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessage,
    ConversationSummary,
    FeedbackRequest,
    FeedbackResponse,
    QuickAction,
    RenameConversationRequest,
    ResetChatResponse,
    WelcomeResponse,
)
from core.config import settings
from core.employee_tools import EmployeeContext
from core.llm_client import is_available
from core.observability import ObservabilityStore
from core.orchestrator import ORCHESTRATOR_BUILD


router = APIRouter()
observability = ObservabilityStore(settings.observability_db_path)

bearer_scheme = HTTPBearer(
    scheme_name="Employee OAuth 2.0 Access Token",
    description="Enter the employee OAuth 2.0 access token without the word Bearer.",
    auto_error=False,
)


def _log_content_enabled() -> bool:
    return str(os.getenv("APP_LOG_CONTENT", "false")).strip().casefold() in {"1", "true", "yes", "on"}


def _build_employee_context(
    credentials: HTTPAuthorizationCredentials | None,
    request: Request | None = None,
) -> EmployeeContext | None:
    """Build identity without ever letting the LLM choose the user.

    Production identity must be placed on request.state.employee_username (and
    ideally request.state.employee_code) by a validated SSO/OAuth middleware or
    trusted gateway. The POC dev username is allowed only outside production.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    access_token = credentials.credentials.strip()
    if not access_token:
        return None

    environment = str(os.getenv("APP_ENVIRONMENT", "development")).strip().casefold()
    verified_username = ""
    verified_code = ""
    if request is not None:
        verified_username = str(getattr(request.state, "employee_username", "") or "").strip().casefold()
        verified_code = str(getattr(request.state, "employee_code", "") or "").strip()

    if verified_username:
        return EmployeeContext(username=verified_username, access_token=access_token, emp_code=verified_code)

    if environment in {"prod", "production"}:
        return None

    dev_username = str(settings.dev_employee_username or "").strip().casefold()
    if not dev_username:
        return None
    dev_code = str(getattr(settings, "dev_employee_code", "") or "").strip()
    return EmployeeContext(username=dev_username, access_token=access_token, emp_code=dev_code)


def _require_context(
    credentials: HTTPAuthorizationCredentials | None,
    request: Request,
) -> EmployeeContext:
    context = _build_employee_context(credentials, request)
    if context is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return context


def _memory(request: Request):
    return request.app.state.orchestrator.memory


def _ask_extra_kwargs(orchestrator, message_id: str) -> dict:
    """Pass message_id only if this orchestrator version accepts it."""
    try:
        params = inspect.signature(orchestrator.ask).parameters
    except (TypeError, ValueError):
        return {}
    return {"message_id": message_id} if "message_id" in params else {}


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Conversation not found")


def _quick_actions(language: str) -> list[QuickAction]:
    if language == "en":
        return [
            QuickAction(id="my_profile", label="My profile", icon="person", message="Show my profile"),
            QuickAction(id="leave_balance", label="Leave balance", icon="beach_access", message="What is my leave balance?"),
            QuickAction(id="meetings_today", label="Today's meetings", icon="calendar_month", message="What meetings do I have today?"),
            QuickAction(id="attendance_today", label="Today's attendance", icon="schedule", message="Show my attendance today"),
            QuickAction(id="employee_search", label="Find an employee", icon="person_search",
                        action_type="prompt_input", placeholder="Type the employee name"),
            QuickAction(id="mazaya", label="Benefits", icon="redeem", message="Show me the available benefits"),
        ]
    return [
        QuickAction(id="my_profile", label="بياناتي", icon="person", message="اعرض بياناتي"),
        QuickAction(id="leave_balance", label="رصيد إجازاتي", icon="beach_access", message="كم رصيد إجازاتي؟"),
        QuickAction(id="meetings_today", label="اجتماعاتي اليوم", icon="calendar_month", message="وش اجتماعاتي اليوم؟"),
        QuickAction(id="attendance_today", label="حضوري اليوم", icon="schedule", message="وش حضوري اليوم؟"),
        QuickAction(id="employee_search", label="البحث عن موظف", icon="person_search",
                    action_type="prompt_input", placeholder="اكتب اسم الموظف"),
        QuickAction(id="mazaya", label="المزايا", icon="redeem", message="اعرض لي المزايا المتاحة"),
    ]


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> ChatResponse:
    orchestrator = request.app.state.orchestrator
    employee_context = _build_employee_context(credentials, request)
    memory = orchestrator.memory

    requested_id = (req.conversation_id or "").strip().lower()
    if requested_id and employee_context is not None:
        # A signed-in user can only continue a conversation that belongs to them.
        if not memory.conversation_exists(employee_context.username, requested_id):
            raise _not_found()
    conversation_id = requested_id or ("c_" + secrets.token_hex(12))

    message_id = "m_" + secrets.token_hex(12)
    request_id = "r_" + secrets.token_hex(12)
    log_content = _log_content_enabled()
    username = employee_context.username if employee_context else "anonymous"
    started = time.perf_counter()

    request_details = {
        "request_id": request_id,
        "message_chars": len(req.message),
        "new_conversation": not requested_id,
    }
    if log_content:
        request_details["user_message"] = req.message

    observability.log_event(
        event_type="chat_request", status="started",
        requester_username=username, conversation_id=conversation_id, message_id=message_id,
        endpoint="POST /chat", model_name=settings.ollama_model,
        orchestrator_build=ORCHESTRATOR_BUILD, details=request_details,
    )

    if employee_context is not None:
        # Register the chat so it appears in the saved-conversations list.
        memory.ensure_conversation(employee_context.username, conversation_id)

    try:
        answer, _, sources = await orchestrator.ask(
            user_message=req.message,
            history=[item.model_dump() for item in req.history],
            employee_context=employee_context,
            conversation_id=conversation_id,
            **_ask_extra_kwargs(orchestrator, message_id),
        )
    except RuntimeError as exc:
        observability.log_event(
            event_type="chat_response", status="failed",
            requester_username=username, conversation_id=conversation_id, message_id=message_id,
            endpoint="POST /chat", duration_ms=int((time.perf_counter() - started) * 1000),
            model_name=settings.ollama_model, orchestrator_build=ORCHESTRATOR_BUILD,
            details={"request_id": request_id, "error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    title = ""
    if employee_context is not None:
        # Feedback (👍/👎) on reloaded history needs the response id on the row.
        memory.tag_last_assistant_message(employee_context.username, conversation_id, message_id)
        conversation = memory.get_conversation(employee_context.username, conversation_id)
        title = str((conversation or {}).get("title") or "")

    observability.log_event(
        event_type="chat_response", status="success",
        requester_username=username, conversation_id=conversation_id, message_id=message_id,
        endpoint="POST /chat", duration_ms=int((time.perf_counter() - started) * 1000),
        model_name=settings.ollama_model, orchestrator_build=ORCHESTRATOR_BUILD,
        details={
            "request_id": request_id,
            "source_count": len(sources),
            "answer_chars": len(answer),
            **({"assistant_answer": answer, "sources": sources} if log_content else {}),
        },
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation_id,
        message_id=message_id,
        title=title,
    )


# ----------------------------------------------------------------------
# Saved conversations
# ----------------------------------------------------------------------

@router.get("/chat/conversations", response_model=ConversationListResponse)
async def list_conversations(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100000),
    q: str = Query(default="", max_length=100, description="Search in titles/previews"),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> ConversationListResponse:
    context = _require_context(credentials, request)
    items, total = _memory(request).list_conversations(
        context.username, limit=limit, offset=offset, query=q,
    )
    return ConversationListResponse(
        items=[ConversationSummary(**item) for item in items],
        total=total, limit=limit, offset=offset,
    )


@router.get("/chat/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    request: Request,
    conversation_id: str = Path(..., pattern=CONVERSATION_ID_PATTERN),
    limit: int = Query(default=50, ge=1, le=200),
    before: int | None = Query(default=None, ge=1, description="Load messages older than this seq"),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> ConversationDetailResponse:
    context = _require_context(credentials, request)
    memory = _memory(request)
    conversation_id = conversation_id.lower()
    conversation = memory.get_conversation(context.username, conversation_id)
    if conversation is None:
        raise _not_found()

    messages, has_more = memory.get_messages(
        context.username, conversation_id, limit=limit, before_seq=before,
    )
    observability.log_event(
        event_type="conversation_opened", status="success",
        requester_username=context.username, conversation_id=conversation_id,
        endpoint="GET /chat/conversations/{id}", model_name=settings.ollama_model,
        orchestrator_build=ORCHESTRATOR_BUILD, details={"returned": len(messages)},
    )
    return ConversationDetailResponse(
        conversation_id=conversation["conversation_id"],
        title=conversation["title"],
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
        message_count=conversation["message_count"],
        messages=[ConversationMessage(**m) for m in messages],
        has_more=has_more,
        next_before=messages[0]["seq"] if (has_more and messages) else None,
    )


@router.patch("/chat/conversations/{conversation_id}", response_model=ConversationActionResponse)
async def rename_conversation(
    body: RenameConversationRequest,
    request: Request,
    conversation_id: str = Path(..., pattern=CONVERSATION_ID_PATTERN),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> ConversationActionResponse:
    context = _require_context(credentials, request)
    if not _memory(request).rename_conversation(context.username, conversation_id.lower(), body.title):
        raise _not_found()
    return ConversationActionResponse()


@router.delete("/chat/conversations/{conversation_id}", response_model=ConversationActionResponse)
async def delete_conversation(
    request: Request,
    conversation_id: str = Path(..., pattern=CONVERSATION_ID_PATTERN),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> ConversationActionResponse:
    context = _require_context(credentials, request)
    if not _memory(request).delete_conversation(context.username, conversation_id.lower()):
        raise _not_found()
    observability.log_event(
        event_type="conversation_deleted", status="success",
        requester_username=context.username, conversation_id=conversation_id.lower(),
        endpoint="DELETE /chat/conversations/{id}", model_name=settings.ollama_model,
        orchestrator_build=ORCHESTRATOR_BUILD,
    )
    return ConversationActionResponse()


# ----------------------------------------------------------------------
# Feedback / reset / welcome / health
# ----------------------------------------------------------------------

@router.post("/chat/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    req: FeedbackRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> FeedbackResponse:
    context = _require_context(credentials, request)
    if not _memory(request).conversation_exists(context.username, req.conversation_id.lower()):
        raise _not_found()

    feedback_id = observability.save_feedback(
        requester_username=context.username,
        conversation_id=req.conversation_id,
        message_id=req.message_id,
        helpful=req.helpful,
        comment=req.comment,
        model_name=settings.ollama_model,
        orchestrator_build=ORCHESTRATOR_BUILD,
    )
    observability.log_event(
        event_type="feedback_submitted", status="success",
        requester_username=context.username, conversation_id=req.conversation_id,
        message_id=req.message_id, endpoint="POST /chat/feedback",
        model_name=settings.ollama_model, orchestrator_build=ORCHESTRATOR_BUILD,
        details={"helpful": req.helpful, "has_comment": bool(req.comment.strip())},
    )
    return FeedbackResponse(feedback_id=feedback_id)


@router.post("/chat/reset", response_model=ResetChatResponse)
async def reset_chat(
    request: Request,
    conversation_id: str | None = Query(default=None, pattern=CONVERSATION_ID_PATTERN),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> ResetChatResponse:
    """Forget the working context. Saved messages stay; use DELETE to remove a chat.

    For a "new chat" button the app does not need this: just send the next
    message without conversation_id.
    """
    context = _require_context(credentials, request)
    memory = _memory(request)
    if conversation_id:
        memory.clear_state(memory.memory_key(context.username, conversation_id))
    else:
        memory.clear_state_prefix(f"{context.username.casefold()}:")
    observability.log_event(
        event_type="chat_reset", status="success",
        requester_username=context.username, conversation_id=conversation_id or "",
        endpoint="POST /chat/reset", model_name=settings.ollama_model,
        orchestrator_build=ORCHESTRATOR_BUILD,
    )
    return ResetChatResponse()


@router.get("/chat/welcome", response_model=WelcomeResponse)
async def chat_welcome(
    request: Request,
    language: str = Query(default="ar", pattern=r"^(ar|en)$"),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> WelcomeResponse:
    context = _require_context(credentials, request)
    payload = await request.app.state.orchestrator.welcome(context, language=language)
    observability.log_event(
        event_type="welcome_viewed", status="success",
        requester_username=context.username, endpoint="GET /chat/welcome",
        model_name=settings.ollama_model, orchestrator_build=ORCHESTRATOR_BUILD,
        details={"language": language},
    )
    return WelcomeResponse(
        message=str(payload.get("message") or ""),
        name=str(payload.get("name") or ""),
        quick_actions=_quick_actions(language),
    )


@router.get("/health")
async def health(request: Request) -> dict:
    orchestrator = request.app.state.orchestrator
    return {
        "status": "ok",
        "ollama_available": is_available(),
        "orchestrator_build": ORCHESTRATOR_BUILD,
        "tools_loaded": orchestrator.tool_names,
        "memory": "sqlite-structured-v6+saved-conversations",
        "observability": "structured-jsonl+sqlserver-feedback-v3",
        "routing": "semantic-parser + deterministic-validator + tool-registry",
    }