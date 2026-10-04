from typing import Literal

from pydantic import BaseModel, Field


CONVERSATION_ID_PATTERN = r"^c_[a-fA-F0-9]{24}$"
MESSAGE_ID_PATTERN = r"^m_[a-fA-F0-9]{24}$"


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, max_length=8000,
        description="The employee's current question or message",
    )
    # Omit to start a NEW conversation. Send the id returned by a previous
    # /chat response (or from /chat/conversations) to continue that chat.
    conversation_id: str | None = Field(default=None, pattern=CONVERSATION_ID_PATTERN)
    # Kept for backward compatibility. The server keeps the history itself.
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Final assistant answer")
    sources: list[str] = Field(default_factory=list)
    conversation_id: str = Field(..., pattern=CONVERSATION_ID_PATTERN)
    message_id: str = Field(..., pattern=MESSAGE_ID_PATTERN)
    # Current title of the conversation, so the app can update its chat list.
    title: str = ""


class ResetChatResponse(BaseModel):
    status: Literal["ok"] = "ok"


class QuickAction(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    label: str = Field(..., min_length=1, max_length=80)
    icon: str = Field(default="", max_length=40)
    action_type: Literal["send_message", "prompt_input"] = "send_message"
    message: str = Field(default="", max_length=500)
    placeholder: str = Field(default="", max_length=120)


class WelcomeResponse(BaseModel):
    message: str
    name: str = ""
    quick_actions: list[QuickAction] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    conversation_id: str = Field(..., pattern=CONVERSATION_ID_PATTERN)
    message_id: str = Field(..., pattern=MESSAGE_ID_PATTERN)
    helpful: bool
    comment: str = Field(default="", max_length=2000)


class FeedbackResponse(BaseModel):
    status: Literal["ok"] = "ok"
    feedback_id: str


# ----------------------------------------------------------------------
# Saved conversations
# ----------------------------------------------------------------------

class ConversationSummary(BaseModel):
    conversation_id: str
    title: str = ""
    last_message_preview: str = ""
    message_count: int = 0
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0


class ConversationMessage(BaseModel):
    # Monotonic position. Use the smallest `seq` you have as `before`
    # to load older messages.
    seq: int
    # Assistant messages carry the m_... id used by /chat/feedback.
    message_id: str = ""
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: str = ""
    created_at: str
    updated_at: str
    message_count: int = 0
    messages: list[ConversationMessage] = Field(default_factory=list)
    has_more: bool = False
    next_before: int | None = None


class RenameConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class ConversationActionResponse(BaseModel):
    status: Literal["ok"] = "ok"