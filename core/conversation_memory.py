"""core/conversation_memory.py



Conversation memory for Sanad (سَنَد).



Two concerns live here:

1. Structured execution state per conversation (focus, pending flows, lists).

2. Saved conversations: the user can list old chats, open one, and continue it.



Changes from the previous version:

- Pending enterprise actions (leave request / HR letter) now keep their

  action/stage/data/options across turns. Previously normalize_state dropped

  them, so multi-step requests restarted after the first question.

- New `conversations` table (per user) with title, preview, counts.

- Messages carry a message_id so feedback works on reloaded history.

- LIKE escaping fixed (SQLite requires a single-character ESCAPE).

- Resetting context no longer deletes the saved chat history.

"""



from __future__ import annotations



import json

import os

import re

import sqlite3

import threading

from copy import deepcopy

from datetime import datetime, timezone

from typing import Any





STATE_SCHEMA_VERSION = 9

CONTEXT_SCHEMA_VERSION = "v76"

MAX_STORED_MESSAGES = 1000          # per conversation, for history display

MAX_MESSAGE_CHARS = 20000

MAX_ENTITY_HISTORY = 12

MAX_GENERAL_TURNS = 10

MAX_LIST_ITEMS = 100

TITLE_MAX_CHARS = 60

PREVIEW_MAX_CHARS = 120

ACTION_PENDING_KIND = "enterprise_action"

IMAGE_MARKER = "[EMPLOYEE_IMAGE_DATA]"

IMAGE_PLACEHOLDER = "[EMPLOYEE_IMAGE]"



_CONVERSATION_ID_RE = re.compile(r"^c_[a-f0-9]{24}$")



_WEAK_TITLES = {

    "هلا", "هلا فيك", "اهلا", "أهلا", "اهلين", "أهلين", "مرحبا", "مرحبًا",

    "السلام عليكم", "صباح الخير", "مساء الخير", "هاي", "هلو",

    "شكرا", "شكرًا", "تمام", "اوكي", "أوكي", "طيب",

    "hi", "hello", "hey", "ok", "okay", "thanks",

}





def _now() -> str:

    return datetime.now(timezone.utc).isoformat()





def _like_escape(value: str) -> str:

    return (

        str(value or "")

        .replace("\\\\", "\\\\\\\\")

        .replace("%", "\\\\%")

        .replace("_", "\\\\_")

    )





def _clean_line(text: Any) -> str:

    return " ".join(str(text or "").split()).strip()





def _make_title(text: Any) -> str:

    value = _clean_line(text).strip(" ؟?!.,،")

    if value.startswith(IMAGE_MARKER) or value.startswith(IMAGE_PLACEHOLDER):

        return ""

    if len(value) > TITLE_MAX_CHARS:

        value = value[: TITLE_MAX_CHARS - 1].rstrip() + "…"

    return value





def _make_preview(text: Any) -> str:

    raw = str(text or "")

    if raw.startswith(IMAGE_MARKER) or raw.startswith(IMAGE_PLACEHOLDER):

        return "🖼️ صورة"

    value = _clean_line(raw)

    if len(value) > PREVIEW_MAX_CHARS:

        value = value[: PREVIEW_MAX_CHARS - 1].rstrip() + "…"

    return value





def _is_weak_title(title: str) -> bool:

    value = _clean_line(title).strip(" ؟?!.,،").casefold()

    return not value or value in {x.casefold() for x in _WEAK_TITLES}





def _storable_content(text: Any) -> str:

    """Keep stored history readable and bounded.



    Image answers are base64 data URLs; truncating them would corrupt them, so

    history keeps a placeholder instead. The client renders it as an image card

    and can re-request the image if needed.

    """

    value = str(text or "").strip()

    if value.startswith(IMAGE_MARKER):

        return IMAGE_PLACEHOLDER

    return value[:MAX_MESSAGE_CHARS]





def _strip_secrets(value: Any) -> Any:

    blocked = {

        "access_token", "authorization", "token", "headers",

        "x-api-key", "api_key", "password",

    }

    if isinstance(value, dict):

        return {

            str(k): _strip_secrets(v)

            for k, v in value.items()

            if str(k).lower() not in blocked

        }

    if isinstance(value, list):

        return [_strip_secrets(v) for v in value]

    return value





def _bounded_employee(row: Any) -> dict[str, Any]:

    if not isinstance(row, dict):

        return {}

    allowed = (

        "username", "userid", "name", "name_ar", "name_en",

        "job_title", "job_title_ar", "job_title_en",

        "department", "department_ar", "department_en",

        "general_department", "general_department_ar", "general_department_en",

        "general_department_source",

        "sector_ar", "sector_en", "email", "mobile", "extension",

    )

    return {k: row.get(k) for k in allowed if row.get(k) not in (None, "")}





def _empty_pending() -> dict[str, Any]:

    return {"kind": "", "query": "", "candidates": [], "request": {}, "message": ""}





def default_state() -> dict[str, Any]:

    return {

        "schema_version": STATE_SCHEMA_VERSION,

        "context_schema_version": CONTEXT_SCHEMA_VERSION,

        "language": "ar",

        # Verified signed-in employee. Independent from conversational focus.

        "authenticated_user": {},

        "focus": {"current": {}, "previous": {}},

        "entity_history": [],

        "relation": {"predicate": "", "subject": {}, "object": {}, "subject_kind": ""},

        "pending": _empty_pending(),

        "last_request": {

            "intent": "", "operation": "", "fields": [], "targets": [],

            "period": "none", "start_date": "", "end_date": "", "time_query": "",

            "query": "", "role_query": "", "department_query": "", "amount": None,

        },

        "meeting_context": {

            "subject_kind": "", "employee": {}, "period": "none",

            "start_date": "", "end_date": "", "time_query": "", "mode": "",

        },

        "task_flow": {

            "kind": "", "stage": "", "period": "none", "time_window": "",

            "duration_minutes": 0, "target": {}, "original_message": "",

        },

        "attendance_context": {"period": "none", "start_date": "", "end_date": "", "mode": ""},

        # Remember the last verified balance only within this conversation.
        "leave_context": {"displayed_balance": None, "source": ""},

        "list_state": {
            "intent": "", "kind": "", "department": "", "items": [], "query": "",
            "selected_index": 0, "selected_detail_intent": "",
            "selected_item": {}, "selected_full_data": {},
        },

        "content_focus": {
            "detail_intent": "", "base_intent": "", "selected_index": 0,
            "item": {}, "full_data": {},
        },

        # Last user-generated writing artifact. This is intentionally separate
        # from employee/content focus so short follow-ups such as "ترجمه",
        # "اختصره" and "خله رسمي أكثر" modify the same draft instead of
        # being re-routed as a new enterprise lookup.
        "writing_context": {
            "kind": "", "draft": "", "language": "",
            "recipient_name": "", "source_message": "",
        },

        "directory_context": {

            "type": "", "general_department_key": "", "general_department": "",

            "department_key": "", "department": "", "unit_type": "",

            "count": 0, "female": 0, "male": 0, "unknown": 0,

        },

        "role_lookup": {"role_query": "", "employee": {}},

        "domain_state": {},

        "general_history": [],

    }





def _as_dict(value: Any) -> dict[str, Any]:

    return value if isinstance(value, dict) else {}





def _as_int(value: Any, default: int = 0) -> int:

    try:

        return int(value or 0)

    except (TypeError, ValueError):

        return default





def normalize_state(value: Any) -> dict[str, Any]:

    raw = _as_dict(value)

    if _as_int(raw.get("schema_version")) != STATE_SCHEMA_VERSION:

        return default_state()



    state = default_state()

    state["context_schema_version"] = str(raw.get("context_schema_version") or CONTEXT_SCHEMA_VERSION)

    language = str(raw.get("language") or "ar")

    state["language"] = language if language in {"ar", "en"} else "ar"

    state["authenticated_user"] = _bounded_employee(raw.get("authenticated_user"))



    focus = _as_dict(raw.get("focus"))

    state["focus"] = {

        "current": _bounded_employee(focus.get("current")),

        "previous": _bounded_employee(focus.get("previous")),

    }



    history = raw.get("entity_history") if isinstance(raw.get("entity_history"), list) else []

    clean_history: list[dict[str, Any]] = []

    seen: set[str] = set()

    for row in reversed(history):

        item = _bounded_employee(row)

        uid = str(item.get("username") or item.get("userid") or "").strip().casefold()

        if not uid or uid in seen:

            continue

        seen.add(uid)

        clean_history.append(item)

        if len(clean_history) >= MAX_ENTITY_HISTORY:

            break

    state["entity_history"] = list(reversed(clean_history))



    relation = _as_dict(raw.get("relation"))

    state["relation"] = {

        "predicate": str(relation.get("predicate") or ""),

        "subject": _bounded_employee(relation.get("subject")),

        "object": _bounded_employee(relation.get("object")),

        "subject_kind": str(relation.get("subject_kind") or ""),

    }



    pending = _as_dict(raw.get("pending"))

    candidates = pending.get("candidates") if isinstance(pending.get("candidates"), list) else []

    state["pending"] = {

        "kind": str(pending.get("kind") or ""),

        "query": str(pending.get("query") or ""),

        "candidates": [_bounded_employee(x) for x in candidates[:10] if isinstance(x, dict)],

        "request": _strip_secrets(_as_dict(pending.get("request"))),

        "message": str(pending.get("message") or ""),

    }

    # FIX: keep the multi-step action workflow fields (leave request, HR letter).

    if state["pending"]["kind"] == ACTION_PENDING_KIND:

        options = pending.get("options") if isinstance(pending.get("options"), list) else []

        state["pending"].update({

            "action": str(pending.get("action") or ""),

            "stage": str(pending.get("stage") or ""),

            "data": _strip_secrets(deepcopy(_as_dict(pending.get("data")))),

            "options": [_strip_secrets(x) for x in options[:MAX_LIST_ITEMS] if isinstance(x, dict)],

        })



    last_request = _as_dict(raw.get("last_request"))

    amount = last_request.get("amount")

    try:

        amount = float(amount) if amount is not None else None

    except (TypeError, ValueError):

        amount = None

    state["last_request"] = {

        "intent": str(last_request.get("intent") or ""),

        "operation": str(last_request.get("operation") or ""),

        "fields": list(last_request.get("fields") or [])[:24],

        "targets": _strip_secrets(list(last_request.get("targets") or [])[:8]),

        "period": str(last_request.get("period") or "none"),

        "start_date": str(last_request.get("start_date") or ""),

        "end_date": str(last_request.get("end_date") or ""),

        "time_query": str(last_request.get("time_query") or ""),

        "query": str(last_request.get("query") or ""),

        "role_query": str(last_request.get("role_query") or ""),

        "department_query": str(last_request.get("department_query") or ""),

        "amount": amount,

    }



    meeting_context = _as_dict(raw.get("meeting_context"))

    state["meeting_context"] = {

        "subject_kind": str(meeting_context.get("subject_kind") or ""),

        "employee": _bounded_employee(meeting_context.get("employee")),

        "period": str(meeting_context.get("period") or "none"),

        "start_date": str(meeting_context.get("start_date") or ""),

        "end_date": str(meeting_context.get("end_date") or ""),

        "time_query": str(meeting_context.get("time_query") or ""),

        "mode": str(meeting_context.get("mode") or ""),

    }



    task_flow = _as_dict(raw.get("task_flow"))

    state["task_flow"] = {

        "kind": str(task_flow.get("kind") or ""),

        "stage": str(task_flow.get("stage") or ""),

        "period": str(task_flow.get("period") or "none"),

        "time_window": str(task_flow.get("time_window") or ""),

        "duration_minutes": max(0, min(_as_int(task_flow.get("duration_minutes")), 480)),

        "target": _strip_secrets(_as_dict(task_flow.get("target"))),

        "original_message": str(task_flow.get("original_message") or "")[:1000],

    }



    attendance_context = _as_dict(raw.get("attendance_context"))

    state["attendance_context"] = {

        "period": str(attendance_context.get("period") or "none"),

        "start_date": str(attendance_context.get("start_date") or ""),

        "end_date": str(attendance_context.get("end_date") or ""),

        "mode": str(attendance_context.get("mode") or ""),

    }



    leave_context = _as_dict(raw.get("leave_context"))
    displayed_balance = leave_context.get("displayed_balance")
    if isinstance(displayed_balance, bool) or not isinstance(displayed_balance, (str, float, int)):
        displayed_balance = None
    if isinstance(displayed_balance, str):
        displayed_balance = displayed_balance[:80]
    state["leave_context"] = {
        "displayed_balance": displayed_balance,
        "source": "leave_balance_lookup" if leave_context.get("source") == "leave_balance_lookup" else "",
    }

    list_state = _as_dict(raw.get("list_state"))

    items = list_state.get("items") if isinstance(list_state.get("items"), list) else []

    state["list_state"] = {

        "intent": str(list_state.get("intent") or ""),

        "kind": str(list_state.get("kind") or ""),

        "department": str(list_state.get("department") or ""),

        "items": [_strip_secrets(x) for x in items[:MAX_LIST_ITEMS] if isinstance(x, dict)],

        "query": str(list_state.get("query") or ""),

        "selected_index": max(0, _as_int(list_state.get("selected_index"))),

        "selected_detail_intent": str(list_state.get("selected_detail_intent") or ""),

        "selected_item": _strip_secrets(_as_dict(list_state.get("selected_item"))),

        "selected_full_data": _strip_secrets(_as_dict(list_state.get("selected_full_data"))),

    }



    content_focus = _as_dict(raw.get("content_focus"))

    state["content_focus"] = {

        "detail_intent": str(content_focus.get("detail_intent") or ""),

        "base_intent": str(content_focus.get("base_intent") or ""),

        "selected_index": max(0, _as_int(content_focus.get("selected_index"))),

        "item": _strip_secrets(_as_dict(content_focus.get("item"))),

        "full_data": _strip_secrets(_as_dict(content_focus.get("full_data"))),

    }



    writing_context = _as_dict(raw.get("writing_context"))

    state["writing_context"] = {

        "kind": str(writing_context.get("kind") or "")[:32],

        "draft": str(writing_context.get("draft") or "")[:MAX_MESSAGE_CHARS],

        "language": str(writing_context.get("language") or "")[:8],

        "recipient_name": str(writing_context.get("recipient_name") or "")[:300],

        "source_message": str(writing_context.get("source_message") or "")[:2000],

    }



    directory_context = _as_dict(raw.get("directory_context"))

    state["directory_context"] = {

        "type": str(directory_context.get("type") or ""),

        "general_department_key": str(directory_context.get("general_department_key") or ""),

        "general_department": str(directory_context.get("general_department") or ""),

        "department_key": str(directory_context.get("department_key") or ""),

        "department": str(directory_context.get("department") or ""),

        "unit_type": str(directory_context.get("unit_type") or ""),

        "count": _as_int(directory_context.get("count")),

        "female": _as_int(directory_context.get("female")),

        "male": _as_int(directory_context.get("male")),

        "unknown": _as_int(directory_context.get("unknown")),

    }



    role_lookup = _as_dict(raw.get("role_lookup"))

    state["role_lookup"] = {

        "role_query": str(role_lookup.get("role_query") or ""),

        "employee": _bounded_employee(role_lookup.get("employee")),

    }



    state["domain_state"] = _strip_secrets(_as_dict(raw.get("domain_state")))



    general_history = raw.get("general_history") if isinstance(raw.get("general_history"), list) else []

    clean_general: list[dict[str, str]] = []

    for turn in general_history[-MAX_GENERAL_TURNS:]:

        if not isinstance(turn, dict):

            continue

        role = str(turn.get("role") or "")

        content = str(turn.get("content") or "").strip()

        if role in {"user", "assistant"} and content:

            clean_general.append({"role": role, "content": content[:8000]})

    state["general_history"] = clean_general

    return state





class ConversationMemory:

    """SQLite-backed conversation state + saved conversation history.



    For multi-node production move this to SQL Server / Redis; the public

    methods are the contract the rest of the app uses.

    """



    def __init__(self, path: str | None = None) -> None:

        self.path = path or os.getenv(

            "APP_CONVERSATION_DB",

            os.path.join(os.getcwd(), "data", "conversation_memory.sqlite3"),

        )

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

        self._lock = threading.RLock()

        self._init_db()



    # ------------------------------------------------------------------

    # Keys

    # ------------------------------------------------------------------



    @staticmethod

    def memory_key(username: str, conversation_id: str) -> str:

        user = str(username or "").strip().casefold()

        cid = str(conversation_id or "").strip().lower()

        return f"{user}:{cid}" if user and cid else ""



    @staticmethod

    def _split_key(memory_key: str) -> tuple[str, str]:

        key = str(memory_key or "")

        if ":" not in key:

            return "", ""

        user, cid = key.rsplit(":", 1)

        return user, cid



    @staticmethod

    def is_valid_conversation_id(value: str) -> bool:

        return bool(_CONVERSATION_ID_RE.fullmatch(str(value or "").strip().lower()))



    # ------------------------------------------------------------------

    # Connection / schema

    # ------------------------------------------------------------------



    def _connect(self) -> sqlite3.Connection:

        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)

        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute("PRAGMA busy_timeout=30000")

        conn.execute("PRAGMA foreign_keys=ON")

        return conn



    def _init_db(self) -> None:

        with self._lock, self._connect() as conn:

            conn.execute(

                """

                CREATE TABLE IF NOT EXISTS conversation_state (

                    memory_key TEXT PRIMARY KEY,

                    state_json TEXT NOT NULL,

                    updated_at TEXT NOT NULL

                )

                """

            )

            conn.execute(

                """

                CREATE TABLE IF NOT EXISTS conversation_messages (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    memory_key TEXT NOT NULL,

                    role TEXT NOT NULL,

                    content TEXT NOT NULL,

                    created_at TEXT NOT NULL

                )

                """

            )

            columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversation_messages)")}

            if "message_id" not in columns:

                conn.execute(

                    "ALTER TABLE conversation_messages ADD COLUMN message_id TEXT NOT NULL DEFAULT ''"

                )

            conn.execute(

                "CREATE INDEX IF NOT EXISTS idx_conv_messages_key_id "

                "ON conversation_messages(memory_key, id)"

            )

            conn.execute(

                """

                CREATE TABLE IF NOT EXISTS conversations (

                    username TEXT NOT NULL,

                    conversation_id TEXT NOT NULL,

                    title TEXT NOT NULL DEFAULT '',

                    title_is_auto INTEGER NOT NULL DEFAULT 1,

                    last_message_preview TEXT NOT NULL DEFAULT '',

                    message_count INTEGER NOT NULL DEFAULT 0,

                    created_at TEXT NOT NULL,

                    updated_at TEXT NOT NULL,

                    PRIMARY KEY (username, conversation_id)

                )

                """

            )

            conn.execute(

                "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated "

                "ON conversations(username, updated_at DESC)"

            )

            self._backfill_conversations(conn)



    def _backfill_conversations(self, conn: sqlite3.Connection) -> None:

        """Register chats that existed before the conversations table."""

        rows = conn.execute(

            """

            SELECT memory_key, MIN(created_at) AS first_at, MAX(created_at) AS last_at,

                   COUNT(*) AS n

            FROM conversation_messages

            GROUP BY memory_key

            """

        ).fetchall()

        for row in rows:

            key = str(row["memory_key"])

            user, cid = self._split_key(key)

            if not user or user == "anonymous" or not self.is_valid_conversation_id(cid):

                continue

            exists = conn.execute(

                "SELECT 1 FROM conversations WHERE username = ? AND conversation_id = ?",

                (user, cid),

            ).fetchone()

            if exists:

                continue

            first_user = conn.execute(

                "SELECT content FROM conversation_messages WHERE memory_key = ? AND role = 'user' "

                "ORDER BY id LIMIT 1",

                (key,),

            ).fetchone()

            last_msg = conn.execute(

                "SELECT content FROM conversation_messages WHERE memory_key = ? ORDER BY id DESC LIMIT 1",

                (key,),

            ).fetchone()

            conn.execute(

                """

                INSERT OR IGNORE INTO conversations

                    (username, conversation_id, title, title_is_auto, last_message_preview,

                     message_count, created_at, updated_at)

                VALUES (?, ?, ?, 1, ?, ?, ?, ?)

                """,

                (

                    user, cid,

                    _make_title(first_user["content"] if first_user else ""),

                    _make_preview(last_msg["content"] if last_msg else ""),

                    int(row["n"] or 0),

                    str(row["first_at"] or _now()),

                    str(row["last_at"] or _now()),

                ),

            )



    # ------------------------------------------------------------------

    # State

    # ------------------------------------------------------------------



    @staticmethod

    def default_state() -> dict[str, Any]:

        return default_state()



    def get_state(self, memory_key: str) -> dict[str, Any]:

        if not memory_key:

            return default_state()

        with self._lock, self._connect() as conn:

            row = conn.execute(

                "SELECT state_json FROM conversation_state WHERE memory_key = ?",

                (memory_key,),

            ).fetchone()

        if not row:

            return default_state()

        try:

            return normalize_state(json.loads(row["state_json"]))

        except (TypeError, json.JSONDecodeError):

            return default_state()



    def save_state(self, memory_key: str, state: dict[str, Any]) -> None:

        if not memory_key:

            return

        clean = normalize_state(_strip_secrets(deepcopy(state)))

        payload = json.dumps(clean, ensure_ascii=False, default=str)

        with self._lock, self._connect() as conn:

            conn.execute(

                """

                INSERT INTO conversation_state(memory_key, state_json, updated_at)

                VALUES (?, ?, ?)

                ON CONFLICT(memory_key) DO UPDATE SET

                    state_json = excluded.state_json,

                    updated_at = excluded.updated_at

                """,

                (memory_key, payload, _now()),

            )



    def clear_state(self, memory_key: str) -> None:

        """Forget the working context of one chat but keep its saved messages."""

        if not memory_key:

            return

        with self._lock, self._connect() as conn:

            conn.execute("DELETE FROM conversation_state WHERE memory_key = ?", (memory_key,))



    def clear_state_prefix(self, prefix: str) -> None:

        if not prefix:

            return

        pattern = _like_escape(prefix) + "%"

        with self._lock, self._connect() as conn:

            conn.execute(

                "DELETE FROM conversation_state WHERE memory_key LIKE ? ESCAPE '\\\\'",

                (pattern,),

            )



    # ------------------------------------------------------------------

    # Messages

    # ------------------------------------------------------------------



    def append_message(

        self,

        memory_key: str,

        role: str,

        content: str,

        *,

        message_id: str = "",

    ) -> None:

        if not memory_key or role not in {"user", "assistant"}:

            return

        text = _storable_content(content)

        if not text:

            return

        now = _now()

        user, cid = self._split_key(memory_key)

        with self._lock, self._connect() as conn:

            conn.execute(

                "INSERT INTO conversation_messages(memory_key, role, content, created_at, message_id) "

                "VALUES (?, ?, ?, ?, ?)",

                (memory_key, role, text, now, str(message_id or "")),

            )

            conn.execute(

                """

                DELETE FROM conversation_messages

                WHERE memory_key = ? AND id NOT IN (

                    SELECT id FROM conversation_messages

                    WHERE memory_key = ? ORDER BY id DESC LIMIT ?

                )

                """,

                (memory_key, memory_key, MAX_STORED_MESSAGES),

            )

            if not user or not cid:

                return

            row = conn.execute(

                "SELECT title, title_is_auto FROM conversations WHERE username = ? AND conversation_id = ?",

                (user, cid),

            ).fetchone()

            if row is None:

                return

            title = str(row["title"] or "")

            if role == "user" and int(row["title_is_auto"] or 0) and _is_weak_title(title):

                candidate = _make_title(text)

                if candidate and (not title or not _is_weak_title(candidate)):

                    title = candidate

            conn.execute(

                """

                UPDATE conversations

                SET title = ?,

                    last_message_preview = ?,

                    message_count = (SELECT COUNT(*) FROM conversation_messages WHERE memory_key = ?),

                    updated_at = ?

                WHERE username = ? AND conversation_id = ?

                """,

                (title, _make_preview(text), memory_key, now, user, cid),

            )



    def tag_last_assistant_message(self, username: str, conversation_id: str, message_id: str) -> None:

        """Attach the response message_id to the newest untagged assistant row.



        Lets the HTTP layer assign feedback ids even when the orchestrator

        appends messages without an id.

        """

        key = self.memory_key(username, conversation_id)

        if not key or not message_id:

            return

        with self._lock, self._connect() as conn:

            row = conn.execute(

                "SELECT id FROM conversation_messages WHERE memory_key = ? AND role = 'assistant' "

                "ORDER BY id DESC LIMIT 1",

                (key,),

            ).fetchone()

            if row is not None:

                conn.execute(

                    "UPDATE conversation_messages SET message_id = ? WHERE id = ? AND message_id = ''",

                    (str(message_id), int(row["id"])),

                )



    def recent_messages(self, memory_key: str, limit: int = 20) -> list[dict[str, str]]:

        """Recent turns used as dialogue context for the language model."""

        if not memory_key:

            return []

        count = max(1, min(int(limit), 100))

        with self._lock, self._connect() as conn:

            rows = conn.execute(

                "SELECT role, content FROM conversation_messages WHERE memory_key = ? "

                "ORDER BY id DESC LIMIT ?",

                (memory_key, count),

            ).fetchall()

        return [

            {"role": str(row["role"]), "content": str(row["content"])}

            for row in reversed(rows)

        ]



    # ------------------------------------------------------------------

    # Saved conversations

    # ------------------------------------------------------------------



    def ensure_conversation(self, username: str, conversation_id: str) -> bool:

        """Create the conversation row if missing. Returns True when created."""

        user = str(username or "").strip().casefold()

        cid = str(conversation_id or "").strip().lower()

        if not user or user == "anonymous" or not self.is_valid_conversation_id(cid):

            return False

        now = _now()

        with self._lock, self._connect() as conn:

            cursor = conn.execute(

                "INSERT OR IGNORE INTO conversations(username, conversation_id, created_at, updated_at) "

                "VALUES (?, ?, ?, ?)",

                (user, cid, now, now),

            )

            return cursor.rowcount == 1



    def conversation_exists(self, username: str, conversation_id: str) -> bool:

        user = str(username or "").strip().casefold()

        cid = str(conversation_id or "").strip().lower()

        if not user or not cid:

            return False

        with self._lock, self._connect() as conn:

            row = conn.execute(

                "SELECT 1 FROM conversations WHERE username = ? AND conversation_id = ?",

                (user, cid),

            ).fetchone()

        return row is not None



    @staticmethod

    def _conversation_row(row: sqlite3.Row) -> dict[str, Any]:

        return {

            "conversation_id": str(row["conversation_id"]),

            "title": str(row["title"] or ""),

            "last_message_preview": str(row["last_message_preview"] or ""),

            "message_count": int(row["message_count"] or 0),

            "created_at": str(row["created_at"]),

            "updated_at": str(row["updated_at"]),

        }



    def list_conversations(

        self,

        username: str,

        *,

        limit: int = 20,

        offset: int = 0,

        query: str = "",

    ) -> tuple[list[dict[str, Any]], int]:

        user = str(username or "").strip().casefold()

        if not user:

            return [], 0

        limit = max(1, min(int(limit), 100))

        offset = max(0, int(offset))

        where = "username = ? AND message_count > 0"

        params: list[Any] = [user]

        search = _clean_line(query)

        if search:

            pattern = "%" + _like_escape(search) + "%"

            where += (

                " AND (title LIKE ? ESCAPE '\\\\' OR last_message_preview LIKE ? ESCAPE '\\\\')"

            )

            params.extend([pattern, pattern])

        with self._lock, self._connect() as conn:

            total = int(conn.execute(

                f"SELECT COUNT(*) FROM conversations WHERE {where}", params

            ).fetchone()[0])

            rows = conn.execute(

                f"""

                SELECT conversation_id, title, last_message_preview, message_count,

                       created_at, updated_at

                FROM conversations

                WHERE {where}

                ORDER BY updated_at DESC

                LIMIT ? OFFSET ?

                """,

                [*params, limit, offset],

            ).fetchall()

        return [self._conversation_row(row) for row in rows], total



    def get_conversation(self, username: str, conversation_id: str) -> dict[str, Any] | None:

        user = str(username or "").strip().casefold()

        cid = str(conversation_id or "").strip().lower()

        if not user or not cid:

            return None

        with self._lock, self._connect() as conn:

            row = conn.execute(

                """

                SELECT conversation_id, title, last_message_preview, message_count,

                       created_at, updated_at

                FROM conversations WHERE username = ? AND conversation_id = ?

                """,

                (user, cid),

            ).fetchone()

        return self._conversation_row(row) if row else None



    def get_messages(

        self,

        username: str,

        conversation_id: str,

        *,

        limit: int = 50,

        before_seq: int | None = None,

    ) -> tuple[list[dict[str, Any]], bool]:

        """Return messages oldest-first plus whether older messages exist.



        Paging: pass the `seq` of the oldest message you already have as

        `before_seq` to load the previous page.

        """

        key = self.memory_key(username, conversation_id)

        if not key:

            return [], False

        limit = max(1, min(int(limit), 200))

        params: list[Any] = [key]

        where = "memory_key = ?"

        if before_seq:

            where += " AND id < ?"

            params.append(int(before_seq))

        with self._lock, self._connect() as conn:

            rows = conn.execute(

                f"""

                SELECT id, message_id, role, content, created_at

                FROM conversation_messages

                WHERE {where}

                ORDER BY id DESC

                LIMIT ?

                """,

                [*params, limit + 1],

            ).fetchall()

        has_more = len(rows) > limit

        rows = rows[:limit]

        messages = [

            {

                "seq": int(row["id"]),

                "message_id": str(row["message_id"] or ""),

                "role": str(row["role"]),

                "content": str(row["content"]),

                "created_at": str(row["created_at"]),

            }

            for row in reversed(rows)

        ]

        return messages, has_more



    def rename_conversation(self, username: str, conversation_id: str, title: str) -> bool:

        user = str(username or "").strip().casefold()

        cid = str(conversation_id or "").strip().lower()

        clean = _clean_line(title)[:120]

        if not user or not cid or not clean:

            return False

        with self._lock, self._connect() as conn:

            cursor = conn.execute(

                "UPDATE conversations SET title = ?, title_is_auto = 0, updated_at = updated_at "

                "WHERE username = ? AND conversation_id = ?",

                (clean, user, cid),

            )

            return cursor.rowcount == 1



    def delete_conversation(self, username: str, conversation_id: str) -> bool:

        user = str(username or "").strip().casefold()

        cid = str(conversation_id or "").strip().lower()

        key = self.memory_key(user, cid)

        if not key:

            return False

        with self._lock, self._connect() as conn:

            cursor = conn.execute(

                "DELETE FROM conversations WHERE username = ? AND conversation_id = ?",

                (user, cid),

            )

            conn.execute("DELETE FROM conversation_state WHERE memory_key = ?", (key,))

            conn.execute("DELETE FROM conversation_messages WHERE memory_key = ?", (key,))

            return cursor.rowcount == 1



    # ------------------------------------------------------------------

    # Backward-compatible full resets (state + messages + saved row)

    # ------------------------------------------------------------------



    def reset(self, memory_key: str) -> None:

        user, cid = self._split_key(memory_key)

        if user and cid:

            self.delete_conversation(user, cid)

            return

        if not memory_key:

            return

        with self._lock, self._connect() as conn:

            conn.execute("DELETE FROM conversation_state WHERE memory_key = ?", (memory_key,))

            conn.execute("DELETE FROM conversation_messages WHERE memory_key = ?", (memory_key,))



    def reset_prefix(self, prefix: str) -> None:

        if not prefix:

            return

        pattern = _like_escape(prefix) + "%"

        user = prefix[:-1] if prefix.endswith(":") else ""

        with self._lock, self._connect() as conn:

            conn.execute(

                "DELETE FROM conversation_state WHERE memory_key LIKE ? ESCAPE '\\\\'", (pattern,)

            )

            conn.execute(

                "DELETE FROM conversation_messages WHERE memory_key LIKE ? ESCAPE '\\\\'", (pattern,)

            )

            if user:

                conn.execute("DELETE FROM conversations WHERE username = ?", (user.casefold(),))