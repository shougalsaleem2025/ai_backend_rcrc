from __future__ import annotations

import json
import os
import re
import sys
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Callable, TextIO

try:
    import pyodbc  # type: ignore
except Exception:  # pragma: no cover - optional outside production
    pyodbc = None  # type: ignore

_RIYADH_TZ = timezone(timedelta(hours=3))
_SECRET_KEYS = {
    "access_token", "authorization", "token", "headers", "x-api-key",
    "api_key", "password", "secret", "client_secret", "refresh_token",
}
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_console_lock = threading.RLock()
_console_installed = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().casefold()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return default


def _now_local() -> datetime:
    return datetime.now(_RIYADH_TZ)


def _safe_string(value: str) -> str:
    value = _BEARER_RE.sub("Bearer [REDACTED]", value)
    value = _JWT_RE.sub("[JWT_REDACTED]", value)
    return value


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.casefold() in _SECRET_KEYS:
                out[key_text] = "[REDACTED]"
            else:
                out[key_text] = _safe_value(item)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(v) for v in value]
    if isinstance(value, str):
        return _safe_string(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _safe_string(str(value))


class _TeeStream:
    def __init__(self, original: TextIO, stream_name: str, callback: Callable[[str, str], None]) -> None:
        self._original = original
        self._stream_name = stream_name
        self._callback = callback
        self._buffer = ""
        self._lock = threading.RLock()
        self._rcrc_combined_tee = True

    def write(self, data: str) -> int:
        text = str(data)
        with self._lock:
            written = self._original.write(text)
            self._original.flush()
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.strip():
                    self._callback(self._stream_name, line)
            return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        with self._lock:
            self._original.flush()
            if self._buffer.strip():
                line = self._buffer.rstrip("\r")
                self._buffer = ""
                self._callback(self._stream_name, line)

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self._original, "encoding", None)

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)



# Load backend environment variables for SQL feedback.
# override=False preserves real process/environment values in production.
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

class ObservabilityStore:
    """Production-safe structured observability.

    * One JSONL file per Riyadh day AND process id. This avoids multi-worker
      processes writing to the same file concurrently.
    * Every structured event carries pid/thread plus user/conversation/message
      identifiers supplied by the HTTP layer.
    * Secrets are recursively redacted.
    * Feedback remains in SQL Server when pyodbc is available.
    """

    backend_name = "jsonl-per-process+sqlserver-feedback-v3"

    def __init__(self, _legacy_sqlite_path: str | None = None) -> None:
        self._lock = threading.RLock()
        self.log_root = Path(os.getenv("APP_LOG_ROOT", "Logs")).resolve()
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.retention_days = max(1, _env_int("APP_LOG_RETENTION_DAYS", 30))
        self.capture_console = _env_bool("APP_CAPTURE_CONSOLE", True)

        self.sql_server = str(os.getenv("APP_SQL_SERVER", "")).strip()
        self.sql_database = str(os.getenv("APP_SQL_DATABASE", "RCRC_AI_Assistant")).strip()
        self.sql_driver = str(os.getenv("APP_SQL_DRIVER", "ODBC Driver 18 for SQL Server")).strip()
        self.sql_timeout_seconds = max(1, _env_int("APP_SQL_TIMEOUT_SECONDS", 10))
        self.trust_server_certificate = _env_bool("APP_SQL_TRUST_SERVER_CERTIFICATE", True)

        self._cleanup_old_logs()
        if self.capture_console:
            self.install_console_capture()

    def _log_path(self) -> Path:
        now = _now_local()
        month_dir = self.log_root / now.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)
        return month_dir / f"RCRC-AI-Assistant-{now:%Y-%m-%d}-pid-{os.getpid()}.jsonl"

    def _cleanup_old_logs(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        try:
            for path in self.log_root.rglob("RCRC-AI-Assistant-*.jsonl"):
                try:
                    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                    if modified < cutoff:
                        path.unlink(missing_ok=True)
                except OSError:
                    continue
        except OSError:
            pass

    def _write_record(self, record: dict[str, Any]) -> None:
        safe = _safe_value(record)
        safe.setdefault("timestamp_local", _now_local().isoformat(timespec="milliseconds"))
        safe.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        safe.setdefault("pid", os.getpid())
        safe.setdefault("thread_id", threading.get_ident())
        line = json.dumps(safe, ensure_ascii=False, default=str, separators=(",", ":"))
        with self._lock:
            path = self._log_path()
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()

    def _capture_console_line(self, stream_name: str, line: str) -> None:
        self._write_record({
            "log_type": "console",
            "level": "ERROR" if stream_name == "stderr" else "INFO",
            "event_type": "console_line",
            "stream": stream_name,
            "message": line,
        })

    def install_console_capture(self) -> None:
        global _console_installed
        with _console_lock:
            if _console_installed:
                return
            if not getattr(sys.stdout, "_rcrc_combined_tee", False):
                sys.stdout = _TeeStream(sys.stdout, "stdout", self._capture_console_line)  # type: ignore[assignment]
            if not getattr(sys.stderr, "_rcrc_combined_tee", False):
                sys.stderr = _TeeStream(sys.stderr, "stderr", self._capture_console_line)  # type: ignore[assignment]
            _console_installed = True

    def log_event(
        self,
        *,
        requester_username: str = "",
        conversation_id: str = "",
        message_id: str = "",
        event_type: str,
        endpoint: str = "",
        status: str = "success",
        duration_ms: int | float | None = None,
        model_name: str = "",
        orchestrator_build: str = "",
        details: dict[str, Any] | None = None,
    ) -> str:
        event_id = "evt_" + uuid.uuid4().hex
        status_text = str(status or "unknown").strip().casefold()
        self._write_record({
            "log_type": "audit",
            "level": "ERROR" if status_text in {"failed", "error"} else "INFO",
            "event_type": event_type,
            "status": status_text,
            "event_id": event_id,
            "requester_username": requester_username,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "endpoint": endpoint,
            "duration_ms": duration_ms,
            "model_name": model_name,
            "orchestrator_build": orchestrator_build,
            "details": details or {},
        })
        return event_id

    def log_application(self, *, event_type: str, status: str = "success", details: dict[str, Any] | None = None) -> None:
        self._write_record({
            "log_type": "application",
            "level": "INFO" if status in {"success", "ok", "completed"} else "ERROR",
            "event_type": event_type,
            "status": status,
            "details": details or {},
        })

    def log_error(
        self,
        *,
        event_type: str,
        error: Exception | str,
        requester_username: str = "",
        conversation_id: str = "",
        message_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        trace = ""
        error_type = "Error"
        if isinstance(error, Exception):
            error_type = error.__class__.__name__
            trace = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
        self._write_record({
            "log_type": "error",
            "level": "ERROR",
            "event_type": event_type,
            "status": "error",
            "requester_username": requester_username,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "error_type": error_type,
            "error": str(error),
            "traceback": trace,
            "details": details or {},
        })

    def _connection_string(self) -> str:
        parts = [
            f"DRIVER={{{self.sql_driver}}}",
            f"SERVER={self.sql_server}",
            f"DATABASE={self.sql_database}",
            "Trusted_Connection=yes",
            f"Connection Timeout={self.sql_timeout_seconds}",
        ]
        if self.sql_driver.casefold().startswith("odbc driver 18"):
            parts += ["Encrypt=yes", f"TrustServerCertificate={'yes' if self.trust_server_certificate else 'no'}"]
        return ";".join(parts) + ";"

    def _connect(self):
        if pyodbc is None:
            raise RuntimeError("pyodbc is not installed.")
        if not self.sql_server:
            raise RuntimeError("APP_SQL_SERVER is not configured.")
        return pyodbc.connect(self._connection_string(), timeout=self.sql_timeout_seconds, autocommit=False)

    def check_sql_connection(self) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                row = conn.cursor().execute("SELECT DB_NAME(), @@SERVERNAME").fetchone()
            return {
                "success": True,
                "database": str(row[0] if row else self.sql_database),
                "server": str(row[1] if row else self.sql_server),
                "driver": self.sql_driver,
            }
        except Exception as exc:
            self.log_error(event_type="sql_connection_check_failed", error=exc)
            return {"success": False, "server": self.sql_server, "database": self.sql_database, "error": str(exc)}

    def save_feedback(
        self,
        *,
        requester_username: str,
        conversation_id: str,
        message_id: str,
        helpful: bool,
        comment: str | None = None,
        model_name: str | None = None,
        orchestrator_build: str | None = None,
    ) -> str:
        username = str(requester_username or "").strip()
        conversation = str(conversation_id or "").strip()
        message = str(message_id or "").strip()
        if not username or not conversation or not message:
            raise ValueError("requester_username, conversation_id and message_id are required")

        feedback_id = str(uuid.uuid4())
        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                existing = cursor.execute(
                    "SELECT FeedbackId FROM dbo.AssistantFeedback WHERE RequesterUsername=? AND MessageId=?",
                    username, message,
                ).fetchone()
                if existing:
                    feedback_id = str(existing[0])
                    cursor.execute(
                        """UPDATE dbo.AssistantFeedback SET UpdatedAt=SYSUTCDATETIME(), ConversationId=?, Helpful=?, Comment=?, ModelName=?, OrchestratorBuild=? WHERE RequesterUsername=? AND MessageId=?""",
                        conversation, 1 if helpful else 0, str(comment or "").strip() or None,
                        str(model_name or "").strip() or None, str(orchestrator_build or "").strip() or None,
                        username, message,
                    )
                    operation = "updated"
                else:
                    cursor.execute(
                        """INSERT INTO dbo.AssistantFeedback (FeedbackId,CreatedAt,UpdatedAt,RequesterUsername,ConversationId,MessageId,Helpful,Comment,ModelName,OrchestratorBuild) VALUES (?,SYSUTCDATETIME(),SYSUTCDATETIME(),?,?,?,?,?,?,?)""",
                        feedback_id, username, conversation, message, 1 if helpful else 0,
                        str(comment or "").strip() or None, str(model_name or "").strip() or None,
                        str(orchestrator_build or "").strip() or None,
                    )
                    operation = "created"
                conn.commit()
        self.log_event(
            event_type="feedback_saved",
            requester_username=username,
            conversation_id=conversation,
            message_id=message,
            endpoint="POST /chat/feedback",
            details={"feedback_id": feedback_id, "operation": operation, "helpful": helpful},
        )
        return feedback_id
