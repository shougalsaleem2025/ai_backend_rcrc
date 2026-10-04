from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    expires_at: float
    value: Any


class TTLCache:
    """Small in-process TTL cache for read-only enterprise API results.

    Cache keys never contain access tokens. Values are deep-copied on read/write
    so one request cannot mutate another request's cached payload.
    """

    DEFAULT_TTLS: dict[str, int] = {
        "get_employee_profile": 600,
        "get_employee_details": 600,
        "get_manager_details": 300,
        "search_employees": 300,
        "find_employee_by_role": 300,
        "get_vacation_types": 3600,
        "get_hr_letter_recipients": 3600,
        "get_hr_letter_types": 3600,
        "get_policy_details": 1800,
        "search_policies": 600,
        "get_news_list": 180,
        "get_news_details": 300,
        "get_announcements_list": 300,
        "get_announcement_details": 600,
        "get_mazaya_list": 600,
        "get_mazaya_details": 900,
        "get_leave_balance": 30,
        "get_vacation_history": 60,
        "get_my_meetings": 60,
        "get_meeting_list": 60,
        "get_today_attendance": 30,
        "get_employee_attendance": 60,
        "get_working_hours": 60,
        "get_tarasol_count": 30,
        "get_tarasol_transactions": 30,
    }

    NEVER_CACHE = {
        "get_employee_image",  # potentially large base64 payload
        "get_intersected_vacations",
        "get_reporting_employees",
        "get_subemployees_attendance",
    }

    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max(32, int(max_entries))
        self._data: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def key(*, requester: str, target: str, tool_name: str, arguments: dict[str, Any]) -> str:
        safe_args = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
        return "|".join([
            str(requester or "").strip().casefold(),
            str(target or "").strip().casefold(),
            str(tool_name or "").strip(),
            safe_args,
        ])

    def ttl_for(self, tool_name: str) -> int:
        if tool_name in self.NEVER_CACHE:
            return 0
        return int(self.DEFAULT_TTLS.get(tool_name, 0))

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._data.pop(key, None)
                return None
            return deepcopy(entry.value)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        ttl_seconds = int(ttl_seconds)
        if ttl_seconds <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._data[key] = _Entry(now + ttl_seconds, deepcopy(value))
            if len(self._data) > self.max_entries:
                self._purge_locked(now)
                if len(self._data) > self.max_entries:
                    oldest = sorted(self._data.items(), key=lambda item: item[1].expires_at)
                    for old_key, _ in oldest[: max(1, len(self._data) - self.max_entries)]:
                        self._data.pop(old_key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [k for k in self._data if k.startswith(prefix)]:
                self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _purge_locked(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        for key in [k for k, entry in self._data.items() if entry.expires_at <= now]:
            self._data.pop(key, None)
