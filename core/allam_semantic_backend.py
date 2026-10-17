
"""Optional ALLaM backend for Sanad semantic parsing only.

Uses the existing comparison virtual environment.
Keeps the model loaded between requests.
Does not call internal RCRC APIs.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from typing import Any


class AllamWorker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._log_file = None
        atexit.register(self.close)

    def _pump(self, stdout: Any) -> None:
        try:
            for line in stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _start(self) -> None:
        python_exe = os.environ.get(
            "SANAD_ALLAM_PYTHON", ""
        ).strip()
        worker_script = os.environ.get(
            "SANAD_ALLAM_WORKER", ""
        ).strip()

        if not python_exe or not worker_script:
            raise RuntimeError(
                "Set SANAD_ALLAM_PYTHON and SANAD_ALLAM_WORKER"
            )

        if not Path(python_exe).is_file():
            raise FileNotFoundError(python_exe)

        if not Path(worker_script).is_file():
            raise FileNotFoundError(worker_script)

        self._responses = queue.Queue()

        log_path = (
            Path(tempfile.gettempdir())
            / "sanad_allam_nlu.log"
        )

        self._log_file = open(
            log_path, "a", encoding="utf-8"
        )

        self._process = subprocess.Popen(
            [python_exe, "-u", worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_file,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert self._process.stdout is not None

        self._reader = threading.Thread(
            target=self._pump,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._reader.start()

    def close(self) -> None:
        process = self._process
        self._process = None

        if process is not None and process.poll() is None:
            process.terminate()

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        with self._lock:
            if (
                self._process is None
                or self._process.poll() is not None
            ):
                self.close()
                self._start()

            assert self._process is not None
            assert self._process.stdin is not None

            try:
                payload = {
                    "messages": messages,
                    "max_new_tokens": 480,
                }

                self._process.stdin.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    ) + "\n"
                )
                self._process.stdin.flush()

                timeout = max(
                    60,
                    int(os.getenv(
                        "SANAD_ALLAM_TIMEOUT",
                        "1200"
                    )),
                )

                line = self._responses.get(
                    timeout=timeout
                )

                if line is None:
                    raise RuntimeError(
                        "ALLaM worker exited; check log"
                    )

                data = json.loads(line)

                if not data.get("ok"):
                    raise RuntimeError(
                        str(data.get("error") or
                            "Unknown ALLaM error")
                    )

                return str(data.get("content") or "")

            except Exception:
                self.close()
                raise


_worker = AllamWorker()


def semantic_chat(
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Same response format as core.llm_client.chat."""

    provider = os.getenv(
        "SANAD_SEMANTIC_PROVIDER",
        "qwen",
    ).strip().lower()

    if provider != "allam":
        from core.llm_client import chat
        return chat(messages)

    result = _worker.generate(messages)

    return {
        "message": {
            "content": result
        }
    }
