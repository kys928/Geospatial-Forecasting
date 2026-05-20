from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalLLMWorkerClient:
    def __init__(self, worker_script: str | None = None, timeout_seconds: float | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.worker_script = str(Path(worker_script) if worker_script else (repo_root / "scripts" / "local_llm_worker.py"))
        self.timeout_seconds = float(timeout_seconds or os.getenv("PLUME_LOCAL_LLM_WORKER_TIMEOUT_SECONDS", "45"))
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._process and self._process.poll() is None:
            return self._process
        self._process = subprocess.Popen(
            ["python", self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        logger.info("[llm] started local LLM worker pid=%s script=%s", self._process.pid, self.worker_script)
        return self._process

    def _kill_worker_locked(self) -> None:
        proc = self._process
        self._process = None
        if not proc:
            return
        if proc.poll() is None:
            proc.kill()
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            self._kill_worker_locked()

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        with self._lock:
            try:
                proc = self._ensure_worker()
                request = {
                    "request_id": str(uuid.uuid4()),
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()

                assert proc.stdout is not None
                start = time.perf_counter()
                line = ""
                while time.perf_counter() - start < self.timeout_seconds:
                    if proc.poll() is not None and not line:
                        code = proc.returncode
                        err = ""
                        if proc.stderr:
                            err = proc.stderr.read().strip()
                        self._process = None
                        return {"ok": False, "error": f"local LLM worker failed (exit_code={code}) {err}".strip()}
                    ready, _, _ = select.select([proc.stdout], [], [], 0.05)
                    if ready:
                        line = proc.stdout.readline()
                        if line:
                            break

                if not line:
                    logger.error("[llm] local LLM worker timeout after %.1fs", self.timeout_seconds)
                    self._kill_worker_locked()
                    return {"ok": False, "error": f"local LLM worker unavailable (timeout after {self.timeout_seconds}s)"}

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    logger.error("[llm] local LLM worker returned invalid JSON")
                    self._kill_worker_locked()
                    return {"ok": False, "error": "local LLM worker failed (invalid JSON response)"}

                return response
            except Exception as exc:
                logger.exception("[llm] local LLM worker call failed")
                self._kill_worker_locked()
                return {"ok": False, "error": f"local LLM worker unavailable: {exc}"}
