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
        if timeout_seconds is None:
            self.timeout_seconds = float(os.getenv("PLUME_LOCAL_LLM_WORKER_TIMEOUT_SECONDS", "120"))
        else:
            self.timeout_seconds = float(timeout_seconds)
        self.startup_timeout_seconds = float(os.getenv("PLUME_LOCAL_LLM_WORKER_STARTUP_TIMEOUT_SECONDS", "240"))
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_tail: dict[int, list[str]] = {}

    def _start_stderr_drain_thread(self, proc: subprocess.Popen[str]) -> None:
        def _drain() -> None:
            if proc.stderr is None:
                return
            tail: list[str] = []
            for raw_line in proc.stderr:
                line = raw_line.rstrip("\n")
                if line:
                    logger.debug("[llm-worker-stderr] %s", line)
                    tail.append(line)
                    if len(tail) > 20:
                        tail.pop(0)
                    self._stderr_tail[proc.pid] = tail.copy()

        thread = threading.Thread(target=_drain, name=f"llm-worker-stderr-{proc.pid}", daemon=True)
        thread.start()

    def _read_json_line_locked(self, proc: subprocess.Popen[str], timeout_seconds: float, phase: str) -> dict[str, Any]:
        assert proc.stdout is not None
        start = time.perf_counter()
        while True:
            if proc.poll() is not None:
                code = proc.returncode
                stderr_summary = " | ".join(self._stderr_tail.get(proc.pid or -1, []))
                suffix = f" stderr={stderr_summary}" if stderr_summary else ""
                raise RuntimeError(f"local LLM worker exited during {phase} (exit_code={code}){suffix}")
            elapsed = time.perf_counter() - start
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise TimeoutError(f"local LLM worker {phase} timeout after {timeout_seconds}s")
            ready, _, _ = select.select([proc.stdout], [], [], min(0.05, remaining))
            if not ready:
                continue
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    code = proc.returncode
                    stderr_summary = " | ".join(self._stderr_tail.get(proc.pid or -1, []))
                    suffix = f" stderr={stderr_summary}" if stderr_summary else ""
                    raise RuntimeError(f"local LLM worker exited during {phase} (exit_code={code}){suffix}")
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"local LLM worker {phase} returned invalid JSON") from exc

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._process and self._process.poll() is None:
            return self._process
        proc = subprocess.Popen(
            ["python", self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_stderr_drain_thread(proc)
        self._process = proc
        logger.info("[llm] started local LLM worker pid=%s script=%s", proc.pid, self.worker_script)

        warmup_request = {"type": "warmup", "request_id": str(uuid.uuid4())}
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(warmup_request) + "\n")
        proc.stdin.flush()

        try:
            warmup_response = self._read_json_line_locked(proc, self.startup_timeout_seconds, "warmup")
        except TimeoutError:
            self._kill_worker_locked()
            raise RuntimeError(f"local LLM worker warmup timeout after {self.startup_timeout_seconds}s")
        except RuntimeError as exc:
            self._kill_worker_locked()
            message = str(exc)
            if "invalid JSON" in message:
                raise RuntimeError("local LLM worker warmup returned invalid JSON") from exc
            raise

        if warmup_response.get("type") != "warmup":
            self._kill_worker_locked()
            raise RuntimeError("local LLM worker warmup returned unexpected response")

        if warmup_response.get("ok") is not True:
            self._kill_worker_locked()
            raise RuntimeError(f"local LLM worker warmup failed: {warmup_response.get('error') or 'unknown error'}")

        logger.info(
            "[llm] local LLM worker warmed up pid=%s model=%s elapsed=%s",
            proc.pid,
            warmup_response.get("model", "unknown"),
            warmup_response.get("elapsed_seconds", "unknown"),
        )
        return proc

    def _kill_worker_locked(self) -> None:
        proc = self._process
        self._process = None
        if not proc:
            return
        self._stderr_tail.pop(proc.pid or -1, None)
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

                response = self._read_json_line_locked(proc, self.timeout_seconds, "request")
                return response
            except TimeoutError:
                logger.error("[llm] local LLM worker timeout after %.1fs", self.timeout_seconds)
                self._kill_worker_locked()
                return {"ok": False, "error": f"local LLM worker unavailable (timeout after {self.timeout_seconds}s)"}
            except Exception as exc:
                logger.exception("[llm] local LLM worker call failed")
                self._kill_worker_locked()
                return {"ok": False, "error": f"local LLM worker unavailable: {exc}"}
