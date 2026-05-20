from __future__ import annotations

from pathlib import Path

import pytest

from plume.services.local_llm_worker_client import LocalLLMWorkerClient


def _write_worker(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_worker_client_success(tmp_path):
    worker = _write_worker(
        tmp_path / "worker_success.py",
        """
import json,sys
for line in sys.stdin:
    req=json.loads(line)
    print(json.dumps({"request_id": req.get("request_id"), "ok": True, "content": "hello", "model": "fake", "elapsed_seconds": 0.01}), flush=True)
""",
    )
    client = LocalLLMWorkerClient(worker_script=worker, timeout_seconds=1)
    result = client.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.1, top_p=0.9)
    assert result["ok"] is True
    assert result["content"] == "hello"
    client.close()


def test_worker_client_crash(tmp_path):
    worker = _write_worker(
        tmp_path / "worker_crash.py",
        """
import os
os._exit(11)
""",
    )
    client = LocalLLMWorkerClient(worker_script=worker, timeout_seconds=1)
    result = client.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.1, top_p=0.9)
    assert result["ok"] is False
    assert "local LLM worker failed" in result["error"]
    client.close()


def test_worker_client_timeout(tmp_path):
    worker = _write_worker(
        tmp_path / "worker_sleep.py",
        """
import json,time,sys
for line in sys.stdin:
    _ = json.loads(line)
    time.sleep(2)
    print(json.dumps({"ok": True, "content": "late"}), flush=True)
""",
    )
    client = LocalLLMWorkerClient(worker_script=worker, timeout_seconds=0.2)
    result = client.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.1, top_p=0.9)
    assert result["ok"] is False
    assert "timeout" in result["error"]
    client.close()
