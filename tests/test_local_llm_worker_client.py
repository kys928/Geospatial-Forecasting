from __future__ import annotations

import json
from pathlib import Path

from plume.services.local_llm_worker_client import LocalLLMWorkerClient


def _write_worker(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_worker_client_success_and_warmup_once(tmp_path):
    count_file = tmp_path / "warmup_count.txt"
    worker = _write_worker(
        tmp_path / "worker_success.py",
        f"""
import json,sys
from pathlib import Path
count_file = Path(r"{count_file}")
for line in sys.stdin:
    req=json.loads(line)
    if req.get("type") == "warmup":
        count = int(count_file.read_text() if count_file.exists() else "0") + 1
        count_file.write_text(str(count))
        print(json.dumps({{"request_id": req.get("request_id"), "ok": True, "type": "warmup", "model": "fake", "elapsed_seconds": 0.01}}), flush=True)
        continue
    print(json.dumps({{"request_id": req.get("request_id"), "ok": True, "content": "hello", "model": "fake", "elapsed_seconds": 0.01}}), flush=True)
""",
    )
    client = LocalLLMWorkerClient(worker_script=worker, timeout_seconds=1)
    result_1 = client.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.1, top_p=0.9)
    result_2 = client.generate(messages=[{"role": "user", "content": "hi again"}], max_tokens=10, temperature=0.1, top_p=0.9)
    assert result_1["ok"] is True
    assert result_1["content"] == "hello"
    assert result_2["ok"] is True
    assert result_2["content"] == "hello"
    assert count_file.read_text() == "1"
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
    assert "local LLM worker" in result["error"]
    client.close()


def test_worker_client_timeout(tmp_path):
    worker = _write_worker(
        tmp_path / "worker_sleep.py",
        """
import json,time,sys
for line in sys.stdin:
    req = json.loads(line)
    if req.get("type") == "warmup":
        print(json.dumps({"request_id": req.get("request_id"), "ok": True, "type": "warmup", "model": "fake", "elapsed_seconds": 0.01}), flush=True)
        continue
    time.sleep(2)
    print(json.dumps({"ok": True, "content": "late"}), flush=True)
""",
    )
    client = LocalLLMWorkerClient(worker_script=worker, timeout_seconds=0.2)
    result = client.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.1, top_p=0.9)
    assert result["ok"] is False
    assert "timeout" in result["error"]
    client.close()
