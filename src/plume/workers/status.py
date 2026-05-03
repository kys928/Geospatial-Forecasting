from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def resolve_worker_status_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    return Path(os.getenv("PLUME_WORKER_STATUS_PATH", "artifacts/worker_status/worker_status.json"))


class WorkerStatusStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_worker_status_path(path)

    def write_status(self, payload: dict[str, object]) -> dict[str, object]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self.path)
        return payload

    def read_status(self) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed worker status JSON at {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Worker status JSON at {self.path} must be an object")
        return payload

    def update_status(self, **fields: Any) -> dict[str, object]:
        current = self.read_status() or {}
        current.update(fields)
        return self.write_status(current)
