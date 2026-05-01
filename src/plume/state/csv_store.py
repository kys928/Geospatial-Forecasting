from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.observation import Observation
from plume.state.base import BaseStateStore


class CsvStateStore(BaseStateStore):
    SESSION_COLUMNS = [
        "session_id",
        "backend_name",
        "model_name",
        "status",
        "created_at",
        "updated_at",
        "last_error",
        "metadata_json",
        "runtime_metadata_json",
        "state_json",
    ]
    LATEST_COLUMNS = ["session_id", "latest_forecast_id", "latest_forecast_artifact_dir", "updated_at"]

    def __init__(self, store_dir: str | Path) -> None:
        self._root = Path(store_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._sessions_csv = self._root / "sessions.csv"
        self._latest_csv = self._root / "session_latest_forecasts.csv"

    def create_session(self, session: BackendSession, state: BackendState) -> None:
        sessions = self._read_sessions()
        sessions[session.session_id] = self._serialize_session(session, state)
        self._write_csv_atomic(self._sessions_csv, self.SESSION_COLUMNS, list(sessions.values()))

    def get_session(self, session_id: str) -> BackendSession | None:
        row = self._read_sessions().get(session_id)
        return self._deserialize_session(row)[0] if row else None

    def save_session(self, session: BackendSession) -> None:
        sessions = self._read_sessions()
        existing = sessions.get(session.session_id)
        if existing is None:
            return
        state_json = existing.get("state_json", "")
        sessions[session.session_id] = self._serialize_session(session, self._state_from_json(session.session_id, state_json))
        self._write_csv_atomic(self._sessions_csv, self.SESSION_COLUMNS, list(sessions.values()))

    def get_state(self, session_id: str) -> BackendState | None:
        row = self._read_sessions().get(session_id)
        return self._deserialize_session(row)[1] if row else None

    def save_state(self, session_id: str, state: BackendState) -> None:
        sessions = self._read_sessions()
        existing = sessions.get(session_id)
        if existing is None:
            return
        session, _ = self._deserialize_session(existing)
        sessions[session_id] = self._serialize_session(session, state)
        self._write_csv_atomic(self._sessions_csv, self.SESSION_COLUMNS, list(sessions.values()))

    def list_sessions(self) -> list[BackendSession]:
        return [self._deserialize_session(row)[0] for row in self._read_sessions().values()]

    def delete_session(self, session_id: str) -> None:
        sessions = self._read_sessions()
        sessions.pop(session_id, None)
        self._write_csv_atomic(self._sessions_csv, self.SESSION_COLUMNS, list(sessions.values()))

        latest = self._read_latest()
        latest.pop(session_id, None)
        self._write_csv_atomic(self._latest_csv, self.LATEST_COLUMNS, list(latest.values()))

    def save_latest_forecast_linkage(self, session_id: str, forecast_id: str, artifact_dir: str | None) -> None:
        latest = self._read_latest()
        latest[session_id] = {
            "session_id": session_id,
            "latest_forecast_id": forecast_id,
            "latest_forecast_artifact_dir": artifact_dir or "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_csv_atomic(self._latest_csv, self.LATEST_COLUMNS, list(latest.values()))

    def get_latest_forecast_linkage(self, session_id: str) -> dict[str, str] | None:
        return self._read_latest().get(session_id)

    def _read_sessions(self) -> dict[str, dict[str, str]]:
        return self._read_csv(self._sessions_csv, "session_id")

    def _read_latest(self) -> dict[str, dict[str, str]]:
        return self._read_csv(self._latest_csv, "session_id")

    def _read_csv(self, path: Path, key: str) -> dict[str, dict[str, str]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            return {row[key]: row for row in rows if row.get(key)}

    def _write_csv_atomic(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _serialize_session(self, session: BackendSession, state: BackendState) -> dict[str, str]:
        return {
            "session_id": session.session_id,
            "backend_name": session.backend_name,
            "model_name": session.model_name or "",
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "last_error": session.last_error or "",
            "metadata_json": json.dumps(session.metadata, separators=(",", ":")),
            "runtime_metadata_json": json.dumps(session.runtime_metadata, separators=(",", ":")),
            "state_json": json.dumps(self._state_to_dict(state), separators=(",", ":")),
        }

    def _deserialize_session(self, row: dict[str, str]) -> tuple[BackendSession, BackendState]:
        session_id = row["session_id"]
        session = BackendSession(
            session_id=session_id,
            backend_name=row["backend_name"],
            model_name=row["model_name"] or None,
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row.get("metadata_json") or "{}"),
            last_error=row.get("last_error") or None,
            runtime_metadata=json.loads(row.get("runtime_metadata_json") or "{}"),
        )
        return session, self._state_from_json(session_id, row.get("state_json", ""))

    def _state_from_json(self, session_id: str, payload: str) -> BackendState:
        if not payload:
            return BackendState(session_id=session_id, last_update_time=datetime.now(timezone.utc), observation_count=0, state_version=0)
        raw = json.loads(payload)
        observations = [
            Observation(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                value=float(item["value"]),
                source_type=item["source_type"],
                pollutant_type=item.get("pollutant_type"),
                metadata=item.get("metadata") or {},
            )
            for item in raw.get("recent_observations", [])
        ]
        return BackendState(
            session_id=session_id,
            last_update_time=datetime.fromisoformat(raw["last_update_time"]),
            observation_count=int(raw["observation_count"]),
            state_version=int(raw["state_version"]),
            internal_state=raw.get("internal_state") or {},
            recent_observations=observations,
            last_prediction_time=datetime.fromisoformat(raw["last_prediction_time"]) if raw.get("last_prediction_time") else None,
            last_ingest_time=datetime.fromisoformat(raw["last_ingest_time"]) if raw.get("last_ingest_time") else None,
            last_observation_time=datetime.fromisoformat(raw["last_observation_time"]) if raw.get("last_observation_time") else None,
            status_message=raw.get("status_message", "initialized"),
            metadata=raw.get("metadata") or {},
        )

    def _state_to_dict(self, state: BackendState) -> dict[str, object]:
        return {
            "last_update_time": state.last_update_time.isoformat(),
            "observation_count": state.observation_count,
            "state_version": state.state_version,
            "internal_state": state.internal_state,
            "recent_observations": [
                {
                    "timestamp": o.timestamp.isoformat(),
                    "latitude": o.latitude,
                    "longitude": o.longitude,
                    "value": o.value,
                    "pollutant_type": o.pollutant_type,
                    "source_type": o.source_type,
                    "metadata": o.metadata,
                }
                for o in state.recent_observations
            ],
            "last_prediction_time": state.last_prediction_time.isoformat() if state.last_prediction_time else None,
            "last_ingest_time": state.last_ingest_time.isoformat() if state.last_ingest_time else None,
            "last_observation_time": state.last_observation_time.isoformat() if state.last_observation_time else None,
            "status_message": state.status_message,
            "metadata": state.metadata,
        }
