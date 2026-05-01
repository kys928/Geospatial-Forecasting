from __future__ import annotations

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.state.base import BaseStateStore


class InMemoryStateStore(BaseStateStore):
    def __init__(self) -> None:
        self._sessions: dict[str, BackendSession] = {}
        self._states: dict[str, BackendState] = {}

    def create_session(self, session: BackendSession, state: BackendState) -> None:
        self._sessions[session.session_id] = session
        self._states[session.session_id] = state

    def get_session(self, session_id: str) -> BackendSession | None:
        return self._sessions.get(session_id)

    def save_session(self, session: BackendSession) -> None:
        if session.session_id in self._sessions:
            self._sessions[session.session_id] = session

    def get_state(self, session_id: str) -> BackendState | None:
        return self._states.get(session_id)

    def save_state(self, session_id: str, state: BackendState) -> None:
        if session_id in self._sessions:
            self._states[session_id] = state

    def list_sessions(self) -> list[BackendSession]:
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._states.pop(session_id, None)

    def save_latest_forecast_linkage(self, session_id: str, forecast_id: str, artifact_dir: str | None) -> None:
        return None

    def get_latest_forecast_linkage(self, session_id: str) -> dict[str, str] | None:
        return None
