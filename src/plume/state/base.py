from __future__ import annotations

from abc import ABC, abstractmethod

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState


class BaseStateStore(ABC):
    @abstractmethod
    def create_session(self, session: BackendSession, state: BackendState) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_session(self, session_id: str) -> BackendSession | None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self, session_id: str) -> BackendState | None:
        raise NotImplementedError

    @abstractmethod
    def save_state(self, session_id: str, state: BackendState) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> list[BackendSession]:
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError
