from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.update_result import UpdateResult
from plume.services.forecast_service import ForecastRunResult


@dataclass(frozen=True)
class RuntimeObservationIngestResult:
    state: BackendState
    auto_update_result: UpdateResult | None


class ForecastRuntimeClient(Protocol):
    def run_batch_forecast(self, payload: dict | None) -> ForecastRunResult: ...

    def create_session(self, payload: dict | None) -> BackendSession: ...

    def list_sessions(self) -> list[BackendSession]: ...

    def get_session(self, session_id: str) -> BackendSession: ...

    def get_session_state(self, session_id: str) -> dict[str, object]: ...

    def ingest_observations(
        self,
        session_id: str,
        payload_dict: dict[str, object],
    ) -> RuntimeObservationIngestResult: ...

    def update_session(self, session_id: str) -> UpdateResult: ...

    def predict_session(self, session_id: str, payload: dict | None) -> ForecastRunResult: ...

    def get_latest_session_forecast_result(self, session_id: str) -> ForecastRunResult: ...
