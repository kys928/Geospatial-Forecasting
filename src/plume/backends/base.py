from __future__ import annotations

from abc import ABC, abstractmethod

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.update_result import UpdateResult


class BaseBackend(ABC):
    @abstractmethod
    def create_session(
        self,
        *,
        model_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BackendSession:
        raise NotImplementedError

    @abstractmethod
    def initialize_state(self, session: BackendSession) -> BackendState:
        raise NotImplementedError

    @abstractmethod
    def ingest_observations(self, state: BackendState, batch: ObservationBatch) -> BackendState:
        raise NotImplementedError

    @abstractmethod
    def update_state(self, state: BackendState) -> UpdateResult:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: BackendState, request: PredictionRequest) -> Forecast:
        raise NotImplementedError

    @abstractmethod
    def summarize_state(self, state: BackendState) -> dict[str, object]:
        raise NotImplementedError
