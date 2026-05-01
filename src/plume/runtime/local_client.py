from __future__ import annotations

from dataclasses import replace

from plume.runtime.client import ForecastRuntimeClient, RuntimeObservationIngestResult
from plume.services.forecast_service import ForecastRunResult, ForecastService
from plume.services.online_forecast_service import OnlineForecastService
from plume.schemas.backend_session import BackendSession
from plume.schemas.update_result import UpdateResult


class LocalForecastRuntimeClient(ForecastRuntimeClient):
    def __init__(
        self,
        *,
        forecast_service: ForecastService,
        online_forecast_service: OnlineForecastService,
        backend_config: dict[str, object],
    ) -> None:
        self._forecast_service = forecast_service
        self._online_forecast_service = online_forecast_service
        self._backend_config = backend_config

    def run_batch_forecast(self, payload: dict | None) -> ForecastRunResult:
        payload = payload or {}
        default_scenario = self._forecast_service.config.load_scenario()
        latitude = float(payload.get("latitude", default_scenario.latitude))
        longitude = float(payload.get("longitude", default_scenario.longitude))
        emissions_rate = float(payload.get("emissions_rate", default_scenario.emissions_rate))
        start = payload.get("start", default_scenario.start)
        end = payload.get("end", default_scenario.end)
        pollution_type = payload.get("pollution_type", default_scenario.pollution_type)
        duration = float(payload.get("duration", default_scenario.duration))
        release_height = float(payload.get("release_height", default_scenario.release_height))
        scenario = replace(
            default_scenario,
            source=(latitude, longitude),
            latitude=latitude,
            longitude=longitude,
            emissions_rate=emissions_rate,
            start=start,
            end=end,
            pollution_type=pollution_type,
            duration=duration,
            release_height=release_height,
        )
        return self._forecast_service.run_forecast(scenario=scenario, run_name=payload.get("run_name"))

    def create_session(self, payload: dict | None) -> BackendSession:
        payload = payload or {}
        backend_name = payload.get("backend_name") or self._backend_config.get("default_backend", "convlstm_online")
        return self._online_forecast_service.create_session(
            backend_name=str(backend_name),
            model_name=payload.get("model_name"),
            metadata=payload.get("metadata") or {},
        )

    def list_sessions(self) -> list[BackendSession]:
        return self._online_forecast_service.list_sessions()

    def get_session(self, session_id: str) -> BackendSession:
        return self._online_forecast_service.get_session(session_id)

    def get_session_state(self, session_id: str) -> dict[str, object]:
        return self._online_forecast_service.get_state_summary(session_id)

    def ingest_observations(self, session_id: str, payload_dict: dict[str, object]) -> RuntimeObservationIngestResult:
        observations_payload = payload_dict.get("observations", [])
        batch = self._online_forecast_service.normalize_observation_batch(session_id, observations_payload)
        state = self._online_forecast_service.ingest_observations(batch)

        update_result = None
        if bool(self._backend_config.get("auto_update_on_ingest", True)):
            update_result = self._online_forecast_service.update_session(session_id)

        return RuntimeObservationIngestResult(state=state, auto_update_result=update_result)

    def update_session(self, session_id: str) -> UpdateResult:
        return self._online_forecast_service.update_session(session_id)

    def predict_session(self, session_id: str, payload: dict | None) -> ForecastRunResult:
        request = self._online_forecast_service.build_prediction_request(session_id=session_id, payload=payload)
        return self._online_forecast_service.predict(request)

    def get_latest_session_forecast_result(self, session_id: str) -> ForecastRunResult:
        return self._online_forecast_service.get_latest_forecast_result(session_id)
