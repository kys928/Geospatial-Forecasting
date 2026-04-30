from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ForecastCreateRequest(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    emissions_rate: float | None = None
    pollution_type: str | None = None
    duration: float | None = None
    release_height: float | None = None
    start: datetime | None = None
    end: datetime | None = None
    run_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastCreateResponse(BaseModel):
    forecast_id: str
    issued_at: datetime | str
    model: str
    model_version: str | None = None
    artifacts: dict[str, Any]
    runtime: dict[str, Any] | None = None
    publishing: dict[str, Any] | None = None


class ForecastListResponse(BaseModel):
    forecasts: list[dict[str, Any]]


class ServiceInfoResponse(BaseModel):
    service_id: str
    label: str
    version: str
    capabilities: list[str]
    artifact_store: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]
    details: dict[str, Any] = Field(default_factory=dict)


class SessionCreateRequest(BaseModel):
    backend_name: str | None = None
    model_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionPredictionRequest(BaseModel):
    scenario: dict[str, Any] | None = None
    grid_spec: dict[str, Any] | None = None
    horizon_seconds: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationPayload(BaseModel):
    timestamp: datetime | str | None = None
    latitude: float | None = None
    longitude: float | None = None
    value: float | None = None
    source_type: str | None = None
    pollutant_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationIngestRequest(BaseModel):
    observations: list[ObservationPayload]
