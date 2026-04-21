from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, ConfigDict


# ----------------------------
# Shared primitives
# ----------------------------

class ORAttributeRef(BaseModel):
    asset_id: str
    attribute_name: str


class ORMetaItem(BaseModel):
    name: str
    value: Any


class ORAttribute(BaseModel):
    """
    Project-side normalized representation of an OpenRemote attribute.
    We keep this simple and JSON-friendly.
    """
    name: str
    value: Any | None = None
    meta: list[ORMetaItem] = Field(default_factory=list)


class ORAssetPayload(BaseModel):
    """
    Project-side normalized asset payload for create/update calls.
    The exact JSON shape sent over HTTP can be adapted in builders/sink.
    """
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str
    type: str
    realm: str | None = None
    parent_id: str | None = None
    attributes: list[ORAttribute] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ORAttributeWrite(BaseModel):
    """
    Maps to OpenRemote single or bulk attribute write semantics.
    """
    asset_id: str
    attribute_name: str
    value: Any


class ORTimestampedAttributeWrite(BaseModel):
    """
    For timestamped writes when you want historical datapoints aligned to a supplied timestamp.
    """
    asset_id: str
    attribute_name: str
    value: Any
    timestamp: datetime


class ORPredictedDatapoint(BaseModel):
    timestamp: datetime
    value: Any


class ORPredictedDatapointWrite(BaseModel):
    asset_id: str
    attribute_name: str
    datapoints: list[ORPredictedDatapoint]


# ----------------------------
# Project asset type enums
# ----------------------------

class ProjectAssetType(str, Enum):
    SITE = "SiteAsset"
    HAZARD_SOURCE = "HazardSourceAsset"
    SENSOR = "SensorAsset"
    FORECAST_RUN = "ForecastRunAsset"
    FORECAST_ZONE = "ForecastZoneAsset"


class ForecastRunStatus(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    PREDICTING = "predicting"
    COMPLETED = "completed"
    ERROR = "error"


class SourceStatus(str, Enum):
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    INACTIVE = "inactive"


class AlertLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityFlag(str, Enum):
    OK = "ok"
    SUSPECT = "suspect"
    INVALID = "invalid"


# ----------------------------
# Domain-facing models
# ----------------------------

class GeoPoint(BaseModel):
    latitude: float
    longitude: float


class BoundingBox(BaseModel):
    min_latitude: float
    min_longitude: float
    max_latitude: float
    max_longitude: float


class HazardSourceAssetModel(BaseModel):
    asset_id: str | None = None
    name: str
    realm: str | None = None
    parent_id: str | None = None

    source_id: str
    location: GeoPoint
    pollutant_type: str
    release_rate: float | None = None
    release_height: float | None = None
    source_status: SourceStatus = SourceStatus.SUSPECTED
    last_observation_time: datetime | None = None
    scenario_metadata: dict[str, Any] = Field(default_factory=dict)

    wind_u: float | None = None
    wind_v: float | None = None
    wind_speed: float | None = None
    wind_direction: float | None = None
    stability_class: str | None = None
    boundary_layer_height: float | None = None


class SensorAssetModel(BaseModel):
    asset_id: str | None = None
    name: str
    realm: str | None = None
    parent_id: str | None = None

    sensor_id: str
    location: GeoPoint
    sensor_type: str
    observed_value: float | None = None
    observed_unit: str | None = None
    pollutant_type: str | None = None
    quality_flag: QualityFlag = QualityFlag.OK
    last_observation_time: datetime | None = None
    observation_metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastRunAssetModel(BaseModel):
    asset_id: str | None = None
    name: str
    realm: str | None = None
    parent_id: str | None = None

    forecast_run_id: str
    session_id: str
    backend_name: str
    model_name: str
    model_version: str | None = None
    run_status: ForecastRunStatus
    issued_at: datetime
    horizon_seconds: int | None = None
    source_asset_id: str | None = None
    site_asset_id: str | None = None

    max_concentration: float | None = None
    mean_concentration: float | None = None
    affected_cells_above_threshold: int | None = None
    affected_area_m2: float | None = None
    affected_area_hectares: float | None = None
    dominant_spread_direction: str | None = None
    threshold_used: float | None = None
    confidence_score: float | None = None
    explanation_text: str | None = None
    alert_level: AlertLevel = AlertLevel.NONE

    plume_footprint_geojson: dict[str, Any] | None = None
    centroid: GeoPoint | None = None
    bounding_box: BoundingBox | None = None

    geojson_url: HttpUrl | None = None
    raster_metadata: dict[str, Any] = Field(default_factory=dict)
    grid_spec: dict[str, Any] = Field(default_factory=dict)
    scenario_snapshot: dict[str, Any] = Field(default_factory=dict)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)
    external_artifact_ref: dict[str, Any] = Field(default_factory=dict)


class ForecastZoneAssetModel(BaseModel):
    asset_id: str | None = None
    name: str
    realm: str | None = None
    parent_id: str | None = None

    zone_id: str
    zone_name: str
    zone_geometry: dict[str, Any]
    zone_type: str

    predicted_max_concentration: float | None = None
    predicted_mean_concentration: float | None = None
    predicted_arrival_time: datetime | None = None
    predicted_peak_time: datetime | None = None
    predicted_clear_time: datetime | None = None
    risk_level: AlertLevel = AlertLevel.NONE
    latest_forecast_run_id: str | None = None