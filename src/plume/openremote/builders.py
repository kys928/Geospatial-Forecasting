from __future__ import annotations

from typing import Any

from plume.openremote.constants import *
from plume.openremote.models import (
    HazardSourceAssetModel,
    SensorAssetModel,
    ForecastRunAssetModel,
    ForecastZoneAssetModel,
    ORAssetPayload,
    ORAttribute,
    ORMetaItem,
    ORAttributeWrite,
    ORTimestampedAttributeWrite,
    ORPredictedDatapointWrite,
)


def _point_to_json(lat: float, lon: float) -> dict[str, float]:
    return {"latitude": lat, "longitude": lon}


def _append_attr(attrs: list[ORAttribute], name: str, value: Any, meta: list[ORMetaItem] | None = None) -> None:
    if value is None:
        return
    attrs.append(ORAttribute(name=name, value=value, meta=meta or []))


def build_hazard_source_asset_payload(model: HazardSourceAssetModel) -> ORAssetPayload:
    attrs: list[ORAttribute] = []

    _append_attr(attrs, ATTR_SOURCE_ID, model.source_id)
    _append_attr(attrs, ATTR_LOCATION, _point_to_json(model.location.latitude, model.location.longitude))
    _append_attr(attrs, ATTR_POLLUTANT_TYPE, model.pollutant_type)
    _append_attr(attrs, ATTR_RELEASE_RATE, model.release_rate)
    _append_attr(attrs, ATTR_RELEASE_HEIGHT, model.release_height)
    _append_attr(attrs, ATTR_SOURCE_STATUS, model.source_status.value)
    _append_attr(attrs, ATTR_LAST_OBSERVATION_TIME, model.last_observation_time.isoformat() if model.last_observation_time else None)
    _append_attr(attrs, ATTR_SCENARIO_METADATA, model.scenario_metadata)

    _append_attr(attrs, ATTR_WIND_U, model.wind_u)
    _append_attr(attrs, ATTR_WIND_V, model.wind_v)
    _append_attr(attrs, ATTR_WIND_SPEED, model.wind_speed)
    _append_attr(attrs, ATTR_WIND_DIRECTION, model.wind_direction)
    _append_attr(attrs, ATTR_STABILITY_CLASS, model.stability_class)
    _append_attr(attrs, ATTR_BOUNDARY_LAYER_HEIGHT, model.boundary_layer_height)

    return ORAssetPayload(
        id=model.asset_id,
        name=model.name,
        type=ASSET_TYPE_HAZARD_SOURCE,
        realm=model.realm,
        parent_id=model.parent_id,
        attributes=attrs,
    )


def build_sensor_asset_payload(model: SensorAssetModel, datapoint_retention_days: int = 30) -> ORAssetPayload:
    attrs: list[ORAttribute] = []

    _append_attr(attrs, ATTR_SENSOR_ID, model.sensor_id)
    _append_attr(attrs, ATTR_LOCATION, _point_to_json(model.location.latitude, model.location.longitude))
    _append_attr(attrs, ATTR_SENSOR_TYPE, model.sensor_type)
    _append_attr(
        attrs,
        ATTR_OBSERVED_VALUE,
        model.observed_value,
        meta=[
            ORMetaItem(name=META_STORE_DATA_POINTS, value=True),
            ORMetaItem(name=META_DATA_POINTS_MAX_AGE_DAYS, value=datapoint_retention_days),
        ],
    )
    _append_attr(attrs, ATTR_OBSERVED_UNIT, model.observed_unit)
    _append_attr(attrs, ATTR_POLLUTANT_TYPE, model.pollutant_type)
    _append_attr(attrs, ATTR_QUALITY_FLAG, model.quality_flag.value)
    _append_attr(attrs, ATTR_LAST_OBSERVATION_TIME, model.last_observation_time.isoformat() if model.last_observation_time else None)
    _append_attr(attrs, ATTR_OBSERVATION_METADATA, model.observation_metadata)

    return ORAssetPayload(
        id=model.asset_id,
        name=model.name,
        type=ASSET_TYPE_SENSOR,
        realm=model.realm,
        parent_id=model.parent_id,
        attributes=attrs,
    )


def build_forecast_run_asset_payload(model: ForecastRunAssetModel) -> ORAssetPayload:
    attrs: list[ORAttribute] = []

    _append_attr(attrs, ATTR_FORECAST_RUN_ID, model.forecast_run_id)
    _append_attr(attrs, ATTR_SESSION_ID, model.session_id)
    _append_attr(attrs, ATTR_BACKEND_NAME, model.backend_name)
    _append_attr(attrs, ATTR_MODEL_NAME, model.model_name)
    _append_attr(attrs, ATTR_MODEL_VERSION, model.model_version)
    _append_attr(attrs, ATTR_RUN_STATUS, model.run_status.value)
    _append_attr(attrs, ATTR_ISSUED_AT, model.issued_at.isoformat())
    _append_attr(attrs, ATTR_HORIZON_SECONDS, model.horizon_seconds)
    _append_attr(attrs, ATTR_SOURCE_ASSET_ID, model.source_asset_id)
    _append_attr(attrs, ATTR_SITE_ASSET_ID, model.site_asset_id)

    _append_attr(attrs, ATTR_MAX_CONCENTRATION, model.max_concentration)
    _append_attr(attrs, ATTR_MEAN_CONCENTRATION, model.mean_concentration)
    _append_attr(attrs, ATTR_AFFECTED_CELLS_ABOVE_THRESHOLD, model.affected_cells_above_threshold)
    _append_attr(attrs, ATTR_AFFECTED_AREA_M2, model.affected_area_m2)
    _append_attr(attrs, ATTR_AFFECTED_AREA_HECTARES, model.affected_area_hectares)
    _append_attr(attrs, ATTR_DOMINANT_SPREAD_DIRECTION, model.dominant_spread_direction)
    _append_attr(attrs, ATTR_THRESHOLD_USED, model.threshold_used)
    _append_attr(attrs, ATTR_CONFIDENCE_SCORE, model.confidence_score)
    _append_attr(attrs, ATTR_EXPLANATION_TEXT, model.explanation_text)
    _append_attr(attrs, ATTR_ALERT_LEVEL, model.alert_level.value)

    _append_attr(attrs, ATTR_PLUME_FOOTPRINT_GEOJSON, model.plume_footprint_geojson)
    _append_attr(
        attrs,
        ATTR_CENTROID,
        _point_to_json(model.centroid.latitude, model.centroid.longitude) if model.centroid else None,
    )
    _append_attr(
        attrs,
        ATTR_BOUNDING_BOX,
        model.bounding_box.model_dump() if model.bounding_box else None,
    )
    _append_attr(attrs, ATTR_GEOJSON_URL, str(model.geojson_url) if model.geojson_url else None)
    _append_attr(attrs, ATTR_RASTER_METADATA, model.raster_metadata)
    _append_attr(attrs, ATTR_GRID_SPEC, model.grid_spec)
    _append_attr(attrs, ATTR_SCENARIO_SNAPSHOT, model.scenario_snapshot)
    _append_attr(attrs, ATTR_EXECUTION_METADATA, model.execution_metadata)
    _append_attr(attrs, ATTR_EXTERNAL_ARTIFACT_REF, model.external_artifact_ref)

    return ORAssetPayload(
        id=model.asset_id,
        name=model.name,
        type=ASSET_TYPE_FORECAST_RUN,
        realm=model.realm,
        parent_id=model.parent_id,
        attributes=attrs,
    )


def build_forecast_zone_asset_payload(
    model: ForecastZoneAssetModel,
    auto_apply_predicted: bool = False,
) -> ORAssetPayload:
    attrs: list[ORAttribute] = []

    predicted_meta = [
        ORMetaItem(name=META_HAS_PREDICTED_DATA_POINTS, value=True),
        ORMetaItem(name=META_APPLY_PREDICTED_DATA_POINTS, value=auto_apply_predicted),
    ]

    _append_attr(attrs, ATTR_ZONE_ID, model.zone_id)
    _append_attr(attrs, ATTR_ZONE_NAME, model.zone_name)
    _append_attr(attrs, ATTR_ZONE_GEOMETRY, model.zone_geometry)
    _append_attr(attrs, ATTR_ZONE_TYPE, model.zone_type)

    _append_attr(attrs, ATTR_PREDICTED_MAX_CONCENTRATION, model.predicted_max_concentration, meta=predicted_meta)
    _append_attr(attrs, ATTR_PREDICTED_MEAN_CONCENTRATION, model.predicted_mean_concentration, meta=predicted_meta)
    _append_attr(attrs, ATTR_PREDICTED_ARRIVAL_TIME, model.predicted_arrival_time.isoformat() if model.predicted_arrival_time else None)
    _append_attr(attrs, ATTR_PREDICTED_PEAK_TIME, model.predicted_peak_time.isoformat() if model.predicted_peak_time else None)
    _append_attr(attrs, ATTR_PREDICTED_CLEAR_TIME, model.predicted_clear_time.isoformat() if model.predicted_clear_time else None)
    _append_attr(attrs, ATTR_RISK_LEVEL, model.risk_level.value)
    _append_attr(attrs, ATTR_LATEST_FORECAST_RUN_ID, model.latest_forecast_run_id)

    return ORAssetPayload(
        id=model.asset_id,
        name=model.name,
        type=ASSET_TYPE_FORECAST_ZONE,
        realm=model.realm,
        parent_id=model.parent_id,
        attributes=attrs,
    )


def build_sensor_observation_write(
    asset_id: str,
    value: float,
    observed_at,
) -> list[ORTimestampedAttributeWrite]:
    return [
        ORTimestampedAttributeWrite(
            asset_id=asset_id,
            attribute_name=ATTR_OBSERVED_VALUE,
            value=value,
            timestamp=observed_at,
        ),
        ORTimestampedAttributeWrite(
            asset_id=asset_id,
            attribute_name=ATTR_LAST_OBSERVATION_TIME,
            value=observed_at.isoformat(),
            timestamp=observed_at,
        ),
    ]


def build_forecast_run_status_writes(
    asset_id: str,
    run_status: str,
    issued_at,
) -> list[ORAttributeWrite]:
    return [
        ORAttributeWrite(asset_id=asset_id, attribute_name=ATTR_RUN_STATUS, value=run_status),
        ORAttributeWrite(asset_id=asset_id, attribute_name=ATTR_ISSUED_AT, value=issued_at.isoformat()),
    ]


def build_zone_predicted_concentration_write(
    asset_id: str,
    datapoints: list[tuple],
) -> ORPredictedDatapointWrite:
    """
    datapoints: list[(timestamp, value)]
    """
    from plume.openremote.models import ORPredictedDatapoint

    return ORPredictedDatapointWrite(
        asset_id=asset_id,
        attribute_name=ATTR_PREDICTED_MAX_CONCENTRATION,
        datapoints=[
            ORPredictedDatapoint(timestamp=ts, value=value)
            for ts, value in datapoints
        ],
    )
