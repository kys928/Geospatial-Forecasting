from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable
from urllib.parse import quote

from plume.openremote.builders import (
    build_forecast_run_asset_payload,
    build_hazard_source_asset_payload,
    build_sensor_asset_payload,
    build_sensor_observation_write,
    build_zone_predicted_concentration_write, build_forecast_zone_asset_payload,
)
from plume.openremote.models import (
    AlertLevel,
    ForecastRunAssetModel,
    ForecastRunStatus,
    ForecastZoneAssetModel,
    GeoPoint,
    HazardSourceAssetModel,
    QualityFlag,
    SensorAssetModel,
)
from plume.openremote.sink import OpenRemoteResultSink


class OpenRemotePublishingService:
    """
    Thin orchestration service which:
    1. converts internal forecast results into OpenRemote-facing asset payloads
    2. publishes them through OpenRemoteResultSink

    This is intentionally narrow and demo-friendly.
    """

    def __init__(
        self,
        sink: OpenRemoteResultSink,
        *,
        realm: str | None = None,
        default_site_asset_id: str | None = None,
        default_site_parent_id: str | None = None,
        forecast_asset_parent_id: str | None = None,
        source_asset_parent_id: str | None = None,
        sensor_asset_parent_id: str | None = None,
        zone_asset_parent_id: str | None = None,
        geojson_base_url: str | None = None,
    ) -> None:
        self.sink = sink
        self.realm = realm
        self.default_site_asset_id = default_site_asset_id
        self.default_site_parent_id = default_site_parent_id
        self.forecast_asset_parent_id = forecast_asset_parent_id
        self.source_asset_parent_id = source_asset_parent_id
        self.sensor_asset_parent_id = sensor_asset_parent_id
        self.zone_asset_parent_id = zone_asset_parent_id
        self.geojson_base_url = geojson_base_url.rstrip("/") if geojson_base_url else None

    # ----------------------------
    # Public API
    # ----------------------------

    async def publish_forecast_result(
        self,
        result: Any,
        *,
        source_asset_id: str | None = None,
        source_name: str | None = None,
        source_status: str = "suspected",
        pollutant_type: str | None = None,
        explanation_text: str | None = None,
        alert_level: str | None = None,
        publish_source_asset: bool = True,
        publish_forecast_asset: bool = True,
    ) -> dict[str, Any]:
        """
        Publish a forecast result into OpenRemote.

        Returns a small summary dict with any created/updated asset identifiers.
        """
        scenario = self._get_attr(result.forecast, "scenario")
        grid_spec = self._get_attr(result.forecast, "grid_spec")
        execution_metadata = self._normalize_mapping(getattr(result, "execution_metadata", {}) or {})
        summary = self._extract_summary_stats(result)

        issued_at = result.issued_at
        forecast_run_id = result.forecast_id

        source_location = GeoPoint(
            latitude=float(self._get_attr(scenario, "latitude")),
            longitude=float(self._get_attr(scenario, "longitude")),
        )

        # 1) optional source asset
        source_publish_result: dict[str, Any] | None = None
        resolved_source_asset_id = source_asset_id

        if publish_source_asset:
            source_model = HazardSourceAssetModel(
                asset_id=source_asset_id,
                name=source_name or f"Source {forecast_run_id}",
                realm=self.realm,
                parent_id=self.source_asset_parent_id or self.default_site_parent_id,
                source_id=source_asset_id or forecast_run_id,
                location=source_location,
                pollutant_type=pollutant_type or self._safe_str(self._get_attr(scenario, "pollution_type", None)) or "unknown",
                release_rate=self._get_optional_float(scenario, "emissions_rate"),
                release_height=self._get_optional_float(scenario, "release_height"),
                source_status=source_status,  # pydantic accepts enum-compatible str
                last_observation_time=issued_at,
                scenario_metadata=self._scenario_metadata(scenario),
            )
            source_payload = build_hazard_source_asset_payload(source_model)
            source_publish_result = await self.sink.upsert_asset(source_payload)
            resolved_source_asset_id = (
                source_publish_result.get("id")
                or source_model.asset_id
                or resolved_source_asset_id
            )

        # 2) forecast run asset
        forecast_publish_result: dict[str, Any] | None = None
        forecast_asset_id: str | None = None

        if publish_forecast_asset:
            forecast_model = ForecastRunAssetModel(
                asset_id=None,  # let OR create it unless you already have a stable ID
                name=f"Forecast {forecast_run_id}",
                realm=self.realm,
                parent_id=self.forecast_asset_parent_id or self.default_site_parent_id,
                forecast_run_id=forecast_run_id,
                session_id=str(execution_metadata.get("session_id", forecast_run_id)),
                backend_name=str(execution_metadata.get("backend_name", result.model_name)),
                model_name=result.model_name,
                model_version=getattr(result, "model_version", None),
                run_status=ForecastRunStatus.COMPLETED,
                issued_at=issued_at,
                horizon_seconds=self._extract_horizon_seconds(result),
                source_asset_id=resolved_source_asset_id,
                site_asset_id=self.default_site_asset_id,
                max_concentration=summary.get("max_concentration"),
                mean_concentration=summary.get("mean_concentration"),
                affected_cells_above_threshold=summary.get("affected_cells_above_threshold"),
                affected_area_m2=summary.get("affected_area_m2"),
                affected_area_hectares=summary.get("affected_area_hectares"),
                dominant_spread_direction=summary.get("dominant_spread_direction"),
                threshold_used=summary.get("threshold_used"),
                confidence_score=summary.get("confidence_score"),
                explanation_text=explanation_text,
                alert_level=self._infer_alert_level(summary, override=alert_level),
                plume_footprint_geojson=self._build_footprint_geojson(result),
                centroid=self._estimate_centroid_from_scenario(scenario),
                bounding_box=self._extract_bounding_box(grid_spec),
                geojson_url=self._build_geojson_url(forecast_run_id),
                raster_metadata=self._extract_raster_metadata(result),
                grid_spec=self._normalize_object(grid_spec),
                scenario_snapshot=self._normalize_object(scenario),
                execution_metadata=execution_metadata,
                external_artifact_ref=self._build_external_artifact_ref(forecast_run_id),
            )
            forecast_payload = build_forecast_run_asset_payload(forecast_model)
            forecast_publish_result = await self.sink.upsert_asset(forecast_payload)
            forecast_asset_id = forecast_publish_result.get("id") or forecast_model.asset_id

        return {
            "source_asset": source_publish_result,
            "forecast_asset": forecast_publish_result,
            "source_asset_id": resolved_source_asset_id,
            "forecast_asset_id": forecast_asset_id,
            "forecast_run_id": forecast_run_id,
        }

    async def upsert_sensor_asset(
        self,
        *,
        sensor_asset_id: str | None,
        sensor_id: str,
        name: str,
        latitude: float,
        longitude: float,
        sensor_type: str,
        pollutant_type: str | None = None,
        observed_unit: str | None = None,
        quality_flag: str = "ok",
        observation_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = SensorAssetModel(
            asset_id=sensor_asset_id,
            name=name,
            realm=self.realm,
            parent_id=self.sensor_asset_parent_id or self.default_site_parent_id,
            sensor_id=sensor_id,
            location=GeoPoint(latitude=latitude, longitude=longitude),
            sensor_type=sensor_type,
            observed_unit=observed_unit,
            pollutant_type=pollutant_type,
            quality_flag=quality_flag,  # enum-compatible str
            observation_metadata=observation_metadata or {},
        )
        payload = build_sensor_asset_payload(model)
        return await self.sink.upsert_asset(payload)

    async def publish_sensor_observation(
        self,
        *,
        sensor_asset_id: str,
        observed_value: float,
        observed_at,
    ) -> list[dict[str, Any]]:
        writes = build_sensor_observation_write(
            asset_id=sensor_asset_id,
            value=observed_value,
            observed_at=observed_at,
        )
        return await self.sink.write_attributes_with_timestamps(writes)

    async def upsert_forecast_zone_asset(
        self,
        *,
        zone_asset_id: str | None,
        zone_id: str,
        zone_name: str,
        zone_geometry: dict[str, Any],
        zone_type: str,
        latest_forecast_run_id: str | None = None,
        risk_level: str = "none",
    ) -> dict[str, Any]:
        model = ForecastZoneAssetModel(
            asset_id=zone_asset_id,
            name=zone_name,
            realm=self.realm,
            parent_id=self.zone_asset_parent_id or self.default_site_parent_id,
            zone_id=zone_id,
            zone_name=zone_name,
            zone_geometry=zone_geometry,
            zone_type=zone_type,
            latest_forecast_run_id=latest_forecast_run_id,
            risk_level=risk_level,  # enum-compatible str
        )
        payload = build_forecast_zone_asset_payload(model)
        return await self.sink.upsert_asset(payload)

    async def publish_zone_predicted_series(
        self,
        *,
        zone_asset_id: str,
        series: Iterable[tuple],
    ) -> dict[str, Any]:
        """
        series: iterable[(timestamp, value)]
        """
        write = build_zone_predicted_concentration_write(
            asset_id=zone_asset_id,
            datapoints=list(series),
        )
        return await self.sink.write_predicted_datapoints(write)

    # ----------------------------
    # Extraction / mapping helpers
    # ----------------------------

    def _extract_summary_stats(self, result: Any) -> dict[str, Any]:
        raw = getattr(result, "summary_statistics", {}) or {}

        if isinstance(raw, dict):
            return raw

        if is_dataclass(raw):
            return asdict(raw)

        if hasattr(raw, "model_dump"):
            return raw.model_dump()

        if hasattr(raw, "__dict__"):
            return dict(raw.__dict__)

        return {}

    def _extract_horizon_seconds(self, result: Any) -> int | None:
        execution_metadata = self._normalize_mapping(getattr(result, "execution_metadata", {}) or {})
        if execution_metadata.get("horizon_seconds") is not None:
            try:
                return int(execution_metadata["horizon_seconds"])
            except (TypeError, ValueError):
                return None
        return None

    def _extract_raster_metadata(self, result: Any) -> dict[str, Any]:
        """
        Keep this minimal and JSON-safe.
        """
        forecast = result.forecast
        grid_spec = self._get_attr(forecast, "grid_spec")
        return {
            "grid_rows": self._get_attr(grid_spec, "number_of_rows", None),
            "grid_columns": self._get_attr(grid_spec, "number_of_columns", None),
            "projection": self._get_attr(grid_spec, "projection", None),
            "boundary_limits": self._normalize_object(self._get_attr(grid_spec, "boundary_limits", None)),
            "grid_center": self._normalize_object(self._get_attr(grid_spec, "grid_center", None)),
        }

    def _build_geojson_url(self, forecast_id: str) -> str | None:
        if not self.geojson_base_url:
            return None
        return f"{self.geojson_base_url}/forecast/{quote(str(forecast_id))}/geojson"

    def _build_external_artifact_ref(self, forecast_id: str) -> dict[str, Any]:
        ref: dict[str, Any] = {"forecastId": forecast_id}
        geojson_url = self._build_geojson_url(forecast_id)
        if geojson_url:
            ref["geojsonUrl"] = geojson_url
        return ref

    def _build_footprint_geojson(self, result: Any) -> dict[str, Any] | None:
        """
        Demo-friendly placeholder:
        build a tiny point footprint from source location if no real contour exists yet.

        Replace later with a real threshold contour/polygon.
        """
        scenario = self._get_attr(result.forecast, "scenario")
        lat = self._get_attr(scenario, "latitude", None)
        lon = self._get_attr(scenario, "longitude", None)
        if lat is None or lon is None:
            return None

        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            },
            "properties": {
                "kind": "forecast_source_centroid_placeholder",
                "forecastId": result.forecast_id,
            },
        }

    def _estimate_centroid_from_scenario(self, scenario: Any) -> GeoPoint | None:
        lat = self._get_attr(scenario, "latitude", None)
        lon = self._get_attr(scenario, "longitude", None)
        if lat is None or lon is None:
            return None
        return GeoPoint(latitude=float(lat), longitude=float(lon))

    def _extract_bounding_box(self, grid_spec: Any):
        boundary_limits = self._get_attr(grid_spec, "boundary_limits", None)
        if not boundary_limits or len(boundary_limits) != 4:
            return None

        from plume.openremote.models import BoundingBox

        min_lat, max_lat, min_lon, max_lon = boundary_limits
        return BoundingBox(
            min_latitude=float(min_lat),
            min_longitude=float(min_lon),
            max_latitude=float(max_lat),
            max_longitude=float(max_lon),
        )

    def _scenario_metadata(self, scenario: Any) -> dict[str, Any]:
        return {
            "start": self._normalize_object(self._get_attr(scenario, "start", None)),
            "end": self._normalize_object(self._get_attr(scenario, "end", None)),
            "duration": self._get_attr(scenario, "duration", None),
        }

    def _infer_alert_level(self, summary: dict[str, Any], override: str | None = None) -> AlertLevel:
        if override:
            return AlertLevel(override)

        max_conc = summary.get("max_concentration")
        if max_conc is None:
            return AlertLevel.NONE

        # Demo heuristic. Replace later with your real domain thresholds.
        try:
            value = float(max_conc)
        except (TypeError, ValueError):
            return AlertLevel.NONE

        if value <= 0:
            return AlertLevel.NONE
        if value < 1e-6:
            return AlertLevel.LOW
        if value < 1e-4:
            return AlertLevel.MEDIUM
        if value < 1e-2:
            return AlertLevel.HIGH
        return AlertLevel.CRITICAL

    # ----------------------------
    # Normalization helpers
    # ----------------------------

    def _get_optional_float(self, obj: Any, attr: str) -> float | None:
        value = self._get_attr(obj, attr, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_str(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _get_attr(self, obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _normalize_mapping(self, obj: Any) -> dict[str, Any]:
        normalized = self._normalize_object(obj)
        return normalized if isinstance(normalized, dict) else {}

    def _normalize_object(self, obj: Any) -> Any:
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                pass

        if isinstance(obj, dict):
            return {str(k): self._normalize_object(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._normalize_object(v) for v in obj]

        if is_dataclass(obj):
            return {k: self._normalize_object(v) for k, v in asdict(obj).items()}

        if hasattr(obj, "model_dump"):
            return self._normalize_object(obj.model_dump())

        if hasattr(obj, "__dict__"):
            return {
                str(k): self._normalize_object(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }

        return str(obj)
