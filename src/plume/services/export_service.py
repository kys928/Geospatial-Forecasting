from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plume.adapters.geojson import forecast_to_geojson
from plume.adapters.openremote import forecast_to_openremote_payload
from plume.adapters.raster import forecast_to_raster_metadata


@dataclass
class RasterMetadata:
    forecast_id: str
    rows: int
    cols: int
    bounds: dict
    projection: str | None
    min_value: float
    max_value: float
    grid_spacing: float


class ExportService:
    def to_summary_json(self, result):
        return {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
            "model": result.model_name,
            "summary_statistics": result.summary_statistics,
        }

    def to_geojson(self, result, *, thresholds=None):
        return forecast_to_geojson(result, thresholds=thresholds)

    def to_raster_metadata(self, result):
        metadata = forecast_to_raster_metadata(result)
        return RasterMetadata(**metadata)

    def to_openremote_payload(self, result):
        return forecast_to_openremote_payload(result)

    def write_geojson(self, result, output_path, *, thresholds=None):
        import json

        geojson = self.to_geojson(result, thresholds=thresholds)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
