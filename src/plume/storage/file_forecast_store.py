from __future__ import annotations

import json
from pathlib import Path

from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastRunResult, ForecastService


class FileForecastStore:
    def __init__(
        self,
        artifact_root: Path,
        *,
        forecast_service: ForecastService,
        export_service: ExportService,
    ):
        self.artifact_root = Path(artifact_root)
        self.forecast_service = forecast_service
        self.export_service = export_service

    def _forecast_dir(self, forecast_id: str) -> Path:
        return self.artifact_root / "forecasts" / forecast_id

    def _read_json(self, path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save(self, result: ForecastRunResult) -> dict[str, object]:
        forecast_dir = self._forecast_dir(result.forecast_id)
        forecast_dir.mkdir(parents=True, exist_ok=True)

        summary = self.forecast_service.summarize_forecast(result)
        geojson = self.export_service.to_geojson(result)
        raster_metadata = self.export_service.to_raster_metadata(result).__dict__
        metadata = {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
            "model": result.model_name,
            "model_version": result.model_version,
            "artifact_root": str(self.artifact_root),
            "artifact_dir": str(forecast_dir),
            "artifacts": {
                "summary": str(forecast_dir / "summary.json"),
                "geojson": str(forecast_dir / "geojson.json"),
                "raster_metadata": str(forecast_dir / "raster_metadata.json"),
                "metadata": str(forecast_dir / "metadata.json"),
            },
        }

        self._write_json(forecast_dir / "summary.json", summary)
        self._write_json(forecast_dir / "geojson.json", geojson)
        self._write_json(forecast_dir / "raster_metadata.json", raster_metadata)
        self._write_json(forecast_dir / "metadata.json", metadata)
        return metadata

    def get_summary(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "summary.json")

    def get_geojson(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "geojson.json")

    def get_raster_metadata(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "raster_metadata.json")

    def get_metadata(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "metadata.json")

    def exists(self, forecast_id: str) -> bool:
        return (self._forecast_dir(forecast_id) / "metadata.json").exists()
