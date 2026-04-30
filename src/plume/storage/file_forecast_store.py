from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastRunResult, ForecastService


logger = logging.getLogger(__name__)


class ForecastArtifactReadError(Exception):
    def __init__(self, *, artifact: str, path: Path, reason: str):
        super().__init__(f"artifact '{artifact}' at '{path}' could not be decoded: {reason}")
        self.artifact = artifact
        self.path = path
        self.reason = reason


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

    def _read_json(self, path: Path, *, artifact: str) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ForecastArtifactReadError(artifact=artifact, path=path, reason=str(exc)) from exc

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _sortable_timestamp(self, value: object) -> datetime:
        if not isinstance(value, str):
            return datetime.min
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min

    def save(self, result: ForecastRunResult) -> dict[str, object]:
        forecasts_root = self.artifact_root / "forecasts"
        forecasts_root.mkdir(parents=True, exist_ok=True)

        forecast_dir = self._forecast_dir(result.forecast_id)
        if forecast_dir.exists():
            raise FileExistsError(f"forecast artifact already exists: {result.forecast_id}")

        summary = self.forecast_service.summarize_forecast(result)
        geojson = self.export_service.to_geojson(result)
        raster_metadata = self.export_service.to_raster_metadata(result).__dict__

        metadata: dict[str, object] = {
            "artifact_schema_version": "forecast_artifact_v1",
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
            "model": result.model_name,
            "model_version": result.model_version,
            "created_at": datetime.now(UTC).isoformat(),
            "artifact_root": str(self.artifact_root),
            "artifact_dir": str(forecast_dir),
            "available_artifacts": ["summary", "geojson", "raster_metadata", "metadata"],
            "artifacts": {
                "summary": str(forecast_dir / "summary.json"),
                "geojson": str(forecast_dir / "geojson.json"),
                "raster_metadata": str(forecast_dir / "raster_metadata.json"),
                "metadata": str(forecast_dir / "metadata.json"),
            },
        }
        if result.execution_metadata is not None:
            try:
                json.dumps(result.execution_metadata)
                metadata["execution_metadata"] = result.execution_metadata
                runtime = result.execution_metadata.get("runtime")
                if isinstance(runtime, dict):
                    metadata["runtime"] = runtime
                else:
                    metadata["runtime"] = result.execution_metadata
            except TypeError:
                logger.warning("forecast.execution_metadata_not_serializable", extra={"forecast_id": result.forecast_id})
        if "summary_statistics" in summary:
            metadata["summary_statistics"] = summary["summary_statistics"]

        temp_dir: Path | None = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{result.forecast_id}-", dir=str(forecasts_root)))
            self._write_json(temp_dir / "summary.json", summary)
            self._write_json(temp_dir / "geojson.json", geojson)
            self._write_json(temp_dir / "raster_metadata.json", raster_metadata)
            self._write_json(temp_dir / "metadata.json", metadata)
            temp_dir.rename(forecast_dir)
        except Exception:
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return metadata

    def list_metadata(self, limit: int = 50) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        metadata_rows: list[dict[str, object]] = []
        forecasts_root = self.artifact_root / "forecasts"
        if not forecasts_root.exists():
            return []

        for folder in forecasts_root.iterdir():
            if not folder.is_dir():
                continue
            metadata_path = folder / "metadata.json"
            try:
                metadata = self._read_json(metadata_path, artifact="metadata")
                if metadata is None:
                    continue
                metadata_rows.append(metadata)
            except Exception as exc:
                logger.warning(
                    "forecast.metadata_malformed",
                    extra={"forecast_dir": str(folder), "error": str(exc)},
                )

        metadata_rows.sort(
            key=lambda row: self._sortable_timestamp(row.get("issued_at") or row.get("created_at")),
            reverse=True,
        )
        return metadata_rows[:limit]

    def get_summary(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "summary.json", artifact="summary")

    def get_geojson(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "geojson.json", artifact="geojson")

    def get_raster_metadata(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "raster_metadata.json", artifact="raster_metadata")

    def get_metadata(self, forecast_id: str) -> dict[str, object] | None:
        return self._read_json(self._forecast_dir(forecast_id) / "metadata.json", artifact="metadata")

    def exists(self, forecast_id: str) -> bool:
        return (self._forecast_dir(forecast_id) / "metadata.json").exists()
