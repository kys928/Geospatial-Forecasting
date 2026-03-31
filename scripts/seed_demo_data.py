from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.utils.config import Config


def seed_mock_forecast_payloads(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    forecast_service = ForecastService(Config())
    export_service = ExportService()

    result = forecast_service.run_forecast(run_name="seed-demo")
    result.forecast_id = "demo-forecast-001"
    fixed_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    result.issued_at = fixed_ts
    result.forecast.timestamp = fixed_ts

    forecast_payload = {
        "forecast_id": result.forecast_id,
        "issued_at": result.issued_at.isoformat(),
        "model": result.model_name,
        "grid_shape": list(result.forecast.concentration_grid.shape),
        "timestamp": result.forecast.timestamp.isoformat(),
    }
    summary_payload = forecast_service.summarize_forecast(result)
    geojson_payload = export_service.to_geojson(result)
    raster_payload = export_service.to_raster_metadata(result).__dict__
    capabilities_payload = {
        "model": ["gaussian_plume"],
        "exports": ["summary", "geojson", "raster-metadata", "openremote"],
    }

    (output_path / "forecast.json").write_text(json.dumps(forecast_payload, indent=2), encoding="utf-8")
    (output_path / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (output_path / "geojson.json").write_text(json.dumps(geojson_payload, indent=2), encoding="utf-8")
    (output_path / "raster-metadata.json").write_text(json.dumps(raster_payload, indent=2), encoding="utf-8")
    (output_path / "capabilities.json").write_text(json.dumps(capabilities_payload, indent=2), encoding="utf-8")


def main(output_dir: str = "frontend/src/mocks") -> None:
    seed_mock_forecast_payloads(output_dir)
    print(f"Mock payloads written to: {output_dir}")


if __name__ == "__main__":
    main()
