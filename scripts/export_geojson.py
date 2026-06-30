from __future__ import annotations

from pathlib import Path

from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.utils.config import Config


def main(config_dir: str | None = None, output_path: str | None = None) -> None:
    forecast_service = ForecastService(Config(config_dir=config_dir))
    export_service = ExportService()

    result = forecast_service.run_forecast()
    path = Path(output_path) if output_path else Path("artifacts") / "forecast.geojson"
    export_service.write_geojson(result, path)
    print(f"GeoJSON written to: {path}")


if __name__ == "__main__":
    main()
