from __future__ import annotations

from plume.api.deps import get_export_service, get_forecast_service
from plume.storage.file_forecast_store import FileForecastStore


def test_file_forecast_store_writes_expected_files(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()
    store = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)

    result = fs.run_forecast(run_name="store-write")
    metadata = store.save(result)

    forecast_dir = tmp_path / "forecasts" / result.forecast_id
    assert (forecast_dir / "summary.json").exists()
    assert (forecast_dir / "geojson.json").exists()
    assert (forecast_dir / "raster_metadata.json").exists()
    assert (forecast_dir / "metadata.json").exists()
    assert metadata["forecast_id"] == result.forecast_id


def test_file_forecast_store_survives_new_instance(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()

    first = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)
    result = fs.run_forecast(run_name="store-reload")
    first.save(result)

    second = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)
    assert second.exists(result.forecast_id)
    summary = second.get_summary(result.forecast_id)
    assert summary is not None
    assert summary["forecast_id"] == result.forecast_id
