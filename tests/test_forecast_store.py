from __future__ import annotations

import pytest

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
    assert metadata["artifact_schema_version"] == "forecast_artifact_v1"
    assert metadata["runtime"]["model_family"] == "gaussian_plume"


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


def test_list_metadata_returns_newest_first(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()
    store = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)

    first = fs.run_forecast(run_name="older")
    second = fs.run_forecast(run_name="newer")
    store.save(first)
    store.save(second)

    listed = store.list_metadata(limit=10)
    assert len(listed) == 2
    assert listed[0]["forecast_id"] == second.forecast_id
    assert listed[1]["forecast_id"] == first.forecast_id


def test_list_metadata_skips_malformed_records(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()
    store = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)

    good = fs.run_forecast(run_name="good")
    store.save(good)

    bad_dir = tmp_path / "forecasts" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "metadata.json").write_text("{not-json", encoding="utf-8")

    listed = store.list_metadata(limit=10)
    assert [row["forecast_id"] for row in listed] == [good.forecast_id]


def test_duplicate_save_rejected(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()
    store = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)

    result = fs.run_forecast(run_name="duplicate")
    store.save(result)
    with pytest.raises(FileExistsError):
        store.save(result)


def test_file_forecast_store_get_explanation_missing_returns_none(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()
    store = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)

    result = fs.run_forecast(run_name="no-explanation")
    store.save(result)

    assert store.get_explanation(result.forecast_id) is None


def test_file_forecast_store_save_with_explanation_writes_artifact(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()
    store = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)

    result = fs.run_forecast(run_name="with-explanation")
    explanation = {"forecast_id": result.forecast_id, "used_llm": False, "summary": {}, "explanation": "ok"}
    metadata = store.save(result, explanation=explanation)

    forecast_dir = tmp_path / "forecasts" / result.forecast_id
    assert (forecast_dir / "explanation.json").exists()
    assert store.get_explanation(result.forecast_id) == explanation
    assert "explanation" in metadata["available_artifacts"]


def test_file_forecast_store_save_explanation_updates_metadata(tmp_path):
    fs = get_forecast_service()
    es = get_export_service()
    store = FileForecastStore(tmp_path, forecast_service=fs, export_service=es)

    result = fs.run_forecast(run_name="post-save-explanation")
    store.save(result)
    explanation = {"forecast_id": result.forecast_id, "used_llm": False, "summary": {}, "explanation": "saved"}
    store.save_explanation(result.forecast_id, explanation)

    metadata = store.get_metadata(result.forecast_id)
    assert metadata is not None
    assert "explanation" in metadata["available_artifacts"]
    assert store.get_explanation(result.forecast_id) == explanation
