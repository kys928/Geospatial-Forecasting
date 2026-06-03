from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_session_client_rejects_stale_ridge_and_wrong_backend_sessions() -> None:
    contents = _read("frontend/src/features/sessions/api/sessionClient.ts")

    assert "activeConvLstmSessionCompatibility" in contents
    assert 'session.backend_name !== "convlstm_online"' in contents
    assert 'predictionEngine === "ridge_baseline"' in contents
    assert "temporary_model_substitution" in contents
    assert "stored session lacks active ConvLSTM model load metadata" in contents
    assert "this.clearSession();" in contents
    assert "createSession(activeSessionCreatePayload())" in contents


def test_session_client_reuses_only_active_convlstm_sessions() -> None:
    contents = _read("frontend/src/features/sessions/api/sessionClient.ts")

    assert "activeModelId" in contents
    assert "checkpointPath" in contents
    assert "resolved_active_model" in contents
    assert "return { compatible: true }" in contents
    assert "frontend_session_contract" in contents


def test_forecast_page_is_raster_first_and_does_not_call_dataset_overlay_endpoints() -> None:
    contents = _read("frontend/src/pages/ForecastPage.tsx")

    assert "hasUsableSelectedRaster" in contents
    assert "selectedFrameRaster?.grid" in contents
    assert "hasUsableSelectedFrame = hasUsableSelectedRaster || hasUsableSelectedFrameGeoJson" in contents
    assert "hasUsableSelectedRaster ? buildPlumeGridRasterOverlay(selectedFrameRaster)" in contents
    assert "/forecast-context/dataset-scenarios/active/overlay" not in contents
    assert "Use dataset playback demo" not in contents
    assert "Use active ConvLSTM forecast" not in contents


def test_forecast_timeline_uses_frame_count_not_geojson_feature_count() -> None:
    contents = _read("frontend/src/pages/ForecastPage.tsx")

    assert "const hasMultiFrameSession = Boolean(framesMetadata && framesMetadata.frame_count > 1);" in contents
    assert "const timelineDisabled = !hasMultiFrameSession;" in contents
    assert "disabled={timelineDisabled}" in contents


def test_forecast_page_disables_dataset_playback_before_active_forecast_start() -> None:
    contents = _read("frontend/src/pages/ForecastPage.tsx")

    disable_index = contents.index("await disableDatasetPlayback();")
    forecast_index = contents.index("sessionClient.runSessionForecast")
    assert disable_index < forecast_index
    assert 'setMapPipelineStatus("disabling_dataset_playback")' in contents


def test_forecast_page_active_error_can_be_retried_without_refresh() -> None:
    contents = _read("frontend/src/pages/ForecastPage.tsx")

    assert "Retry active ConvLSTM forecast" in contents
    assert "hasAutoBootstrappedRef.current = false; void runActiveForecast();" in contents
    assert "activeForecastError ?" in contents


def test_decision_support_page_bootstraps_active_session_when_opened_first() -> None:
    contents = _read("frontend/src/pages/DecisionSupportPage.tsx")

    assert "sessionClient.runSessionForecast" in contents
    assert "forecast_overview" in contents
    assert "setActiveSessionId(runResult.sessionId)" in contents
    assert "setLatestForecastBundle(runResult.sessionId, bundle)" in contents
    assert "Active ConvLSTM unavailable:" in contents
    assert "/forecast-context/latest?source=session" in contents
