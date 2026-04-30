from plume.openremote.forecast_asset_contract import (
    DEFAULT_ATTRIBUTE_NAMES,
    FORECAST_ASSET_TYPE,
    build_forecast_attribute_payloads,
)


def test_forecast_asset_contract_constants():
    assert FORECAST_ASSET_TYPE == "PlumeForecast"
    assert DEFAULT_ATTRIBUTE_NAMES["summary"] == "forecastSummary"


def test_build_forecast_attribute_payloads_maps_expected_keys():
    payloads = build_forecast_attribute_payloads(
        forecast_id="f-123",
        issued_at="2026-04-30T00:00:00Z",
        summary={"summary_statistics": {"max_concentration": 1.0}},
        geojson={"type": "FeatureCollection", "features": []},
        raster_metadata={"rows": 10, "cols": 12},
        runtime={"path": "batch"},
    )

    assert payloads == {
        "forecastId": "f-123",
        "forecastIssuedAt": "2026-04-30T00:00:00Z",
        "forecastSummary": {"summary_statistics": {"max_concentration": 1.0}},
        "forecastGeoJson": {"type": "FeatureCollection", "features": []},
        "forecastRasterMetadata": {"rows": 10, "cols": 12},
        "forecastRuntime": {"path": "batch"},
        "forecastRiskLevel": "unknown",
    }


def test_build_forecast_attribute_payloads_custom_names():
    payloads = build_forecast_attribute_payloads(
        forecast_id="f-9",
        issued_at="2026-04-30T00:00:00Z",
        summary={},
        geojson=None,
        raster_metadata=None,
        runtime=None,
        attribute_names={"summary": "sumAttr", "forecast_id": "idAttr"},
    )
    assert payloads["idAttr"] == "f-9"
    assert payloads["sumAttr"] == {}
