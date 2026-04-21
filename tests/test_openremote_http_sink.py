from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from plume.openremote.models import (
    ORAssetPayload,
    ORAttribute,
    ORMetaItem,
    ORTimestampedAttributeWrite,
)
from plume.openremote.sink import HttpOpenRemoteResultSink


class _FakeAsyncClient:
    responses: list[httpx.Response] = []
    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def request(self, method, url, json=None):
        _FakeAsyncClient.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
            }
        )
        if not _FakeAsyncClient.responses:
            request = httpx.Request(method, url)
            return httpx.Response(200, request=request, json={})
        return _FakeAsyncClient.responses.pop(0)


def _run(coro):
    return asyncio.run(coro)


def test_http_sink_upsert_asset_shapes_payload_and_uses_post(monkeypatch):
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://or.example/api/master/asset"),
            json={"id": "asset-created"},
        )
    ]
    monkeypatch.setattr("plume.openremote.sink.httpx.AsyncClient", _FakeAsyncClient)

    sink = HttpOpenRemoteResultSink(base_url="https://or.example/api/master", access_token="token")
    payload = ORAssetPayload(
        name="Forecast Asset",
        type="ForecastRunAsset",
        realm="master",
        parent_id="site-123",
        attributes=[ORAttribute(name="key", value="value", meta=[ORMetaItem(name="x", value=True)])],
    )

    response = _run(sink.upsert_asset(payload))

    assert response["id"] == "asset-created"
    assert len(_FakeAsyncClient.calls) == 1
    call = _FakeAsyncClient.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://or.example/api/master/asset"
    assert call["json"]["parentId"] == "site-123"
    assert "parent_id" not in call["json"]


def test_http_sink_timestamped_write_emits_epoch_milliseconds(monkeypatch):
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        httpx.Response(
            200,
            request=httpx.Request("PUT", "https://or.example/api/master/asset/attributes/timestamp"),
            json=[{"ok": True}],
        )
    ]
    monkeypatch.setattr("plume.openremote.sink.httpx.AsyncClient", _FakeAsyncClient)

    sink = HttpOpenRemoteResultSink(base_url="https://or.example/api/master", access_token="token")
    writes = [
        ORTimestampedAttributeWrite(
            asset_id="sensor-1",
            attribute_name="observedValue",
            value=2.5,
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
    ]

    _run(sink.write_attributes_with_timestamps(writes))

    call = _FakeAsyncClient.calls[0]
    assert call["method"] == "PUT"
    assert call["json"][0]["timestamp"] == 1767225600000


def test_http_sink_maps_http_error_with_debug_context(monkeypatch):
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        httpx.Response(
            400,
            request=httpx.Request("PUT", "https://or.example/api/master/asset/attributes"),
            text="bad payload",
        )
    ]
    monkeypatch.setattr("plume.openremote.sink.httpx.AsyncClient", _FakeAsyncClient)

    sink = HttpOpenRemoteResultSink(base_url="https://or.example/api/master", access_token="token")

    with pytest.raises(RuntimeError) as exc:
        _run(
            sink.write_attributes_with_timestamps(
                [
                    ORTimestampedAttributeWrite(
                        asset_id="sensor-1",
                        attribute_name="observedValue",
                        value=1.2,
                        timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                    )
                ]
            )
        )

    message = str(exc.value)
    assert "timestamped attribute write" in message
    assert "HTTP 400" in message
    assert "bad payload" in message
