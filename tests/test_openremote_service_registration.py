from __future__ import annotations

import asyncio
import httpx

from plume.openremote.service_registration import (
    OpenRemoteServiceRegistrar,
    OpenRemoteServiceRegistrationSettings,
)


def _settings(**overrides) -> OpenRemoteServiceRegistrationSettings:
    base = OpenRemoteServiceRegistrationSettings(
        enabled=True,
        manager_api_url="https://openremote.example/api/master",
        service_id="geospatial-plume-forecast",
        label="Geospatial Plume Forecast",
        version="0.1.0",
        icon="mdi-map-marker-radius",
        homepage_url="https://plume.example",
        global_service=False,
        heartbeat_interval_seconds=30,
        access_token="token",
    )
    return OpenRemoteServiceRegistrationSettings(**{**base.__dict__, **overrides})


def _client_factory(handler):
    transport = httpx.MockTransport(handler)
    return lambda: httpx.AsyncClient(transport=transport)


def test_disabled_registrar_noops():
    registrar = OpenRemoteServiceRegistrar(_settings(enabled=False))
    assert asyncio.run(registrar.register()) is None
    assert asyncio.run(registrar.heartbeat()) is False
    assert asyncio.run(registrar.deregister()) is False
    status = registrar.status()
    assert status["enabled"] is False
    assert status["registered"] is False


def test_register_posts_expected_payload_and_stores_instance_id():
    def handler(request: httpx.Request):
        assert request.url.path == "/api/master/service"
        assert request.headers["Authorization"] == "Bearer token"
        payload = __import__("json").loads(request.content.decode("utf-8"))
        assert payload["serviceId"] == "geospatial-plume-forecast"
        assert payload["homepageUrl"] == "https://plume.example"
        return httpx.Response(200, json={"instanceId": 9})

    registrar = OpenRemoteServiceRegistrar(_settings(), client_factory=_client_factory(handler))
    asyncio.run(registrar.register())
    assert registrar.registered is True
    assert registrar.instance_id == 9


def test_global_service_posts_to_global_path():
    def handler(request: httpx.Request):
        assert request.url.path == "/api/master/service/global"
        return httpx.Response(200, json={"instanceId": 11})

    registrar = OpenRemoteServiceRegistrar(_settings(global_service=True), client_factory=_client_factory(handler))
    asyncio.run(registrar.register())
    assert registrar.instance_id == 11


def test_heartbeat_sends_put():
    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={})

    registrar = OpenRemoteServiceRegistrar(_settings(), client_factory=_client_factory(handler))
    registrar.registered = True
    registrar.instance_id = 15
    assert asyncio.run(registrar.heartbeat()) is True
    assert calls == [("PUT", "/api/master/service/geospatial-plume-forecast/15")]


def test_heartbeat_404_clears_registration():
    def handler(request: httpx.Request):
        return httpx.Response(404, json={})

    registrar = OpenRemoteServiceRegistrar(_settings(), client_factory=_client_factory(handler))
    registrar.registered = True
    registrar.instance_id = 15
    assert asyncio.run(registrar.heartbeat()) is False
    assert registrar.registered is False
    assert registrar.instance_id is None


def test_deregister_sends_delete():
    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path))
        return httpx.Response(204)

    registrar = OpenRemoteServiceRegistrar(_settings(), client_factory=_client_factory(handler))
    registrar.registered = True
    registrar.instance_id = 17
    assert asyncio.run(registrar.deregister()) is True
    assert calls == [("DELETE", "/api/master/service/geospatial-plume-forecast/17")]


def test_register_failure_records_last_error_without_raise():
    def handler(request: httpx.Request):
        return httpx.Response(500, json={"error": "boom"})

    registrar = OpenRemoteServiceRegistrar(_settings(), client_factory=_client_factory(handler))
    result = asyncio.run(registrar.register())
    assert result is None
    assert registrar.last_error is not None
