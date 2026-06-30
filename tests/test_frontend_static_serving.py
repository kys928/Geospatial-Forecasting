from __future__ import annotations

from fastapi.testclient import TestClient

from plume.api.main import create_app


FRONTEND_HTML = "<html><body>frontend shell</body></html>"


def _write_fake_dist(tmp_path):
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text(FRONTEND_HTML, encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('frontend asset');", encoding="utf-8")
    return dist_dir


def test_frontend_static_serving_is_disabled_by_default(monkeypatch, tmp_path):
    dist_dir = _write_fake_dist(tmp_path)
    monkeypatch.delenv("PLUME_SERVE_FRONTEND", raising=False)
    monkeypatch.setenv("PLUME_FRONTEND_DIST_DIR", str(dist_dir))

    client = TestClient(create_app())

    assert client.get("/app").status_code == 404
    assert client.get("/health").status_code < 500


def test_frontend_static_serving_is_scoped_to_app_without_shadowing_api(monkeypatch, tmp_path):
    dist_dir = _write_fake_dist(tmp_path)
    monkeypatch.setenv("PLUME_SERVE_FRONTEND", "true")
    monkeypatch.setenv("PLUME_FRONTEND_DIST_DIR", str(dist_dir))

    client = TestClient(create_app())

    for path in [
        "/app",
        "/app/",
        "/app/forecast",
        "/app/decision-support",
        "/app/ops",
        "/app/some/frontend/path",
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert "frontend shell" in response.text

    asset_response = client.get("/app/assets/app.js")
    assert asset_response.status_code == 200
    assert "frontend asset" in asset_response.text

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    assert openapi_response.headers["content-type"].startswith("application/json")
    assert "frontend shell" not in openapi_response.text

    runtime_response = client.get("/runtime/status")
    assert runtime_response.status_code == 200
    assert runtime_response.headers["content-type"].startswith("application/json")
    assert "frontend shell" not in runtime_response.text

    for path in [
        "/forecast",
        "/decision-support/latest",
        "/ops/status",
    ]:
        response = client.get(path)
        assert "frontend shell" not in response.text
