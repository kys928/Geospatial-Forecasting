from __future__ import annotations

from fastapi.testclient import TestClient

from plume.api.main import create_app


def test_frontend_static_serving_is_disabled_by_default(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<html>frontend</html>", encoding="utf-8")
    monkeypatch.delenv("PLUME_SERVE_FRONTEND", raising=False)
    monkeypatch.setenv("PLUME_FRONTEND_DIST_DIR", str(dist_dir))

    client = TestClient(create_app())

    assert client.get("/").status_code == 404
    assert client.get("/health").status_code < 500


def test_frontend_static_serving_handles_spa_paths_without_shadowing_api(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html>frontend shell</html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('frontend asset');", encoding="utf-8")
    monkeypatch.setenv("PLUME_SERVE_FRONTEND", "true")
    monkeypatch.setenv("PLUME_FRONTEND_DIST_DIR", str(dist_dir))

    client = TestClient(create_app())

    assert client.get("/health").status_code < 500
    assert client.get("/openapi.json").status_code == 200
    assert "frontend shell" in client.get("/").text
    assert "frontend shell" in client.get("/some/frontend/path").text
    asset_response = client.get("/assets/app.js")
    assert asset_response.status_code == 200
    assert "frontend asset" in asset_response.text

    for path in [
        "/forecast/unknown-test-path",
        "/sessions/unknown-test-path",
        "/ops/unknown-test-path",
        "/docs",
        "/openapi.json",
    ]:
        response = client.get(path)
        assert "frontend shell" not in response.text
