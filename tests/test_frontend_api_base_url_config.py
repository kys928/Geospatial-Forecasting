from __future__ import annotations

from pathlib import Path


def test_frontend_client_uses_vite_api_base_url_with_fallback() -> None:
    client_path = Path(__file__).resolve().parents[1] / "frontend" / "src" / "services" / "api" / "client.ts"
    contents = client_path.read_text(encoding="utf-8")

    assert "resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL)" in contents
    assert "const configuredBaseUrl = envValue?.trim();" in contents
    assert "return configuredBaseUrl ? configuredBaseUrl : DEFAULT_API_BASE_URL;" in contents
    assert 'const DEFAULT_API_BASE_URL = "http://localhost:8000";' in contents
