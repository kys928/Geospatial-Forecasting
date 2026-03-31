# Geospatial Forecasting

## Overview
Geospatial Forecasting is an early proof-of-concept Python project for airborne hazard dispersion forecasting. The current implementation is a Gaussian plume baseline that supports both local script execution and a backend HTTP/JSON boundary.

This is **not** a production-ready atmospheric dispersion system. It is a practical baseline intended for local validation and integration scaffolding.

## What is implemented now

### Forecasting baseline
- Scenario + grid schema loading from YAML configs
- Input validation and grid construction
- Gaussian plume concentration grid generation
- Forecast summary statistics (`max_concentration`, `mean_concentration`)

### Backend boundary
- Service layer for forecast execution and summarization
- Export service and adapters for:
  - GeoJSON-like payloads
  - raster metadata payloads
  - provisional OpenRemote-oriented payloads
- FastAPI app exposing forecast creation and retrieval endpoints

### Scripts
- `scripts/run_local_inference.py`: local baseline flow + optional plotting
- `scripts/run_demo_forecast.py`: service-layer demo summary output
- `scripts/export_geojson.py`: generate and write GeoJSON output
- `scripts/seed_demo_data.py`: deterministic mock payload generation

## Architecture at a glance
Backend-first layering used in this repo:
- `src/plume/models`, `src/plume/inference`, `src/plume/schemas`: forecasting core
- `src/plume/services`: orchestration and application-layer behavior
- `src/plume/adapters`: external payload translation
- `src/plume/api`: HTTP API boundary
- `scripts`: thin local entry points

The backend is intentionally decoupled from frontend concerns.

## Repository structure
```text
.
├── configs/                 Runtime/scenario configuration examples
├── docs/                    Architecture and API documentation
├── scripts/                 Thin local/demo entry points
├── src/plume/
│   ├── adapters/            Export/payload translators (GeoJSON, raster, OpenRemote)
│   ├── api/                 FastAPI app and dependency wiring
│   ├── inference/           Validation, grid handling, inference orchestration
│   ├── models/              Forecast model implementations
│   ├── schemas/             Core dataclasses
│   ├── services/            Forecast/explain/export services
│   └── utils/               Supporting utilities
├── tests/                   Automated tests
├── pyproject.toml
└── requirements.txt
```

## Installation
Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[test]"
```

## Run local script paths

### Baseline local inference
```bash
python scripts/run_local_inference.py
```

### Service-layer demo run
```bash
python scripts/run_demo_forecast.py
```

### Export GeoJSON artifact
```bash
python scripts/export_geojson.py
```

### Seed deterministic demo payloads
```bash
python scripts/seed_demo_data.py
```

## Run API path
The FastAPI app entrypoint is `src/plume/api/main.py` (`app = create_app()`).

If your environment has an ASGI server available (for example `uvicorn`), run:

```bash
uvicorn plume.api.main:app --reload
```

Implemented endpoints:
- `GET /health`
- `GET /capabilities`
- `POST /forecast`
- `GET /forecast/{forecast_id}`
- `GET /forecast/{forecast_id}/summary`
- `GET /forecast/{forecast_id}/geojson`
- `GET /forecast/{forecast_id}/raster-metadata`

See `docs/api-contract.md` for concrete response shapes.

## Testing
```bash
pytest
```

Current tests cover baseline inference behavior plus service/export/API contracts.

## Current limitations
- Gaussian plume baseline is simplified (not full atmospheric simulation)
- API forecast storage is in-memory only
- No auth layer
- No database persistence
- OpenRemote adapter is a **provisional generic payload translation** only:
  - not a validated OpenRemote schema contract
  - not a live OpenRemote integration client
