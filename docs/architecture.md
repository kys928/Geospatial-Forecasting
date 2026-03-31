# Architecture

## Purpose and current state
This repository is an early proof-of-concept for geospatial forecasting of airborne hazard dispersion. The currently implemented model path is a Gaussian plume baseline.

The codebase now supports both:
- a local script-driven workflow, and
- a backend-first HTTP/JSON boundary for running and retrieving forecasts.

It is still **not** a production atmospheric dispersion platform.

## Backend-first layering (implemented)
The current Python backend follows a layered separation:

1. **Forecasting core** (`src/plume/models`, `src/plume/inference`, `src/plume/schemas`)
   - Numeric model implementation (`GaussianPlume`)
   - Validation and grid preparation (`Validator`, `GridBuilder`, `InferenceEngine`)
   - Core dataclasses (`Scenario`, `GridSpec`, `Forecast`)

2. **Service layer** (`src/plume/services`)
   - `ForecastService` orchestrates model+engine execution and returns a canonical `ForecastRunResult`
   - `ExplainService` builds deterministic summary/explanation output and can optionally call `LLMService`
   - `ExportService` delegates format conversion to adapters and writes GeoJSON files

3. **Export adapters** (`src/plume/adapters`)
   - `geojson.py` translates forecasts into a GeoJSON-like `FeatureCollection`
   - `raster.py` translates forecasts into lightweight raster metadata
   - `openremote.py` translates forecasts into a provisional generic integration payload

4. **HTTP API boundary** (`src/plume/api`)
   - FastAPI app exposes the current baseline over HTTP/JSON
   - Route handlers are intentionally thin and rely on service-layer calls
   - Created forecasts are stored in-memory in the app process

5. **Thin scripts** (`scripts`)
   - Local CLI-style entry points that call the service layer
   - No duplicated forecasting logic in scripts

## Implemented execution paths

### A) Local script path
- `scripts/run_local_inference.py` runs the baseline via core inference path.
- `scripts/run_demo_forecast.py` runs via `ForecastService` and prints a readable summary.
- `scripts/export_geojson.py` runs a forecast and writes a GeoJSON artifact.
- `scripts/seed_demo_data.py` writes deterministic mock payloads for local/demo use.

### B) HTTP API path
- `src/plume/api/main.py` creates a FastAPI app exposing health, capabilities, forecast creation, and retrieval/export endpoints.
- The API currently supports the Gaussian plume baseline only.

## Data/export boundary notes
- Core forecast objects remain internal Python dataclasses/NumPy arrays.
- External payloads are produced through adapters (`geojson`, `raster`, `openremote`) to avoid coupling export shapes to core model objects.
- The OpenRemote payload is intentionally documented as a **provisional generic translation**, not a validated upstream schema contract.

## Scope and limitations
- Forecast results are in-memory only (no persistence layer).
- No authentication/authorization layer in the API.
- No frontend coupling in backend services.
- No live OpenRemote client/auth/session integration in the adapter.
- Gaussian plume baseline is simplified and intended for local validation, not full atmospheric realism.
