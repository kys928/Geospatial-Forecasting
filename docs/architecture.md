# Architecture

## Purpose and current state
This repository is an early proof-of-concept for airborne hazard dispersion forecasting. It separates:

- **Batch forecasting** (one-off Gaussian baseline runs), and
- **Online backend runtime flow** (session/state lifecycle).

The project remains non-production and intentionally lightweight.

## Layered architecture

1. **Forecasting core** (`src/plume/models`, `src/plume/inference`, `src/plume/schemas`)
   - Numeric/model logic and canonical data structures.
   - No HTTP concerns.

2. **Backend runtimes** (`src/plume/backends`)
   - `BaseBackend` defines runtime interface.
   - `MockOnlineBackend` simulates online behavior (ingest/update/predict).
   - `GaussianFallbackBackend` wraps existing Gaussian plume path as interchangeable fallback backend.

3. **State layer** (`src/plume/state`)
   - `BaseStateStore` abstraction.
   - `InMemoryStateStore` process-local session/state persistence.
   - API wiring uses a singleton in-memory store per process to preserve session continuity across requests.

4. **Service layer** (`src/plume/services`)
   - `OnlineForecastService`: session lifecycle, ingest, update, predict orchestration via backend + state store.
   - `ObservationService`: dedicated observation normalization/validation boundary.
   - `ForecastService`: batch one-off forecast path retained for current scripts and batch API routes.
   - `ExplainService` and `ExportService`: unchanged responsibility for explanation/export concerns.

5. **HTTP API** (`src/plume/api`)
   - Thin FastAPI route handlers.
   - Session lifecycle endpoints delegate to service methods.
   - Batch endpoints remain available and isolated from online runtime concerns.

## Session lifecycle

Lightweight status transitions are explicit:

1. Session creation -> `created`
2. Observation ingest -> `active`
3. State update -> `updated`
4. Prediction request in-flight -> `predicting`
5. Prediction success -> `idle`
6. Prediction failure -> `error` (+ `last_error`)

This keeps runtime behavior inspectable without introducing a heavy finite-state machine.

## Runtime state summaries

Backend state summaries now include:
- backend name and session id
- observation count and state version
- timestamp block (`last_update_time`, `last_ingest_time`, `last_observation_time`, `last_prediction_time`)
- internal-state snapshot
- capability hints and explicit backend limitations

## Observation ingestion boundary

Observation payloads are normalized before backend calls:
- type conversion and timestamp parsing
- coordinate/value range checks
- pollutant normalization
- metadata normalization
- optional timestamp ordering within a batch

This keeps route handlers thin and prevents backend-specific parsing behavior from leaking into API code.

## Why this separation

The Gaussian plume baseline remains useful, but the architecture centers on backend runtime and state lifecycle rather than a single batch inference path. This allows incremental backend evolution while preserving current batch behavior.

## Scope boundaries

- No databases/Redis/background workers/WebSockets.
- No live OpenRemote integration client.
- No claim of real online learning yet.
- OpenRemote payload remains a provisional generic translation.
