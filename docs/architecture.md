# Architecture

## Purpose and current state
This repository is an early proof-of-concept for airborne hazard dispersion forecasting. It now separates:

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

4. **Service layer** (`src/plume/services`)
   - `OnlineForecastService`: session lifecycle, ingest, update, predict orchestration via backend + state store.
   - `ForecastService`: batch one-off forecast path retained for current scripts and batch API routes.
   - `ExplainService` and `ExportService`: unchanged responsibility for explanation/export concerns.

5. **HTTP API** (`src/plume/api`)
   - Thin FastAPI route handlers.
   - Session lifecycle endpoints delegate to `OnlineForecastService`.
   - Batch endpoints remain available and isolated from online runtime concerns.

## Session lifecycle

Online flow:
1. Create session (`POST /sessions`), selecting backend runtime.
2. Ingest observations (`POST /sessions/{id}/observations`).
3. Trigger state updates (`POST /sessions/{id}/update`) or rely on auto-update-on-ingest config.
4. Request prediction (`POST /sessions/{id}/predict`).
5. Inspect state summary (`GET /sessions/{id}/state`).

This creates a practical online-backend skeleton without adding external infra.

## Why this separation

The Gaussian plume baseline remains useful, but the architecture now centers on backend runtime and state lifecycle rather than a single batch inference path. This makes future backend evolution possible while preserving current batch behavior.

## Scope boundaries

- No databases/Redis/background workers/WebSockets.
- No live OpenRemote integration client.
- No claim of full online learning yet.
- OpenRemote payload remains a provisional generic translation.
