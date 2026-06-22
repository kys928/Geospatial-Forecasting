# Architecture

## Purpose and current state

This repository is an early proof-of-concept for airborne hazard dispersion forecasting. The current application shape is a **modular monolith with internal runtime boundaries**: one FastAPI application composes the batch forecast path, online/session prediction path, forecast-context assembly, decision-support layer, optional dataset playback/demo support, optional OpenRemote-facing integrations, and local worker process boundaries.

The project remains intentionally lightweight and non-production. It does **not** claim production atmospheric dispersion modeling, real online learning, a distributed microservice deployment, or a live-validated OpenRemote schema contract.

## Current runtime shape

The main runtime is composed in `src/plume/api/main.py`:

1. FastAPI route modules expose HTTP endpoints for forecasts, sessions, service/runtime status, forecast context, decision support, and ops.
2. `ForecastRuntimeClient` is the internal runtime seam.
3. `LocalForecastRuntimeClient` is the current implementation of that seam.
4. The local runtime delegates batch requests to `ForecastService` and online/session requests to `OnlineForecastService`.
5. Backend implementations are selected through the backend registry.
6. Forecast artifacts, jobs, session state, model registry state, and ops state are stored through local app-owned stores.

This keeps the codebase deployable as one application while making the control/runtime boundaries explicit for future extraction if needed.

## Layered architecture

1. **Forecasting core** (`src/plume/models`, `src/plume/inference`, `src/plume/schemas`)
   - Numeric/model logic and canonical data structures.
   - No HTTP concerns.

2. **Runtime seam** (`src/plume/runtime`)
   - `ForecastRuntimeClient` defines the internal interface used by API routes and higher-level services.
   - `LocalForecastRuntimeClient` currently implements that interface in-process.
   - Batch requests are delegated to `ForecastService`.
   - Online/session requests are delegated to `OnlineForecastService`.

3. **Backend runtimes** (`src/plume/backends`)
   - `convlstm_online` is the currently configured online backend.
   - `gaussian_fallback` is the configured fallback backend.
   - `mock_online` remains available for development/test support.
   - Backend aliases are resolved through the backend registry.

4. **State and storage layers** (`src/plume/state`, `src/plume/storage`, job/ops stores)
   - Forecast artifacts are file-backed under the configured artifact root.
   - Sessions use process-local in-memory storage by default.
   - Sessions can use an optional app-owned CSV store through config/environment.
   - Ops metadata/job/registry/event stores are local app-owned stores; optional SQLite-backed Ops stores may be used where configured.
   - No broker or external database is required for the current proof-of-concept runtime.
   - No OpenRemote database mirroring is implemented.

5. **Service layer** (`src/plume/services`)
   - `ForecastService`: batch one-off Gaussian baseline path retained for scripts, API, and worker execution.
   - `OnlineForecastService`: session lifecycle, ingest, update, predict orchestration via backend + state store.
   - `ForecastContextService`: backend layer that assembles normalized context for UI and decision support.
   - `DecisionSupportService`: backend layer that derives operator-facing summaries/chat from forecast context and optional LLM interpretation.
   - `ObservationService`: observation normalization/validation boundary.
   - `ExplainService` and `ExportService`: explanation/export concerns.

6. **HTTP API** (`src/plume/api`)
   - Thin FastAPI route handlers.
   - Routes call the runtime seam and service layer instead of duplicating model logic.

7. **Worker process boundary** (`src/plume/workers`)
   - Forecast and retraining worker modes are local process-mode execution boundaries.
   - Workers coordinate with the API through shared app-owned local stores/artifacts.
   - This is not a brokered or distributed queue system.

## Backend setup

The current backend configuration uses:

- `convlstm_online` as the configured online backend.
- `gaussian_fallback` as fallback.
- `mock_online` for development/test workflows.

The ConvLSTM path can load from configured checkpoints or the active model registry when registry mode is enabled and usable model assets exist. This should be described as proof-of-concept inference/runtime behavior, not production-validated atmospheric modeling and not real online learning.

The Gaussian fallback path is a fallback/baseline path. It must not be described as active model serving.

## Dataset playback and demo context support

Dataset playback is demo/context support for UI and decision-support workflows. It is managed by `DatasetScenarioService` and exposed through forecast-context routes.

When dataset playback is enabled, `ForecastContextService.latest(source="auto")` may return dataset playback context instead of a session forecast. Dataset playback outputs are labeled with `dataset_playback` provenance and should be described as demo/dataset playback only.

Dataset playback is **not** live sensor data, **not** OpenRemote data, and **not** active ConvLSTM inference.

## Forecast context and decision support

`ForecastContextService` is a real backend layer. It assembles forecast summaries, runtime metadata, provenance, source/meteorology fields, plume metrics, uncertainty, limitations, and raw supporting payloads into a normalized context object.

`DecisionSupportService` is also a real backend layer. It uses forecast context to produce latest decision-support payloads and chat answers. It may call an optional LLM through the explanation service, but LLM output is grounded in the supplied forecast context and is not autonomous forecasting, independent model execution, or a source of new plume predictions.

Decision-support wording must preserve provenance truth:

- `forecast_source=dataset_playback` means demo/dataset playback.
- `fallback_used=true` means fallback served the result.
- Active ConvLSTM inference should only be claimed when provenance supports `forecast_source=active_model_inference` and `model_family=ConvLSTM`.

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

Backend state summaries include:

- backend name and session id
- observation count and state version
- timestamp block (`last_update_time`, `last_ingest_time`, `last_observation_time`, `last_prediction_time`)
- internal-state snapshot
- capability hints and backend limitations

## Observation ingestion boundary

Observation payloads are normalized before backend calls:

- type conversion and timestamp parsing
- coordinate/value range checks
- pollutant normalization
- metadata normalization
- optional timestamp ordering within a batch

This keeps route handlers thin and prevents backend-specific parsing behavior from leaking into API code.

## OpenRemote boundary

OpenRemote support is optional and provisional:

- Optional service registration and heartbeat lifecycle exist.
- Optional HTTP publishing components exist.
- These paths are disabled by default unless configured.
- This repository does not claim a live-validated OpenRemote schema or contract.
- This repository does not mirror or copy the OpenRemote database.
- OpenRemote internal database tables are not this app's persistence contract.

## Scope boundaries

- Not production atmospheric dispersion modeling.
- Not real online learning.
- Not a brokered distributed queue.
- Not two independently deployed services yet.
- No external database is required.
- Optional app-owned SQLite-backed Ops stores may exist where configured.
- No OpenRemote DB mirroring is implemented.
- Dataset playback is demo/context support, not live sensor/OpenRemote data.
- Gaussian fallback is fallback behavior, not active ConvLSTM serving.
