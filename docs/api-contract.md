# API Contract

This document describes the currently implemented FastAPI HTTP/JSON contract at a high level. It is a proof-of-concept API contract, not a production stability guarantee.

## Scope

- Protocol: HTTP/JSON.
- Runtime shape: modular monolith with internal runtime boundaries.
- Internal runtime seam: `ForecastRuntimeClient`, currently implemented by `LocalForecastRuntimeClient`.
- Batch path: Gaussian plume baseline through `ForecastService`.
- Online/session path: configured backend through `OnlineForecastService`.
- Current configured online backend: `convlstm_online`.
- Fallback backend: `gaussian_fallback`.
- Development/test backend: `mock_online`.

## Storage model

- Forecast artifacts are file-backed under the configured artifact root and can be listed/retrieved by forecast id.
- Online sessions use the configured state store: in-memory by default, optional local CSV via config/environment.
- Forecast/retraining jobs and Ops metadata use app-owned local stores.
- Optional app-owned SQLite-backed Ops stores may exist where configured.
- Dataset playback state is local demo/runtime state.
- No broker or external database is required.
- No OpenRemote database mirroring is implemented.

## Runtime/provenance truth rules

Consumers should use forecast context `provenance` and `runtime` fields as the source of truth for prediction source:

- `forecast_source=dataset_playback` means demo/dataset playback, not live sensor data and not active ConvLSTM inference.
- `fallback_used=true` means fallback logic served the result.
- Active ConvLSTM should only be claimed when provenance supports `forecast_source=active_model_inference` and `model_family=ConvLSTM`.
- LLM explanation/decision-support output is optional and grounded in forecast context; it is not autonomous forecasting.

## Core endpoints

### `GET /health`

Returns basic API health.

Example:

```json
{"status": "ok"}
```

### `GET /capabilities`

Returns currently advertised API capabilities. The exact payload may evolve as proof-of-concept features are added.

### `GET /service/runtime-status`

Returns runtime status details such as forecast artifact store status, session store mode, model runtime configuration, OpenRemote service-registration status, and dataset playback availability.

## Forecast artifact routes

### `POST /forecast`

Runs a synchronous batch forecast inline through the runtime seam and persists standard forecast artifacts.

### `GET /forecasts?limit=50`

Lists persisted forecast metadata, newest first. `limit` must be greater than 0 and no more than 500.

### `GET /forecast/{forecast_id}`

Returns the persisted forecast summary for `forecast_id`.

### `GET /forecast/{forecast_id}/summary`

Alias for the persisted forecast summary.

### `GET /forecast/{forecast_id}/geojson`

Returns the persisted GeoJSON artifact for `forecast_id`.

### `GET /forecast/{forecast_id}/raster-metadata`

Returns persisted raster metadata for `forecast_id`.

### `GET /forecast/{forecast_id}/explanation`

Returns a persisted explanation artifact when one exists. Explanation persistence is opt-in; when no `explanation.json` exists and live reconstruction is unavailable, the endpoint returns a conflict response instead of inventing an explanation.

## Forecast job routes

These routes provide an asynchronous control/execution boundary without introducing a broker.

### `POST /forecast/jobs`

Creates a queued forecast job from a forecast request payload.

### `GET /forecast/jobs?limit=50`

Lists forecast jobs. `limit` must be greater than 0 and no more than 500.

### `GET /forecast/jobs/{job_id}`

Returns a single forecast job record or a not-found error.

## Online session routes

### `POST /sessions`

Creates a runtime session. If `backend_name` is omitted, the configured default backend is used.

Request body example:

```json
{
  "backend_name": "convlstm_online",
  "model_name": "optional-model-name",
  "metadata": {"site": "demo"}
}
```

Response includes session id, backend name, model name, status, timestamps, metadata, capabilities, and runtime metadata.

### `GET /sessions`

Returns all known sessions in the configured session store.

### `GET /sessions/{session_id}`

Returns a single session record.

### `GET /sessions/{session_id}/state`

Returns backend state summary, including backend/session ids, observation counters, timestamps, internal-state summaries, capabilities, and limitations.

### `POST /sessions/{session_id}/observations`

Ingests observations for a session.

Validation behavior:

- `timestamp` is required and parseable.
- latitude must be in `[-90, 90]`.
- longitude must be in `[-180, 180]`.
- value must be numeric, non-NaN, and non-negative.
- `source_type` is required and non-empty.
- optional `pollutant_type` is normalized to lowercase.
- `metadata` defaults to `{}`.
- observations are sorted by timestamp within a batch.

Malformed observation payloads return HTTP 400.

### `POST /sessions/{session_id}/update`

Manually triggers backend update for a session.

### `POST /sessions/{session_id}/predict`

Requests prediction for a session. The response follows the forecast summary style used by forecast outputs.

## Forecast-context routes

`ForecastContextService` backs these routes. It assembles normalized context for UI and decision support from session forecasts and/or dataset playback.

### `GET /forecast-context/latest`

Returns latest forecast context.

Query parameters:

- `session_id` optional session id.
- `source`: `auto`, `dataset`, or `session`.

When `source=auto`, enabled dataset playback may supply the context before session forecast context. Dataset playback responses must remain labeled as demo/dataset playback.

### Dataset scenario routes

- `GET /forecast-context/dataset-scenarios`
- `GET /forecast-context/dataset-scenarios/{scenario_id}`
- `POST /forecast-context/dataset-scenarios/{scenario_id}/activate`
- `GET /forecast-context/dataset-scenarios/{scenario_id}/overlay`
- `GET /forecast-context/dataset-scenarios/{scenario_id}/raster`
- `GET /forecast-context/dataset-scenarios/{scenario_id}/frames`
- `GET /forecast-context/dataset-scenarios/{scenario_id}/frames/{frame_index}/raster`
- `GET /forecast-context/dataset-scenarios/{scenario_id}/frames/{frame_index}/overlay`

### Active dataset scenario routes

- `GET /forecast-context/dataset-scenarios/active`
- `GET /forecast-context/dataset-scenarios/active/overlay`
- `GET /forecast-context/dataset-scenarios/active/raster`
- `GET /forecast-context/dataset-scenarios/active/frames`
- `GET /forecast-context/dataset-scenarios/active/frames/{frame_index}/raster`
- `GET /forecast-context/dataset-scenarios/active/frames/{frame_index}/overlay`

### Dataset playback routes

- `GET /forecast-context/dataset-playback/state`
- `POST /forecast-context/dataset-playback/state`
- `POST /forecast-context/dataset-playback/start`
- `POST /forecast-context/dataset-playback/stop`
- `POST /forecast-context/dataset-playback/next`

Dataset playback is demo/context support only. It is not live sensor data, not OpenRemote data, and not active ConvLSTM inference.

## Decision-support routes

`DecisionSupportService` backs these routes. It uses forecast context and optional LLM interpretation to produce operator-facing summaries/chat. LLM output is grounded in the supplied context and does not create independent forecasts.

### `GET /decision-support/latest`

Returns the latest decision-support payload for an optional `session_id`.

### `POST /decision-support/chat`

Answers a forecast-context-grounded question.

Request body:

```json
{
  "message": "Who is doing the prediction?",
  "session_id": "optional-session-id"
}
```

Prediction-source questions should be answered from provenance fields. The service should not claim active ConvLSTM inference for dataset playback or fallback results.

## OpenRemote notes

OpenRemote support is optional and provisional:

- Service registration and heartbeat lifecycle exist.
- HTTP publishing components exist.
- These paths are disabled by default unless configured.
- No live-validated OpenRemote schema/contract is claimed.
- No OpenRemote database mirroring is implemented.

## Error behavior notes

- Missing forecasts or jobs return not-found errors.
- Missing sessions return HTTP 404.
- Malformed observation/prediction payloads return HTTP 400.
- Missing persisted explanations return conflict when live reconstruction is not implemented.
