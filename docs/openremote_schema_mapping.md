# OpenRemote schema mapping notes

## Purpose

This document records practical OpenRemote database/schema findings for integration mapping and documents this repository's app-owned local session CSV store shape.

It is intentionally documentation-only: no direct OpenRemote database integration is introduced, and no OpenRemote database behavior is reimplemented here.

## Current OpenRemote alignment

OpenRemote support in this repository is optional and provisional:

- Optional service registration and heartbeat lifecycle exist.
- Optional HTTP publishing components exist.
- These paths are disabled by default unless configured.
- This repository does not claim a live-validated OpenRemote schema or contract.
- This repository does not mirror or copy OpenRemote's database.
- OpenRemote internal database tables are not this app's runtime persistence contract.

## What OpenRemote stores

OpenRemote Manager uses PostgreSQL internally, including migration files under:

- `manager/src/main/resources/org/openremote/manager/setup/database`

From reviewed schema facts:

- `ASSET` stores core asset identity and current attribute state.
- `ASSET.ATTRIBUTES` is `jsonb` containing the latest/current attribute objects.
- `ASSET_DATAPOINT` stores datapoint time-series values.
- `ASSET_PREDICTED_DATAPOINT` stores predicted datapoint time-series values.
- OpenRemote converts datapoint tables into Timescale hypertables.

## Important OpenRemote tables/columns

The following columns are important for conceptual mapping and technical honesty when discussing integration.

### `ASSET`

- `ID`
- `ATTRIBUTES`
- `CREATED_ON`
- `NAME`
- `PARENT_ID`
- `PATH`
- `REALM`
- `TYPE`
- `ACCESS_PUBLIC_READ`
- `VERSION`

### `ASSET_DATAPOINT`

- `TIMESTAMP`
- `ENTITY_ID`
- `ATTRIBUTE_NAME`
- `VALUE`

### `ASSET_PREDICTED_DATAPOINT`

- `TIMESTAMP`
- `ENTITY_ID`
- `ATTRIBUTE_NAME`
- `VALUE`

## Attribute JSON shape

`ASSET.ATTRIBUTES` (jsonb) stores current attributes using a map keyed by attribute name.

```json
{
  "<attributeName>": {
    "name": "<attributeName>",
    "type": "<attributeType>",
    "value": "<json>",
    "timestamp": "<epoch_ms>",
    "meta": {}
  }
}
```

Notes:

- The top-level key is the attribute name.
- `value` is JSON-valued payload content.
- `timestamp` is epoch milliseconds.

## What we should NOT copy

For this project:

- Do **not** copy, mirror, or clone OpenRemote's PostgreSQL schema.
- Do **not** write directly to OpenRemote internal tables.
- Do **not** treat OpenRemote internal database structures as this app's persistence contract.
- Do **not** build local SQL/SQLite replicas of OpenRemote tables for runtime behavior.

OpenRemote's database remains internal to OpenRemote.

## Recommended integration pattern

Use OpenRemote through API/service boundaries only:

- keep service registration and heartbeat lifecycle in the existing OpenRemote integration path,
- keep optional HTTP publishing provisional until validated against a target OpenRemote deployment,
- keep this repository's domain/runtime state independent from OpenRemote internal storage.

This preserves separation of concerns and avoids coupling this app to OpenRemote internal schema evolution.

## Local app-owned CSV session store

This section documents the implemented optional local CSV session/state store used when the configured state store is CSV, for example through `PLUME_STATE_STORE=csv` or equivalent backend configuration.

This mapping is for this repository's operational needs; it is **not** an OpenRemote database mirror and not an OpenRemote-compatible persistence contract.

### `sessions.csv`

Columns:

- `session_id`
- `backend_name`
- `model_name`
- `status`
- `created_at`
- `updated_at`
- `last_error`
- `metadata_json`
- `runtime_metadata_json`
- `state_json`

`state_json` stores the current backend state summary, including recent observations needed by the app-owned state store.

### `session_latest_forecasts.csv`

Columns:

- `session_id`
- `latest_forecast_id`
- `latest_forecast_artifact_dir`
- `updated_at`

### Observation persistence note

There is currently no separate implemented `observations.csv`. Recent observations and state details are serialized inside `state_json` in `sessions.csv`. A separate observations export/import file could be added later only if replay or recovery requirements justify it.

### Contract rules and non-goals

- Scope: local session recovery/export for this app only.
- Non-goal: OpenRemote-compatible database mirroring.
- JSON columns should store compact JSON strings.
- CSV is acceptable for proof-of-concept and local development.
- CSV is not intended for high-concurrency production storage.
- Optional app-owned SQLite-backed Ops stores may exist elsewhere in the application; those are also not OpenRemote DB mirrors.
