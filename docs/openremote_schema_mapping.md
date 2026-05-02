# OpenRemote schema mapping notes

## Purpose
This document records practical OpenRemote database/schema findings for integration mapping and sets a lightweight local CSV session-store contract for this repository.

It is intentionally documentation-only: no direct database integration is introduced, and no OpenRemote database behavior is reimplemented here.

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
The following columns are important for conceptual mapping and technical honesty when discussing integration:

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
- publish forecast/attribute payloads through OpenRemote APIs,
- keep this repository's domain/runtime state independent from OpenRemote internal storage.

This preserves separation of concerns and avoids coupling this app to OpenRemote internal schema evolution.

## CSV mapping if we need local export/import
If a local durable export/import layer is needed, use app-owned CSV/JSON files with app-owned columns. This can assist local recovery and operator workflows without introducing a database dependency.

This mapping is for this repository's operational needs; it is **not** an OpenRemote database mirror.

## Proposed local CSV session-store contract
This section defines a documentation contract for a future lightweight local session store.

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

### `session_latest_forecasts.csv`
Columns:
- `session_id`
- `latest_forecast_id`
- `latest_forecast_artifact_dir`
- `updated_at`

### Optional `observations.csv`
Use only if local replay/recovery requires per-observation persistence.

Columns:
- `session_id`
- `timestamp`
- `latitude`
- `longitude`
- `value`
- `pollutant_type`
- `source_type`
- `metadata_json`

### Contract rules and non-goals
- Scope: local session recovery/export for this app only.
- Non-goal: OpenRemote-compatible database mirroring.
- JSON columns should store compact JSON strings.
- CSV is acceptable for proof-of-concept and local development.
- CSV is not intended for high-concurrency production storage.
