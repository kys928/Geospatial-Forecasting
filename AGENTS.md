# AGENTS.md

## Repository purpose
This repository is an early proof-of-concept for geospatial forecasting of airborne hazard dispersion. The implemented baseline validates a release scenario, builds a grid, runs a Gaussian plume forecast, and exposes results through both local scripts and a backend HTTP API.

## Current maturity
- Early proof of concept
- Gaussian plume baseline, not a full atmospheric dispersion simulator
- Backend-first workflow with lightweight exports and API
- Not production hardened

## Development guidance
- Keep changes focused and practical.
- Do not overengineer or add platform-level complexity without clear need.
- Preserve existing layering and separation of concerns.
- Prefer small, testable additions over broad refactors.
- Keep local scripts and API behavior working.
- Keep CI passing.
- If you add or change runnable commands, update `README.md` in the same change.
- Keep docs/config aligned with actual implementation.

## Architectural separation to preserve
- **Forecasting core**: `src/plume/models`, `src/plume/inference`, `src/plume/schemas`
  - numeric/model logic only
  - no HTTP concerns
- **Service layer**: `src/plume/services`
  - orchestration/application behavior
  - no frontend coupling
- **Export adapters**: `src/plume/adapters`
  - translate canonical forecast results into external payload formats
  - keep format-specific logic out of core model/inference code
- **HTTP API**: `src/plume/api`
  - thin route handlers calling service layer
- **Scripts**: `scripts`
  - thin local entry points only; do not duplicate core/service logic

## OpenRemote adapter status
- Treat `src/plume/adapters/openremote.py` as a **provisional generic payload translator**.
- Do not describe it as a validated OpenRemote schema contract.
- Do not implement live OpenRemote auth/session/client behavior unless explicitly requested.

## Code organization
- `src/plume/schemas`: dataclasses and core data structures
- `src/plume/models`: forecasting model implementations
- `src/plume/inference`: orchestration, validation, and grid handling
- `src/plume/services`: forecast/explain/export orchestration
- `src/plume/adapters`: payload/export translators
- `src/plume/api`: FastAPI boundary and dependencies
- `scripts`: runnable local/demo entry points
- `configs`: scenario and runtime configuration examples
- `docs`: architecture and API documentation
- `tests`: automated validation of current behavior

## Working style
- Prefer minimal, production-sensible improvements for the current maturity level.
- Do not add deployment, Docker, cloud infrastructure, or secrets in routine tasks.
- Do not describe future ideas as already implemented.
- Keep terminology consistent across code, configs, tests, and docs.
