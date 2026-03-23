# AGENTS.md

## Repository purpose
This repository is an early proof-of-concept for geospatial forecasting of airborne hazard dispersion. The current system is a local baseline that validates a release scenario, builds a grid, runs a Gaussian plume forecast, and returns a forecast object for local inspection or visualization.

## Current maturity
- Early proof of concept
- Local baseline workflow, not a production-ready dispersion platform
- Current forecast model is a Gaussian plume baseline, not a fully realistic atmospheric simulation

## Development guidance
- Make focused, practical changes.
- Do not overengineer or introduce platform-level complexity without a clear need.
- Preserve the current structure unless a small structural change is clearly justified.
- Prefer small, testable additions over broad refactors.
- Keep local demo scripts working.
- Keep CI passing.
- If you add or change setup steps or runnable commands, update `README.md` in the same change.
- Keep documentation and config files aligned with the actual implementation.

## Code organization
- `src/plume/schemas`: dataclasses and core data structures
- `src/plume/models`: forecasting model implementations
- `src/plume/inference`: orchestration, validation, and grid handling
- `scripts`: runnable local and demo entry points
- `configs`: scenario and runtime configuration examples
- `docs`: architecture and design documentation
- `tests`: automated validation of current behavior

## Working style
- Prefer minimal, production-sensible improvements for the current maturity level.
- Do not add deployment, Docker, cloud infrastructure, or secrets in routine tasks.
- Do not describe future ideas as already implemented.
- Keep terminology consistent across code, configs, and docs.
