# Geospatial Forecasting

## Overview
Geospatial Forecasting is an early proof-of-concept Python project for forecasting airborne hazard dispersion to support geospatial decision-making. The current repository focuses on a local baseline workflow: define a release scenario, build a grid around the source, run a Gaussian plume calculation, and return a forecast grid that can be inspected locally.

This project is **not** a production-ready atmospheric dispersion platform. The current implementation is a **Gaussian plume baseline** intended to provide a simple, testable foundation for later work.

## Current capabilities
Today the repository provides a small but usable baseline system for:
- representing release scenarios and forecast grids
- validating scenario and grid inputs before inference
- building latitude/longitude coordinate arrays and mesh grids
- generating a Gaussian plume concentration forecast over a local grid
- running a local inference/demo script that prints summary statistics and plots the concentration grid

The current model is intentionally simple. It does **not** yet provide realistic atmospheric transport, terrain interaction, uncertainty quantification, or operational service interfaces.

## Repository structure
```text
.
├── configs/                 Example runtime and scenario configuration files
├── docs/                    Project documentation
├── scripts/                 Local runnable entry points and demos
├── src/plume/
│   ├── inference/           Validation, grid handling, and inference orchestration
│   ├── models/              Forecasting model implementations
│   ├── schemas/             Dataclasses for scenarios, grids, and forecasts
│   ├── services/            Optional external service integrations
│   └── utils/               Supporting utilities
├── tests/                   Automated baseline tests
├── pyproject.toml           Minimal Python project configuration
└── requirements.txt         Runtime dependencies
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

This installs the project in editable mode so imports work from the repository root while preserving the current local development flow.

## Running local inference
Run the local demo script from the repository root:

```bash
python scripts/run_local_inference.py
```

The demo script:
- builds a sample scenario near Utrecht
- creates a local latitude/longitude grid
- runs the Gaussian plume baseline model
- prints the forecast grid shape, maximum concentration, mean concentration, and forecast timestamp
- opens a local matplotlib heatmap for visual inspection

The plotting window is intended for local use only and is **not** part of CI.

## Running tests
Run the automated test suite from the repository root:

```bash
pytest
```

The tests are deterministic and cover the current baseline behavior for grid construction, validation, inference orchestration, and Gaussian plume output.

## Continuous integration
GitHub Actions runs a single CI workflow on pushes to `main` and pull requests targeting `main`. The workflow installs dependencies with Python 3.11 and runs `pytest` so basic import, syntax, and baseline regression issues fail quickly.

## Roadmap / next steps
Near-term extensions for this proof of concept are expected to include:
- wind-aware plume behavior and less symmetric spread patterns
- more realistic physical dispersion assumptions
- explicit uncertainty handling around forecasts
- cleaner service or API integration for downstream applications
- eventual integration into a geospatial dashboard or OpenRemote-like environment

Those items are future directions only. The current repository remains a local, early-stage baseline for geospatial airborne hazard forecasting.
