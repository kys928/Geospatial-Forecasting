# Architecture

## Purpose
This repository explores geospatial forecasting for airborne hazard dispersion in a form that is easy to run locally and easy to extend. The current focus is an early proof-of-concept baseline for generating a concentration grid from a simple release scenario.

This is **not** yet a fully realistic atmospheric dispersion platform. It currently uses a Gaussian plume baseline so contributors can validate the end-to-end flow before adding more sophisticated physics or services.

## Current baseline flow
1. **Scenario input**
   A `Scenario` dataclass captures the release source, timing, emission rate, pollutant type, and release height.
2. **Validation**
   `Validator` checks core scenario and grid assumptions such as latitude, longitude, duration, projection, and grid dimensions.
3. **Grid creation**
   `GridBuilder` creates one-dimensional latitude and longitude coordinate arrays and converts them into a mesh grid for model evaluation.
4. **Forecast generation**
   `InferenceEngine` orchestrates validation and grid preparation, then calls the configured model.
5. **Gaussian plume baseline**
   `GaussianPlume` transforms the source and grid into a local projected coordinate system and computes a concentration field using a simple Gaussian formulation.
6. **Forecast output**
   The result is returned as a `Forecast` object containing the concentration grid, timestamp, scenario, and grid specification.
7. **Local visualization/demo**
   `scripts/run_local_inference.py` provides a local demo that runs the baseline forecast, prints summary statistics, and displays a matplotlib heatmap.

## Package and folder responsibilities
- `src/plume/schemas`
  Defines the core dataclasses used to pass structured inputs and outputs through the pipeline.
- `src/plume/inference`
  Contains validation logic, grid construction, and the inference engine that coordinates the baseline run.
- `src/plume/models`
  Holds forecasting model implementations. Right now this is the Gaussian plume baseline.
- `src/plume/services`
  Contains optional external-service integration code. This is not part of the numeric baseline path.
- `src/plume/utils`
  Reserved for small supporting utilities.
- `scripts`
  Local entry points for manual runs and demos.
- `configs`
  Minimal YAML examples showing how scenarios, grids, and runtime settings can be organized as the project grows.
- `tests`
  Automated checks for the current baseline behavior.
- `docs`
  Documentation describing the current architecture and intended direction.

## Near-term direction
The next meaningful improvements should stay grounded in the existing baseline and focus on:
- better physical realism than the current symmetric Gaussian approximation
- wind-aware behavior so plume transport reflects direction and advection more clearly
- uncertainty handling around scenario assumptions and forecast outputs
- service or API exposure for downstream consumers when the local baseline stabilizes
- eventual integration into a geospatial dashboard or OpenRemote-like environment

Those items are intended future steps. They are not already implemented in this repository.
