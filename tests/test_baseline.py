from __future__ import annotations

import numpy as np
import pytest

from plume.inference.engine import InferenceEngine
from plume.inference.grid_builder import GridBuilder
from plume.inference.validator import Validator
from plume.models.gaussian_plume import GaussianPlume


def test_grid_builder_build_coordinate_arrays_returns_expected_lengths(sample_grid_spec):
    builder = GridBuilder(sample_grid_spec)

    latitudes, longitudes = builder.build_coordinate_arrays()

    assert len(latitudes) == sample_grid_spec.number_of_rows
    assert len(longitudes) == sample_grid_spec.number_of_columns


def test_grid_builder_return_grid_as_mesh_returns_expected_shapes(sample_grid_spec):
    builder = GridBuilder(sample_grid_spec)

    lat_grid, lon_grid = builder.return_grid_as_mesh()

    expected_shape = (
        sample_grid_spec.number_of_rows,
        sample_grid_spec.number_of_columns,
    )
    assert lat_grid.shape == expected_shape
    assert lon_grid.shape == expected_shape


def test_validator_accepts_valid_scenario_and_grid(sample_scenario, sample_grid_spec):
    validator = Validator(sample_scenario, sample_grid_spec)

    assert validator.validate_scenario_latitude(sample_scenario) is True
    assert validator.validate_scenario_longitude(sample_scenario) is True
    assert validator.validate_grid_spec_projection(sample_grid_spec) is True


def test_validator_rejects_invalid_latitude(sample_scenario, sample_grid_spec):
    validator = Validator(sample_scenario, sample_grid_spec)
    sample_scenario.latitude = 95.0

    with pytest.raises(ValueError, match="latitude"):
        validator.validate_scenario_latitude(sample_scenario)


def test_validator_rejects_invalid_longitude(sample_scenario, sample_grid_spec):
    validator = Validator(sample_scenario, sample_grid_spec)
    sample_scenario.longitude = 181.0

    with pytest.raises(ValueError, match="longitude"):
        validator.validate_scenario_longitude(sample_scenario)


def test_inference_engine_returns_forecast_with_expected_grid_shape(
    sample_scenario,
    sample_grid_spec,
):
    model = GaussianPlume(grid_spec=sample_grid_spec, scenario=sample_scenario)
    engine = InferenceEngine(model=model)

    forecast = engine.run_inference(sample_scenario, sample_grid_spec)

    assert forecast.concentration_grid.shape == (
        sample_grid_spec.number_of_rows,
        sample_grid_spec.number_of_columns,
    )


def test_gaussian_plume_returns_non_negative_concentrations(sample_scenario, sample_grid_spec):
    model = GaussianPlume(grid_spec=sample_grid_spec, scenario=sample_scenario)

    forecast = model.predict_scenario(sample_scenario, sample_grid_spec)

    assert np.all(forecast.concentration_grid >= 0)


def test_gaussian_plume_concentration_grid_has_finite_maximum(sample_scenario, sample_grid_spec):
    model = GaussianPlume(grid_spec=sample_grid_spec, scenario=sample_scenario)

    forecast = model.predict_scenario(sample_scenario, sample_grid_spec)

    assert np.isfinite(np.max(forecast.concentration_grid))
