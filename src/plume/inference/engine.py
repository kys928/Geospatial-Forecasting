from .grid_builder import GridBuilder
from ..models.base import BaseForecastModel
from ..schemas.scenario import Scenario
from ..schemas.grid import GridSpec
from validator import Validator
import numpy as np
import pandas as pd


class InferenceEngine:
    def __init__(self, model: BaseForecastModel):
      self.model = model

    def run_inference(self, scenario: Scenario, grid_spec: GridSpec):
        self.scenario = scenario
        self.grid_spec = grid_spec

        self.validate_inputs(scenario, grid_spec)
        self.prepare_grid(grid_spec)
        return self.model.predict_scenario(scenario, grid_spec)


    def validate_inputs(self, scenario: Scenario, grid_spec: GridSpec):
        validator = Validator(self.scenario, self.grid_spec)

        validator.validate_scenario_existence(scenario)
        validator.validate_grid_spec_existence(grid_spec)
        validator.validate_scenario_duration(scenario)
        validator.validate_scenario_start_time(scenario)
        validator.validate_scenario_longitude(scenario)
        validator.validate_scenario_latitude(scenario)
        validator.validate_grid_spec_number_of_rows(grid_spec)
        validator.validate_grid_spec_number_of_columns(grid_spec)
        validator.validate_grid_spec_projection(grid_spec)

    def prepare_grid(self, grid_spec: GridSpec):
        grid_builder = GridBuilder(grid_spec)
        grid_builder.build_coordinate_arrays()




