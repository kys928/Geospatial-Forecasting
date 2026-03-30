from .grid_builder import GridBuilder
from .validator import Validator
from ..models.base import BaseForecastModel
from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario


class InferenceEngine:
    def __init__(self, model: BaseForecastModel, validate_inputs: bool = True):
        self.model = model
        self.validate_before_run = validate_inputs

    def run_inference(self, scenario: Scenario, grid_spec: GridSpec):
        if self.validate_before_run:
            self.validate_inputs(scenario, grid_spec)
        self.prepare_grid(grid_spec)
        return self.model.predict_scenario(scenario, grid_spec)

    def validate_inputs(self, scenario: Scenario, grid_spec: GridSpec):
        validator = Validator(scenario, grid_spec)

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
