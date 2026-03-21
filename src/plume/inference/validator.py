from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario

class Validator:
    def __init__(self, scenario: Scenario, grid_spec: GridSpec):
        self.scenario = scenario
        self.grid_spec = grid_spec

    def validate_scenario_existence(self, scenario: Scenario):
        if scenario.start is None:
            raise ValueError("Invalid scenario")
        return True

    def validate_grid_spec_existence(self, grid_spec: GridSpec):
        if grid_spec.grid_center is None:
            raise ValueError("Invalid grid specification.")
        return True

    def validate_grid_spec_number_of_rows(self, grid_spec: GridSpec):
        if grid_spec.number_of_rows is None:
            raise ValueError("Invalid grid specification for number of rows.")
        if grid_spec.number_of_rows < 0:
            raise ValueError("Negative grid specification for number of rows.")
        return True

    def validate_grid_spec_number_of_columns(self, grid_spec: GridSpec):
        if grid_spec.number_of_columns is None:
            raise ValueError("Invalid grid specification for number of columns.")
        if grid_spec.number_of_columns < 0:
            raise ValueError("Negative grid specification for number of columns.")
        return True

    def validate_scenario_duration(self, scenario: Scenario):
        if scenario.duration is None:
            raise ValueError("Invalid scenario duration.")
        if scenario.duration < 0:
            raise ValueError("Negative scenario duration.")
        return True

    def validate_scenario_start_time(self, scenario: Scenario):
        if scenario.start is None:
            raise ValueError("Invalid scenario start time.")
        if scenario.start > scenario.end:
            raise ValueError("Scenario start cannot be earlier than scenario end.")
        return True

    def validate_scenario_longitude(self, scenario: Scenario):
        if scenario.longitude is None:
            raise ValueError("Invalid scenario longitude.")
        if not (-180 <= scenario.longitude <= 180):
            raise ValueError("Invalid scenario longitude range.")
        return True

    def validate_scenario_latitude(self, scenario: Scenario):
        if scenario.latitude is None:
            raise ValueError("Invalid scenario latitude.")
        if not (-90 <= scenario.latitude <= 90):
            raise ValueError("Invalid scenario latitude range.")
        return True

    def validate_grid_spec_projection(self, grid_spec: GridSpec):
        if grid_spec.projection is None:
            raise ValueError("Invalid grid specification for projection.")
        if grid_spec.projection != "EPSG:4326":
            raise ValueError("Invalid grid specification for projection. Expected EPSG:4326 but got {}".format(grid_spec.projection))
        return True



