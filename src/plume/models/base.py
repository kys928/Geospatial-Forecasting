from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario


class BaseForecastModel:
    def __init__(self):
        pass

    def predict_scenario(self, scenario: Scenario, grid_spec: GridSpec):
        raise NotImplementedError("Subclasses must implement this method")

        #Needs to return type Forecast class
