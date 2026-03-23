from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario
from ..schemas.forecast import Forecast
from abc import ABC, abstractmethod

class BaseForecastModel(ABC):

    @abstractmethod
    def predict_scenario(self, scenario: Scenario, grid_spec: GridSpec) -> Forecast:
        pass

        #Needs to return type Forecast class
