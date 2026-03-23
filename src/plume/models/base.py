from abc import ABC, abstractmethod

from ..schemas.forecast import Forecast
from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario


class BaseForecastModel(ABC):
    @abstractmethod
    def predict_scenario(self, scenario: Scenario, grid_spec: GridSpec) -> Forecast:
        raise NotImplementedError
