from dataclasses import dataclass
from scenario import Scenario
from grid import GridSpec
import datetime


@dataclass
class Forecast:
    predictions: list[float]
    timestamp: datetime.datetime
    scenario: Scenario
    grid_spec: GridSpec