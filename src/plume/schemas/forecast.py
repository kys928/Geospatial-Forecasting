from dataclasses import dataclass
from scenario import Scenario
from grid import GridSpec
import datetime


@dataclass
class Forecast:
    predictions: list[float]
    timestamp: datetime.datetime
    metadata: dict[Scenario.start, Scenario.pollution_type, GridSpec.grid_spacing, GridSpec.grid_center]