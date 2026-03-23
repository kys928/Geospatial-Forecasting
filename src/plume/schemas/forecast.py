from dataclasses import dataclass
from .scenario import Scenario
from .grid import GridSpec
import datetime
import numpy as np


@dataclass
class Forecast:
    concentration_grid: np.ndarray
    timestamp: datetime.datetime
    scenario: Scenario
    grid_spec: GridSpec