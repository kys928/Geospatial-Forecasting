from dataclasses import dataclass
import datetime

import numpy as np

from .grid import GridSpec
from .scenario import Scenario


@dataclass
class Forecast:
    concentration_grid: np.ndarray
    timestamp: datetime.datetime
    scenario: Scenario
    grid_spec: GridSpec
    concentration_sequence: np.ndarray | None = None
    metadata: dict[str, object] | None = None
