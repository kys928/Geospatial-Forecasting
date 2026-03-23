from dataclasses import dataclass
import datetime

@dataclass
class Scenario:
    source: tuple[float, float]
    latitude: float
    longitude: float
    start: datetime.datetime
    end: datetime.datetime
    emissions_rate: float
    pollution_type: str
    duration: float
    release_height: float

