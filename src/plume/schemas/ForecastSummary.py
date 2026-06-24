from dataclasses import dataclass


@dataclass
class ForecastSummary:
    source_latitude: float
    source_longitude: float
    grid_rows: int
    grid_columns: int
    projection: str | None
    max_concentration: float
    mean_concentration: float
    affected_cells_above_threshold: int
    affected_area_m2: float
    affected_area_hectares: float
    dominant_spread_direction: str
    threshold_used: float
    note: str | None = None