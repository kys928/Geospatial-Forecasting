from dataclasses import dataclass

@dataclass
class GridSpec:
    grid_height: float
    grid_width: float
    grid_center: tuple[float, float]
    grid_spacing: float
    number_of_rows: int
    number_of_columns: int
    projection: str
    boundary_limits: tuple[float, float, float, float]
