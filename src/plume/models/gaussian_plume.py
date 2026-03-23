import numpy as np
from pyproj import CRS, Transformer
import datetime as datetime
from ..schemas.forecast import Forecast
from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario
from ..inference.grid_builder import GridBuilder
from .base import BaseForecastModel


class GaussianPlume(BaseForecastModel):
    def __init__(self, grid_spec: GridSpec, scenario: Scenario):
        self.grid_spec = grid_spec
        self.scenario = scenario

    def predict_scenario(self, scenario=None, grid_spec=None) -> Forecast:
        if scenario is None:
            scenario = self.scenario
        if grid_spec is None:
            grid_spec = self.grid_spec

        concentration_grid = self.calculate_grids(scenario, grid_spec)

        forecast = Forecast(
            concentration_grid=concentration_grid,
            timestamp=datetime.datetime.now(),
            scenario=scenario,
            grid_spec=grid_spec,
        )
        return forecast

    def calculate_grids(self, scenario: Scenario, grid_spec: GridSpec) -> np.ndarray:
        source_lat, source_lon = scenario.source
        emission_rate = scenario.emissions_rate

        grid_builder = GridBuilder(grid_spec)
        lat_grid, lon_grid = grid_builder.return_grid_as_mesh()

        transformer = self._build_utm_transformer(source_lat, source_lon)

        source_x, source_y = transformer.transform(source_lon, source_lat)
        x_grid, y_grid = transformer.transform(lon_grid, lat_grid)

        dx = x_grid - source_x
        dy = y_grid - source_y

        sigma_x = 100.0
        sigma_y = 100.0

        concentration_grid = (emission_rate / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
            -((dx ** 2) / (2 * sigma_x ** 2) + (dy ** 2) / (2 * sigma_y ** 2))
        )

        return concentration_grid

    def _build_utm_transformer(self, lat: float, lon: float) -> Transformer:
        utm_zone = int((lon + 180) / 6) + 1
        epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        utm_crs = CRS.from_epsg(epsg_code)
        return Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)