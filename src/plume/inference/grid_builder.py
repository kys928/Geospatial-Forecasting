from ..schemas.grid import GridSpec
import numpy as np


class GridBuilder:
    def __init__(self, grid_spec: GridSpec):
        self.grid_spec = grid_spec

    def build_coordinate_arrays(self):
        grid_spec = self.grid_spec

        center_lat, center_lon = grid_spec.grid_center
        start_lat = center_lat - grid_spec.grid_height / 2
        start_lon = center_lon - grid_spec.grid_width / 2

        latitudes = []
        longitudes = []

        for row in range(grid_spec.number_of_rows):
            lat = start_lat + (row * grid_spec.grid_spacing)
            latitudes.append(lat)

        for col in range(grid_spec.number_of_columns):
            lon = start_lon + (col * grid_spec.grid_spacing)
            longitudes.append(lon)

        return np.array(latitudes), np.array(longitudes)

    def return_grid_as_mesh(self):
        latitudes, longitudes = self.build_coordinate_arrays()
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        return lat_grid, lon_grid