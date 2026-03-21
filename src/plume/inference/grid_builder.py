from ..schemas.grid import GridSpec
from numpy import linspace
from numpy import meshgrid

class GridBuilder:
    def __init__(self, grid_spec: GridSpec):
        self.grid_spec = grid_spec

    def build_coordinate_arrays(self):
        grid_spec = self.grid_spec
        arrays = []
        center_lat, center_lon = grid_spec.grid_center
        start_lat = center_lat - grid_spec.grid_height / 2
        start_lon = center_lon - grid_spec.grid_width / 2

        for row in range(grid_spec.number_of_rows):
            lat = start_lat + (row * grid_spec.grid_spacing)
            for col in range(grid_spec.number_of_columns):
                lon = start_lon + (col * grid_spec.grid_spacing)
                arrays.append((lat, lon))
        return arrays

    def return_grid_as_mesh(self):
        lats, lons = zip(*self.build_coordinate_arrays())
        lat_grid, lon_grid = meshgrid(lats, lons)
        return lat_grid, lon_grid