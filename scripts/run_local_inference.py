from src.plume.inference.engine import InferenceEngine
from src.plume.schemas.grid import GridSpec
from src.plume.schemas.scenario import Scenario
from src.plume.models.gaussian_plume import GaussianPlume
import datetime as datetime
import matplotlib.pyplot as plt
import numpy as np


def main():
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(minutes=60)

    scenario = Scenario(
        source=(52.0907, 5.1214),
        latitude=52.0907,
        longitude=5.1214,
        start=start_time,
        end=end_time,
        emissions_rate=100.0,
        pollution_type="smoke",
        duration=60.0,
        release_height=10.0,
    )


    grid_spec = GridSpec(
        grid_center=(52.0907, 5.1214),
        number_of_rows=50,
        number_of_columns=50,
        grid_height=0.02,
        grid_width=0.02,
        grid_spacing=0.0004,
        projection="EPSG:4326",
        boundary_limits=(52.0807, 52.1007, 5.1114, 5.1314)
    )

    model = GaussianPlume(grid_spec=grid_spec, scenario=scenario)
    engine = InferenceEngine(model=model)

    forecast = engine.run_inference(scenario, grid_spec)

    concentration_grid = forecast.concentration_grid

    print("Forecast generated successfully.")
    print(f"Grid shape: {concentration_grid.shape}")
    print(f"Max concentration: {np.max(concentration_grid):.6f}")
    print(f"Mean concentration: {np.mean(concentration_grid):.6f}")
    print(f"Timestamp: {forecast.timestamp}")

    plt.figure(figsize=(8, 6))
    plt.imshow(concentration_grid, origin="lower")
    plt.colorbar(label="Concentration")
    plt.title("Gaussian Plume Forecast")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()