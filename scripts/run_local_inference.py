from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plume.inference.engine import InferenceEngine
from plume.models.gaussian_plume import GaussianPlume
from plume.utils.config import Config


def main(config_dir: str | Path | None = None):
    config = Config(config_dir=config_dir)
    scenario = config.load_scenario()
    grid_spec = config.load_grid()
    base = config.load_base()
    inference = config.load_inference()

    if base.model != "gaussian_plume":
        raise ValueError(f"Unsupported model in base.yaml: {base.model}")

    model = GaussianPlume(grid_spec=grid_spec, scenario=scenario)
    engine = InferenceEngine(model=model, validate_inputs=inference.validate_inputs)

    forecast = engine.run_inference(scenario, grid_spec)
    concentration_grid = forecast.concentration_grid

    print("Forecast generated successfully.")
    print(f"Run name: {base.run_name}")
    print(f"Model: {base.model}")
    print(f"Grid shape: {concentration_grid.shape}")
    print(f"Timestamp: {forecast.timestamp}")

    if "max_concentration" in inference.summary_statistics:
        print(f"Max concentration: {np.max(concentration_grid):.6f}")

    if "mean_concentration" in inference.summary_statistics:
        print(f"Mean concentration: {np.mean(concentration_grid):.6f}")

    if inference.plot.enabled:
        plt.figure(figsize=(8, 6))
        plt.imshow(concentration_grid, origin="lower")
        plt.colorbar(label="Concentration")
        plt.title(f"{base.model} Forecast")
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
