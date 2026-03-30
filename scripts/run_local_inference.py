import matplotlib.pyplot as plt
import numpy as np
from src.plume.inference.engine import InferenceEngine
from src.plume.models.gaussian_plume import GaussianPlume
from src.plume.utils.config import Config
from src.plume.inference.postprocessor import ForecastPostprocessor


def main():
    config = Config()
    scenario = config.load_scenario()
    grid_spec = config.load_grid()
    base = config.load_base()
    inference = config.load_inference()

    if base.model != "gaussian_plume":
        raise ValueError(f"Unsupported model in base.yaml: {base.model}")

    model = GaussianPlume(grid_spec=grid_spec, scenario=scenario)
    engine = InferenceEngine(model=model, validate_inputs=inference.validate_inputs)

    postprocessor = ForecastPostprocessor(inference)

    forecast = engine.run_inference(scenario, grid_spec)
    summary_statistics = postprocessor.process(
        forecast,
        title=f"{base.model} Forecast",
    )

    print("Forecast generated successfully.")
    print(f"Run name: {base.run_name}")
    print(f"Model: {base.model}")
    print(f"Grid shape: {forecast.concentration_grid.shape}")
    print(f"Timestamp: {forecast.timestamp}")

    for stat_name, value in summary_statistics.items():
        print(f"{stat_name}: {value:.6f}")


if __name__ == "__main__":
    main()