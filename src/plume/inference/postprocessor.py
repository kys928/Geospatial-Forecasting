from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..schemas.forecast import Forecast
from ..schemas.Inference import Inference


class ForecastPostprocessor:
    def __init__(self, inference_config: Inference) -> None:
        self.inference_config = inference_config

    def compute_summary_statistics(self, forecast: Forecast) -> dict[str, float]:
        concentration_grid = forecast.concentration_grid
        requested = set(self.inference_config.summary_statistics)

        summary_statistics: dict[str, float] = {}

        if "max_concentration" in requested:
            summary_statistics["max_concentration"] = float(np.max(concentration_grid))

        if "mean_concentration" in requested:
            summary_statistics["mean_concentration"] = float(np.mean(concentration_grid))

        return summary_statistics

    def should_plot(self) -> bool:
        return bool(self.inference_config.plot.enabled)

    def plot_forecast(self, forecast: Forecast, title: str = "Forecast Concentration Grid") -> None:
        concentration_grid = forecast.concentration_grid

        plt.figure(figsize=(8, 6))
        plt.imshow(concentration_grid, origin="lower")
        plt.colorbar(label="Concentration")
        plt.title(title)
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")
        plt.tight_layout()
        plt.show()

    def process(self, forecast: Forecast, title: str = "Forecast Concentration Grid") -> dict[str, float]:
        summary_statistics = self.compute_summary_statistics(forecast)

        if self.should_plot():
            self.plot_forecast(forecast, title=title)

        return summary_statistics