from dataclasses import dataclass


@dataclass
class Plot:
    enabled: bool
    interactive: str

@dataclass
class Inference:
    mode: str
    validate_inputs: bool
    return_forecast_object: bool
    summary_statistics: list[str]
    plot: Plot



