from dataclasses import dataclass

@dataclass
class LLMConfig:
    enabled: bool
    provider: str
    model: str
    forecast_summary_only: bool
    timeout_seconds: float
