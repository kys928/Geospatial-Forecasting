from __future__ import annotations

from plume.services.explain_service import ExplainService
from plume.services.forecast_service import ForecastService
from plume.utils.config import Config


def main(config_dir: str | None = None) -> None:
    service = ForecastService(Config(config_dir=config_dir))
    explain_service = ExplainService()

    result = service.run_forecast()
    summary = service.summarize_forecast(result)

    print("Forecast generated successfully.")
    print(f"Forecast ID: {summary['forecast_id']}")
    print(f"Run name: {summary['run_name']}")
    print(f"Model: {summary['model']}")
    print(f"Issued at: {summary['issued_at']}")
    print(f"Grid: {summary['grid']['rows']}x{summary['grid']['columns']}")
    for key, value in summary["summary_statistics"].items():
        print(f"{key}: {value:.6e}")

    explanation = explain_service.explain(result, use_llm=False)
    print("Explanation:", explanation.explanation["summary"])


if __name__ == "__main__":
    main()
