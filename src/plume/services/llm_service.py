from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import yaml
from huggingface_hub import InferenceClient

from ..schemas.LLMConfig import LLMConfig
from ..schemas.ForecastSummary import ForecastSummary
from ..schemas.LLMInterpretationResult import LLMInterpretationResult
from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario


def load_llm_config(config_path: str | Path) -> LLMConfig:
    """
    Load LLM configuration from api.yaml and return a typed config object.
    """
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid LLM config in {path}: expected a mapping/dictionary.")

    return LLMConfig(**config_dict)


class LLMService:
    """
    API-backed LLM service using Hugging Face Inference.

    Responsibility:
    - Accept a prepared forecast summary
    - Send it to an external LLM through API
    - Return a structured interpretation
    """

    def __init__(self, llm_config: LLMConfig, api_key: str | None = None, temperature: float = 0.2, max_output_tokens: int = 500):
        self.llm_config = llm_config
        self.model_name = llm_config.model
        self.provider = llm_config.provider
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = llm_config.timeout_seconds
        self.forecast_summary_only = llm_config.forecast_summary_only
        self.enabled = llm_config.enabled

        if not self.enabled:
            raise ValueError("LLMService was initialized, but LLM config has enabled=False.")

        supported_providers = {"auto", "hf-inference"}

        if self.provider not in supported_providers:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"This service currently supports only: {sorted(supported_providers)}."
            )

        resolved_api_key = (
            api_key
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if not resolved_api_key:
            raise ValueError(
                "HF_TOKEN is not set. Pass api_key explicitly or set HF_TOKEN in the environment."
            )

        self.client = InferenceClient(
            model=self.model_name,
            token=resolved_api_key,
            provider=self.provider,
            timeout=self.timeout_seconds,
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path, api_key: str | None = None, temperature: float = 0.2, max_output_tokens: int = 500):
        llm_config = load_llm_config(config_path)
        return cls(
            llm_config=llm_config,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def interpret_forecast(
            self,
            forecast_summary: ForecastSummary,
    ) -> LLMInterpretationResult:
        """
        Accepts already-prepared forecast summary data
        and returns a structured interpretation.
        """
        try:
            system_prompt = self._build_instructions()
            user_input = self._build_user_input(forecast_summary)

            completion = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )

            raw_text = self._extract_chat_text(completion).strip()

            if not raw_text:
                return LLMInterpretationResult(
                    success=False,
                    summary=None,
                    risk_level=None,
                    recommendation=None,
                    uncertainty_note=None,
                    raw_text=None,
                    error="The model returned an empty response.",
                    provider=self.provider,
                    model=self.model_name,
                )

            parsed = self._safe_parse_json(raw_text)
            if parsed is None:
                return LLMInterpretationResult(
                    success=True,
                    summary=None,
                    risk_level=None,
                    recommendation=None,
                    uncertainty_note=None,
                    raw_text=raw_text,
                    error="Model returned text, but not valid JSON in the expected format.",
                    provider=self.provider,
                    model=self.model_name,
                )

            return LLMInterpretationResult(
                success=True,
                summary=self._safe_get_str(parsed, "summary"),
                risk_level=self._safe_get_str(parsed, "risk_level"),
                recommendation=self._safe_get_str(parsed, "recommendation"),
                uncertainty_note=self._safe_get_str(parsed, "uncertainty_note"),
                raw_text=raw_text,
                error=None,
                provider=self.provider,
                model=self.model_name,
            )

        except Exception as e:
            return LLMInterpretationResult(
                success=False,
                summary=None,
                risk_level=None,
                recommendation=None,
                uncertainty_note=None,
                raw_text=None,
                error=str(e),
                provider=self.provider,
                model=self.model_name,
            )

    def interpret_forecast_stream(
        self,
        forecast_summary: ForecastSummary,
    ) -> Iterator[str]:
        system_prompt = self._build_instructions()
        user_input = self._build_user_input(forecast_summary)

        stream = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            try:
                choice = chunk.choices[0]
                delta = getattr(choice.delta, "content", None)
                if delta:
                    yield delta
            except Exception:
                continue

    def summarize_from_scenario_and_grid(
        self,
        scenario: Scenario,
        grid_spec: GridSpec,
        *,
        max_concentration: float,
        mean_concentration: float,
        affected_cells_above_threshold: int,
        dominant_spread_direction: str,
        threshold_used: float,
        note: str | None = None,
    ) -> ForecastSummary:
        return ForecastSummary(
            source_latitude=float(scenario.latitude),
            source_longitude=float(scenario.longitude),
            grid_rows=int(grid_spec.number_of_rows),
            grid_columns=int(grid_spec.number_of_columns),
            projection=getattr(grid_spec, "projection", None),
            max_concentration=float(max_concentration),
            mean_concentration=float(mean_concentration),
            affected_cells_above_threshold=int(affected_cells_above_threshold),
            dominant_spread_direction=str(dominant_spread_direction),
            threshold_used=float(threshold_used),
            note=note,
        )

    def _build_instructions(self) -> str:
        return (
            "You are a geospatial hazard interpretation assistant. "
            "You receive a structured forecast summary from a forecasting pipeline. "
            "Your task is to interpret the summary conservatively. "
            "Do not invent measurements, physics, times, or locations that were not provided. "
            "Return ONLY valid JSON with exactly these fields: "
            "summary, risk_level, recommendation, uncertainty_note. "
            "Keep summary concise and operational. "
            "Use risk_level as one of: low, moderate, high, critical."
        )

    def _build_user_input(self, forecast_summary: ForecastSummary) -> str:
        payload = {
            "source_latitude": forecast_summary.source_latitude,
            "source_longitude": forecast_summary.source_longitude,
            "grid_rows": forecast_summary.grid_rows,
            "grid_columns": forecast_summary.grid_columns,
            "projection": forecast_summary.projection,
            "max_concentration": forecast_summary.max_concentration,
            "mean_concentration": forecast_summary.mean_concentration,
            "affected_cells_above_threshold": forecast_summary.affected_cells_above_threshold,
            "dominant_spread_direction": forecast_summary.dominant_spread_direction,
            "threshold_used": forecast_summary.threshold_used,
            "note": forecast_summary.note,
        }

        return (
            "Interpret the following forecast summary and return strict JSON only.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

    @staticmethod
    def _extract_chat_text(completion: Any) -> str:
        try:
            content = completion.choices[0].message.content
            if isinstance(content, str):
                return content
            return ""
        except Exception:
            return ""

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _safe_get_str(data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        return str(value).strip()