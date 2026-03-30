from __future__ import annotations
import json
import os
import yaml
from typing import Any
from huggingface_hub import InferenceClient
from ..schemas.LLMConfig import LLMConfig
from ..schemas.ForecastSummary import ForecastSummary
from ..schemas.scenario import Scenario
from ..schemas.grid import GridSpec
from ..schemas.LLMInterpretationResult import LLMInterpretationResult



def load_llm_config(self) -> LLMConfig:
    self.config = "../configs/api.yaml"
    with open(self.config, "r") as f:
        self.config_dict = yaml.safe_load(f)
    return LLMConfig(**self.config_dict)





class LLMService:
    """
    API-backed LLM service using Hugging Face Inference.

    Responsibility:
    - Accept a prepared forecast summary
    - Send it to an external LLM through API
    - Return a structured interpretation
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 500,
        provider: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.provider = provider

        resolved_api_key = (
            api_key
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if not resolved_api_key:
            raise ValueError(
                "HF_TOKEN is not set. Pass api_key explicitly or set HF_TOKEN in the environment."
            )

        # model=... lets HF route the request to the hosted inference backend
        # provider=... is optional; omit it unless you want to pin a specific provider
        self.client = InferenceClient(
            model=self.model_name,
            token=resolved_api_key,
            provider=self.provider,
        )

    def interpret_forecast(self, forecast_summary: ForecastSummary) -> LLMInterpretationResult:
        """
        Main public method. Accepts already-prepared forecast summary data
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
                model=self.model_name,
            )

    def interpret_forecast_stream(self, forecast_summary: ForecastSummary):

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
            delta = None

            try:
                choice = chunk.choices[0]
                delta = getattr(choice.delta, "content", None)
            except Exception:
                delta = None

            if delta:
                yield delta

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
            'summary, risk_level, recommendation, uncertainty_note. '
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
            return completion.choices[0].message.content or ""
        except Exception:
            return ""

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any] | None:
        """
        Attempts to parse JSON. Also handles the common case where the model
        wraps JSON in markdown fences.
        """
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