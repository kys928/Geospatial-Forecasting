from __future__ import annotations

import json
import os
import subprocess
import tempfile
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
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid LLM config in {path}: expected a mapping/dictionary.")

    return LLMConfig(**config_dict)


class LLMService:
    def __init__(
        self,
        llm_config: LLMConfig,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 500,
    ):
        self.llm_config = llm_config
        raw_provider = os.getenv("PLUME_LLM_PROVIDER") or llm_config.provider
        provider_aliases = {
            "huggingface": "hf-inference",
            "llama-cpp": "local-gguf",
            "llama_cpp": "local-gguf",
        }
        self.provider = provider_aliases.get(raw_provider, raw_provider)
        self.local_gguf_path: str | None = None
        self.llama_cpp_bin: str | None = None
        self.model_name = llm_config.model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = llm_config.timeout_seconds
        self.forecast_summary_only = llm_config.forecast_summary_only
        self.enabled = llm_config.enabled

        if not self.enabled:
            raise ValueError("LLMService was initialized, but LLM config has enabled=False.")

        supported_providers = {"auto", "hf-inference", "local-gguf"}

        if self.provider not in supported_providers:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"This service currently supports only: {sorted(supported_providers)}."
            )

        if self.provider == "local-gguf":
            self.local_gguf_path = os.getenv(
                "PLUME_LOCAL_LLM_GGUF_PATH",
                "/workspace/llm_runtime/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf",
            )
            self.llama_cpp_bin = os.getenv(
                "PLUME_LLAMA_CPP_BIN",
                "/workspace/llm_runtime/tools/llama.cpp/build/bin/llama-cli",
            )
            gguf = Path(self.local_gguf_path)
            llama_bin = Path(self.llama_cpp_bin)
            if not gguf.exists():
                raise ValueError(f"Local GGUF path does not exist: {gguf}")
            if not llama_bin.exists():
                raise ValueError(f"llama.cpp binary does not exist: {llama_bin}")
            if not os.access(llama_bin, os.X_OK):
                raise ValueError(f"llama.cpp binary is not executable: {llama_bin}")
            self.model_name = os.getenv("PLUME_LOCAL_LLM_MODEL_NAME") or gguf.name
            self.client = None
        else:
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
    def from_yaml(
        cls,
        config_path: str | Path,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 500,
    ):
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
                    success=False,
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


    def interpret_context(self, *, system_prompt: str, context: dict[str, Any]) -> LLMInterpretationResult:
        try:
            user_prompt = "Forecast context:\n" + json.dumps(context, indent=2)
            if self.provider == "local-gguf":
                raw_text = self._run_local_gguf_prompt(system_prompt, user_prompt).strip()
            else:
                completion = self.client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
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
                    success=False,
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

    def answer_context_question(self, *, system_prompt: str, context: dict[str, Any], question: str) -> dict[str, Any]:
        try:
            user_prompt = f"Question: {question}\n\nForecast context:\n{json.dumps(context, indent=2)}"
            if self.provider == "local-gguf":
                answer = self._run_local_gguf_prompt(system_prompt, user_prompt).strip()
            else:
                response = self.client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=int(os.getenv("PLUME_LOCAL_LLM_MAX_TOKENS", "300")),
                    temperature=float(os.getenv("PLUME_LOCAL_LLM_TEMPERATURE", str(self.temperature))),
                )
                answer = self._extract_chat_text(response).strip()
            if not answer:
                return {"success": False, "answer": None, "error": "The model returned an empty response."}
            return {"success": True, "answer": answer, "error": None}
        except Exception as exc:
            return {"success": False, "answer": None, "error": str(exc)}
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
        affected_area_m2: float,
        affected_area_hectares: float,
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
            affected_area_m2=float(affected_area_m2),
            affected_area_hectares=float(affected_area_hectares),
            dominant_spread_direction=str(dominant_spread_direction),
            threshold_used=float(threshold_used),
            note=note,
        )

    def _build_instructions(self) -> str:
        return (
            "You are a geospatial hazard explanation assistant for non-experts. "
            "Write like a calm operations briefing, not like a lab report. "
            "Do not invent times, places, weather, physics, casualties, or actions that were not provided. "
            "Return ONLY valid JSON with exactly these fields: "
            "summary, risk_level, recommendation, uncertainty_note. "
            "Rules for summary: "
            "1) Use 2 or 3 short sentences. "
            "2) Focus on what is happening, where the plume is moving, and how serious it feels. "
            "3) Avoid repeating raw numbers unless they are absolutely necessary. "
            "4) Prefer phrases like 'small affected area', 'moderate plume', 'strong inner core', "
            "'spreading mainly to the north-east', 'staying close to the source'. "
            "5) Do not repeat the exact area numbers or concentration numbers if the UI can show structural details elsewhere. "
            "6) If no meaningful plume is present, say that clearly and simply. "
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
            "affected_area_m2": forecast_summary.affected_area_m2,
            "affected_area_hectares": forecast_summary.affected_area_hectares,
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
        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                cleaned = cleaned[start:end + 1]

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    def _run_local_gguf_prompt(self, system_prompt: str, user_prompt: str) -> str:
        if not self.local_gguf_path or not self.llama_cpp_bin:
            raise RuntimeError("Local GGUF provider is not initialized.")
        prompt_file_path = None
        max_tokens = int(os.getenv("PLUME_LOCAL_LLM_MAX_TOKENS", str(self.max_output_tokens)))
        temperature = float(os.getenv("PLUME_LOCAL_LLM_TEMPERATURE", str(self.temperature)))
        top_p = float(os.getenv("PLUME_LOCAL_LLM_TOP_P", "0.9"))
        timeout_seconds = float(os.getenv("PLUME_LOCAL_LLM_TIMEOUT_SECONDS", "240"))
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as prompt_file:
                prompt_file.write(
                    "<|im_start|>system\n"
                    f"{system_prompt}\n"
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{user_prompt}\n"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                prompt_file_path = prompt_file.name
            proc = subprocess.run(
                [
                    self.llama_cpp_bin,
                    "-m", self.local_gguf_path,
                    "-f", prompt_file_path,
                    "-no-cnv",
                    "-n", str(max_tokens),
                    "--temp", str(temperature),
                    "--top-p", str(top_p),
                    "--repeat-penalty", "1.05",
                    "--no-display-prompt",
                    "--no-show-timings",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "local GGUF process failed")
            stdout = proc.stdout.strip()
            if not stdout:
                raise RuntimeError("local GGUF produced empty output")
            return stdout
        finally:
            if prompt_file_path:
                Path(prompt_file_path).unlink(missing_ok=True)

    @staticmethod
    def _safe_get_str(data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        return str(value).strip()
