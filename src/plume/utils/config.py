from pathlib import Path

import yaml

from ..schemas.Base import Base
from ..schemas.Inference import Inference, Plot
from ..schemas.LLMConfig import LLMConfig
from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario


class Config:
    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir is None:
            self.config_dir = Path(__file__).resolve().parents[3] / "configs"
        else:
            self.config_dir = Path(config_dir)

    def load_grid(self) -> GridSpec:
        grid_yaml = self.config_dir / "grid.yaml"
        with grid_yaml.open("r", encoding="utf-8") as f:
            grid = yaml.safe_load(f)
        return GridSpec(**grid)

    def load_scenario(self) -> Scenario:
        scenario_yaml = self.config_dir / "scenario.yaml"
        with scenario_yaml.open("r", encoding="utf-8") as f:
            scenario = yaml.safe_load(f)
        return Scenario(**scenario)

    def load_inference(self) -> Inference:
        inference_yaml = self.config_dir / "inference.yaml"
        with inference_yaml.open("r", encoding="utf-8") as f:
            inference = yaml.safe_load(f)
        inference["plot"] = Plot(**inference["plot"])
        return Inference(**inference)

    def load_base(self) -> Base:
        base_yaml = self.config_dir / "base.yaml"
        with base_yaml.open("r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        return Base(**base)

    def load_llm(self) -> LLMConfig:
        api_yaml = self.config_dir / "api.yaml"
        with api_yaml.open("r", encoding="utf-8") as f:
            llm = yaml.safe_load(f)
        return LLMConfig(**llm)
