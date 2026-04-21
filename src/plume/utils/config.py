from pathlib import Path
import yaml

from ..schemas.Inference import Inference, Plot
from ..schemas.grid import GridSpec
from ..schemas.scenario import Scenario
from ..schemas.Base import Base


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
        grid["grid_center"] = tuple(grid["grid_center"])
        grid["boundary_limits"] = tuple(grid["boundary_limits"])
        return GridSpec(**grid)

    def load_scenario(self) -> Scenario:
        scenario_yaml = self.config_dir / "scenario.yaml"
        with scenario_yaml.open("r", encoding="utf-8") as f:
            scenario = yaml.safe_load(f)
        scenario["source"] = tuple(scenario["source"])
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

    def load_backend(self) -> dict[str, object]:
        backend_yaml = self.config_dir / "backend.yaml"
        with backend_yaml.open("r", encoding="utf-8") as f:
            backend = yaml.safe_load(f)
        return backend

    def load_openremote(self) -> dict[str, object]:
        openremote_yaml = self.config_dir / "openremote.yaml"
        if openremote_yaml.exists():
            with openremote_yaml.open("r", encoding="utf-8") as f:
                openremote = yaml.safe_load(f) or {}
        else:
            openremote = {}

        defaults: dict[str, object] = {
            "enabled": False,
            "sink_mode": "disabled",
            "base_url": "",
            "realm": None,
            "site_asset_id": None,
            "parent_asset_id": None,
            "geojson_public_base_url": None,
            "access_token_env_var": "OPENREMOTE_ACCESS_TOKEN",
        }
        defaults.update(openremote)
        return defaults
