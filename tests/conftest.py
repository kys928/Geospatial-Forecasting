from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario


@pytest.fixture
def sample_grid_spec() -> GridSpec:
    return GridSpec(
        grid_height=1000.0,
        grid_width=1000.0,
        grid_center=(34.05, -118.25),
        grid_spacing=100.0,
        number_of_rows=32,
        number_of_columns=32,
        projection="EPSG:4326",
        boundary_limits=(33.9, -118.4, 34.2, -118.1),
    )


@pytest.fixture
def sample_scenario() -> Scenario:
    start = datetime.now(timezone.utc).replace(microsecond=0)
    end = start + timedelta(hours=1)
    return Scenario(
        source=(34.05, -118.25),
        latitude=34.05,
        longitude=-118.25,
        start=start,
        end=end,
        emissions_rate=25.0,
        pollution_type="SO2",
        duration=3600.0,
        release_height=10.0,
    )


def make_lightweight_convlstm_config(tmp_path: Path):
    import shutil
    import yaml
    from plume.utils.config import Config

    cfg_dir = tmp_path / "config"
    shutil.copytree(REPO_ROOT / "configs", cfg_dir)
    backend_yaml = cfg_dir / "backend.yaml"
    backend = yaml.safe_load(backend_yaml.read_text(encoding="utf-8"))
    backend["convlstm_prediction_engine"] = "convlstm"
    backend["convlstm_init_mode"] = "random_init"
    backend["convlstm_checkpoint_path"] = None
    backend["convlstm_device"] = "cpu"
    backend["use_model_registry"] = False
    backend_yaml.write_text(yaml.safe_dump(backend), encoding="utf-8")
    return Config(config_dir=cfg_dir)


@pytest.fixture
def lightweight_convlstm_config(tmp_path: Path):
    return make_lightweight_convlstm_config(tmp_path)
