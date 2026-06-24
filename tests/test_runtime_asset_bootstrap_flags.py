import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_runtime_assets.py"
_SPEC = importlib.util.spec_from_file_location("bootstrap_runtime_assets", _MODULE_PATH)
bootstrap = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bootstrap
assert _SPEC.loader is not None
_SPEC.loader.exec_module(bootstrap)


def test_default_model_assets_are_huggingface_configured(monkeypatch):
    for name in (
        "PLUME_LLM_HF_REPO_ID",
        "PLUME_LLM_HF_FILENAME",
        "PLUME_CONVLSTM_HF_REPO_ID",
        "PLUME_CONVLSTM_HF_FILENAME",
        "PLUME_CONVLSTM_SHA256_EXPECTED",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = bootstrap.Config.from_env()

    assert cfg.download_assets is True
    assert cfg.download_model_assets is True
    assert cfg.download_dataset is False
    assert cfg.require_dataset is False
    assert cfg.convlstm_checkpoint_path.as_posix().endswith(
        "artifacts/models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"
    )
    assert cfg.convlstm_sha256_expected == bootstrap.DEFAULT_CONVLSTM_SHA256
    assert bootstrap.defaulted_env("PLUME_LLM_HF_REPO_ID", bootstrap.DEFAULT_HF_REPO_ID) == (
        "DavidDulovic/geospatial-plume-runtime-assets"
    )
    assert bootstrap.defaulted_env("PLUME_CONVLSTM_HF_FILENAME", bootstrap.DEFAULT_CONVLSTM_HF_FILENAME) == (
        "models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"
    )


def test_empty_convlstm_sha_disables_validation(monkeypatch):
    monkeypatch.setenv("PLUME_CONVLSTM_SHA256_EXPECTED", "")

    cfg = bootstrap.Config.from_env()

    assert cfg.convlstm_sha256_expected is None


def test_optional_dataset_does_not_download_without_dataset_flag(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("PLUME_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(tmp_path / "model.gguf"))
    monkeypatch.setenv("PLUME_CONVLSTM_CHECKPOINT_PATH", str(tmp_path / "checkpoint.pt"))
    monkeypatch.setenv("PLUME_SETUP_DOWNLOAD_MODEL_ASSETS", "false")
    monkeypatch.setenv("PLUME_SETUP_DOWNLOAD_DATASET", "false")
    monkeypatch.setattr(bootstrap, "download_kaggle_dataset", lambda target: calls.append(target))

    cfg = bootstrap.Config.from_env()
    bootstrap.maybe_download_assets(cfg)

    assert calls == []
