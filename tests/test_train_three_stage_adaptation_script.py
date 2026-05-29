from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_three_stage_adaptation.py"
    spec = importlib.util.spec_from_file_location("train_three_stage_adaptation_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def _write_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input=np.zeros((3, 10, 64, 64), dtype=np.float32),
        target=np.zeros((4, 1, 64, 64), dtype=np.float32),
    )
    return path


def _register_accepted(buffer: AdaptationBuffer, tmp_path: Path, sample_id: str) -> None:
    source = _write_npz(tmp_path / "source" / f"{sample_id}.npz")
    buffer.register_npz_window(source, sample_id=sample_id)
    buffer.accept_pending_sample(sample_id)


def test_parse_args_requires_output_dir() -> None:
    parser = script.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--reference-dataset-dir", "/tmp/reference"])


def test_dry_run_writes_manifest_preview(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _write_npz(reference / "train" / "train-0.npz")
    _write_npz(reference / "train" / "train-1.npz")
    _write_npz(reference / "val" / "val-0.npz")
    output_dir = tmp_path / "run"

    result = script.main(
        [
            "--reference-dataset-dir",
            str(reference),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert result == 0
    preview_path = output_dir / "dataset_manifest_preview.json"
    assert preview_path.exists()
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    assert preview["counts"]["reference_train"] == 2
    assert preview["counts"]["reference_val"] == 1
    assert preview["counts"]["train_total"] == 2
    assert preview["counts"]["val_total"] == 1
    assert not (output_dir / "dataset_manifest.json").exists()


def test_dry_run_fails_or_reports_no_samples(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reference = tmp_path / "empty_reference"
    reference.mkdir()
    output_dir = tmp_path / "run"

    result = script.main(
        [
            "--reference-dataset-dir",
            str(reference),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "no train/validation sample pair" in captured.err
    preview_path = output_dir / "dataset_manifest_preview.json"
    assert preview_path.exists()
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    assert preview["counts"]["train_total"] == 0
    assert preview["counts"]["val_total"] == 0
    assert "no train samples selected" in preview["warnings"]
    assert "no validation samples selected" in preview["warnings"]


def test_script_builds_manifest_from_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLUME_ADAPTATION_BUFFER_DIR", raising=False)
    buffer_root = tmp_path / "buffer"
    buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=buffer_root))
    for index in range(5):
        _register_accepted(buffer, tmp_path, f"fresh-{index}")
    output_dir = tmp_path / "run"

    result = script.main(
        [
            "--buffer-root",
            str(buffer_root),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert result == 0
    preview = json.loads((output_dir / "dataset_manifest_preview.json").read_text(encoding="utf-8"))
    assert preview["counts"]["fresh_buffer_train"] == 4
    assert preview["counts"]["fresh_buffer_val"] == 1
    assert preview["counts"]["train_total"] == 4
    assert preview["counts"]["val_total"] == 1
    assert {sample["source"] for sample in preview["train_samples"] + preview["val_samples"]} == {"fresh_buffer"}
