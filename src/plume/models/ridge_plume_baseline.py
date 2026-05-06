from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


EXPECTED_INPUT_SHAPE = (3, 10, 64, 64)


def downsample_grid(grid: np.ndarray, factor: int = 4) -> np.ndarray:
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {grid.shape}")
    height, width = grid.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"Grid shape {grid.shape} must be divisible by factor={factor}")
    return grid.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def extract_features(input_window: np.ndarray) -> np.ndarray:
    if input_window.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"Ridge baseline expects input shape {EXPECTED_INPUT_SHAPE}, got {input_window.shape}")
    concentration_frames = input_window[:, 0, :, :]
    concentration_downsampled = [downsample_grid(frame, factor=4).ravel() for frame in concentration_frames]
    concentration_flat = np.concatenate(concentration_downsampled)

    meteorology = input_window[:, 1:10, :, :]
    meteorology_summary = meteorology.mean(axis=(2, 3)).ravel()

    concentration_stats = np.concatenate(
        [concentration_frames.mean(axis=(1, 2)), concentration_frames.max(axis=(1, 2))]
    )
    return np.concatenate([concentration_flat, meteorology_summary, concentration_stats])


def load_ridge_artifact(path: str | Path) -> dict[str, object]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Ridge baseline prediction engine is enabled, but model artifact was not found at: {artifact_path}"
        )
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("Ridge baseline artifact must be a dict with metadata")
    if "model" not in artifact:
        raise ValueError("Ridge baseline artifact is missing required key: 'model'")
    downsample_factor = int(artifact.get("downsample_factor", 4))
    if downsample_factor != 4:
        raise ValueError(f"Ridge baseline artifact downsample_factor must be 4, got {downsample_factor}")
    artifact["downsample_factor"] = downsample_factor
    return artifact


def predict_ridge_plume(input_window: np.ndarray, artifact: dict[str, object]) -> np.ndarray:
    features = extract_features(input_window)
    model = artifact["model"]
    pred = np.asarray(model.predict(features[None, :]), dtype=float).reshape(-1)
    if pred.size != 16 * 16:
        raise ValueError(f"Ridge baseline prediction output must have 256 values, got {pred.size}")
    coarse = pred.reshape(16, 16)
    upsampled = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)
    cleaned = np.nan_to_num(upsampled, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(cleaned, a_min=0.0, a_max=None)
