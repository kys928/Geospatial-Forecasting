from __future__ import annotations

import numpy as np

# Canonical ConvLSTM contract for plume-path inference.
CONVLSTM_CONTRACT_VERSION = "convlstm_canonical_v1"
CONVLSTM_SEQUENCE_LENGTH = 3
CONVLSTM_INPUT_CHANNELS = 10
CONVLSTM_GRID_HEIGHT = 64
CONVLSTM_GRID_WIDTH = 64
CONVLSTM_RUNTIME_BATCHED_INPUT_SHAPE = (
    "B",
    CONVLSTM_INPUT_CHANNELS,
    CONVLSTM_SEQUENCE_LENGTH,
    CONVLSTM_GRID_HEIGHT,
    CONVLSTM_GRID_WIDTH,
)
CONVLSTM_STORED_INPUT_SHAPE = (
    CONVLSTM_SEQUENCE_LENGTH,
    CONVLSTM_INPUT_CHANNELS,
    CONVLSTM_GRID_HEIGHT,
    CONVLSTM_GRID_WIDTH,
)
CONVLSTM_STORED_TARGET_SHAPE = (1, CONVLSTM_INPUT_CHANNELS, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH)
CONVLSTM_PLUME_OUTPUT_SHAPE = ("B", 1, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH)
CONVLSTM_PLUME_TARGET_SHAPE = ("B", 1, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH)

CONVLSTM_CHANNEL_MANIFEST: tuple[str, ...] = (
    "plume_concentration",
    "u10m_ms",
    "v10m_ms",
    "wspd10_ms",
    "wdir_sin",
    "wdir_cos",
    "pblh_m",
    "sfcp_hpa",
    "rh2m_pct",
    "t02m_k",
)

CONVLSTM_TEMPORAL_SPACING = "hourly"
CONVLSTM_TEMPORAL_PATTERN = "x[t-2], x[t-1], x[t] -> x[t+1]"
CONVLSTM_NORMALIZATION_MODE = "none"
CONVLSTM_NORMALIZATION_DETAILS = "no extra normalization in adapter/backend by default"

PLUME_LOG1P_SCALE = 1e12

CONVLSTM_TARGET_POLICY = "stored_future_full_target_plume_supervision_first"
CONVLSTM_PLUME_TARGET_CHANNEL = 0


def plume_to_model_space(raw_concentration: np.ndarray | float) -> np.ndarray:
    """Apply canonical plume transform: log1p(1e12 * x)."""
    raw = np.asarray(raw_concentration, dtype=float)
    return np.log1p(PLUME_LOG1P_SCALE * raw)


def plume_to_physical_space(model_output: np.ndarray | float, *, clamp_non_negative: bool = True) -> np.ndarray:
    """Invert canonical plume transform: (exp(pred) - 1) / 1e12."""
    transformed = np.asarray(model_output, dtype=float)
    raw = (np.exp(transformed) - 1.0) / PLUME_LOG1P_SCALE
    if clamp_non_negative:
        raw = np.clip(raw, a_min=0.0, a_max=None)
    return raw
