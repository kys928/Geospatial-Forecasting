"""Training-facing import for the robust multi-step ConvLSTM model.

This module intentionally re-exports the model implementation used by serving
so adaptation trainers do not duplicate architecture code.
"""

from __future__ import annotations

from plume.models.torch_robust_multistep_convlstm import (  # noqa: F401
    DEFAULT_INPUT_SHAPE,
    DEFAULT_OUTPUT_SHAPE,
    ROBUST_MODEL_NAME,
    RobustMultiStepConvLSTMForecaster,
)

__all__ = [
    "DEFAULT_INPUT_SHAPE",
    "DEFAULT_OUTPUT_SHAPE",
    "ROBUST_MODEL_NAME",
    "RobustMultiStepConvLSTMForecaster",
]
