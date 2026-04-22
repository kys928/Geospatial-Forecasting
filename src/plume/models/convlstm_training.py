from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.convlstm_contract import (
    CONVLSTM_GRID_HEIGHT,
    CONVLSTM_GRID_WIDTH,
    CONVLSTM_INPUT_CHANNELS,
    CONVLSTM_NORMALIZATION_MODE,
    CONVLSTM_PLUME_TARGET_CHANNEL,
    CONVLSTM_SEQUENCE_LENGTH,
    CONVLSTM_STORED_INPUT_SHAPE,
    CONVLSTM_STORED_TARGET_SHAPE,
)


@dataclass(frozen=True)
class ConvLSTMTrainingConfig:
    """Narrow Phase-D training configuration for plume-only supervision."""

    learning_rate: float = 1e-3
    target_policy: str = "plume_only"
    normalization_mode: str = CONVLSTM_NORMALIZATION_MODE
    loss_space: str = "transformed_plume"
    raw_interpretation_formula: str = "raw = (exp(pred) - 1) / 1e12"


class CanonicalConvLSTMSampleDataset:
    """Loader/validator for stored canonical ConvLSTM samples.

    Stored per-sample contract:
    - input: (3, 10, 64, 64)
    - target: (1, 10, 64, 64)

    Supervised objective in this phase slices plume-only from stored target.
    """

    def __init__(self, sample_paths: list[str | Path]):
        self.sample_paths = [Path(p) for p in sample_paths]

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        path = self.sample_paths[idx]
        with np.load(path, allow_pickle=False) as sample:
            if "input" not in sample or "target" not in sample:
                raise ValueError(f"Malformed sample {path}: expected keys 'input' and 'target'")
            input_tensor = np.asarray(sample["input"], dtype=float)
            target_tensor = np.asarray(sample["target"], dtype=float)
        self.validate_sample(input_tensor=input_tensor, target_tensor=target_tensor, source=str(path))
        return input_tensor, target_tensor

    @staticmethod
    def validate_sample(*, input_tensor: np.ndarray, target_tensor: np.ndarray, source: str = "<memory>") -> None:
        if input_tensor.shape != CONVLSTM_STORED_INPUT_SHAPE:
            raise ValueError(
                f"Malformed sample {source}: input shape must be {CONVLSTM_STORED_INPUT_SHAPE}, got {input_tensor.shape}"
            )
        if target_tensor.shape != CONVLSTM_STORED_TARGET_SHAPE:
            raise ValueError(
                f"Malformed sample {source}: target shape must be {CONVLSTM_STORED_TARGET_SHAPE}, got {target_tensor.shape}"
            )
        if not np.isfinite(input_tensor).all():
            raise ValueError(f"Malformed sample {source}: input contains non-finite values")
        if not np.isfinite(target_tensor).all():
            raise ValueError(f"Malformed sample {source}: target contains non-finite values")

        if target_tensor.shape[1] <= CONVLSTM_PLUME_TARGET_CHANNEL:
            raise ValueError(
                f"Malformed sample {source}: plume channel index {CONVLSTM_PLUME_TARGET_CHANNEL} is missing in target"
            )
        plume_channel = target_tensor[:, CONVLSTM_PLUME_TARGET_CHANNEL, :, :]
        if plume_channel.shape != (1, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH):
            raise ValueError(
                f"Malformed sample {source}: plume channel slice shape must be "
                f"(1, {CONVLSTM_GRID_HEIGHT}, {CONVLSTM_GRID_WIDTH}), got {plume_channel.shape}"
            )
        if not np.isfinite(plume_channel).all():
            raise ValueError(f"Malformed sample {source}: plume channel contains non-finite values")



def slice_plume_target(stored_target: np.ndarray) -> np.ndarray:
    """Slice plume-only supervision target while preserving leading dimensions.

    Input:  (B, 1, 10, 64, 64)
    Output: (B, 1, 1, 64, 64)
    """

    expected = ("B", *CONVLSTM_STORED_TARGET_SHAPE)
    if stored_target.ndim != 5:
        raise ValueError(f"Expected batched stored target with shape {expected}, got {stored_target.shape}")
    if tuple(stored_target.shape[1:]) != CONVLSTM_STORED_TARGET_SHAPE:
        raise ValueError(
            f"Batched stored target must preserve canonical shape {CONVLSTM_STORED_TARGET_SHAPE} after batch dimension, "
            f"got {stored_target.shape[1:]}"
        )
    plume = stored_target[:, :, CONVLSTM_PLUME_TARGET_CHANNEL : CONVLSTM_PLUME_TARGET_CHANNEL + 1, :, :]
    return plume


class ConvLSTMPlumeTrainer:
    """Phase-D trainer path: plume-only next-frame supervision with no extra normalization."""

    def __init__(self, model: MinimalConvLSTMModel, config: ConvLSTMTrainingConfig | None = None):
        self.model = model
        self.config = config or ConvLSTMTrainingConfig()
        if self.config.target_policy != "plume_only":
            raise ValueError(f"Phase-D trainer requires target_policy='plume_only', got {self.config.target_policy}")
        if self.config.normalization_mode != CONVLSTM_NORMALIZATION_MODE:
            raise ValueError(
                "Phase-D trainer requires canonical normalization_mode='none'; "
                f"got {self.config.normalization_mode}"
            )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "target_policy": self.config.target_policy,
            "stored_target_contract": CONVLSTM_STORED_TARGET_SHAPE,
            "supervised_target_contract": (1, 1, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH),
            "normalization_mode": self.config.normalization_mode,
            "loss_space": self.config.loss_space,
            "raw_interpretation_formula": self.config.raw_interpretation_formula,
        }

    def train_step(self, batch_input: np.ndarray, batch_target: np.ndarray) -> float:
        self._validate_batch(batch_input=batch_input, batch_target=batch_target)
        plume_target = slice_plume_target(batch_target)

        batch_size = batch_input.shape[0]
        grad_w_out = np.zeros_like(self.model.w_out)
        grad_b_out = 0.0
        total_loss = 0.0

        for i in range(batch_size):
            sequence = batch_input[i]
            hidden = self._last_hidden_state(sequence)
            pred = np.einsum("h,hrc->rc", self.model.w_out, hidden) + self.model.b_out
            pred = np.clip(pred, a_min=0.0, a_max=None)
            target = plume_target[i, 0, 0]
            error = pred - target
            total_loss += float(np.mean(error**2))

            grad_map = (2.0 / error.size) * error
            grad_w_out += np.einsum("rc,hrc->h", grad_map, hidden)
            grad_b_out += float(np.sum(grad_map))

        grad_w_out /= batch_size
        grad_b_out /= batch_size
        self.model.w_out -= self.config.learning_rate * grad_w_out
        self.model.b_out -= self.config.learning_rate * grad_b_out

        return total_loss / batch_size

    def _validate_batch(self, *, batch_input: np.ndarray, batch_target: np.ndarray) -> None:
        if batch_input.ndim != 5:
            raise ValueError(
                f"Expected batched input with shape (B, {CONVLSTM_SEQUENCE_LENGTH}, {CONVLSTM_INPUT_CHANNELS}, 64, 64), "
                f"got {batch_input.shape}"
            )
        if batch_target.ndim != 5:
            raise ValueError(
                f"Expected batched target with shape (B, 1, {CONVLSTM_INPUT_CHANNELS}, 64, 64), got {batch_target.shape}"
            )
        if tuple(batch_input.shape[1:]) != CONVLSTM_STORED_INPUT_SHAPE:
            raise ValueError(
                f"Batched input must preserve canonical shape {CONVLSTM_STORED_INPUT_SHAPE} after batch dimension, "
                f"got {batch_input.shape[1:]}"
            )
        if tuple(batch_target.shape[1:]) != CONVLSTM_STORED_TARGET_SHAPE:
            raise ValueError(
                f"Batched target must preserve canonical shape {CONVLSTM_STORED_TARGET_SHAPE} after batch dimension, "
                f"got {batch_target.shape[1:]}"
            )
        if batch_input.shape[0] != batch_target.shape[0]:
            raise ValueError(
                f"Input/target batch size mismatch: input batch={batch_input.shape[0]}, target batch={batch_target.shape[0]}"
            )
        if not np.isfinite(batch_input).all() or not np.isfinite(batch_target).all():
            raise ValueError("Batched input/target must be finite")

    def _last_hidden_state(self, sequence: np.ndarray) -> np.ndarray:
        h_t = np.zeros((self.model.hidden_channels, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH), dtype=float)
        c_t = np.zeros((self.model.hidden_channels, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH), dtype=float)

        for t in range(CONVLSTM_SEQUENCE_LENGTH):
            x_t = sequence[t]
            gates = (
                np.einsum("gc,chw->ghw", self.model.w_x, x_t)
                + np.einsum("gc,chw->ghw", self.model.w_h, h_t)
                + self.model.b[:, np.newaxis, np.newaxis]
            )
            i, f, o, g = np.split(gates, 4, axis=0)
            c_t = self.model._sigmoid(f) * c_t + self.model._sigmoid(i) * np.tanh(g)
            h_t = self.model._sigmoid(o) * np.tanh(c_t)
        return h_t
