from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.convlstm_contract import (
    CONVLSTM_GRID_HEIGHT,
    CONVLSTM_GRID_WIDTH,
    CONVLSTM_INPUT_CHANNELS,
    CONVLSTM_CONTRACT_VERSION,
    CONVLSTM_NORMALIZATION_MODE,
    CONVLSTM_PLUME_TARGET_CHANNEL,
    CONVLSTM_SEQUENCE_LENGTH,
    CONVLSTM_STORED_INPUT_SHAPE,
    CONVLSTM_STORED_TARGET_SHAPE,
    plume_to_physical_space,
)


@dataclass(frozen=True)
class ConvLSTMTrainingConfig:
    """Phase-E training configuration for plume-only supervision."""

    learning_rate: float = 1e-3
    target_policy: str = "plume_only"
    normalization_mode: str = CONVLSTM_NORMALIZATION_MODE
    loss_space: str = "transformed_plume"
    raw_interpretation_formula: str = "raw = (exp(pred) - 1) / 1e12"
    trainable_parameter_scope: str = "full_model"
    eval_metric: str = "val_mse"
    eval_direction: str = "min"
    checkpoint_metric: str = "val_mse"
    checkpoint_direction: str = "min"
    save_best_only: bool = True


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
    """Phase-E trainer path: full end-to-end plume-only supervision with no extra normalization."""

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
        if self.config.trainable_parameter_scope != "full_model":
            raise ValueError(
                "Phase-E trainer requires trainable_parameter_scope='full_model' for end-to-end optimization; "
                f"got {self.config.trainable_parameter_scope}"
            )
        if self.config.eval_direction not in {"min", "max"}:
            raise ValueError(f"eval_direction must be 'min' or 'max', got {self.config.eval_direction}")
        if self.config.checkpoint_direction not in {"min", "max"}:
            raise ValueError(
                f"checkpoint_direction must be 'min' or 'max', got {self.config.checkpoint_direction}"
            )

        self.best_checkpoint_metric_name: str | None = None
        self.best_checkpoint_metric_value: float | None = None
        self.best_checkpoint_epoch: int | None = None
        self.best_checkpoint_step: int | None = None

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": self.config.target_policy,
            "stored_target_contract": CONVLSTM_STORED_TARGET_SHAPE,
            "supervised_target_contract": (1, 1, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH),
            "normalization_mode": self.config.normalization_mode,
            "loss_space": self.config.loss_space,
            "raw_interpretation_formula": self.config.raw_interpretation_formula,
            "trainable_parameter_scope": self.config.trainable_parameter_scope,
            "trainable_parameters": self._trainable_parameter_names(),
            "eval_metric": self.config.eval_metric,
            "eval_direction": self.config.eval_direction,
            "checkpoint_metric": self.config.checkpoint_metric,
            "checkpoint_direction": self.config.checkpoint_direction,
            "save_best_only": self.config.save_best_only,
            "best_checkpoint_metric_name": self.best_checkpoint_metric_name,
            "best_checkpoint_metric_value": self.best_checkpoint_metric_value,
            "best_checkpoint_epoch": self.best_checkpoint_epoch,
            "best_checkpoint_step": self.best_checkpoint_step,
        }

    def train_step(self, batch_input: np.ndarray, batch_target: np.ndarray) -> float:
        self._validate_batch(batch_input=batch_input, batch_target=batch_target)
        plume_target = slice_plume_target(batch_target)

        batch_size = batch_input.shape[0]
        grads = self._zero_parameter_grads()
        total_loss = 0.0

        for i in range(batch_size):
            sequence = batch_input[i]
            cache = self._forward_with_cache(sequence)
            pred_linear = np.einsum("h,hrc->rc", self.model.w_out, cache["h_last"]) + self.model.b_out
            pred = np.clip(pred_linear, a_min=0.0, a_max=None)
            target = plume_target[i, 0, 0]
            error = pred - target
            total_loss += float(np.mean(error**2))

            grad_pred = (2.0 / error.size) * error
            grad_pred *= (pred_linear > 0.0).astype(float)
            sample_grads = self._backward_through_time(cache=cache, grad_pred=grad_pred)
            self._accumulate_grads(grads=grads, sample_grads=sample_grads)

        self._apply_gradients(grads=grads, batch_size=batch_size)

        return total_loss / batch_size

    def train_epoch(self, batch_iterable: list[tuple[np.ndarray, np.ndarray]]) -> float:
        """Run a narrow epoch helper over already-batched tensors."""
        if not batch_iterable:
            raise ValueError("train_epoch requires at least one batch")
        losses = [self.train_step(batch_input=x, batch_target=y) for x, y in batch_iterable]
        return float(np.mean(losses))

    def evaluate_batch(
        self, *, batch_input: np.ndarray, batch_target: np.ndarray, metric_prefix: str = "val", include_raw_space: bool = False
    ) -> dict[str, float]:
        self._validate_batch(batch_input=batch_input, batch_target=batch_target)
        plume_target = slice_plume_target(batch_target)[:, 0, 0]
        predictions = self._predict_batch(batch_input)
        transformed_metrics = self._regression_metrics(
            pred=predictions, target=plume_target, metric_prefix=metric_prefix, metric_space="transformed"
        )
        if not include_raw_space:
            return transformed_metrics

        raw_pred = plume_to_physical_space(predictions)
        raw_target = plume_to_physical_space(plume_target)
        raw_metrics = self._regression_metrics(pred=raw_pred, target=raw_target, metric_prefix=metric_prefix, metric_space="raw")
        return {**transformed_metrics, **raw_metrics}

    def evaluate_epoch(
        self, batch_iterable: list[tuple[np.ndarray, np.ndarray]], metric_prefix: str = "val", include_raw_space: bool = False
    ) -> dict[str, float]:
        if not batch_iterable:
            raise ValueError("evaluate_epoch requires at least one batch")

        total_sq_error = 0.0
        total_abs_error = 0.0
        total_count = 0
        total_raw_sq_error = 0.0
        total_raw_abs_error = 0.0

        for batch_input, batch_target in batch_iterable:
            self._validate_batch(batch_input=batch_input, batch_target=batch_target)
            target = slice_plume_target(batch_target)[:, 0, 0]
            pred = self._predict_batch(batch_input)
            error = pred - target
            total_sq_error += float(np.sum(error**2))
            total_abs_error += float(np.sum(np.abs(error)))
            total_count += int(error.size)

            if include_raw_space:
                raw_pred = plume_to_physical_space(pred)
                raw_target = plume_to_physical_space(target)
                raw_error = raw_pred - raw_target
                total_raw_sq_error += float(np.sum(raw_error**2))
                total_raw_abs_error += float(np.sum(np.abs(raw_error)))

        metrics = self._aggregate_metrics(
            total_sq_error=total_sq_error,
            total_abs_error=total_abs_error,
            total_count=total_count,
            metric_prefix=metric_prefix,
            metric_space="transformed",
        )
        if include_raw_space:
            metrics.update(
                self._aggregate_metrics(
                    total_sq_error=total_raw_sq_error,
                    total_abs_error=total_raw_abs_error,
                    total_count=total_count,
                    metric_prefix=metric_prefix,
                    metric_space="raw",
                )
            )
        return metrics

    def update_best_checkpoint(self, *, metrics: dict[str, float], epoch: int, step: int | None = None) -> bool:
        metric_name = self.config.checkpoint_metric
        if metric_name not in metrics:
            raise ValueError(f"Checkpoint metric '{metric_name}' was not found in metrics: {sorted(metrics.keys())}")
        metric_value = float(metrics[metric_name])

        if self.best_checkpoint_metric_value is None:
            improved = True
        elif self.config.checkpoint_direction == "min":
            improved = metric_value < self.best_checkpoint_metric_value
        else:
            improved = metric_value > self.best_checkpoint_metric_value

        if improved:
            self.best_checkpoint_metric_name = metric_name
            self.best_checkpoint_metric_value = metric_value
            self.best_checkpoint_epoch = int(epoch)
            self.best_checkpoint_step = None if step is None else int(step)
        return improved

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metrics: dict[str, float] | None = None,
        epoch: int | None = None,
        step: int | None = None,
        is_best: bool | None = None,
    ) -> dict[str, object]:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        metric_name = self.config.checkpoint_metric
        metric_value = None
        if metrics is not None and metric_name in metrics:
            metric_value = float(metrics[metric_name])

        if is_best is None:
            is_best = metric_value is not None and self.best_checkpoint_metric_value is not None and metric_value == self.best_checkpoint_metric_value

        metadata = {
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": self.config.target_policy,
            "normalization_mode": self.config.normalization_mode,
            "trainable_parameter_scope": self.config.trainable_parameter_scope,
            "selected_metric_name": metric_name,
            "selected_metric_value": metric_value,
            "epoch": epoch,
            "step": step,
            "is_best": bool(is_best),
            "checkpoint_direction": self.config.checkpoint_direction,
        }

        state = self.model.state_dict()
        np.savez(
            checkpoint_path,
            **state,
            checkpoint_metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
            contract_version=np.array(CONVLSTM_CONTRACT_VERSION, dtype=np.str_),
        )
        return {"path": str(checkpoint_path), "metadata": metadata}

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, object]:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ConvLSTM trainer checkpoint not found: {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            payload = {key: checkpoint[key] for key in checkpoint.files}

        metadata_json = payload.pop("checkpoint_metadata_json", np.array("{}", dtype=np.str_))
        payload.pop("contract_version", None)
        metadata = json.loads(str(np.asarray(metadata_json).item()))
        self.model.load_state_dict(payload, strict=strict)
        return {"path": str(checkpoint_path), "metadata": metadata}

    @staticmethod
    def _aggregate_metrics(
        *, total_sq_error: float, total_abs_error: float, total_count: int, metric_prefix: str, metric_space: str
    ) -> dict[str, float]:
        if total_count <= 0:
            raise ValueError("Cannot aggregate metrics for empty tensors")
        mse = total_sq_error / total_count
        mae = total_abs_error / total_count
        rmse = float(np.sqrt(mse))

        suffix = "" if metric_space == "transformed" else f"_{metric_space}"
        return {
            f"{metric_prefix}{suffix}_mse": float(mse),
            f"{metric_prefix}{suffix}_mae": float(mae),
            f"{metric_prefix}{suffix}_rmse": rmse,
        }

    def _regression_metrics(
        self, *, pred: np.ndarray, target: np.ndarray, metric_prefix: str, metric_space: str
    ) -> dict[str, float]:
        error = pred - target
        return self._aggregate_metrics(
            total_sq_error=float(np.sum(error**2)),
            total_abs_error=float(np.sum(np.abs(error))),
            total_count=int(error.size),
            metric_prefix=metric_prefix,
            metric_space=metric_space,
        )

    def _predict_batch(self, batch_input: np.ndarray) -> np.ndarray:
        return np.stack([self.model.forward(batch_input[i]) for i in range(batch_input.shape[0])], axis=0)

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
        return self._forward_with_cache(sequence)["h_last"]

    def _trainable_parameter_names(self) -> tuple[str, ...]:
        return ("w_x", "w_h", "b", "w_out", "b_out")

    def _zero_parameter_grads(self) -> dict[str, np.ndarray | float]:
        return {
            "w_x": np.zeros_like(self.model.w_x),
            "w_h": np.zeros_like(self.model.w_h),
            "b": np.zeros_like(self.model.b),
            "w_out": np.zeros_like(self.model.w_out),
            "b_out": 0.0,
        }

    @staticmethod
    def _accumulate_grads(
        *, grads: dict[str, np.ndarray | float], sample_grads: dict[str, np.ndarray | float]
    ) -> None:
        grads["w_x"] += sample_grads["w_x"]
        grads["w_h"] += sample_grads["w_h"]
        grads["b"] += sample_grads["b"]
        grads["w_out"] += sample_grads["w_out"]
        grads["b_out"] = float(grads["b_out"]) + float(sample_grads["b_out"])

    def _apply_gradients(self, *, grads: dict[str, np.ndarray | float], batch_size: int) -> None:
        scale = self.config.learning_rate / float(batch_size)
        self.model.w_x -= scale * grads["w_x"]
        self.model.w_h -= scale * grads["w_h"]
        self.model.b -= scale * grads["b"]
        self.model.w_out -= scale * grads["w_out"]
        self.model.b_out -= scale * float(grads["b_out"])

    def _forward_with_cache(self, sequence: np.ndarray) -> dict[str, object]:
        h_t = np.zeros((self.model.hidden_channels, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH), dtype=float)
        c_t = np.zeros((self.model.hidden_channels, CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH), dtype=float)
        steps: list[dict[str, np.ndarray]] = []

        for t in range(CONVLSTM_SEQUENCE_LENGTH):
            x_t = sequence[t]
            h_prev = h_t
            c_prev = c_t
            gates = (
                np.einsum("gc,chw->ghw", self.model.w_x, x_t)
                + np.einsum("gc,chw->ghw", self.model.w_h, h_t)
                + self.model.b[:, np.newaxis, np.newaxis]
            )
            i_raw, f_raw, o_raw, g_raw = np.split(gates, 4, axis=0)
            i = self.model._sigmoid(i_raw)
            f = self.model._sigmoid(f_raw)
            o = self.model._sigmoid(o_raw)
            g_tanh = np.tanh(g_raw)
            c_t = f * c_prev + i * g_tanh
            h_t = o * np.tanh(c_t)
            steps.append(
                {
                    "x_t": x_t,
                    "h_prev": h_prev,
                    "c_prev": c_prev,
                    "i_raw": i_raw,
                    "f_raw": f_raw,
                    "o_raw": o_raw,
                    "g_raw": g_raw,
                    "i": i,
                    "f": f,
                    "o": o,
                    "g_tanh": g_tanh,
                    "h_t": h_t,
                    "c_t": c_t,
                }
            )
        return {"h_last": h_t, "steps": steps}

    def _backward_through_time(
        self, *, cache: dict[str, object], grad_pred: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        grads = self._zero_parameter_grads()
        h_last = np.asarray(cache["h_last"], dtype=float)
        grads["w_out"] = np.einsum("rc,hrc->h", grad_pred, h_last)
        grads["b_out"] = float(np.sum(grad_pred))

        d_h = np.einsum("h,rc->hrc", self.model.w_out, grad_pred)
        d_c_next = np.zeros_like(h_last)
        d_h_next = np.zeros_like(h_last)

        steps = cache["steps"]
        assert isinstance(steps, list)
        for step in reversed(steps):
            h_prev = np.asarray(step["h_prev"], dtype=float)
            c_prev = np.asarray(step["c_prev"], dtype=float)
            i = np.asarray(step["i"], dtype=float)
            f = np.asarray(step["f"], dtype=float)
            o = np.asarray(step["o"], dtype=float)
            g_tanh = np.asarray(step["g_tanh"], dtype=float)
            c_t = np.asarray(step["c_t"], dtype=float)
            x_t = np.asarray(step["x_t"], dtype=float)
            g_raw = np.asarray(step["g_raw"], dtype=float)

            tanh_c = np.tanh(c_t)
            d_h_total = d_h + d_h_next
            d_o = d_h_total * tanh_c
            d_c = d_h_total * o * (1.0 - tanh_c**2) + d_c_next
            d_f = d_c * c_prev
            d_i = d_c * g_tanh
            d_g_tanh = d_c * i

            d_i_raw = d_i * i * (1.0 - i)
            d_f_raw = d_f * f * (1.0 - f)
            d_o_raw = d_o * o * (1.0 - o)
            d_g_raw = d_g_tanh * (1.0 - np.tanh(g_raw) ** 2)
            d_gates = np.concatenate((d_i_raw, d_f_raw, d_o_raw, d_g_raw), axis=0)

            grads["w_x"] += np.einsum("ghw,chw->gc", d_gates, x_t)
            grads["w_h"] += np.einsum("ghw,khw->gk", d_gates, h_prev)
            grads["b"] += np.einsum("ghw->g", d_gates)

            d_h_next = np.einsum("gk,ghw->khw", self.model.w_h, d_gates)
            d_c_next = d_c * f

        return grads
