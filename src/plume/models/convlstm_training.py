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
    physics_loss_enabled: bool = False
    lambda_smooth: float = 0.0
    lambda_mass: float = 0.0
    smoothness_loss_mode: str = "finite_difference_l2"
    mass_loss_mode: str = "mean_mass_mse"
    mass_loss_space: str = "transformed"
    physics_schedule_enabled: bool = False
    physics_schedule_type: str = "epoch"
    physics_schedule_stage_boundaries: tuple[int, ...] = (0,)
    physics_schedule_lambda_smooth: tuple[float, ...] = (0.0,)
    physics_schedule_lambda_mass: tuple[float, ...] = (0.0,)
    smoothness_ramp_type: str = "none"
    smoothness_ramp_start: int = 0
    smoothness_ramp_end: int = 0
    mass_ramp_type: str = "none"
    mass_ramp_start: int = 0
    mass_ramp_end: int = 0
    metric_stage_progression_enabled: bool = False
    metric_stage_monitor: str = "val_mse"
    metric_stage_direction: str = "min"
    metric_stage_thresholds: tuple[float, ...] = ()
    metric_stage_min_epoch_per_stage: int = 0
    metric_stage_patience: int = 0
    plume_mass_metric_enabled: bool = True
    plume_mass_metric_include_raw: bool = False
    plume_support_metric_enabled: bool = True
    plume_support_threshold_space: str = "transformed"
    plume_support_threshold_value: float = 0.1
    plume_centroid_metric_enabled: bool = True
    plume_centroid_metric_space: str = "transformed"


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
        if self.config.smoothness_loss_mode != "finite_difference_l2":
            raise ValueError(
                "Only smoothness_loss_mode='finite_difference_l2' is supported in Phase-G, "
                f"got {self.config.smoothness_loss_mode}"
            )
        if self.config.mass_loss_mode != "mean_mass_mse":
            raise ValueError(
                "Only mass_loss_mode='mean_mass_mse' is supported in Phase-G, "
                f"got {self.config.mass_loss_mode}"
            )
        if self.config.mass_loss_space not in {"transformed", "raw"}:
            raise ValueError(
                "mass_loss_space must be 'transformed' or 'raw', "
                f"got {self.config.mass_loss_space}"
            )
        if self.config.lambda_smooth < 0.0:
            raise ValueError(f"lambda_smooth must be >= 0, got {self.config.lambda_smooth}")
        if self.config.lambda_mass < 0.0:
            raise ValueError(f"lambda_mass must be >= 0, got {self.config.lambda_mass}")
        if self.config.physics_schedule_type != "epoch":
            raise ValueError(
                "Only physics_schedule_type='epoch' is supported in Phase-H, "
                f"got {self.config.physics_schedule_type}"
            )
        if self.config.smoothness_ramp_type not in {"none", "linear"}:
            raise ValueError(
                "smoothness_ramp_type must be 'none' or 'linear', "
                f"got {self.config.smoothness_ramp_type}"
            )
        if self.config.mass_ramp_type not in {"none", "linear"}:
            raise ValueError(
                "mass_ramp_type must be 'none' or 'linear', "
                f"got {self.config.mass_ramp_type}"
            )
        if self.config.plume_support_threshold_space not in {"transformed", "raw"}:
            raise ValueError(
                "plume_support_threshold_space must be 'transformed' or 'raw', "
                f"got {self.config.plume_support_threshold_space}"
            )
        if self.config.plume_centroid_metric_space not in {"transformed", "raw"}:
            raise ValueError(
                "plume_centroid_metric_space must be 'transformed' or 'raw', "
                f"got {self.config.plume_centroid_metric_space}"
            )
        if self.config.plume_support_threshold_value < 0.0:
            raise ValueError(
                "plume_support_threshold_value must be >= 0, "
                f"got {self.config.plume_support_threshold_value}"
            )
        self._validate_physics_schedule_config()
        self._validate_metric_stage_progression_config()

        self.best_checkpoint_metric_name: str | None = None
        self.best_checkpoint_metric_value: float | None = None
        self.best_checkpoint_epoch: int | None = None
        self.best_checkpoint_step: int | None = None
        self.last_train_step_metrics: dict[str, float] | None = None
        self._train_steps_completed: int = 0
        self._last_effective_lambda_smooth: float = 0.0
        self._last_effective_lambda_mass: float = 0.0
        self._last_active_stage: int = 0
        self._metric_active_stage: int = 0
        self._metric_stage_start_epoch: int = 0
        self._metric_stage_satisfaction_streak: int = 0
        self._metric_stage_last_value: float | None = None
        self._metric_stage_last_advanced: bool = False
        self._metric_stage_last_update_epoch: int | None = None

    @property
    def metadata(self) -> dict[str, object]:
        epochs_in_current_stage = 0
        if self.config.metric_stage_progression_enabled and self._metric_stage_last_update_epoch is not None:
            epochs_in_current_stage = max(0, self._metric_stage_last_update_epoch - self._metric_stage_start_epoch)
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
            "physics_loss_enabled": self.config.physics_loss_enabled,
            "lambda_smooth": self.config.lambda_smooth,
            "lambda_mass": self.config.lambda_mass,
            "smoothness_loss_mode": self.config.smoothness_loss_mode,
            "mass_loss_mode": self.config.mass_loss_mode,
            "mass_loss_space": self.config.mass_loss_space,
            "physics_schedule_enabled": self.config.physics_schedule_enabled,
            "physics_schedule_type": self.config.physics_schedule_type,
            "physics_schedule_stage_boundaries": self.config.physics_schedule_stage_boundaries,
            "physics_schedule_lambda_smooth": self.config.physics_schedule_lambda_smooth,
            "physics_schedule_lambda_mass": self.config.physics_schedule_lambda_mass,
            "smoothness_ramp_type": self.config.smoothness_ramp_type,
            "smoothness_ramp_start": self.config.smoothness_ramp_start,
            "smoothness_ramp_end": self.config.smoothness_ramp_end,
            "mass_ramp_type": self.config.mass_ramp_type,
            "mass_ramp_start": self.config.mass_ramp_start,
            "mass_ramp_end": self.config.mass_ramp_end,
            "metric_stage_progression_enabled": self.config.metric_stage_progression_enabled,
            "metric_stage_monitor": self.config.metric_stage_monitor,
            "metric_stage_direction": self.config.metric_stage_direction,
            "metric_stage_thresholds": self.config.metric_stage_thresholds,
            "metric_stage_min_epoch_per_stage": self.config.metric_stage_min_epoch_per_stage,
            "metric_stage_patience": self.config.metric_stage_patience,
            "plume_mass_metric_enabled": self.config.plume_mass_metric_enabled,
            "plume_mass_metric_include_raw": self.config.plume_mass_metric_include_raw,
            "plume_support_metric_enabled": self.config.plume_support_metric_enabled,
            "plume_support_threshold_space": self.config.plume_support_threshold_space,
            "plume_support_threshold_value": self.config.plume_support_threshold_value,
            "plume_centroid_metric_enabled": self.config.plume_centroid_metric_enabled,
            "plume_centroid_metric_space": self.config.plume_centroid_metric_space,
            "metric_stage_last_value": self._metric_stage_last_value,
            "metric_stage_last_advanced": self._metric_stage_last_advanced,
            "epochs_in_current_stage": epochs_in_current_stage,
            "metric_stage_satisfaction_streak": self._metric_stage_satisfaction_streak,
            "effective_lambda_smooth": self._last_effective_lambda_smooth,
            "effective_lambda_mass": self._last_effective_lambda_mass,
            "active_stage": self._last_active_stage,
            "best_checkpoint_metric_name": self.best_checkpoint_metric_name,
            "best_checkpoint_metric_value": self.best_checkpoint_metric_value,
            "best_checkpoint_epoch": self.best_checkpoint_epoch,
            "best_checkpoint_step": self.best_checkpoint_step,
        }

    def train_step(
        self,
        batch_input: np.ndarray,
        batch_target: np.ndarray,
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> float:
        metrics = self.train_step_with_metrics(batch_input=batch_input, batch_target=batch_target, epoch=epoch, step=step)
        return float(metrics["train_total_loss"])

    def train_step_with_metrics(
        self,
        *,
        batch_input: np.ndarray,
        batch_target: np.ndarray,
        epoch: int | None = None,
        step: int | None = None,
    ) -> dict[str, float]:
        self._validate_batch(batch_input=batch_input, batch_target=batch_target)
        plume_target = slice_plume_target(batch_target)
        effective_weights = self._effective_physics_weights(epoch=epoch, step=step)

        batch_size = batch_input.shape[0]
        grads = self._zero_parameter_grads()
        supervised_loss_total = 0.0
        smoothness_loss_total = 0.0
        mass_loss_total = 0.0

        for i in range(batch_size):
            sequence = batch_input[i]
            cache = self._forward_with_cache(sequence)
            pred_linear = np.einsum("h,hrc->rc", self.model.w_out, cache["h_last"]) + self.model.b_out
            pred = np.clip(pred_linear, a_min=0.0, a_max=None)
            target = plume_target[i, 0, 0]
            components = self._loss_components(
                pred=pred,
                target=target,
                effective_lambda_smooth=effective_weights["lambda_smooth"],
                effective_lambda_mass=effective_weights["lambda_mass"],
            )
            supervised_loss_total += components["supervised_loss"]
            smoothness_loss_total += components["smoothness_loss"]
            mass_loss_total += components["mass_loss"]

            grad_pred = self._loss_grad_wrt_prediction(
                pred=pred,
                target=target,
                effective_lambda_smooth=effective_weights["lambda_smooth"],
                effective_lambda_mass=effective_weights["lambda_mass"],
            )
            grad_pred *= (pred_linear > 0.0).astype(float)
            sample_grads = self._backward_through_time(cache=cache, grad_pred=grad_pred)
            self._accumulate_grads(grads=grads, sample_grads=sample_grads)

        self._apply_gradients(grads=grads, batch_size=batch_size)

        metrics = self._mean_loss_metrics(
            metric_prefix="train",
            supervised_loss_total=supervised_loss_total,
            smoothness_loss_total=smoothness_loss_total,
            mass_loss_total=mass_loss_total,
            count=batch_size,
            effective_lambda_smooth=effective_weights["lambda_smooth"],
            effective_lambda_mass=effective_weights["lambda_mass"],
            active_stage=effective_weights["active_stage"],
        )
        self._last_effective_lambda_smooth = float(effective_weights["lambda_smooth"])
        self._last_effective_lambda_mass = float(effective_weights["lambda_mass"])
        self._last_active_stage = int(effective_weights["active_stage"])
        self.last_train_step_metrics = metrics
        self._train_steps_completed += 1
        return metrics

    def update_stage_from_validation(self, val_metrics: dict[str, float], *, epoch: int) -> bool:
        """Update metric-gated stage progression from validation metrics.

        Returns True when stage advanced, otherwise False.
        """
        self._metric_stage_last_advanced = False
        if not self.config.metric_stage_progression_enabled:
            return False
        metric_name = self.config.metric_stage_monitor
        if metric_name not in val_metrics:
            raise ValueError(
                f"Metric-gated stage progression requires '{metric_name}' in validation metrics, "
                f"got keys: {sorted(val_metrics.keys())}"
            )
        metric_value = float(val_metrics[metric_name])
        self._metric_stage_last_value = metric_value
        self._metric_stage_last_update_epoch = int(epoch)

        current_stage = self._metric_active_stage
        max_stage = len(self.config.physics_schedule_stage_boundaries) - 1
        if current_stage >= max_stage:
            self._metric_stage_satisfaction_streak = 0
            return False

        epochs_in_current_stage = max(0, int(epoch) - self._metric_stage_start_epoch)
        if epochs_in_current_stage < self.config.metric_stage_min_epoch_per_stage:
            self._metric_stage_satisfaction_streak = 0
            return False

        threshold = float(self.config.metric_stage_thresholds[current_stage])
        threshold_satisfied = (
            metric_value <= threshold
            if self.config.metric_stage_direction == "min"
            else metric_value >= threshold
        )
        if threshold_satisfied:
            self._metric_stage_satisfaction_streak += 1
        else:
            self._metric_stage_satisfaction_streak = 0

        required_streak = max(1, self.config.metric_stage_patience + 1)
        if self._metric_stage_satisfaction_streak < required_streak:
            return False

        self._metric_active_stage += 1
        self._metric_stage_start_epoch = int(epoch)
        self._metric_stage_satisfaction_streak = 0
        self._metric_stage_last_advanced = True
        self._last_active_stage = self._metric_active_stage
        return True

    def train_epoch(self, batch_iterable: list[tuple[np.ndarray, np.ndarray]], *, epoch: int = 0) -> float:
        """Run a narrow epoch helper over already-batched tensors."""
        if not batch_iterable:
            raise ValueError("train_epoch requires at least one batch")
        losses = [self.train_step(batch_input=x, batch_target=y, epoch=epoch) for x, y in batch_iterable]
        return float(np.mean(losses))

    def evaluate_batch(
        self,
        *,
        batch_input: np.ndarray,
        batch_target: np.ndarray,
        metric_prefix: str = "val",
        include_raw_space: bool = False,
        include_loss_components: bool = False,
    ) -> dict[str, float]:
        self._validate_batch(batch_input=batch_input, batch_target=batch_target)
        plume_target = slice_plume_target(batch_target)[:, 0, 0]
        predictions = self._predict_batch(batch_input)
        transformed_metrics = self._regression_metrics(
            pred=predictions, target=plume_target, metric_prefix=metric_prefix, metric_space="transformed"
        )
        transformed_metrics.update(
            self._plume_specific_metrics(
                pred=predictions,
                target=plume_target,
                metric_prefix=metric_prefix,
                include_raw_space=include_raw_space,
            )
        )
        if include_loss_components:
            transformed_metrics.update(
                self._evaluate_loss_component_metrics(
                    pred=predictions,
                    target=plume_target,
                    metric_prefix=metric_prefix,
                )
            )
        if not include_raw_space:
            return transformed_metrics

        raw_pred = plume_to_physical_space(predictions)
        raw_target = plume_to_physical_space(plume_target)
        raw_metrics = self._regression_metrics(pred=raw_pred, target=raw_target, metric_prefix=metric_prefix, metric_space="raw")
        return {**transformed_metrics, **raw_metrics}

    def evaluate_epoch(
        self,
        batch_iterable: list[tuple[np.ndarray, np.ndarray]],
        metric_prefix: str = "val",
        include_raw_space: bool = False,
        include_loss_components: bool = False,
    ) -> dict[str, float]:
        if not batch_iterable:
            raise ValueError("evaluate_epoch requires at least one batch")

        total_sq_error = 0.0
        total_abs_error = 0.0
        total_count = 0
        total_raw_sq_error = 0.0
        total_raw_abs_error = 0.0
        total_supervised_loss = 0.0
        total_smoothness_loss = 0.0
        total_mass_loss = 0.0
        total_samples = 0
        plume_metric_totals: dict[str, float] = {}
        plume_metric_counts: dict[str, int] = {}

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

            plume_metrics = self._plume_specific_metrics(
                pred=pred,
                target=target,
                metric_prefix=metric_prefix,
                include_raw_space=include_raw_space,
            )
            for metric_name, metric_value in plume_metrics.items():
                plume_metric_totals[metric_name] = plume_metric_totals.get(metric_name, 0.0) + float(metric_value) * pred.shape[0]
                plume_metric_counts[metric_name] = plume_metric_counts.get(metric_name, 0) + int(pred.shape[0])
            if include_loss_components:
                for i in range(pred.shape[0]):
                    components = self._loss_components(pred=pred[i], target=target[i])
                    total_supervised_loss += components["supervised_loss"]
                    total_smoothness_loss += components["smoothness_loss"]
                    total_mass_loss += components["mass_loss"]
                total_samples += int(pred.shape[0])

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
        for metric_name, metric_total in plume_metric_totals.items():
            count = plume_metric_counts[metric_name]
            metrics[metric_name] = float(metric_total / float(count))
        if include_loss_components:
            effective_weights = self._effective_physics_weights(epoch=0, step=self._train_steps_completed)
            metrics.update(
                self._mean_loss_metrics(
                    metric_prefix=metric_prefix,
                    supervised_loss_total=total_supervised_loss,
                    smoothness_loss_total=total_smoothness_loss,
                    mass_loss_total=total_mass_loss,
                    count=total_samples,
                    effective_lambda_smooth=effective_weights["lambda_smooth"],
                    effective_lambda_mass=effective_weights["lambda_mass"],
                    active_stage=int(effective_weights["active_stage"]),
                )
            )
        return metrics

    def build_epoch_report(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float] | None = None,
        val_metrics: dict[str, float] | None = None,
        is_best_checkpoint: bool | None = None,
    ) -> dict[str, object]:
        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}
        active_stage = int(train_metrics.get("train_active_stage", self._last_active_stage))
        effective_lambda_smooth = float(train_metrics.get("train_effective_lambda_smooth", self._last_effective_lambda_smooth))
        effective_lambda_mass = float(train_metrics.get("train_effective_lambda_mass", self._last_effective_lambda_mass))
        return {
            "epoch": int(epoch),
            "active_stage": active_stage,
            "effective_lambdas": {
                "lambda_smooth": effective_lambda_smooth,
                "lambda_mass": effective_lambda_mass,
            },
            "train": dict(train_metrics),
            "validation": dict(val_metrics),
            "checkpoint": {
                "metric_name": self.best_checkpoint_metric_name,
                "metric_value": self.best_checkpoint_metric_value,
                "epoch": self.best_checkpoint_epoch,
                "step": self.best_checkpoint_step,
                "is_best": is_best_checkpoint,
            },
        }

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

    def _physics_terms_active(self, *, effective_lambda_smooth: float, effective_lambda_mass: float) -> bool:
        return bool(self.config.physics_loss_enabled and (effective_lambda_smooth > 0.0 or effective_lambda_mass > 0.0))

    def _loss_components(
        self,
        *,
        pred: np.ndarray,
        target: np.ndarray,
        effective_lambda_smooth: float | None = None,
        effective_lambda_mass: float | None = None,
    ) -> dict[str, float]:
        if effective_lambda_smooth is None or effective_lambda_mass is None:
            defaults = self._effective_physics_weights(epoch=0, step=self._train_steps_completed)
            effective_lambda_smooth = defaults["lambda_smooth"]
            effective_lambda_mass = defaults["lambda_mass"]
        supervised_error = pred - target
        supervised_loss = float(np.mean(supervised_error**2))
        if not self._physics_terms_active(
            effective_lambda_smooth=effective_lambda_smooth,
            effective_lambda_mass=effective_lambda_mass,
        ):
            return {
                "supervised_loss": supervised_loss,
                "smoothness_loss": 0.0,
                "mass_loss": 0.0,
                "total_loss": supervised_loss,
            }

        smoothness_loss = 0.0
        mass_loss = 0.0
        if effective_lambda_smooth > 0.0:
            smoothness_loss, _ = self._smoothness_loss_and_grad(pred)
        if effective_lambda_mass > 0.0:
            mass_loss, _ = self._mass_loss_and_grad(pred=pred, target=target)

        total_loss = supervised_loss + effective_lambda_smooth * smoothness_loss + effective_lambda_mass * mass_loss
        return {
            "supervised_loss": supervised_loss,
            "smoothness_loss": float(smoothness_loss),
            "mass_loss": float(mass_loss),
            "total_loss": float(total_loss),
        }

    def _loss_grad_wrt_prediction(
        self,
        *,
        pred: np.ndarray,
        target: np.ndarray,
        effective_lambda_smooth: float,
        effective_lambda_mass: float,
    ) -> np.ndarray:
        grad = (2.0 / pred.size) * (pred - target)
        if not self._physics_terms_active(
            effective_lambda_smooth=effective_lambda_smooth,
            effective_lambda_mass=effective_lambda_mass,
        ):
            return grad
        if effective_lambda_smooth > 0.0:
            _, smooth_grad = self._smoothness_loss_and_grad(pred)
            grad += effective_lambda_smooth * smooth_grad
        if effective_lambda_mass > 0.0:
            _, mass_grad = self._mass_loss_and_grad(pred=pred, target=target)
            grad += effective_lambda_mass * mass_grad
        return grad

    def _smoothness_loss_and_grad(self, pred: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(pred)
        dx = pred[:, 1:] - pred[:, :-1]
        dy = pred[1:, :] - pred[:-1, :]
        loss_x = float(np.mean(dx**2))
        loss_y = float(np.mean(dy**2))
        loss = loss_x + loss_y

        if dx.size > 0:
            coeff_x = 2.0 / dx.size
            grad[:, 1:] += coeff_x * dx
            grad[:, :-1] -= coeff_x * dx
        if dy.size > 0:
            coeff_y = 2.0 / dy.size
            grad[1:, :] += coeff_y * dy
            grad[:-1, :] -= coeff_y * dy
        return loss, grad

    def _mass_loss_and_grad(self, *, pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(pred)
        if self.config.mass_loss_space == "transformed":
            pred_mean = float(np.mean(pred))
            target_mean = float(np.mean(target))
            diff = pred_mean - target_mean
            loss = diff**2
            grad.fill((2.0 * diff) / pred.size)
            return float(loss), grad

        raw_pred = plume_to_physical_space(pred, clamp_non_negative=False)
        raw_target = plume_to_physical_space(target, clamp_non_negative=False)
        diff = float(np.mean(raw_pred) - np.mean(raw_target))
        loss = diff**2
        d_raw_d_pred = np.exp(pred) / 1e12
        grad = ((2.0 * diff) / pred.size) * d_raw_d_pred
        return float(loss), grad

    def _mean_loss_metrics(
        self,
        *,
        metric_prefix: str,
        supervised_loss_total: float,
        smoothness_loss_total: float,
        mass_loss_total: float,
        count: int,
        effective_lambda_smooth: float,
        effective_lambda_mass: float,
        active_stage: int,
    ) -> dict[str, float]:
        if count <= 0:
            raise ValueError("Cannot aggregate loss components for empty tensors")
        supervised_loss = supervised_loss_total / float(count)
        smoothness_loss = smoothness_loss_total / float(count)
        mass_loss = mass_loss_total / float(count)
        total_loss = supervised_loss + effective_lambda_smooth * smoothness_loss + effective_lambda_mass * mass_loss
        return {
            f"{metric_prefix}_supervised_loss": float(supervised_loss),
            f"{metric_prefix}_smoothness_loss": float(smoothness_loss),
            f"{metric_prefix}_mass_loss": float(mass_loss),
            f"{metric_prefix}_total_loss": float(total_loss),
            f"{metric_prefix}_effective_lambda_smooth": float(effective_lambda_smooth),
            f"{metric_prefix}_effective_lambda_mass": float(effective_lambda_mass),
            f"{metric_prefix}_active_stage": float(active_stage),
        }

    def _evaluate_loss_component_metrics(self, *, pred: np.ndarray, target: np.ndarray, metric_prefix: str) -> dict[str, float]:
        supervised_total = 0.0
        smoothness_total = 0.0
        mass_total = 0.0
        for i in range(pred.shape[0]):
            components = self._loss_components(pred=pred[i], target=target[i])
            supervised_total += components["supervised_loss"]
            smoothness_total += components["smoothness_loss"]
            mass_total += components["mass_loss"]
        effective_weights = self._effective_physics_weights(epoch=0, step=self._train_steps_completed)
        return self._mean_loss_metrics(
            metric_prefix=metric_prefix,
            supervised_loss_total=supervised_total,
            smoothness_loss_total=smoothness_total,
            mass_loss_total=mass_total,
            count=int(pred.shape[0]),
            effective_lambda_smooth=effective_weights["lambda_smooth"],
            effective_lambda_mass=effective_weights["lambda_mass"],
            active_stage=effective_weights["active_stage"],
        )

    def _plume_specific_metrics(
        self, *, pred: np.ndarray, target: np.ndarray, metric_prefix: str, include_raw_space: bool
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.config.plume_mass_metric_enabled:
            metrics[f"{metric_prefix}_mass_abs_error_transformed"] = self._mass_abs_error(pred=pred, target=target)
            if include_raw_space or self.config.plume_mass_metric_include_raw:
                raw_pred = plume_to_physical_space(pred)
                raw_target = plume_to_physical_space(target)
                metrics[f"{metric_prefix}_mass_abs_error_raw"] = self._mass_abs_error(pred=raw_pred, target=raw_target)
        if self.config.plume_support_metric_enabled:
            support_pred, support_target = self._to_metric_space(
                pred=pred,
                target=target,
                space=self.config.plume_support_threshold_space,
            )
            metrics[f"{metric_prefix}_support_iou_{self.config.plume_support_threshold_space}"] = self._support_iou(
                pred=support_pred,
                target=support_target,
                threshold=self.config.plume_support_threshold_value,
            )
        if self.config.plume_centroid_metric_enabled:
            centroid_pred, centroid_target = self._to_metric_space(
                pred=pred,
                target=target,
                space=self.config.plume_centroid_metric_space,
            )
            metrics[f"{metric_prefix}_centroid_distance_raster_{self.config.plume_centroid_metric_space}"] = (
                self._centroid_distance_raster(pred=centroid_pred, target=centroid_target)
            )
        return metrics

    @staticmethod
    def _to_metric_space(*, pred: np.ndarray, target: np.ndarray, space: str) -> tuple[np.ndarray, np.ndarray]:
        if space == "transformed":
            return pred, target
        if space == "raw":
            return plume_to_physical_space(pred), plume_to_physical_space(target)
        raise ValueError(f"Unsupported metric space '{space}'")

    @staticmethod
    def _mass_abs_error(*, pred: np.ndarray, target: np.ndarray) -> float:
        pred_mass = np.sum(pred, axis=(1, 2))
        target_mass = np.sum(target, axis=(1, 2))
        return float(np.mean(np.abs(pred_mass - target_mass)))

    @staticmethod
    def _support_iou(*, pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
        pred_mask = pred >= threshold
        target_mask = target >= threshold
        intersection = np.sum(pred_mask & target_mask, axis=(1, 2)).astype(float)
        union = np.sum(pred_mask | target_mask, axis=(1, 2)).astype(float)
        iou = np.ones_like(union, dtype=float)
        valid = union > 0.0
        iou[valid] = intersection[valid] / union[valid]
        return float(np.mean(iou))

    @staticmethod
    def _centroid_distance_raster(*, pred: np.ndarray, target: np.ndarray) -> float:
        distances: list[float] = []
        max_distance = float(np.sqrt((CONVLSTM_GRID_HEIGHT - 1) ** 2 + (CONVLSTM_GRID_WIDTH - 1) ** 2))
        for i in range(pred.shape[0]):
            pred_center = ConvLSTMPlumeTrainer._center_of_mass(pred[i])
            target_center = ConvLSTMPlumeTrainer._center_of_mass(target[i])
            if pred_center is None and target_center is None:
                distances.append(0.0)
            elif pred_center is None or target_center is None:
                distances.append(max_distance)
            else:
                distances.append(float(np.linalg.norm(np.array(pred_center) - np.array(target_center))))
        return float(np.mean(distances))

    @staticmethod
    def _center_of_mass(field: np.ndarray) -> tuple[float, float] | None:
        positive = np.clip(field, a_min=0.0, a_max=None)
        mass = float(np.sum(positive))
        if mass <= 0.0:
            return None
        row_coords = np.arange(field.shape[0], dtype=float)[:, np.newaxis]
        col_coords = np.arange(field.shape[1], dtype=float)[np.newaxis, :]
        row = float(np.sum(positive * row_coords) / mass)
        col = float(np.sum(positive * col_coords) / mass)
        return row, col

    def _validate_physics_schedule_config(self) -> None:
        if not self.config.physics_schedule_enabled:
            return
        boundaries = self.config.physics_schedule_stage_boundaries
        lambda_smooth = self.config.physics_schedule_lambda_smooth
        lambda_mass = self.config.physics_schedule_lambda_mass
        if not boundaries:
            raise ValueError("physics_schedule_stage_boundaries must be non-empty when physics_schedule_enabled=True")
        if boundaries[0] != 0:
            raise ValueError("physics_schedule_stage_boundaries must start at 0")
        if any(b < 0 for b in boundaries):
            raise ValueError("physics_schedule_stage_boundaries must be non-negative")
        if tuple(sorted(boundaries)) != tuple(boundaries):
            raise ValueError("physics_schedule_stage_boundaries must be sorted in ascending order")
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("physics_schedule_stage_boundaries must not contain duplicates")
        if len(lambda_smooth) != len(boundaries):
            raise ValueError("physics_schedule_lambda_smooth length must match stage boundaries length")
        if len(lambda_mass) != len(boundaries):
            raise ValueError("physics_schedule_lambda_mass length must match stage boundaries length")
        if any(v < 0.0 for v in lambda_smooth):
            raise ValueError("physics_schedule_lambda_smooth values must be >= 0")
        if any(v < 0.0 for v in lambda_mass):
            raise ValueError("physics_schedule_lambda_mass values must be >= 0")
        self._validate_linear_ramp_bounds(self.config.smoothness_ramp_type, self.config.smoothness_ramp_start, self.config.smoothness_ramp_end)
        self._validate_linear_ramp_bounds(self.config.mass_ramp_type, self.config.mass_ramp_start, self.config.mass_ramp_end)

    def _validate_metric_stage_progression_config(self) -> None:
        if not self.config.metric_stage_progression_enabled:
            return
        if not self.config.physics_schedule_enabled:
            raise ValueError(
                "metric_stage_progression_enabled=True requires physics_schedule_enabled=True"
            )
        if self.config.metric_stage_direction not in {"min", "max"}:
            raise ValueError("metric_stage_direction must be 'min' or 'max'")
        if self.config.metric_stage_min_epoch_per_stage < 0:
            raise ValueError("metric_stage_min_epoch_per_stage must be >= 0")
        if self.config.metric_stage_patience < 0:
            raise ValueError("metric_stage_patience must be >= 0")
        expected_thresholds = len(self.config.physics_schedule_stage_boundaries) - 1
        if len(self.config.metric_stage_thresholds) != expected_thresholds:
            raise ValueError(
                "metric_stage_thresholds length must equal number of stage transitions "
                f"({expected_thresholds})"
            )

    @staticmethod
    def _validate_linear_ramp_bounds(ramp_type: str, ramp_start: int, ramp_end: int) -> None:
        if ramp_start < 0 or ramp_end < 0:
            raise ValueError("Ramp boundaries must be non-negative")
        if ramp_type == "linear" and ramp_end < ramp_start:
            raise ValueError("Linear ramp requires ramp_end >= ramp_start")

    def _effective_physics_weights(self, *, epoch: int | None, step: int | None) -> dict[str, float]:
        if not self.config.physics_loss_enabled:
            return {"lambda_smooth": 0.0, "lambda_mass": 0.0, "active_stage": 0.0}

        if not self.config.physics_schedule_enabled:
            return {
                "lambda_smooth": float(self.config.lambda_smooth),
                "lambda_mass": float(self.config.lambda_mass),
                "active_stage": 0.0,
            }

        progress = 0 if epoch is None else int(epoch)
        boundaries = self.config.physics_schedule_stage_boundaries
        if self.config.metric_stage_progression_enabled:
            active_stage = min(self._metric_active_stage, len(boundaries) - 1)
        else:
            active_stage = 0
            for idx, start in enumerate(boundaries):
                if progress >= start:
                    active_stage = idx
                else:
                    break
        lambda_smooth = float(self.config.physics_schedule_lambda_smooth[active_stage])
        lambda_mass = float(self.config.physics_schedule_lambda_mass[active_stage])
        lambda_smooth = self._apply_linear_ramp(
            value=lambda_smooth,
            progress=progress,
            ramp_type=self.config.smoothness_ramp_type,
            ramp_start=self.config.smoothness_ramp_start,
            ramp_end=self.config.smoothness_ramp_end,
        )
        lambda_mass = self._apply_linear_ramp(
            value=lambda_mass,
            progress=progress,
            ramp_type=self.config.mass_ramp_type,
            ramp_start=self.config.mass_ramp_start,
            ramp_end=self.config.mass_ramp_end,
        )
        return {
            "lambda_smooth": float(lambda_smooth),
            "lambda_mass": float(lambda_mass),
            "active_stage": float(active_stage),
        }

    @staticmethod
    def _apply_linear_ramp(*, value: float, progress: int, ramp_type: str, ramp_start: int, ramp_end: int) -> float:
        if ramp_type == "none":
            return value
        if progress <= ramp_start:
            return 0.0
        if progress >= ramp_end:
            return value
        if ramp_end == ramp_start:
            return value
        alpha = (progress - ramp_start) / float(ramp_end - ramp_start)
        return float(alpha * value)

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
