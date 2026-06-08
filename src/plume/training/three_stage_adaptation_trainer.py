"""Standalone robust three-stage ConvLSTM adaptation trainer.

The trainer consumes Phase 5A adaptation dataset manifests and writes local run
artifacts only. It does not promote models, mutate registries, start workers, or
change serving behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Literal

from plume.training.adaptation_dataset import AdaptationNPZDataset, AdaptationSample

try:  # Keep module importable when torch is absent.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


StageKey = Literal["stage1", "stage2", "stage3"]
ResumeMode = Literal["none", "model_only"]


class TrainingCancelled(RuntimeError):
    """Raised when cooperative adaptation training cancellation is requested."""


MODEL_CONTRACT: dict[str, Any] = {
    "model_name": "RobustMultiStepConvLSTMForecaster",
    "forecast_mode": "direct_plus_autoregressive_multistep",
    "input_shape": [3, 10, 64, 64],
    "output_shape": [4, 1, 64, 64],
    "plume_channel": 0,
    "wind_u_channel": 1,
    "wind_v_channel": 2,
    "has_direct_branch": True,
    "has_autoregressive_branch": True,
    "residual_rollout": True,
}


@dataclass
class LossWeights:
    direct_data: float = 0.0
    rollout_data: float = 0.0
    consistency: float = 0.0
    mass: float = 0.0
    temporal: float = 0.0
    smooth: float = 0.0
    nonneg: float = 0.0
    bg: float = 0.0


@dataclass
class StageConfig:
    name: str
    enabled: bool = True
    max_epochs: int = 1
    min_epochs: int = 1
    patience: int = 1
    learning_rate: float = 3e-4
    train_direct: bool = True
    train_rollout: bool = True
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 1.0
    horizon_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    loss_weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class NoiseConfig:
    enabled_stage3: bool = True
    plume_gaussian_std: float = 0.02
    plume_dropout_prob: float = 0.03
    plume_random_shift_pixels: int = 1
    clip_min: float = 0.0


@dataclass
class SelectionGateConfig:
    enabled: bool = True
    reference_stage_name: str = "stage2_autoregressive_teacher_forcing"
    max_worse_rollout_weighted_mse_percent: float = 3.0
    max_worse_t3_weighted_mse_percent: float = 5.0
    max_worse_t4_weighted_mse_percent: float = 5.0


@dataclass
class ThreeStageTrainerConfig:
    run_name: str = "convlstm_three_stage_adaptation"
    initial_batch_size: int = 16
    min_batch_size: int = 1
    auto_reduce_batch_on_oom: bool = True
    allow_cpu_fallback_on_cuda_oom: bool = False
    num_workers: int = 0
    plume_threshold: float = 1e-6
    plume_weight: float = 5.0
    background_target_threshold: float = 1e-6
    model: dict[str, Any] = field(default_factory=lambda: {
        "encoder_channels": 32,
        "hidden_channels": 64,
        "num_encoder_lstm_layers": 2,
        "decoder_channels": 32,
        "kernel_size": 3,
        "groupnorm_groups": 4,
        "residual_rollout": True,
        "detach_feedback": True,
    })
    optimizer: dict[str, Any] = field(default_factory=lambda: {
        "name": "AdamW",
        "weight_decay": 1.0e-4,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "gradient_clip": 1.0,
    })
    physics_contract: dict[str, Any] = field(default_factory=lambda: {
        "advection_enabled": False,
        "advection_status": "disabled_unconfirmed_physical_grid",
        "wind_u_channel_confirmed": 1,
        "wind_v_channel_confirmed": 2,
        "wind_units": "m/s",
        "dt_seconds_generation_inferred": 3600,
        "dx_meters_confirmed": None,
        "dy_meters_confirmed": None,
        "plume_value_space": "log_transformed_or_unknown",
        "notes": [
            "safe physics losses are mass/temporal smoothness/nonnegative/background only",
            "wind advection residual remains disabled until physical grid metadata is confirmed",
        ],
    })
    stage1: StageConfig = field(
        default_factory=lambda: StageConfig(
            name="stage1_direct_multihorizon",
            max_epochs=5,
            min_epochs=2,
            patience=2,
            learning_rate=1e-3,
            train_direct=True,
            train_rollout=False,
            teacher_forcing_start=0.0,
            teacher_forcing_end=0.0,
            horizon_weights=[1.0, 0.8, 0.6, 0.4],
            loss_weights=LossWeights(direct_data=1.0),
        )
    )
    stage2: StageConfig = field(
        default_factory=lambda: StageConfig(
            name="stage2_autoregressive_teacher_forcing",
            max_epochs=20,
            min_epochs=6,
            patience=5,
            learning_rate=5e-4,
            train_direct=False,
            train_rollout=True,
            teacher_forcing_start=1.0,
            teacher_forcing_end=0.2,
            horizon_weights=[1.0, 0.9, 0.8, 0.7],
            loss_weights=LossWeights(rollout_data=1.0),
        )
    )
    stage3: StageConfig = field(
        default_factory=lambda: StageConfig(
            name="stage3_mixed_robust_physics_v2",
            max_epochs=8,
            min_epochs=3,
            patience=3,
            learning_rate=2e-4,
            train_direct=True,
            train_rollout=True,
            teacher_forcing_start=0.50,
            teacher_forcing_end=0.10,
            loss_weights=LossWeights(
                rollout_data=1.0,
                direct_data=0.60,
                consistency=0.20,
                mass=0.05,
                temporal=0.05,
                smooth=0.01,
                nonneg=0.10,
                bg=0.02,
            ),
            horizon_weights=[1.0, 1.0, 1.0, 1.0],
        )
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    selection_gates: SelectionGateConfig = field(default_factory=SelectionGateConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRunSummary:
    run_name: str
    created_at: str
    finished_at: str | None
    status: str
    best_overall_checkpoint: str | None = None
    best_stage1_checkpoint: str | None = None
    best_stage2_checkpoint: str | None = None
    best_stage3_checkpoint: str | None = None
    final_checkpoint: str | None = None
    best_metrics: dict[str, Any] = field(default_factory=dict)
    stage_summaries: list[dict[str, Any]] = field(default_factory=list)
    selection_gate_summary: dict[str, Any] = field(default_factory=dict)
    dataset_counts: dict[str, int] = field(default_factory=dict)
    resume_checkpoint_path: str | None = None
    resume_mode: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is required for three-stage ConvLSTM adaptation training")
    return torch


def load_checkpoint_payload(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    require_torch()
    try:
        raw = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Older torch versions do not support weights_only.
        raw = torch.load(path, map_location=map_location)
    if not isinstance(raw, dict):
        raise ValueError(f"Selected robust ConvLSTM checkpoint payload is not a dict: {path}")
    return raw


def teacher_forcing_prob(epoch: int, total_epochs: int, start: float, end: float) -> float:
    progress = epoch / max(1, total_epochs - 1)
    return float(start + progress * (end - start))


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def reduce_batch_size_after_oom(current_batch_size: int, min_batch_size: int) -> int:
    if current_batch_size <= min_batch_size:
        raise RuntimeError(
            f"CUDA out of memory at minimum batch size {min_batch_size}; cannot reduce further"
        )
    return max(min_batch_size, current_batch_size // 2)


def selection_score(metrics: dict[str, Any]) -> float:
    weights = {
        "val_rollout_weighted_mse": 1.0,
        "val_rollout_mae": 0.25,
        "val_rollout_mass_abs_error": 0.001,
        "val_rollout_peak_location_error": 0.002,
        "val_rollout_background_false_positive_area": 0.01,
        "val_rollout_plume_iou": -0.25,
        "val_direct_weighted_mse": 0.35,
        "val_free_rollout_gap": 0.50,
    }
    return float(sum(float(metrics.get(key, 0.0)) * weight for key, weight in weights.items()))


def stage3_passes_selection_gates(
    reference_metrics: dict[str, float] | None,
    candidate_metrics: dict[str, float],
    gate_config: SelectionGateConfig | None = None,
) -> tuple[bool, list[str]]:
    cfg = gate_config or SelectionGateConfig()
    if not cfg.enabled or reference_metrics is None:
        return True, []

    checks = [
        (
            "val_rollout_weighted_mse",
            cfg.max_worse_rollout_weighted_mse_percent,
            "rollout_weighted_mse",
        ),
        ("val_rollout_weighted_mse_t3", cfg.max_worse_t3_weighted_mse_percent, "t3_weighted_mse"),
        ("val_rollout_weighted_mse_t4", cfg.max_worse_t4_weighted_mse_percent, "t4_weighted_mse"),
    ]
    reasons: list[str] = []
    for key, max_worse_percent, label in checks:
        ref = float(reference_metrics.get(key, math.inf))
        cand = float(candidate_metrics.get(key, math.inf))
        allowed = ref * (1.0 + max_worse_percent / 100.0)
        if cand > allowed:
            reasons.append(f"{label} {cand:.6g} exceeded allowed {allowed:.6g} from reference {ref:.6g}")
    return not reasons, reasons


def weighted_plume_mse(
    prediction: Any,
    target: Any,
    *,
    plume_threshold: float = 1e-6,
    plume_weight: float = 5.0,
    sample_weights: Any | None = None,
    horizon_weights: list[float] | None = None,
) -> Any:
    require_torch()
    plume_weights = torch.where(target > plume_threshold, torch.as_tensor(plume_weight, device=target.device), torch.ones_like(target))
    if horizon_weights is not None:
        horizon = torch.as_tensor(horizon_weights, dtype=target.dtype, device=target.device).view(1, -1, 1, 1, 1)
        plume_weights = plume_weights * horizon
    loss_by_element = plume_weights * (prediction - target).pow(2)
    if sample_weights is None:
        return loss_by_element.sum() / plume_weights.sum().clamp_min(1.0)
    sample = sample_weights.to(device=target.device, dtype=target.dtype).view(-1, 1, 1, 1, 1)
    weighted_loss = loss_by_element * sample
    weighted_norm = plume_weights * sample
    return weighted_loss.sum() / weighted_norm.sum().clamp_min(1.0)


def mass_abs_error(prediction: Any, target: Any) -> Any:
    spatial = tuple(range(2, prediction.ndim))
    pred_mass = prediction.sum(dim=spatial)
    target_mass = target.sum(dim=spatial)
    norm = target_mass.abs().mean().detach().clamp_min(1.0)
    return (pred_mass - target_mass).abs().mean() / norm


def temporal_smoothness_loss(prediction: Any, target: Any) -> Any:
    if prediction.shape[1] < 2:
        return prediction.new_tensor(0.0)
    return ((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])).pow(2).mean()


def spatial_smoothness_loss(prediction: Any) -> Any:
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]).pow(2).mean()
    dx = (prediction[..., :, 1:] - prediction[..., :, :-1]).pow(2).mean()
    return dx + dy


def background_penalty(prediction: Any, target: Any, plume_threshold: float) -> Any:
    mask = target <= plume_threshold
    if not bool(mask.any()):
        return prediction.new_tensor(0.0)
    return prediction[mask].pow(2).mean()


def apply_stage3_noise(inputs: Any, config: NoiseConfig) -> Any:
    if not config.enabled_stage3:
        return inputs
    noisy = inputs.clone()
    plume = noisy[:, :, 0:1, :, :]
    if config.plume_gaussian_std > 0:
        plume = plume + torch.randn_like(plume) * config.plume_gaussian_std
    if config.plume_dropout_prob > 0:
        keep = (torch.rand_like(plume) >= config.plume_dropout_prob).to(plume.dtype)
        plume = plume * keep
    max_shift = int(config.plume_random_shift_pixels)
    if max_shift > 0:
        shift_y = int(torch.randint(-max_shift, max_shift + 1, (), device=inputs.device).item())
        shift_x = int(torch.randint(-max_shift, max_shift + 1, (), device=inputs.device).item())
        plume = torch.roll(plume, shifts=(shift_y, shift_x), dims=(-2, -1))
    noisy[:, :, 0:1, :, :] = plume.clamp_min(config.clip_min)
    return noisy


def trainer_side_rollout(model: Any, x: Any, target: Any | None, teacher_forcing_probability: float) -> Any:
    state = model.encode(x)
    feedback = x[:, -1, 0:1, :, :]
    outputs: list[Any] = []
    for step in range(model.future_steps):
        cell_input = feedback.detach() if model.detach_feedback else feedback
        state = model.rollout_cell(cell_input, state)
        delta = model.rollout_delta_head(state[0])
        frame = feedback + delta if model.residual_rollout else delta
        frame = model._apply_output_activation(frame)
        outputs.append(frame)
        if target is not None and teacher_forcing_probability > 0.0 and step < model.future_steps - 1:
            if teacher_forcing_probability >= 1.0 or bool(torch.rand((), device=x.device) < teacher_forcing_probability):
                feedback = target[:, step, :, :, :]
            else:
                feedback = frame
        else:
            feedback = frame
    return torch.stack(outputs, dim=1)


class ThreeStageAdaptationTrainer:
    def __init__(
        self,
        *,
        train_samples: list[AdaptationSample],
        val_samples: list[AdaptationSample],
        output_dir: str | Path,
        config: ThreeStageTrainerConfig | None = None,
        resume_checkpoint_path: str | Path | None = None,
        resume_mode: ResumeMode = "none",
        start_stage: StageKey = "stage1",
        device: str = "auto",
        cancel_callback: Callable[[], bool] | None = None,
    ) -> None:
        require_torch()
        if resume_mode not in {"none", "model_only"}:
            raise ValueError(f"Unsupported resume_mode {resume_mode!r}; only 'none' and 'model_only' are supported")
        self.train_samples = list(train_samples)
        self.val_samples = list(val_samples)
        if not self.train_samples:
            raise ValueError("train_samples must not be empty")
        if not self.val_samples:
            raise ValueError("val_samples must not be empty")
        self.output_dir = Path(output_dir)
        self.config = config or ThreeStageTrainerConfig()
        self.resume_checkpoint_path = Path(resume_checkpoint_path) if resume_checkpoint_path else None
        self.resume_mode = resume_mode
        self.start_stage = start_stage
        self.cancel_callback = cancel_callback
        self.device = self._resolve_device(device)
        self.batch_size = int(self.config.initial_batch_size)
        self.global_epoch = 0
        self.created_at = utc_now()
        self.best_overall_score = math.inf
        self.best_overall_metrics: dict[str, Any] = {}
        self.best_stage2_reference_metrics: dict[str, float] | None = None
        self.selection_gate_summary: dict[str, Any] = {
            "enabled": self.config.selection_gates.enabled,
            "reference_stage_name": self.config.selection_gates.reference_stage_name,
            "stage3_rejected_by_gates": False,
            "rejection_reasons": [],
        }

        from plume.training.robust_convlstm_model import RobustMultiStepConvLSTMForecaster

        model_kwargs = self._model_kwargs()
        self.model = RobustMultiStepConvLSTMForecaster(**model_kwargs).to(self.device)
        if self.resume_checkpoint_path is not None:
            if self.resume_mode != "model_only":
                raise ValueError("resume_checkpoint_path requires resume_mode='model_only' in this trainer phase")
            self._load_model_only_checkpoint(self.resume_checkpoint_path)

    def _model_kwargs(self) -> dict[str, Any]:
        defaults = {
            "input_channels": 10,
            "input_frames": 3,
            "future_steps": 4,
            "encoder_channels": 32,
            "hidden_channels": 64,
            "decoder_channels": 32,
            "kernel_size": 3,
            "groupnorm_groups": 4,
            "residual_rollout": True,
            "detach_feedback": True,
            "output_activation": "softplus",
            "num_encoder_lstm_layers": 2,
        }
        return {**defaults, **dict(self.config.model)}

    def _resolve_device(self, requested: str) -> Any:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({requested}) but CUDA is not available")
        return torch.device(requested)

    def _load_model_only_checkpoint(self, path: Path) -> None:
        raw = load_checkpoint_payload(path, map_location=self.device)
        if not isinstance(raw, dict) or "model_state_dict" not in raw:
            raise ValueError(f"Checkpoint {path} does not contain model_state_dict")
        contract = raw.get("model_contract")
        if isinstance(contract, dict):
            self._validate_model_contract(contract)
        state = raw["model_state_dict"]
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Robust ConvLSTM state_dict mismatch during model-only resume: {exc}") from exc

    @staticmethod
    def _validate_model_contract(contract: dict[str, Any]) -> None:
        expected = MODEL_CONTRACT
        for key in ("model_name", "forecast_mode", "input_shape", "output_shape", "plume_channel"):
            if contract.get(key) != expected[key]:
                raise ValueError(f"checkpoint model_contract.{key} mismatch: expected {expected[key]!r}, got {contract.get(key)!r}")
        if contract.get("has_direct_branch") is not True or contract.get("has_autoregressive_branch") is not True:
            raise ValueError("checkpoint model_contract must include direct and autoregressive branches")

    def _make_loader(self, samples: list[AdaptationSample], shuffle: bool) -> Any:
        dataset = AdaptationNPZDataset(samples)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.config.num_workers)

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[Any, Any, Any | None]:
        inputs = batch["input"]
        targets = batch["target"]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets, dtype=torch.float32)
        weights = batch.get("weight")
        if weights is not None and not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, dtype=torch.float32)
        return inputs.to(self.device), targets.to(self.device), weights.to(self.device) if weights is not None else None

    def train(self) -> TrainingRunSummary:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.output_dir / "config.json", self.config.to_dict())
        metrics_path = self.output_dir / "metrics.jsonl"
        events_path = self.output_dir / "events.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()
        if events_path.exists():
            events_path.unlink()
        self._append_event({"event": "run_start", "run_name": self.config.run_name, "created_at": self.created_at})

        summary = TrainingRunSummary(
            run_name=self.config.run_name,
            created_at=self.created_at,
            finished_at=None,
            status="failed",
            dataset_counts={"train_total": len(self.train_samples), "val_total": len(self.val_samples)},
            resume_checkpoint_path=str(self.resume_checkpoint_path) if self.resume_checkpoint_path else None,
            resume_mode=self.resume_mode,
            selection_gate_summary=dict(self.selection_gate_summary),
        )
        try:
            self._raise_if_cancelled()
            stages = self._stages_from_start()
            for stage_index, stage in stages:
                self._raise_if_cancelled()
                if not stage.enabled:
                    continue
                self._append_event({"event": "stage_start", "stage": stage.name, "stage_index": stage_index})
                stage_summary = self._run_stage(stage_index, stage)
                summary.stage_summaries.append(stage_summary)
                self._append_event({"event": "stage_complete", "stage": stage.name, "stage_index": stage_index, "summary": stage_summary})
            self._raise_if_cancelled()
            final_metrics = self._evaluate(self._make_loader(self.val_samples, shuffle=False))
            final_ckpt = self._save_checkpoint("final_full_checkpoint.pt", "final", 0, final_metrics)
            self._ensure_best_aliases()
            summary.final_checkpoint = str(final_ckpt)
            summary.status = "completed"
            summary.best_overall_checkpoint = str(self.output_dir / "best_overall_full_checkpoint.pt") if (self.output_dir / "best_overall_full_checkpoint.pt").exists() else None
            summary.best_stage1_checkpoint = self._existing_path("best_stage1_direct_full_checkpoint.pt")
            summary.best_stage2_checkpoint = self._existing_path("best_stage2_rollout_full_checkpoint.pt")
            summary.best_stage3_checkpoint = self._existing_path("best_stage3_robust_full_checkpoint.pt")
            summary.best_metrics = dict(self.best_overall_metrics)
            summary.selection_gate_summary = dict(self.selection_gate_summary)
            summary.finished_at = utc_now()
            self._write_json(self.output_dir / "training_summary.json", summary.to_dict())
            self._append_event({"event": "run_complete", "final_checkpoint": summary.final_checkpoint, "best_overall_checkpoint": summary.best_overall_checkpoint})
            return summary
        except Exception:
            summary.status = "failed"
            summary.finished_at = utc_now()
            summary.selection_gate_summary = dict(self.selection_gate_summary)
            self._write_json(self.output_dir / "training_summary.json", summary.to_dict())
            self._append_event({"event": "run_failed", "finished_at": summary.finished_at})
            raise

    def _raise_if_cancelled(self) -> None:
        if self.cancel_callback is not None and self.cancel_callback():
            raise TrainingCancelled("Training cancelled by operator.")

    def _existing_path(self, name: str) -> str | None:
        path = self.output_dir / name
        return str(path) if path.exists() else None

    def _stages_from_start(self) -> list[tuple[int, StageConfig]]:
        all_stages: list[tuple[int, StageKey, StageConfig]] = [
            (1, "stage1", self.config.stage1),
            (2, "stage2", self.config.stage2),
            (3, "stage3", self.config.stage3),
        ]
        start_index = {"stage1": 1, "stage2": 2, "stage3": 3}[self.start_stage]
        return [(idx, cfg) for idx, _key, cfg in all_stages if idx >= start_index]

    def _run_stage(self, stage_index: int, stage: StageConfig) -> dict[str, Any]:
        stage_start_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        while True:
            try:
                return self._run_stage_once(stage_index, stage)
            except RuntimeError as exc:
                if not (self.config.auto_reduce_batch_on_oom and is_cuda_oom(exc)):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model.load_state_dict(stage_start_state, strict=True)
                self.model.to(self.device)
                old = self.batch_size
                self.batch_size = reduce_batch_size_after_oom(self.batch_size, self.config.min_batch_size)
                if self.batch_size == old:
                    raise

    def _run_stage_once(self, stage_index: int, stage: StageConfig) -> dict[str, Any]:
        train_loader = self._make_loader(self.train_samples, shuffle=True)
        val_loader = self._make_loader(self.val_samples, shuffle=False)
        opt_cfg = self.config.optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=stage.learning_rate,
            weight_decay=float(opt_cfg.get("weight_decay", 1.0e-4)),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
            eps=float(opt_cfg.get("eps", 1.0e-8)),
        )
        best_stage_score = math.inf
        best_stage_metrics: dict[str, Any] = {}
        best_epoch = 0
        epochs_without_improvement = 0
        completed_epochs = 0

        for epoch_in_stage in range(1, stage.max_epochs + 1):
            self._raise_if_cancelled()
            tf_prob = teacher_forcing_prob(
                epoch_in_stage - 1,
                stage.max_epochs,
                stage.teacher_forcing_start,
                stage.teacher_forcing_end,
            )
            train_loss = self._train_one_epoch(train_loader, optimizer, stage, stage_index, tf_prob)
            self._raise_if_cancelled()
            val_metrics = self._evaluate(val_loader)
            self._raise_if_cancelled()
            val_metrics.update(
                {
                    "stage_name": stage.name,
                    "stage_index": stage_index,
                    "epoch_in_stage": epoch_in_stage,
                    "global_epoch": self.global_epoch + 1,
                    "train_loss": train_loss,
                    "teacher_forcing_prob": tf_prob,
                    "learning_rate": stage.learning_rate,
                }
            )
            val_metrics["selection_score"] = selection_score(val_metrics)
            self.global_epoch += 1
            completed_epochs = epoch_in_stage
            self._append_metrics(val_metrics)

            score = float(val_metrics["selection_score"])
            if score < best_stage_score:
                best_stage_score = score
                best_stage_metrics = dict(val_metrics)
                best_epoch = epoch_in_stage
                epochs_without_improvement = 0
                self._save_stage_best(stage_index, stage, epoch_in_stage, val_metrics)
                if stage_index == 2:
                    self.best_stage2_reference_metrics = dict(val_metrics)
                self._maybe_save_best_overall(stage_index, stage, epoch_in_stage, val_metrics)
            else:
                epochs_without_improvement += 1

            if epoch_in_stage >= stage.min_epochs and epochs_without_improvement >= stage.patience:
                break

        transition = self._save_checkpoint(
            f"stage_transition_after_{stage.name}_full_checkpoint.pt",
            stage.name,
            completed_epochs,
            best_stage_metrics,
            stage_index=stage_index,
        )
        return {
            "stage_name": stage.name,
            "stage_index": stage_index,
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "best_selection_score": best_stage_score,
            "best_metrics": best_stage_metrics,
            "transition_checkpoint": str(transition),
            "batch_size": self.batch_size,
        }

    def _train_one_epoch(self, loader: Any, optimizer: Any, stage: StageConfig, stage_index: int, tf_prob: float) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for batch in loader:
            inputs, targets, weights = self._prepare_batch(batch)
            if stage_index == 3:
                inputs = apply_stage3_noise(inputs, self.config.noise)
            optimizer.zero_grad(set_to_none=True)
            direct_pred = self.model.direct(inputs) if stage.train_direct or stage.loss_weights.direct_data > 0 else None
            rollout_pred = (
                trainer_side_rollout(self.model, inputs, targets, tf_prob)
                if stage.train_rollout or stage.loss_weights.rollout_data > 0
                else None
            )
            loss = self._compute_loss(direct_pred, rollout_pred, targets, weights, stage.loss_weights, stage.horizon_weights)
            loss.backward()
            gradient_clip = float(self.config.optimizer.get("gradient_clip", 1.0))
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        return total / max(1, count)

    def _compute_loss(self, direct_pred: Any | None, rollout_pred: Any | None, targets: Any, weights: Any | None, lw: LossWeights, horizon_weights: list[float]) -> Any:
        loss = targets.new_tensor(0.0)
        if direct_pred is not None and lw.direct_data:
            loss = loss + lw.direct_data * weighted_plume_mse(
                direct_pred, targets, plume_threshold=self.config.plume_threshold, plume_weight=self.config.plume_weight, sample_weights=weights, horizon_weights=horizon_weights
            )
        if rollout_pred is not None and lw.rollout_data:
            loss = loss + lw.rollout_data * weighted_plume_mse(
                rollout_pred, targets, plume_threshold=self.config.plume_threshold, plume_weight=self.config.plume_weight, sample_weights=weights, horizon_weights=horizon_weights
            )
        if direct_pred is not None and rollout_pred is not None and lw.consistency:
            loss = loss + lw.consistency * (direct_pred - rollout_pred).pow(2).mean()
        physics_pred = rollout_pred if rollout_pred is not None else direct_pred
        if physics_pred is not None:
            if lw.mass:
                loss = loss + lw.mass * mass_abs_error(physics_pred, targets)
            if lw.temporal:
                loss = loss + lw.temporal * temporal_smoothness_loss(physics_pred, targets)
            if lw.smooth:
                loss = loss + lw.smooth * spatial_smoothness_loss(physics_pred)
            if lw.nonneg:
                loss = loss + lw.nonneg * torch.relu(-physics_pred).mean()
            if lw.bg:
                loss = loss + lw.bg * background_penalty(physics_pred, targets, self.config.plume_threshold)
        return loss

    def _evaluate(self, loader: Any) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        batches = 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets, weights = self._prepare_batch(batch)
                direct_pred = self.model.direct(inputs)
                rollout_pred = self.model.rollout(inputs)
                metrics = self._batch_metrics(direct_pred, rollout_pred, targets, weights)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                batches += 1
        averaged = {key: value / max(1, batches) for key, value in totals.items()}
        averaged["val_loss"] = averaged.get("val_rollout_weighted_mse", 0.0)
        return averaged

    def _batch_metrics(self, direct_pred: Any, rollout_pred: Any, targets: Any, weights: Any | None) -> dict[str, float]:
        free_rollout_gap = (rollout_pred - direct_pred).pow(2).mean()
        bg_mask = targets <= self.config.background_target_threshold
        bg_false_positive = rollout_pred[bg_mask].abs().mean() if bool(bg_mask.any()) else targets.new_tensor(0.0)
        metrics: dict[str, float] = {
            "val_direct_weighted_mse": float(weighted_plume_mse(direct_pred, targets, plume_threshold=self.config.plume_threshold, plume_weight=self.config.plume_weight, sample_weights=weights).cpu()),
            "val_rollout_weighted_mse": float(weighted_plume_mse(rollout_pred, targets, plume_threshold=self.config.plume_threshold, plume_weight=self.config.plume_weight, sample_weights=weights).cpu()),
            "val_rollout_mae": float((rollout_pred - targets).abs().mean().cpu()),
            "val_rollout_mass_abs_error": float(mass_abs_error(rollout_pred, targets).cpu()),
            "val_rollout_peak_location_error": float(self._peak_location_error(rollout_pred, targets).cpu()),
            "val_rollout_plume_iou": float(self._plume_iou(rollout_pred, targets).cpu()),
            "val_rollout_background_false_positive_area": float(bg_false_positive.cpu()),
            "val_free_rollout_gap": float(free_rollout_gap.cpu()),
        }
        for i in range(4):
            metrics[f"val_rollout_weighted_mse_t{i + 1}"] = float(weighted_plume_mse(
                rollout_pred[:, i : i + 1], targets[:, i : i + 1], plume_threshold=self.config.plume_threshold, plume_weight=self.config.plume_weight, sample_weights=weights
            ).cpu())
            metrics[f"val_direct_weighted_mse_t{i + 1}"] = float(weighted_plume_mse(
                direct_pred[:, i : i + 1], targets[:, i : i + 1], plume_threshold=self.config.plume_threshold, plume_weight=self.config.plume_weight, sample_weights=weights
            ).cpu())
        return metrics


    def _plume_iou(self, prediction: Any, target: Any) -> Any:
        pred_mask = prediction > self.config.plume_threshold
        target_mask = target > self.config.plume_threshold
        inter = (pred_mask & target_mask).flatten(start_dim=2).sum(dim=2).float()
        union = (pred_mask | target_mask).flatten(start_dim=2).sum(dim=2).float()
        return torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union)).mean()

    @staticmethod
    def _peak_location_error(prediction: Any, target: Any) -> Any:
        b, t, _c, h, w = prediction.shape
        pred_flat = prediction.reshape(b, t, -1).argmax(dim=-1).float()
        target_flat = target.reshape(b, t, -1).argmax(dim=-1).float()
        pred_y = torch.floor(pred_flat / w)
        pred_x = pred_flat % w
        target_y = torch.floor(target_flat / w)
        target_x = target_flat % w
        return torch.sqrt((pred_y - target_y).pow(2) + (pred_x - target_x).pow(2)).mean()

    def _save_stage_best(self, stage_index: int, stage: StageConfig, epoch_in_stage: int, metrics: dict[str, Any]) -> None:
        names = {
            1: "best_stage1_direct_full_checkpoint.pt",
            2: "best_stage2_rollout_full_checkpoint.pt",
            3: "best_stage3_robust_full_checkpoint.pt",
        }
        self._save_checkpoint(names[stage_index], stage.name, epoch_in_stage, metrics, stage_index=stage_index)

    def _maybe_save_best_overall(self, stage_index: int, stage: StageConfig, epoch_in_stage: int, metrics: dict[str, Any]) -> None:
        gate_passed = True
        gate_reasons: list[str] = []
        if stage_index == 3:
            gate_passed, gate_reasons = stage3_passes_selection_gates(
                self.best_stage2_reference_metrics,
                metrics,
                self.config.selection_gates,
            )
            if not gate_passed:
                self.selection_gate_summary["stage3_rejected_by_gates"] = True
                self.selection_gate_summary["rejection_reasons"] = gate_reasons
        score = float(metrics["selection_score"])
        if gate_passed and score < self.best_overall_score:
            self.best_overall_score = score
            self.best_overall_metrics = dict(metrics)
            self._save_checkpoint("best_overall_full_checkpoint.pt", stage.name, epoch_in_stage, metrics, stage_index=stage_index)

    def _checkpoint_payload(self, stage_name: str, epoch_in_stage: int, metrics: dict[str, Any], stage_index: int | None) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "model_contract": dict(MODEL_CONTRACT),
            "stage_name": stage_name,
            "stage_index": stage_index,
            "epoch_in_stage": epoch_in_stage,
            "global_epoch": self.global_epoch,
            "metrics": dict(metrics),
            "created_at": utc_now(),
        }

    def _save_checkpoint(
        self,
        filename: str,
        stage_name: str,
        epoch_in_stage: int,
        metrics: dict[str, Any],
        *,
        stage_index: int | None = None,
    ) -> Path:
        path = self.output_dir / filename
        payload = self._checkpoint_payload(stage_name, epoch_in_stage, metrics, stage_index)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        if filename == "best_overall_full_checkpoint.pt":
            self._copy_checkpoint_alias(path, self.output_dir / "best_full_checkpoint.pt")
        elif filename == "best_full_checkpoint.pt":
            self._copy_checkpoint_alias(path, self.output_dir / "best_overall_full_checkpoint.pt")
        self._append_event({"event": "checkpoint_saved", "path": str(path), "stage": stage_name, "stage_index": stage_index, "global_epoch": self.global_epoch})
        return path

    def _copy_checkpoint_alias(self, source: Path, destination: Path) -> None:
        if source.resolve(strict=False) == destination.resolve(strict=False):
            return
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, tmp)
        os.replace(tmp, destination)

    def _ensure_best_aliases(self) -> None:
        best_overall = self.output_dir / "best_overall_full_checkpoint.pt"
        best_contract = self.output_dir / "best_full_checkpoint.pt"
        if best_overall.exists() and not best_contract.exists():
            self._copy_checkpoint_alias(best_overall, best_contract)
        elif best_contract.exists() and not best_overall.exists():
            self._copy_checkpoint_alias(best_contract, best_overall)

    def _append_metrics(self, metrics: dict[str, Any]) -> None:
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    def _append_event(self, event: dict[str, Any]) -> None:
        payload = {"timestamp": utc_now(), **event}
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)


def train_three_stage_adaptation(
    *,
    train_samples: list[AdaptationSample],
    val_samples: list[AdaptationSample],
    output_dir: str | Path,
    config: ThreeStageTrainerConfig | None = None,
    resume_checkpoint_path: str | Path | None = None,
    resume_mode: ResumeMode = "none",
    start_stage: StageKey = "stage1",
    device: str = "auto",
    cancel_callback: Callable[[], bool] | None = None,
) -> TrainingRunSummary:
    trainer = ThreeStageAdaptationTrainer(
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=output_dir,
        config=config,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_mode=resume_mode,
        start_stage=start_stage,
        device=device,
        cancel_callback=cancel_callback,
    )
    return trainer.train()


__all__ = [
    "LossWeights",
    "NoiseConfig",
    "SelectionGateConfig",
    "StageConfig",
    "ThreeStageAdaptationTrainer",
    "ThreeStageTrainerConfig",
    "TrainingRunSummary",
    "TrainingCancelled",
    "apply_stage3_noise",
    "reduce_batch_size_after_oom",
    "selection_score",
    "stage3_passes_selection_gates",
    "teacher_forcing_prob",
    "train_three_stage_adaptation",
    "trainer_side_rollout",
    "load_checkpoint_payload",
    "weighted_plume_mse",
]
