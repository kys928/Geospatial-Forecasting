#!/usr/bin/env python3
"""
train_convlstm_plume_multistep_three_stage_robust.py

Three-stage robust ConvLSTM trainer for 4-step plume forecasting.

This is built from the previous uploaded two-stage trainer contract, but fixes the
curriculum semantics:

Stage 1: direct multi-horizon prediction only
    input:  real t-2, t-1, t
    output: t+1, t+2, t+3, t+4
    no autoregressive feedback

Stage 2: autoregressive rollout with scheduled teacher forcing
    the model sometimes feeds real previous plume frames and sometimes its own
    previous prediction, so training starts to resemble inference.

Stage 3: mixed robust training
    direct branch + autoregressive branch + noisy plume inputs + safe physics
    losses + consistency loss. Advection/PDE loss stays disabled because dx/dy
    and raw concentration units are not confirmed.

No argparse is used. Edit CONFIG below.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# =============================================================================
# CONFIG - edit here
# =============================================================================

CONFIG: Dict[str, Any] = {
    "run_name": "convlstm_multistep_three_stage_robust_v1",
    "data": {
        "windows_dir": "/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026/windows",
        "npz_glob": "*.npz",
        "input_key": "input",
        "target_key": "target",
        "input_frames": 3,
        "input_channels": 10,
        "height": 64,
        "width": 64,
        "future_steps": 4,
        "plume_channel": 0,
        "wind_u_channel": 1,
        "wind_v_channel": 2,
        "validation_split": 0.20,
        "random_seed": 42,
        "require_consecutive_window_ids": True,
        "max_files": None,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "drop_last": False,
    },
    "model": {
        "encoder_channels": 32,
        "hidden_channels": 64,
        "num_encoder_lstm_layers": 2,
        "decoder_channels": 32,
        "kernel_size": 3,
        "groupnorm_groups": 4,
        "residual_rollout": True,
        "detach_feedback": True,
    },
    "optimizer": {
        "name": "AdamW",
        "learning_rate_stage1": 1.0e-3,
        "learning_rate_stage2": 5.0e-4,
        "learning_rate_stage3": 2.0e-4,
        "weight_decay": 1.0e-4,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "gradient_clip": 1.0,
        "gradient_accumulation_steps": 1,
    },
    "precision": {
        "amp": True,
        "prefer_bf16_if_supported": True,
    },
    "physics_contract": {
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
    },
    "loss": {
        "plume_threshold": 1.0e-6,
        "plume_weight": 5.0,
        "background_target_threshold": 1.0e-6,
        "horizon_weights_stage1": [1.0, 0.8, 0.6, 0.4],
        "horizon_weights_stage2": [1.0, 0.9, 0.8, 0.7],
        "horizon_weights_stage3": [1.0, 1.0, 1.0, 1.0],
    },
    "noise": {
        "enabled_stage1": False,
        "enabled_stage2": False,
        "enabled_stage3": True,
        "plume_gaussian_std": 0.02,
        "plume_dropout_prob": 0.03,
        "plume_random_shift_pixels": 1,
        "clip_min": 0.0,
    },
    "curriculum": {
        "stages": [
            {
                "name": "stage1_direct_multihorizon",
                "mode": "direct_only",
                "max_epochs": 18,
                "min_epochs": 5,
                "patience": 5,
                "scheduler_patience": 3,
                "lr_key": "learning_rate_stage1",
                "horizon_weights_key": "horizon_weights_stage1",
                "teacher_forcing": {"enabled": False, "start": 0.0, "end": 0.0},
                "loss_weights": {
                    "direct_data": 1.00,
                    "rollout_data": 0.00,
                    "consistency": 0.00,
                    "mass": 0.00,
                    "temporal": 0.00,
                    "smooth": 0.00,
                    "nonneg": 0.00,
                    "bg": 0.00,
                },
            },
            {
                "name": "stage2_autoregressive_teacher_forcing",
                "mode": "rollout_only",
                "max_epochs": 24,
                "min_epochs": 6,
                "patience": 7,
                "scheduler_patience": 3,
                "lr_key": "learning_rate_stage2",
                "horizon_weights_key": "horizon_weights_stage2",
                "teacher_forcing": {"enabled": True, "start": 1.0, "end": 0.2},
                "loss_weights": {
                    "direct_data": 0.00,
                    "rollout_data": 1.00,
                    "consistency": 0.00,
                    "mass": 0.00,
                    "temporal": 0.00,
                    "smooth": 0.00,
                    "nonneg": 0.00,
                    "bg": 0.00,
                },
            },
            {
                "name": "stage3_mixed_robust_physics",
                "mode": "mixed",
                "max_epochs": 30,
                "min_epochs": 8,
                "patience": 8,
                "scheduler_patience": 3,
                "lr_key": "learning_rate_stage3",
                "horizon_weights_key": "horizon_weights_stage3",
                "teacher_forcing": {"enabled": True, "start": 0.5, "end": 0.1},
                "loss_weights": {
                    "direct_data": 0.60,
                    "rollout_data": 1.00,
                    "consistency": 0.20,
                    "mass": 0.05,
                    "temporal": 0.05,
                    "smooth": 0.01,
                    "nonneg": 0.10,
                    "bg": 0.02,
                },
            },
        ]
    },
    "selection": {
        "score_weights": {
            "val_rollout_weighted_mse": 1.00,
            "val_rollout_mae": 0.25,
            "val_rollout_mass_abs_error": 0.001,
            "val_rollout_peak_location_error": 0.002,
            "val_rollout_background_false_positive_area": 0.01,
            "val_rollout_plume_iou": -0.25,
            "val_direct_weighted_mse": 0.35,
            "val_free_rollout_gap": 0.50,
        }
    },
    "checkpointing": {
        "output_dir": "/workspace/Geospatial-Forecasting/runs/convlstm_multistep_three_stage_robust_v1",
        "artifact_model_dir": "/workspace/Geospatial-Forecasting/artifacts/models/convlstm_multistep_three_stage_robust_v1",
        "save_every_epochs": 2,
        "best_name": "best_full_checkpoint.pt",
        "final_name": "final_full_checkpoint.pt",
        "weights_only_pattern": "weights_epoch_{epoch:03d}_stage_{stage}.pt",
        "stage_transition_pattern": "stage_transition_after_{stage}_full_checkpoint.pt",
        "metrics_name": "metrics.jsonl",
        "events_name": "events.jsonl",
        "summary_name": "training_summary.json",
        "config_name": "config.json",
    },
}


# =============================================================================
# Utilities
# =============================================================================


def fixed_path(path: str | Path) -> Path:
    p = Path(path)
    if str(p).startswith("workspace/"):
        return Path("/") / p
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return obj


def choose_device_and_precision(config: Dict[str, Any]) -> Tuple[torch.device, bool, torch.dtype, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["precision"]["amp"] and device.type == "cuda")
    amp_dtype = torch.float32
    use_scaler = False
    if amp_enabled:
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if config["precision"]["prefer_bf16_if_supported"] and bf16_supported:
            amp_dtype = torch.bfloat16
            use_scaler = False
        else:
            amp_dtype = torch.float16
            use_scaler = True
    return device, amp_enabled, amp_dtype, use_scaler


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return nullcontext()


def teacher_forcing_prob(epoch_in_stage: int, total_epochs: int, start: float, end: float) -> float:
    progress = (epoch_in_stage - 1) / max(1, total_epochs - 1)
    return float(start + progress * (end - start))


# =============================================================================
# Dataset
# =============================================================================


@dataclass(frozen=True)
class WindowRecord:
    path: Path
    scenario_id: str
    window_id: int


_FILENAME_RE = re.compile(r"(?P<scenario>\d+)_(?P<window>\d+)\.npz$")


def _to_scalar_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def infer_record(path: Path) -> WindowRecord:
    scenario_id: Optional[str] = None
    window_id: Optional[int] = None
    match = _FILENAME_RE.search(path.name)
    if match:
        scenario_id = str(int(match.group("scenario")))
        window_id = int(match.group("window"))
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "scenario_id" in npz:
                scenario_id = _to_scalar_str(npz["scenario_id"])
            if "window_id" in npz:
                raw_window = _to_scalar_str(npz["window_id"])
                try:
                    window_id = int(float(raw_window))
                except ValueError:
                    pass
    except Exception:
        pass
    if scenario_id is None or window_id is None:
        raise ValueError(f"Could not infer scenario_id/window_id from {path}")
    return WindowRecord(path=path, scenario_id=scenario_id, window_id=window_id)


def discover_sequence_samples(config: Dict[str, Any]) -> List[Tuple[WindowRecord, List[WindowRecord]]]:
    data_cfg = config["data"]
    windows_dir = fixed_path(data_cfg["windows_dir"])
    files = sorted(windows_dir.rglob(data_cfg["npz_glob"]))
    if data_cfg["max_files"] is not None:
        files = files[: int(data_cfg["max_files"])]
    if not files:
        raise FileNotFoundError(f"No NPZ files found under {windows_dir}")
    records = [infer_record(p) for p in files]
    by_scenario: Dict[str, List[WindowRecord]] = defaultdict(list)
    for rec in records:
        by_scenario[rec.scenario_id].append(rec)
    future_steps = int(data_cfg["future_steps"])
    samples: List[Tuple[WindowRecord, List[WindowRecord]]] = []
    for group in by_scenario.values():
        group = sorted(group, key=lambda r: r.window_id)
        for i in range(0, len(group)):
            future = group[i : i + future_steps]
            if len(future) < future_steps:
                continue
            if bool(data_cfg["require_consecutive_window_ids"]):
                expected = list(range(group[i].window_id, group[i].window_id + future_steps))
                actual = [r.window_id for r in future]
                if actual != expected:
                    continue
            samples.append((group[i], future))
    if not samples:
        raise RuntimeError(f"No multistep samples could be built for future_steps={future_steps}.")
    return samples


class MultiStepWindowDataset(Dataset):
    def __init__(self, samples: List[Tuple[WindowRecord, List[WindowRecord]]], config: Dict[str, Any]):
        self.samples = samples
        self.config = config
        self.data_cfg = config["data"]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npz(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as npz:
            x = np.asarray(npz[self.data_cfg["input_key"]], dtype=np.float32)
            y = np.asarray(npz[self.data_cfg["target_key"]], dtype=np.float32)
        return x, y

    def _prepare_input(self, x: np.ndarray, path: Path) -> np.ndarray:
        expected = (
            self.data_cfg["input_frames"],
            self.data_cfg["input_channels"],
            self.data_cfg["height"],
            self.data_cfg["width"],
        )
        if x.shape != expected:
            raise ValueError(f"{path}: expected input shape {expected}, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: input contains non-finite values")
        return x

    def _extract_plume_target(self, target: np.ndarray, path: Path) -> np.ndarray:
        plume_ch = int(self.data_cfg["plume_channel"])
        h = int(self.data_cfg["height"])
        w = int(self.data_cfg["width"])
        if target.ndim == 4:
            if target.shape[-2:] != (h, w):
                raise ValueError(f"{path}: target spatial shape mismatch, got {target.shape}")
            if target.shape[1] <= plume_ch:
                raise ValueError(f"{path}: plume channel {plume_ch} missing in target {target.shape}")
            plume = target[0, plume_ch, :, :]
        elif target.ndim == 5:
            if target.shape[-2:] != (h, w):
                raise ValueError(f"{path}: target spatial shape mismatch, got {target.shape}")
            plume = target[0, 0, plume_ch, :, :] if target.shape[1] == 1 else target[0, plume_ch, :, :]
        elif target.ndim == 3:
            plume = target[plume_ch]
        elif target.ndim == 2:
            plume = target
        else:
            raise ValueError(f"{path}: unsupported target shape {target.shape}")
        plume = np.asarray(plume, dtype=np.float32)
        if plume.shape != (h, w):
            raise ValueError(f"{path}: extracted plume target shape must be {(h, w)}, got {plume.shape}")
        if not np.isfinite(plume).all():
            raise ValueError(f"{path}: plume target contains non-finite values")
        return plume[None, :, :]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_rec, future_recs = self.samples[idx]
        x_raw, _ = self._load_npz(input_rec.path)
        x = self._prepare_input(x_raw, input_rec.path)
        ys: List[np.ndarray] = []
        for rec in future_recs:
            _, y_raw = self._load_npz(rec.path)
            ys.append(self._extract_plume_target(y_raw, rec.path))
        y_seq = np.stack(ys, axis=0)
        return torch.from_numpy(x), torch.from_numpy(y_seq)


def build_train_val_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    samples = discover_sequence_samples(config)
    seed = int(config["data"]["random_seed"])
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(samples) * float(config["data"]["validation_split"]))))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    dataset = MultiStepWindowDataset(samples, config)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    loader_cfg = dict(config["loader"])
    if int(loader_cfg["num_workers"]) == 0:
        loader_cfg["persistent_workers"] = False
        loader_cfg.pop("prefetch_factor", None)
    kwargs = dict(
        batch_size=int(loader_cfg["batch_size"]),
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
        drop_last=bool(loader_cfg["drop_last"]),
    )
    if int(loader_cfg["num_workers"]) > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", None)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_kwargs = dict(kwargs)
    val_kwargs["drop_last"] = False
    val_loader = DataLoader(val_ds, shuffle=False, **val_kwargs)
    stats = {
        "total_sequence_samples": len(samples),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "future_steps": config["data"]["future_steps"],
        "example_first_sample": {
            "input": str(samples[0][0].path),
            "future_targets": [str(r.path) for r in samples[0][1]],
        },
    }
    return train_loader, val_loader, stats


# =============================================================================
# Model
# =============================================================================


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        c = torch.zeros_like(h)
        return h, c

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class RobustMultiStepConvLSTMForecaster(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        data = config["data"]
        model = config["model"]
        self.input_frames = int(data["input_frames"])
        self.input_channels = int(data["input_channels"])
        self.future_steps = int(data["future_steps"])
        self.height = int(data["height"])
        self.width = int(data["width"])
        self.plume_channel = int(data["plume_channel"])
        self.hidden_channels = int(model["hidden_channels"])
        self.residual_rollout = bool(model.get("residual_rollout", True))
        self.detach_feedback = bool(model.get("detach_feedback", True))
        encoder_channels = int(model["encoder_channels"])
        kernel_size = int(model["kernel_size"])
        groups = int(model["groupnorm_groups"])
        decoder_channels = int(model["decoder_channels"])

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=encoder_channels),
            nn.GELU(),
        )
        self.encoder_lstm_layers = nn.ModuleList()
        for i in range(int(model["num_encoder_lstm_layers"])):
            in_ch = encoder_channels if i == 0 else self.hidden_channels
            self.encoder_lstm_layers.append(ConvLSTMCell(in_ch, self.hidden_channels, kernel_size=kernel_size))

        self.direct_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_channels, decoder_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=groups, num_channels=decoder_channels),
                nn.GELU(),
                nn.Conv2d(decoder_channels, 1, kernel_size=1),
            )
            for _ in range(self.future_steps)
        ])

        self.rollout_cell = ConvLSTMCell(1, self.hidden_channels, kernel_size=kernel_size)
        self.rollout_delta_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.output_activation = nn.Softplus()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        expected = (self.input_frames, self.input_channels, self.height, self.width)
        if (t, c, h, w) != expected:
            raise ValueError(f"Expected x after batch {expected}, got {(t,c,h,w)}")
        seq = [self.encoder(x[:, i]) for i in range(t)]
        final_h: Optional[torch.Tensor] = None
        final_c: Optional[torch.Tensor] = None
        for cell in self.encoder_lstm_layers:
            h_t, c_t = cell.init_hidden(b, h, w, x.device, x.dtype)
            out_seq = []
            for frame in seq:
                h_t, c_t = cell(frame, h_t, c_t)
                out_seq.append(h_t)
            seq = out_seq
            final_h, final_c = h_t, c_t
        assert final_h is not None and final_c is not None
        return final_h, final_c

    def direct(self, h: torch.Tensor) -> torch.Tensor:
        preds = [self.output_activation(head(h)) for head in self.direct_heads]
        return torch.stack(preds, dim=1)

    def rollout(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        targets: Optional[torch.Tensor],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        prev = x[:, -1, self.plume_channel : self.plume_channel + 1, :, :]
        preds: List[torch.Tensor] = []
        h_dec, c_dec = h, c
        for step in range(self.future_steps):
            h_dec, c_dec = self.rollout_cell(prev, h_dec, c_dec)
            raw_delta = self.rollout_delta_head(h_dec)
            if self.residual_rollout:
                pred = self.output_activation(prev + raw_delta)
            else:
                pred = self.output_activation(raw_delta)
            preds.append(pred)
            use_teacher = (
                self.training
                and targets is not None
                and teacher_forcing_ratio > 0.0
                and random.random() < teacher_forcing_ratio
            )
            next_prev = targets[:, step] if use_teacher else pred
            if self.detach_feedback and not use_teacher:
                next_prev = next_prev.detach()
            prev = next_prev
        return torch.stack(preds, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        return_direct: bool = True,
        return_rollout: bool = True,
    ) -> Dict[str, torch.Tensor]:
        h, c = self.encode(x)
        out: Dict[str, torch.Tensor] = {}
        if return_direct:
            out["direct"] = self.direct(h)
        if return_rollout:
            out["rollout"] = self.rollout(x, h, c, targets, teacher_forcing_ratio)
        return out


# =============================================================================
# Losses and metrics
# =============================================================================


def safe_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.new_tensor(0.0)


def horizon_weighted_plume_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    plume_weight: float,
    horizon_weights: List[float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    weights = torch.tensor(horizon_weights, dtype=pred.dtype, device=pred.device).view(1, -1, 1, 1, 1)
    diff = pred - target
    diff2 = diff.pow(2)
    plume_mask = target > threshold
    bg_mask = ~plume_mask
    plume_w = 1.0 + plume_weight * plume_mask.to(pred.dtype)
    loss = (weights * plume_w * diff2).sum() / (weights * plume_w).sum().clamp_min(1.0)
    metrics: Dict[str, torch.Tensor] = {
        "weighted_mse": loss.detach(),
        "mse": diff2.mean().detach(),
        "mae": diff.abs().mean().detach(),
        "plume_mse": safe_mean(diff2, plume_mask).detach(),
        "background_mse": safe_mean(diff2, bg_mask).detach(),
    }
    for k in range(pred.shape[1]):
        d2 = diff[:, k].pow(2)
        metrics[f"weighted_mse_t{k+1}"] = ((1.0 + plume_weight * plume_mask[:, k].to(pred.dtype)) * d2).mean().detach()
        metrics[f"mae_t{k+1}"] = diff[:, k].abs().mean().detach()
    return loss, metrics


def mass_balance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mass = pred.sum(dim=(-1, -2, -3))
    target_mass = target.sum(dim=(-1, -2, -3))
    return F.mse_loss(pred_mass, target_mass)


def temporal_evolution_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    true_diff = target[:, 1:] - target[:, :-1]
    return F.l1_loss(pred_diff, true_diff)


def spatial_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    dx = pred[..., :, 1:] - pred[..., :, :-1]
    dy = pred[..., 1:, :] - pred[..., :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


def nonnegative_concentration_loss(pred: torch.Tensor) -> torch.Tensor:
    return F.relu(-pred).pow(2).mean()


def background_false_positive_loss(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    return safe_mean(pred.pow(2), target <= threshold)


def peak_location_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b, f, _, h, w = pred.shape
    pred_idx = pred.reshape(b, f, -1).argmax(dim=-1)
    tgt_idx = target.reshape(b, f, -1).argmax(dim=-1)
    pred_y = pred_idx // w
    pred_x = pred_idx % w
    tgt_y = tgt_idx // w
    tgt_x = tgt_idx % w
    return ((pred_x.float() - tgt_x.float()).pow(2) + (pred_y.float() - tgt_y.float()).pow(2)).sqrt().mean()


def plume_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    p = pred > threshold
    t = target > threshold
    inter = (p & t).sum(dim=(-1, -2, -3)).float()
    union = (p | t).sum(dim=(-1, -2, -3)).float()
    return torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union)).mean()


def background_false_positive_area(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    return ((pred > threshold) & (target <= threshold)).sum(dim=(-1, -2, -3)).float().mean()


def branch_metrics(prefix: str, pred: torch.Tensor, target: torch.Tensor, config: Dict[str, Any], horizon_weights: List[float]) -> Dict[str, torch.Tensor]:
    threshold = float(config["loss"]["plume_threshold"])
    plume_weight = float(config["loss"]["plume_weight"])
    _, base = horizon_weighted_plume_loss(pred, target, threshold, plume_weight, horizon_weights)
    mass_abs = (pred.sum(dim=(-1, -2, -3)) - target.sum(dim=(-1, -2, -3))).abs().mean()
    out = {f"{prefix}_{k}": v for k, v in base.items()}
    out.update({
        f"{prefix}_mass_loss": mass_balance_loss(pred, target).detach(),
        f"{prefix}_temporal_loss": temporal_evolution_loss(pred, target).detach(),
        f"{prefix}_smoothness_loss": spatial_smoothness_loss(pred).detach(),
        f"{prefix}_nonnegative_loss": nonnegative_concentration_loss(pred).detach(),
        f"{prefix}_background_loss": background_false_positive_loss(pred, target, float(config["loss"]["background_target_threshold"])).detach(),
        f"{prefix}_mass_abs_error": mass_abs.detach(),
        f"{prefix}_peak_location_error": peak_location_error(pred, target).detach(),
        f"{prefix}_plume_iou": plume_iou(pred, target, threshold).detach(),
        f"{prefix}_background_false_positive_area": background_false_positive_area(pred, target, threshold).detach(),
    })
    return out


def compute_stage_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    config: Dict[str, Any],
    stage: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    threshold = float(config["loss"]["plume_threshold"])
    plume_weight = float(config["loss"]["plume_weight"])
    horizon_weights = list(config["loss"][stage["horizon_weights_key"]])
    lw = stage["loss_weights"]
    device = target.device
    total = torch.zeros((), dtype=target.dtype, device=device)
    metrics: Dict[str, torch.Tensor] = {}

    if "direct" in outputs:
        direct_loss, _ = horizon_weighted_plume_loss(outputs["direct"], target, threshold, plume_weight, horizon_weights)
        total = total + float(lw.get("direct_data", 0.0)) * direct_loss
        metrics.update(branch_metrics("direct", outputs["direct"], target, config, horizon_weights))
        metrics["direct_data_loss"] = direct_loss.detach()

    if "rollout" in outputs:
        rollout_loss, _ = horizon_weighted_plume_loss(outputs["rollout"], target, threshold, plume_weight, horizon_weights)
        total = total + float(lw.get("rollout_data", 0.0)) * rollout_loss
        metrics.update(branch_metrics("rollout", outputs["rollout"], target, config, horizon_weights))
        metrics["rollout_data_loss"] = rollout_loss.detach()

        total = total + float(lw.get("mass", 0.0)) * mass_balance_loss(outputs["rollout"], target)
        total = total + float(lw.get("temporal", 0.0)) * temporal_evolution_loss(outputs["rollout"], target)
        total = total + float(lw.get("smooth", 0.0)) * spatial_smoothness_loss(outputs["rollout"])
        total = total + float(lw.get("nonneg", 0.0)) * nonnegative_concentration_loss(outputs["rollout"])
        total = total + float(lw.get("bg", 0.0)) * background_false_positive_loss(
            outputs["rollout"], target, float(config["loss"]["background_target_threshold"])
        )

    if "direct" in outputs and "rollout" in outputs:
        consistency = F.l1_loss(outputs["rollout"], outputs["direct"].detach())
        total = total + float(lw.get("consistency", 0.0)) * consistency
        metrics["consistency_loss"] = consistency.detach()

    metrics["loss"] = total.detach()
    if "direct" in outputs and "rollout" in outputs:
        metrics["free_rollout_gap"] = (
            metrics["rollout_weighted_mse"] - metrics["direct_weighted_mse"]
        ).abs().detach()
    else:
        metrics["free_rollout_gap"] = target.new_tensor(0.0)
    return total, metrics


def aggregate_metrics(totals: Dict[str, float], batch_metrics: Dict[str, torch.Tensor], batch_size: int) -> None:
    for k, v in batch_metrics.items():
        totals[k] = totals.get(k, 0.0) + float(v.detach().cpu().item()) * batch_size


def finalize_metrics(totals: Dict[str, float], n: int, prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v / max(1, n) for k, v in totals.items()}


def corrupt_plume_inputs(x: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    noise_cfg = config["noise"]
    plume_ch = int(config["data"]["plume_channel"])
    out = x.clone()
    plume = out[:, :, plume_ch : plume_ch + 1]
    std = float(noise_cfg["plume_gaussian_std"])
    if std > 0:
        plume = plume + torch.randn_like(plume) * std
    drop_prob = float(noise_cfg["plume_dropout_prob"])
    if drop_prob > 0:
        keep = torch.rand_like(plume) > drop_prob
        plume = plume * keep.to(plume.dtype)
    max_shift = int(noise_cfg["plume_random_shift_pixels"])
    if max_shift > 0:
        shift_y = random.randint(-max_shift, max_shift)
        shift_x = random.randint(-max_shift, max_shift)
        plume = torch.roll(plume, shifts=(shift_y, shift_x), dims=(-2, -1))
    plume = torch.clamp(plume, min=float(noise_cfg["clip_min"]))
    out[:, :, plume_ch : plume_ch + 1] = plume
    return out


# =============================================================================
# Checkpointing / scoring
# =============================================================================


def composite_score(metrics: Dict[str, float], config: Dict[str, Any]) -> float:
    score = 0.0
    for key, weight in config["selection"]["score_weights"].items():
        value = metrics.get(key)
        if value is None:
            continue
        score += float(weight) * float(value)
    return float(score)


def make_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_epoch: int,
    stage_index: int,
    stage_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    best_score: float,
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch_in_stage": epoch,
        "global_epoch": global_epoch,
        "stage_index": stage_index,
        "stage_name": stage_name,
        "best_score": best_score,
        "metrics": metrics,
        "config": clean_for_json(config),
        "model_contract": {
            "model_name": "RobustMultiStepConvLSTMForecaster",
            "forecast_mode": "direct_plus_autoregressive_multistep",
            "input_shape": [
                config["data"]["input_frames"],
                config["data"]["input_channels"],
                config["data"]["height"],
                config["data"]["width"],
            ],
            "output_shape": [
                config["data"]["future_steps"],
                1,
                config["data"]["height"],
                config["data"]["width"],
            ],
            "plume_channel": config["data"]["plume_channel"],
            "wind_u_channel": config["data"]["wind_u_channel"],
            "wind_v_channel": config["data"]["wind_v_channel"],
            "has_direct_branch": True,
            "has_autoregressive_branch": True,
            "residual_rollout": config["model"].get("residual_rollout", True),
        },
    }


def validate_advection_contract(config: Dict[str, Any]) -> None:
    if bool(config["physics_contract"].get("advection_enabled", False)):
        raise RuntimeError("Advection is disabled until dx/dy/raw plume units are confirmed.")


# =============================================================================
# Train / Validate
# =============================================================================


def branch_flags_for_stage(stage: Dict[str, Any]) -> Tuple[bool, bool]:
    mode = stage["mode"]
    if mode == "direct_only":
        return True, False
    if mode == "rollout_only":
        return False, True
    if mode == "mixed":
        return True, True
    raise ValueError(f"Unknown stage mode: {mode}")


def train_one_epoch(
    model: RobustMultiStepConvLSTMForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    config: Dict[str, Any],
    stage: Dict[str, Any],
    epoch_in_stage: int,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n = 0
    grad_accum = int(config["optimizer"]["gradient_accumulation_steps"])
    return_direct, return_rollout = branch_flags_for_stage(stage)
    tf_cfg = stage["teacher_forcing"]
    tf_ratio = teacher_forcing_prob(epoch_in_stage, int(stage["max_epochs"]), float(tf_cfg["start"]), float(tf_cfg["end"])) if bool(tf_cfg["enabled"]) else 0.0
    noise_enabled = bool(config["noise"].get(f"enabled_stage{stage['name'][5]}", False))

    optimizer.zero_grad(set_to_none=True)
    for step, (x_cpu, y_cpu) in enumerate(loader, start=1):
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        if noise_enabled:
            x = corrupt_plume_inputs(x, config)
        bs = x.shape[0]
        with autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(x, targets=y, teacher_forcing_ratio=tf_ratio, return_direct=return_direct, return_rollout=return_rollout)
            loss, metrics = compute_stage_loss(outputs, y, config, stage)
            loss = loss / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimizer"]["gradient_clip"]))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics["teacher_forcing_ratio"] = torch.tensor(tf_ratio, device=device)
        aggregate_metrics(totals, metrics, bs)
        n += bs
    return finalize_metrics(totals, n, "train")


@torch.no_grad()
def validate_one_epoch(
    model: RobustMultiStepConvLSTMForecaster,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    config: Dict[str, Any],
    stage: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    # Always validate both branches free-running so Stage 1/2/3 are comparable.
    eval_stage = dict(stage)
    eval_stage["mode"] = "mixed"
    eval_stage["loss_weights"] = {
        "direct_data": 1.0,
        "rollout_data": 1.0,
        "consistency": 0.0,
        "mass": 0.0,
        "temporal": 0.0,
        "smooth": 0.0,
        "nonneg": 0.0,
        "bg": 0.0,
    }
    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        bs = x.shape[0]
        with autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(x, targets=None, teacher_forcing_ratio=0.0, return_direct=True, return_rollout=True)
            _, metrics = compute_stage_loss(outputs, y, config, eval_stage)
        aggregate_metrics(totals, metrics, bs)
        n += bs
    return finalize_metrics(totals, n, "val")


def main() -> None:
    config = CONFIG
    validate_advection_contract(config)
    set_seed(int(config["data"]["random_seed"]))
    torch.backends.cudnn.benchmark = True

    output_dir = fixed_path(config["checkpointing"]["output_dir"])
    artifact_dir = fixed_path(config["checkpointing"]["artifact_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / config["checkpointing"]["metrics_name"]
    events_path = output_dir / config["checkpointing"]["events_name"]
    summary_path = output_dir / config["checkpointing"]["summary_name"]
    config_path = output_dir / config["checkpointing"]["config_name"]
    metrics_path.write_text("", encoding="utf-8")
    events_path.write_text("", encoding="utf-8")
    atomic_json_save(clean_for_json(config), config_path)

    device, amp_enabled, amp_dtype, use_scaler = choose_device_and_precision(config)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None
    train_loader, val_loader, data_stats = build_train_val_loaders(config)

    model = RobustMultiStepConvLSTMForecaster(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate_stage1"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
        betas=tuple(config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
    )

    summary: Dict[str, Any] = {
        "run_name": config["run_name"],
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "use_grad_scaler": bool(use_scaler),
        "trainable_parameters": total_params,
        "data_stats": data_stats,
        "physics_contract": config["physics_contract"],
        "stage_summaries": [],
        "best": {"score": math.inf, "path": None, "artifact_path": None, "stage": None, "global_epoch": None, "metrics": None},
    }
    atomic_json_save(clean_for_json(summary), summary_path)

    print("=" * 96)
    print("Three-stage robust ConvLSTM multi-step training")
    print("=" * 96)
    print(f"device={device} gpu={summary['gpu_name']} amp={amp_enabled} dtype={summary['amp_dtype']}")
    print(f"run_output_dir={output_dir}")
    print(f"artifact_model_dir={artifact_dir}")
    print(f"samples: train={data_stats['train_samples']} val={data_stats['val_samples']} future_steps={config['data']['future_steps']}")
    print(f"trainable_params={total_params:,}")
    print("=" * 96)

    append_jsonl(events_path, {"event": "run_start", "summary": clean_for_json(summary)})
    best_score = math.inf
    global_epoch = 0

    for stage_index, stage in enumerate(config["curriculum"]["stages"], start=1):
        stage_name = stage["name"]
        for group in optimizer.param_groups:
            group["lr"] = float(config["optimizer"][stage["lr_key"]])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(stage["scheduler_patience"]), min_lr=1.0e-6
        )
        print(f"\n--- {stage_name} ---")
        append_jsonl(events_path, {"event": "stage_start", "stage": stage_name, "stage_index": stage_index})
        stage_best_score = math.inf
        stage_best_epoch = None
        epochs_without_improve = 0
        stage_records: List[Dict[str, float]] = []

        for epoch in range(1, int(stage["max_epochs"]) + 1):
            global_epoch += 1
            t0 = time.time()
            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_enabled, amp_dtype, config, stage, epoch)
            val_metrics = validate_one_epoch(model, val_loader, device, amp_enabled, amp_dtype, config, stage)
            all_metrics: Dict[str, float] = {
                "global_epoch": float(global_epoch),
                "epoch_in_stage": float(epoch),
                "stage_index": float(stage_index),
                **train_metrics,
                **val_metrics,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - t0,
            }
            score = composite_score(all_metrics, config)
            all_metrics["selection_score"] = score
            scheduler.step(float(all_metrics.get("val_rollout_weighted_mse", all_metrics.get("val_loss", score))))
            append_jsonl(metrics_path, {"stage": stage_name, **all_metrics})
            stage_records.append(all_metrics)

            improved = score < best_score
            stage_improved = score < stage_best_score
            if improved:
                best_score = score
                best_path = output_dir / config["checkpointing"]["best_name"]
                artifact_best_path = artifact_dir / config["checkpointing"]["best_name"]
                payload = make_checkpoint_payload(model, optimizer, scheduler, epoch, global_epoch, stage_index, stage_name, config, all_metrics, best_score)
                atomic_torch_save(payload, best_path)
                atomic_torch_save(payload, artifact_best_path)
                summary["best"] = {
                    "score": best_score,
                    "path": str(best_path),
                    "artifact_path": str(artifact_best_path),
                    "stage": stage_name,
                    "global_epoch": global_epoch,
                    "metrics": all_metrics,
                }
                append_jsonl(events_path, {"event": "new_best_full_checkpoint", "stage": stage_name, "global_epoch": global_epoch, "score": best_score, "path": str(best_path), "artifact_path": str(artifact_best_path)})

            if stage_improved:
                stage_best_score = score
                stage_best_epoch = epoch
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epoch % int(config["checkpointing"]["save_every_epochs"]) == 0:
                weights_path = output_dir / config["checkpointing"]["weights_only_pattern"].format(epoch=global_epoch, stage=stage_name)
                atomic_torch_save(model.state_dict(), weights_path)
                append_jsonl(events_path, {"event": "weights_only_checkpoint", "stage": stage_name, "global_epoch": global_epoch, "path": str(weights_path)})

            print(
                f"{stage_name} epoch {epoch:03d}/{stage['max_epochs']:03d} global={global_epoch:03d} "
                f"tf={train_metrics.get('train_teacher_forcing_ratio', 0.0):.3f} "
                f"val_rollout_wmse={all_metrics.get('val_rollout_weighted_mse', math.nan):.6g} "
                f"val_direct_wmse={all_metrics.get('val_direct_weighted_mse', math.nan):.6g} "
                f"val_iou={all_metrics.get('val_rollout_plume_iou', math.nan):.4f} "
                f"gap={all_metrics.get('val_free_rollout_gap', math.nan):.6g} "
                f"score={score:.6g} {'BEST' if improved else ''}"
            )

            summary["status"] = "running"
            summary["current_stage"] = stage_name
            summary["last_metrics"] = all_metrics
            atomic_json_save(clean_for_json(summary), summary_path)

            if epoch >= int(stage["min_epochs"]) and epochs_without_improve >= int(stage["patience"]):
                append_jsonl(events_path, {"event": "stage_early_stop", "stage": stage_name, "global_epoch": global_epoch, "epochs_without_improve": epochs_without_improve})
                print(f"[stage early stop] {stage_name} after {epoch} epochs.")
                break

        transition_path = output_dir / config["checkpointing"]["stage_transition_pattern"].format(stage=stage_name)
        payload = make_checkpoint_payload(model, optimizer, scheduler, int(stage_records[-1]["epoch_in_stage"]), global_epoch, stage_index, stage_name, config, stage_records[-1], best_score)
        atomic_torch_save(payload, transition_path)
        append_jsonl(events_path, {"event": "stage_complete_checkpoint", "stage": stage_name, "global_epoch": global_epoch, "path": str(transition_path)})
        summary["stage_summaries"].append({
            "stage": stage_name,
            "stage_index": stage_index,
            "epochs_ran": len(stage_records),
            "stage_best_score": stage_best_score,
            "stage_best_epoch": stage_best_epoch,
            "last_record": stage_records[-1] if stage_records else None,
        })
        atomic_json_save(clean_for_json(summary), summary_path)

    final_path = output_dir / config["checkpointing"]["final_name"]
    artifact_final_path = artifact_dir / config["checkpointing"]["final_name"]
    final_metrics = summary.get("last_metrics", {})
    payload = make_checkpoint_payload(model, optimizer, scheduler, int(final_metrics.get("epoch_in_stage", 0)), global_epoch, len(config["curriculum"]["stages"]), str(summary.get("current_stage", "unknown")), config, final_metrics, best_score)
    atomic_torch_save(payload, final_path)
    atomic_torch_save(payload, artifact_final_path)
    summary["status"] = "complete"
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["final_checkpoint_path"] = str(final_path)
    summary["artifact_final_checkpoint_path"] = str(artifact_final_path)
    atomic_json_save(clean_for_json(summary), summary_path)
    append_jsonl(events_path, {"event": "run_complete", "final_checkpoint": str(final_path), "artifact_final_checkpoint": str(artifact_final_path), "best": summary["best"]})

    print("=" * 96)
    print("Training complete")
    print(f"best run checkpoint:      {summary['best']['path']}")
    print(f"best artifact checkpoint: {summary['best']['artifact_path']}")
    print(f"final run checkpoint:     {final_path}")
    print(f"final artifact checkpoint:{artifact_final_path}")
    print(f"metrics:                  {metrics_path}")
    print(f"summary:                  {summary_path}")
    print("=" * 96)


if __name__ == "__main__":
    main()
