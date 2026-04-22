from __future__ import annotations

from pathlib import Path

import numpy as np


class MinimalConvLSTMModel:
    """Lightweight ConvLSTM-style model wrapper for online backend integration.

    This uses random/untrained weights for demo/runtime plumbing only.

    Inference contract:
    - Input shape: (T, C, H, W)
    - Output shape: (H, W)
    - Output meaning: one non-negative concentration grid prediction for the next forecast step.
    """

    def __init__(self, input_channels: int, hidden_channels: int = 8, seed: int = 7):
        if input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        rng = np.random.default_rng(seed)
        self.w_x = rng.normal(0.0, 0.1, size=(4 * hidden_channels, input_channels))
        self.w_h = rng.normal(0.0, 0.1, size=(4 * hidden_channels, hidden_channels))
        self.b = np.zeros((4 * hidden_channels,), dtype=float)
        self.w_out = rng.normal(0.0, 0.1, size=(hidden_channels,))
        self.b_out = 0.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.ndim != 4:
            raise ValueError(f"Expected input shape (T, C, H, W), got {sequence.shape}")
        t_steps, channels, rows, cols = sequence.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        if t_steps <= 0:
            raise ValueError("Expected at least one time step")

        h_t = np.zeros((self.hidden_channels, rows, cols), dtype=float)
        c_t = np.zeros((self.hidden_channels, rows, cols), dtype=float)
        for t in range(t_steps):
            x_t = sequence[t]
            gates = (
                np.einsum("gc,chw->ghw", self.w_x, x_t)
                + np.einsum("gc,chw->ghw", self.w_h, h_t)
                + self.b[:, np.newaxis, np.newaxis]
            )
            i, f, o, g = np.split(gates, 4, axis=0)
            c_t = self._sigmoid(f) * c_t + self._sigmoid(i) * np.tanh(g)
            h_t = self._sigmoid(o) * np.tanh(c_t)

        concentration = np.einsum("h,hrc->rc", self.w_out, h_t) + self.b_out
        return np.clip(concentration, a_min=0.0, a_max=None)

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "w_x": self.w_x,
            "w_h": self.w_h,
            "b": self.b,
            "w_out": self.w_out,
            "b_out": float(self.b_out),
        }

    def load_state_dict(self, payload: dict[str, np.ndarray | float], *, strict: bool = True) -> None:
        required = ("w_x", "w_h", "b", "w_out", "b_out")
        missing = [key for key in required if key not in payload]
        if missing and strict:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

        def _load_array(name: str, current: np.ndarray) -> np.ndarray:
            if name not in payload:
                return current
            loaded = np.asarray(payload[name], dtype=float)
            if loaded.shape != current.shape:
                raise ValueError(
                    f"Checkpoint tensor shape mismatch for {name}: expected {current.shape}, got {loaded.shape}"
                )
            return loaded

        self.w_x = _load_array("w_x", self.w_x)
        self.w_h = _load_array("w_h", self.w_h)
        self.b = _load_array("b", self.b)
        self.w_out = _load_array("w_out", self.w_out)
        if "b_out" in payload:
            self.b_out = float(np.asarray(payload["b_out"]).item())

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, object]:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ConvLSTM checkpoint not found: {checkpoint_path}")

        if checkpoint_path.suffix != ".npz":
            raise ValueError(f"Unsupported checkpoint format for ConvLSTM checkpoint: {checkpoint_path.suffix}")

        try:
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                payload = {key: checkpoint[key] for key in checkpoint.files}
        except Exception as exc:
            raise ValueError(f"Failed to read ConvLSTM checkpoint: {checkpoint_path}") from exc

        self.load_state_dict(payload, strict=strict)
        version = str(payload.get("model_version", "unknown"))
        return {
            "path": str(checkpoint_path),
            "format": "npz",
            "strict": strict,
            "model_version": version,
        }
