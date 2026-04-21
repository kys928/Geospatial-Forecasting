from __future__ import annotations

import numpy as np


class MinimalConvLSTMModel:
    """Lightweight ConvLSTM-style model wrapper for online backend integration.

    This uses random/untrained weights for demo/runtime plumbing only.
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
