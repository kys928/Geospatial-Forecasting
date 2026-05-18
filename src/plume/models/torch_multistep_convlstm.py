from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class ConvLSTMCellTorch(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            b, _, h, w = x_t.shape
            h_t = torch.zeros((b, self.hidden_channels, h, w), device=x_t.device, dtype=x_t.dtype)
            c_t = torch.zeros((b, self.hidden_channels, h, w), device=x_t.device, dtype=x_t.dtype)
        else:
            h_t, c_t = state
        gates = self.gates(torch.cat([x_t, h_t], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class TorchMultiStepConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int = 10,
        encoder_channels: int = 32,
        hidden_channels: int = 64,
        num_encoder_layers: int = 2,
        future_steps: int = 4,
    ):
        super().__init__()
        if num_encoder_layers != 2:
            raise ValueError("TorchMultiStepConvLSTM currently requires num_encoder_layers=2")
        self.input_channels = input_channels
        self.future_steps = future_steps
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(),
        )
        self.encoder_lstm_layers = nn.ModuleList(
            [
                ConvLSTMCellTorch(input_channels=encoder_channels, hidden_channels=hidden_channels),
                ConvLSTMCellTorch(input_channels=hidden_channels, hidden_channels=hidden_channels),
            ]
        )
        self.decoder_cell = ConvLSTMCellTorch(input_channels=1, hidden_channels=hidden_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, encoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(),
            nn.Conv2d(encoder_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if c <= 0:
            raise ValueError("Expected channel dimension > 0")

        layer_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.encoder_lstm_layers)
        for ti in range(t):
            encoded = self.encoder(x[:, ti, :, :, :])
            lstm_input = encoded
            for li, layer in enumerate(self.encoder_lstm_layers):
                layer_states[li] = layer(lstm_input, layer_states[li])
                lstm_input = layer_states[li][0]

        final_state = layer_states[-1]
        if final_state is None:
            zero = torch.zeros((b, 64, h, w), device=x.device, dtype=x.dtype)
            final_state = (zero, zero)

        decoder_input = x[:, -1, 0:1, :, :] if (t > 0 and c >= 1) else torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
        dec_state = final_state
        outputs: list[torch.Tensor] = []
        for _ in range(self.future_steps):
            dec_state = self.decoder_cell(decoder_input, dec_state)
            frame = self.decoder(dec_state[0])
            outputs.append(frame)
            decoder_input = frame
        return torch.stack(outputs, dim=1)


class TorchMultiStepConvLSTMCheckpoint:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu", checkpoint_strict: bool = True):
        self.checkpoint_path = str(Path(checkpoint_path))
        self.device = device
        self.checkpoint_strict = checkpoint_strict
        path = Path(self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"ConvLSTM checkpoint not found: {path}")
        raw = torch.load(path, map_location=device)
        if not isinstance(raw, dict):
            raise ValueError("Expected checkpoint payload to be a dict")
        model_state = raw.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint missing model_state_dict")
        cleaned_state = {k.removeprefix("module."): v for k, v in model_state.items()}

        config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
        contract = raw.get("model_contract") if isinstance(raw.get("model_contract"), dict) else {}

        future_steps = int(contract.get("future_steps") or config.get("future_steps") or 4)
        model = TorchMultiStepConvLSTM(
            input_channels=int(contract.get("input_channels") or config.get("input_channels") or 10),
            encoder_channels=int(contract.get("encoder_channels") or config.get("encoder_channels") or 32),
            hidden_channels=int(contract.get("hidden_channels") or config.get("hidden_channels") or 64),
            num_encoder_layers=int(contract.get("num_encoder_layers") or config.get("num_encoder_layers") or 2),
            future_steps=future_steps,
        )
        model.to(device)
        load_result = model.load_state_dict(cleaned_state, strict=checkpoint_strict)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        model.eval()

        self.model = model
        self.future_steps = future_steps
        self.metadata: dict[str, Any] = {
            "checkpoint_path": self.checkpoint_path,
            "device": device,
            "global_epoch": raw.get("global_epoch"),
            "stage_name": raw.get("stage_name"),
            "best_score": raw.get("best_score"),
            "metrics": raw.get("metrics"),
            "config": config,
            "model_contract": contract,
            "future_steps": future_steps,
            "load_missing_keys": missing,
            "load_unexpected_keys": unexpected,
        }

    def predict(self, sequence_np: np.ndarray) -> np.ndarray:
        arr = np.asarray(sequence_np, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"Expected input shape (T,C,H,W), got {arr.shape}")
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
        out = pred.squeeze(0).squeeze(1).detach().cpu().numpy().astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, a_min=0.0, a_max=None)
        return out
