from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util

if importlib.util.find_spec("torch") is None:
    torch = None

    class _NN:
        class Module:
            pass

    nn = _NN()
else:
    import torch
    from torch import nn
    import torch.nn.functional as F


ROBUST_MODEL_NAME = "RobustMultiStepConvLSTMForecaster"
DEFAULT_INPUT_SHAPE = [3, 10, 64, 64]
DEFAULT_OUTPUT_SHAPE = [4, 1, 64, 64]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
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
        i, f, o, g = torch.chunk(self.gates(torch.cat([x_t, h_t], dim=1)), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class RobustMultiStepConvLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_channels: int = 10,
        input_frames: int = 3,
        future_steps: int = 4,
        encoder_channels: int = 32,
        hidden_channels: int = 64,
        decoder_channels: int = 32,
        kernel_size: int = 3,
        groupnorm_groups: int = 4,
        residual_rollout: bool = True,
        detach_feedback: bool = True,
        output_activation: str | None = "softplus",
        num_encoder_lstm_layers: int = 2,
    ):
        super().__init__()
        if num_encoder_lstm_layers != 2:
            raise ValueError("RobustMultiStepConvLSTMForecaster currently requires num_encoder_lstm_layers=2")
        self.input_channels = input_channels
        self.input_frames = input_frames
        self.future_steps = future_steps
        self.encoder_channels = encoder_channels
        self.hidden_channels = hidden_channels
        self.decoder_channels = decoder_channels
        self.kernel_size = kernel_size
        self.groupnorm_groups = groupnorm_groups
        self.residual_rollout = residual_rollout
        self.detach_feedback = detach_feedback
        self.output_activation = output_activation

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groupnorm_groups, encoder_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder_lstm_layers = nn.ModuleList(
            [
                ConvLSTMCell(encoder_channels, hidden_channels, kernel_size=kernel_size),
                ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=kernel_size),
            ]
        )
        self.direct_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_channels, decoder_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(groupnorm_groups, decoder_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(decoder_channels, 1, kernel_size=1),
                )
                for _ in range(future_steps)
            ]
        )
        self.rollout_cell = ConvLSTMCell(1, hidden_channels, kernel_size=kernel_size)
        self.rollout_delta_head = nn.Sequential(
            nn.Conv2d(hidden_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groupnorm_groups, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.output_activation.lower() if isinstance(self.output_activation, str) else self.output_activation
        if activation == "softplus":
            return F.softplus(x)
        if activation == "relu":
            return torch.relu(x)
        if activation in {None, "none", "linear"}:
            return x
        raise ValueError(f"Unsupported output_activation: {self.output_activation}")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.input_frames:
            raise ValueError(f"Expected {self.input_frames} input frames, got {t}")
        if c != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {c}")

        layer_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.encoder_lstm_layers)
        for ti in range(t):
            lstm_input = self.encoder(x[:, ti, :, :, :])
            for li, layer in enumerate(self.encoder_lstm_layers):
                layer_states[li] = layer(lstm_input, layer_states[li])
                lstm_input = layer_states[li][0]

        final_state = layer_states[-1]
        if final_state is None:
            zero = torch.zeros((b, self.hidden_channels, h, w), device=x.device, dtype=x.dtype)
            return zero, zero
        return final_state

    def direct(self, x: torch.Tensor) -> torch.Tensor:
        h_t, _ = self.encode(x)
        outputs = [self._apply_output_activation(head(h_t)) for head in self.direct_heads]
        return torch.stack(outputs, dim=1)

    def predict_direct(self, x: torch.Tensor) -> torch.Tensor:
        return self.direct(x)

    def rollout(self, x: torch.Tensor) -> torch.Tensor:
        state = self.encode(x)
        feedback = x[:, -1, 0:1, :, :]
        outputs: list[torch.Tensor] = []
        for _ in range(self.future_steps):
            cell_input = feedback.detach() if self.detach_feedback else feedback
            state = self.rollout_cell(cell_input, state)
            delta = self.rollout_delta_head(state[0])
            frame = feedback + delta if self.residual_rollout else delta
            frame = self._apply_output_activation(frame)
            outputs.append(frame)
            feedback = frame
        return torch.stack(outputs, dim=1)

    def predict_rollout(self, x: torch.Tensor) -> torch.Tensor:
        return self.rollout(x)

    def forward(self, x: torch.Tensor, mode: str = "rollout") -> torch.Tensor:
        if mode == "rollout":
            return self.rollout(x)
        if mode == "direct":
            return self.direct(x)
        raise ValueError(f"Unsupported robust ConvLSTM forward mode: {mode}")


class RobustMultiStepConvLSTMCheckpoint:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        if torch is None:
            raise ModuleNotFoundError("torch is required for RobustMultiStepConvLSTMCheckpoint")
        self.checkpoint_path = str(Path(checkpoint_path))
        self.device = device
        path = Path(self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Robust ConvLSTM checkpoint not found: {path}")

        try:
            raw = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            raw = torch.load(path, map_location=device)
        if not isinstance(raw, dict):
            raise ValueError("Expected robust ConvLSTM checkpoint payload to be a dict")
        model_state = raw.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError("Robust ConvLSTM checkpoint missing model_state_dict")
        cleaned_state = {str(k).removeprefix("module."): v for k, v in model_state.items()}

        config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
        model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
        data_config = config.get("data") if isinstance(config.get("data"), dict) else {}
        contract = raw.get("model_contract") if isinstance(raw.get("model_contract"), dict) else {}

        init_kwargs = self._model_kwargs(model_config, data_config, config, contract)
        self._validate_contract(
            contract=contract,
            init_kwargs=init_kwargs,
            model_config=model_config,
            data_config=data_config,
            config=config,
        )
        model = RobustMultiStepConvLSTMForecaster(**init_kwargs)
        model.to(device)
        try:
            model.load_state_dict(cleaned_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Robust ConvLSTM state_dict mismatch: missing/unexpected keys or shape mismatch: {exc}"
            ) from exc
        model.eval()

        self.model = model
        self.metadata: dict[str, Any] = {
            "checkpoint_path": self.checkpoint_path,
            "device": device,
            "stage_name": raw.get("stage_name"),
            "global_epoch": raw.get("global_epoch"),
            "model_contract": contract,
            "metrics": raw.get("metrics"),
            "config": config,
            **init_kwargs,
        }

    @staticmethod
    def _coalesce_int(*values: object, default: int) -> int:
        for value in values:
            if value is not None:
                return int(value)
        return default

    @staticmethod
    def _coalesce_bool(*values: object, default: bool) -> bool:
        for value in values:
            if value is not None:
                return bool(value)
        return default

    @classmethod
    def _model_kwargs(
        cls,
        model_config: dict[str, Any],
        data_config: dict[str, Any],
        config: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "input_channels": cls._coalesce_int(
                model_config.get("input_channels"),
                data_config.get("input_channels"),
                config.get("input_channels"),
                default=10,
            ),
            "input_frames": cls._coalesce_int(
                model_config.get("input_frames"),
                data_config.get("input_frames"),
                config.get("input_frames"),
                default=3,
            ),
            "future_steps": cls._coalesce_int(
                model_config.get("future_steps"),
                data_config.get("future_steps"),
                config.get("future_steps"),
                default=4,
            ),
            "encoder_channels": cls._coalesce_int(model_config.get("encoder_channels"), default=32),
            "hidden_channels": cls._coalesce_int(model_config.get("hidden_channels"), default=64),
            "decoder_channels": cls._coalesce_int(model_config.get("decoder_channels"), default=32),
            "kernel_size": cls._coalesce_int(model_config.get("kernel_size"), default=3),
            "groupnorm_groups": cls._coalesce_int(model_config.get("groupnorm_groups"), default=4),
            "residual_rollout": cls._coalesce_bool(
                model_config.get("residual_rollout"),
                contract.get("residual_rollout"),
                default=True,
            ),
            "detach_feedback": cls._coalesce_bool(model_config.get("detach_feedback"), default=True),
            "output_activation": model_config.get("output_activation", "softplus"),
            "num_encoder_lstm_layers": cls._coalesce_int(model_config.get("num_encoder_lstm_layers"), default=2),
        }

    @staticmethod
    def _configured_spatial_shape(
        model_config: dict[str, Any],
        data_config: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[int, int]:
        height = (
            model_config.get("grid_height")
            or model_config.get("height")
            or data_config.get("grid_height")
            or config.get("grid_height")
            or 64
        )
        width = (
            model_config.get("grid_width")
            or model_config.get("width")
            or data_config.get("grid_width")
            or config.get("grid_width")
            or 64
        )
        return int(height), int(width)

    def _validate_contract(
        self,
        contract: dict[str, Any],
        init_kwargs: dict[str, Any],
        model_config: dict[str, Any],
        data_config: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        if not contract:
            return
        model_name = contract.get("model_name")
        if model_name != ROBUST_MODEL_NAME:
            raise ValueError(
                f"Robust ConvLSTM checkpoint model_contract.model_name must be {ROBUST_MODEL_NAME}, "
                f"got {model_name!r}"
            )
        height, width = self._configured_spatial_shape(model_config, data_config, config)
        expected_input = [int(init_kwargs["input_frames"]), int(init_kwargs["input_channels"]), height, width]
        expected_output = [int(init_kwargs["future_steps"]), 1, height, width]
        input_shape = contract.get("input_shape")
        if input_shape is not None and list(input_shape) != expected_input:
            raise ValueError(f"Robust ConvLSTM model_contract.input_shape must be {expected_input}, got {input_shape}")
        output_shape = contract.get("output_shape")
        if output_shape is not None and list(output_shape) != expected_output:
            raise ValueError(f"Robust ConvLSTM model_contract.output_shape must be {expected_output}, got {output_shape}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Robust ConvLSTM prediction expects a torch.Tensor with shape (B, T, C, H, W)")
        with torch.no_grad():
            return self.model(x.to(self.device))
