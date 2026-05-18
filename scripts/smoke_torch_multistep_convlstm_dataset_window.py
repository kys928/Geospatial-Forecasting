from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plume.models.torch_multistep_convlstm import TorchMultiStepConvLSTMCheckpoint


def _stats(name: str, arr: np.ndarray) -> str:
    return (
        f"{name}: shape={arr.shape} min={float(arr.min()):.6g} "
        f"max={float(arr.max()):.6g} mean={float(arr.mean()):.6g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Torch multi-step ConvLSTM on dataset windows")
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/models/convlstm_multistep_autoreg_two_stage_v1/best_full_checkpoint.pt",
    )
    parser.add_argument(
        "--windows-dir",
        default="/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026/windows",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    ckpt = TorchMultiStepConvLSTMCheckpoint(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        checkpoint_strict=False,
    )

    windows_dir = Path(args.windows_dir)
    files = sorted(windows_dir.glob("*.npz"))[: args.limit]
    if not files:
        print(f"WARNING: no .npz windows found in {windows_dir}")
        return 0

    for file in files:
        with np.load(file) as arr:
            x = np.asarray(arr["input"], dtype=np.float32)
            target = np.asarray(arr["target"], dtype=np.float32) if "target" in arr.files else None

        pred = ckpt.predict(x)
        print(f"file={file.name}")
        print(_stats("input", x))
        if target is not None:
            print(_stats("target", target))
        print(f"pred: shape={pred.shape} min={float(pred.min()):.6g} max={float(pred.max()):.6g} mean={float(pred.mean()):.6g} sum={float(pred.sum()):.6g}")

        if float(pred.max()) > 0.0:
            print("FOUND NONZERO PREDICTION")
            return 0

    print("WARNING: all checked predictions were zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
