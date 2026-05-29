#!/usr/bin/env python3
"""Manual CLI/smoke runner for robust three-stage ConvLSTM adaptation training.

Example continuation command (paths are examples only, not defaults):

python scripts/train_three_stage_adaptation.py \
  --reference-dataset-dir /workspace/online_sets/online_learning_subset \
  --output-dir /workspace/Geospatial-Forecasting/runs/convlstm_multistep_three_stage_robust_v2_from_stage2 \
  --resume-checkpoint /workspace/Geospatial-Forecasting/runs/convlstm_multistep_three_stage_robust_v1/stage_transition_after_stage2_autoregressive_teacher_forcing_full_checkpoint.pt \
  --resume-mode model_only \
  --start-stage stage3 \
  --device cuda \
  --max-epochs-stage3 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.training.adaptation_dataset import build_adaptation_dataset_manifest
from plume.training.three_stage_adaptation_trainer import (
    ThreeStageTrainerConfig,
    train_three_stage_adaptation,
)


_DEFAULT_RUN_NAME = "convlstm_multistep_three_stage_robust_manual"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an adaptation dataset manifest and optionally run the standalone robust three-stage trainer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example continuation command (example paths only):\n"
            "  python scripts/train_three_stage_adaptation.py \\\n"
            "    --reference-dataset-dir /workspace/online_sets/online_learning_subset \\\n"
            "    --output-dir /workspace/Geospatial-Forecasting/runs/convlstm_multistep_three_stage_robust_v2_from_stage2 \\\n"
            "    --resume-checkpoint /workspace/Geospatial-Forecasting/runs/convlstm_multistep_three_stage_robust_v1/"
            "stage_transition_after_stage2_autoregressive_teacher_forcing_full_checkpoint.pt \\\n"
            "    --resume-mode model_only \\\n"
            "    --start-stage stage3 \\\n"
            "    --device cuda \\\n"
            "    --max-epochs-stage3 8"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Run output directory")
    parser.add_argument("--reference-dataset-dir", type=Path, default=None, help="Optional reference dataset directory")
    parser.add_argument("--buffer-root", type=Path, default=None, help="Optional AdaptationBuffer root directory")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Optional robust checkpoint for model-only resume")
    parser.add_argument("--resume-mode", choices=("none", "model_only"), default="none")
    parser.add_argument("--start-stage", choices=("stage1", "stage2", "stage3"), default="stage1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-name", default=_DEFAULT_RUN_NAME)
    parser.add_argument("--initial-batch-size", type=int, default=16)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-epochs-stage1", type=int, default=None, help="Optional stage 1 max epoch override")
    parser.add_argument("--max-epochs-stage2", type=int, default=None, help="Optional stage 2 max epoch override")
    parser.add_argument("--max-epochs-stage3", type=int, default=None, help="Optional stage 3 max epoch override")
    parser.add_argument(
        "--stage3-only-gentle",
        action="store_true",
        help="Use stage 3 as the effective start stage unless --start-stage is explicitly provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and write dataset_manifest_preview.json without starting training.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.stage3_only_gentle and "--start-stage" not in raw_argv:
        args.start_stage = "stage3"
    return args


def _build_trainer_config(args: argparse.Namespace) -> ThreeStageTrainerConfig:
    config = ThreeStageTrainerConfig(
        run_name=args.run_name,
        initial_batch_size=args.initial_batch_size,
        min_batch_size=args.min_batch_size,
    )
    if args.max_epochs_stage1 is not None:
        config.stage1.max_epochs = args.max_epochs_stage1
    if args.max_epochs_stage2 is not None:
        config.stage2.max_epochs = args.max_epochs_stage2
    if args.max_epochs_stage3 is not None:
        config.stage3.max_epochs = args.max_epochs_stage3
    return config


def _write_manifest(path: Path, manifest_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_dict, indent=2, sort_keys=True), encoding="utf-8")


def _print_manifest_summary(manifest_counts: dict[str, int], output_dir: Path, device: str) -> None:
    print(f"Selected device: {device}")
    print(f"Train samples: {manifest_counts.get('train_total', 0)}")
    print(f"Validation samples: {manifest_counts.get('val_total', 0)}")
    print(f"Output dir: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reference_dataset_dir is None and args.buffer_root is None:
        print("ERROR: provide at least one data source: --reference-dataset-dir or --buffer-root", file=sys.stderr)
        return 2
    if args.resume_checkpoint is not None and args.resume_mode != "model_only":
        print("ERROR: --resume-checkpoint requires --resume-mode model_only", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adaptation_buffer = None
    if args.buffer_root is not None:
        adaptation_buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=args.buffer_root))

    manifest = build_adaptation_dataset_manifest(
        reference_dataset_dir=args.reference_dataset_dir,
        adaptation_buffer=adaptation_buffer,
    )
    manifest_dict = manifest.to_dict()
    manifest_path = output_dir / ("dataset_manifest_preview.json" if args.dry_run else "dataset_manifest.json")
    _write_manifest(manifest_path, manifest_dict)

    _print_manifest_summary(manifest.counts, output_dir, args.device)
    if args.resume_checkpoint is not None:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
    if manifest.warnings:
        print("Manifest warnings:")
        for warning in manifest.warnings:
            print(f"- {warning}")

    has_train_and_val = manifest.counts.get("train_total", 0) > 0 and manifest.counts.get("val_total", 0) > 0
    if args.dry_run:
        if not has_train_and_val:
            print("DRY RUN: no train/validation sample pair is available; training would fail.", file=sys.stderr)
        print(f"Dry run complete. Wrote manifest preview: {manifest_path}")
        return 0

    if not has_train_and_val:
        print("ERROR: no train/validation sample pair found; need train_total > 0 and val_total > 0", file=sys.stderr)
        return 2

    config = _build_trainer_config(args)
    print("Starting three-stage adaptation trainer...")
    try:
        summary = train_three_stage_adaptation(
            train_samples=manifest.train_samples,
            val_samples=manifest.val_samples,
            output_dir=output_dir,
            config=config,
            resume_checkpoint_path=args.resume_checkpoint,
            resume_mode=args.resume_mode,
            start_stage=args.start_stage,
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: three-stage adaptation trainer failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"Final status: {summary.status}")
    print(f"Best overall checkpoint: {summary.best_overall_checkpoint}")
    print(f"Final checkpoint: {summary.final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
