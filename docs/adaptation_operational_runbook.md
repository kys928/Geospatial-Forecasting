# Adaptation Operational Runbook

## 1. Purpose

This runbook gives operators a repeatable path for verifying the automatic ConvLSTM adaptation loop without changing runtime behavior. The loop is:

```text
adaptation buffer → readiness → dataset manifest → training → candidate → promotion policy → Ops API/UI
```

At a high level:

- The **adaptation buffer** stores accepted local windows and raw-observation derived samples for later retraining.
- **Readiness** checks decide whether there are enough usable samples, compatible checkpoints, available storage, and no conflicting jobs.
- The **dataset manifest** mixes reference data with accepted buffer samples for training.
- The robust **three-stage trainer** can produce a candidate checkpoint.
- The worker or manual trainer can register a **candidate**.
- The **promotion policy** evaluates candidate metrics before activation.
- The **Ops API and frontend Ops UI** expose readiness, training status, registry candidates, evaluation, policy application, approval/rejection, storage warnings, and checkpoint-file cleanup controls.

Phase 8 adds only smoke-test tooling and this runbook. It does not add new training, serving, API, worker, or frontend behavior.

## 2. Safety rules

Use these rules when operating the adaptation loop:

- `GET /ops/adaptation/readiness` and `POST /ops/adaptation/check-now` evaluate readiness only. They do **not** start training.
- Candidate `evaluate` is non-mutating. It reads registry/candidate state and reports the promotion decision.
- `apply-policy` may mutate registry state according to the configured promotion policy. Do not use it as a read-only check.
- Manual `approve` can activate a candidate only after the compatibility check passes.
- `reject` changes candidate metadata/status but keeps checkpoint files.
- `delete-file` removes only the checkpoint file for the selected model record and keeps registry metadata/history.
- There is no automatic checkpoint deletion in this phase.
- Inference continues using the current active model while adaptation training runs.
- The smoke script defaults to dry-run checks and starts real training only with `--run-tiny-training`.

## 3. Required environment / files

Before running the smoke test or manually training, identify the following:

- Repository checkout with `configs/adaptation.yaml`.
- Adaptation buffer root, if local buffer verification is needed.
- Reference dataset directory containing canonical `.npz` windows, if reference data should be included.
- Active or latest robust ConvLSTM checkpoint for model-only resume, if continuation is needed.
- Enough free disk space for run output, candidate checkpoints, logs, manifests, and registry metadata.
- GPU/VRAM: use the environment-specific recommendation for the deployed ConvLSTM size. CPU is acceptable for dry-run checks, but real training should use a suitable GPU where available.

Example paths in this document are examples only. They are not hardcoded defaults.

## 4. Quick frontend workflow

1. Open the frontend **Ops Overview** page.
2. Check the adaptation readiness card:
   - Green means the system is ready for adaptation work.
   - Yellow/red should be read as a reason to investigate before launching training.
3. Use **Check now** to refresh readiness. This is safe and does not start training.
4. Open the **Training** tab.
5. Trigger or manually launch adaptation training only if readiness, resource availability, and operator intent are clear.
6. Open the **Registry** tab after a run.
7. Select a candidate and use **Evaluate** to get the promotion decision without mutating the registry.
8. Use **Apply policy**, **Approve**, or **Reject** only when the operator intends to mutate candidate state.
9. Review storage warnings before deleting checkpoint files. The delete-file action removes the checkpoint file only and preserves metadata/history.

## 5. CLI smoke-test workflow

Dry run, including local checks and optional API checks:

```bash
python scripts/smoke_adaptation_loop.py \
  --repo-root /workspace/Geospatial-Forecasting \
  --reference-dataset-dir /workspace/online_sets/online_learning_subset \
  --api-base-url http://localhost:8000 \
  --dry-run
```

Tiny training, explicit opt-in:

```bash
python scripts/smoke_adaptation_loop.py \
  --repo-root /workspace/Geospatial-Forecasting \
  --reference-dataset-dir /workspace/online_sets/online_learning_subset \
  --resume-checkpoint /path/to/robust_checkpoint.pt \
  --run-tiny-training \
  --start-stage stage3 \
  --max-epochs-stage3 1 \
  --device cuda
```

The smoke script prints a `PASS`/`WARN`/`FAIL`/`SKIP` report and returns:

- `0` when there are no failed checks.
- `1` when one or more checks fail.
- `2` for invalid CLI usage or an unexpected invocation-level error.

Use `--json-report /path/to/report.json` to write the structured report. If `--ops-token` is provided, the token value is redacted in the report.

## 6. Manual trainer CLI workflow

Use the existing manual trainer when you intentionally want to build a manifest or run robust adaptation training outside the worker:

```bash
python scripts/train_three_stage_adaptation.py \
  --reference-dataset-dir /path/to/reference_subset \
  --buffer-root /path/to/adaptation_buffer \
  --output-dir /path/to/run_output \
  --resume-checkpoint /path/to/robust_checkpoint.pt \
  --resume-mode model_only \
  --start-stage stage3 \
  --device cuda \
  --max-epochs-stage3 8
```

For a non-training preview, add `--dry-run`. The dry run writes `dataset_manifest_preview.json` and does not start training.

## 7. Expected success signs

A healthy end-to-end verification usually shows:

- Readiness is green, or yellow/red has a known and accepted operator reason.
- The dataset manifest has nonzero train and validation counts.
- The trainer dry-run writes a dataset manifest preview.
- Tiny training, when explicitly requested, writes `training_summary.json`.
- A worker-managed completed adaptation run registers a candidate.
- The Registry tab shows the candidate.
- Candidate evaluation returns a decision without mutating registry state.

## 8. Failure troubleshooting

Common symptoms and next checks:

- **Buffer missing**: confirm the buffer root and `PLUME_ADAPTATION_BUFFER_DIR`; missing local buffer is a warning for smoke checks but can block readiness.
- **Reference dataset missing**: pass the correct `--reference-dataset-dir` or configure the runtime reference dataset path.
- **Not enough fresh samples**: check accepted train/validation counts, minimum fresh sample settings, and whether reserve reuse is allowed.
- **CUDA unavailable / low VRAM**: run dry-run checks on CPU; for real training use `--device cpu` for very small tests or move to a GPU node with sufficient VRAM.
- **Checkpoint incompatible**: inspect model contract fields, input shape, output shape, and contract version; use a compatible robust checkpoint for model-only resume.
- **Trainer OOM**: reduce batch size in the manual trainer, use a smaller smoke dataset, or move to a larger GPU.
- **Candidate uncertain because active metrics are missing**: ensure the active model has comparable metrics before relying on automatic promotion decisions.
- **Candidate rejected due to t+3/t+4 regression**: review horizon-specific metrics; the policy intentionally guards later-step degradation.
- **Frontend auth error**: verify the Ops token, role, API base URL, and whether the backend requires read or operator access for the action.
- **Checkpoint file deleted but metadata remains**: this is expected for delete-file; registry/history still show the record, but the checkpoint file cannot be loaded or activated.

## 9. Rollback

Rollback uses the previous active model recorded in registry history. For robust adaptation models, rollback includes a compatibility check before activation. A deleted checkpoint file cannot be a rollback target because the file is no longer available, but metadata and history remain for auditability.

## 10. What Phase 8 does not do

Phase 8 does **not** add:

- New training logic.
- Frontend redesign.
- HYSPLIT integration.
- Automatic checkpoint deletion.
- LLM diagnostics.
