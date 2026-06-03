# Task 10B Discovery Notes

Repository inspected before Task 10B code changes. Current branch already contains partial Task 10A-style changes in the latest commit, but several required behaviors/tests/audit checks are incomplete.

## Verified config facts
- `configs/adaptation.yaml` has `training.retry_cooldown_seconds: 3600` and `training.min_seconds_between_training_runs: 3600`; cooldown is 1 hour.
- `configs/backend.yaml` has `default_backend: convlstm_online`, `convlstm_prediction_engine: torch_multistep`, explicit `convlstm_checkpoint_path`, `use_model_registry: false`, `model_registry_path: null`, and `convlstm_ridge_model_path: artifacts/models/ridge_plume_baseline.pkl`.

## Discovery checklist answers
1. Automatic retraining jobs are currently enqueued by `maybe_enqueue_automatic_adaptation_job(...)` in `src/plume/services/convlstm_operations.py`, called from `run_retraining_worker_once(...)` in `src/plume/workers/retraining_worker.py`. This was already partially present.
2. Manual retraining jobs are enqueued by `/ops/retraining/trigger` in `src/plume/api/routes/ops.py`, which calls `submit_retraining_job(...)`.
3. The worker claims retraining jobs in `RetrainingJobStore.claim_next_queued_job(...)`, called by `run_retraining_worker_once(...)`.
4. Adaptation readiness is evaluated in `_adaptation_readiness(...)` in `src/plume/api/routes/ops.py` and by `AdaptationReadinessService.evaluate(...)` in automatic enqueue/training paths.
5. Latest training status is assembled by `_adaptation_training_status(...)` in `src/plume/api/routes/ops.py`.
6. Live training logs are assembled in the frontend by `collectLogs(...)` in `frontend/src/features/ops/components/OpsTrainingTab.tsx`; current code partially prefers `latest_job.log_tail`.
7. Current code partially writes real trainer logs to `<output_dir>/training.log` in `run_adaptation_retraining_job(...)`, but needs stronger metadata/status tests and log fallback details.
8. `/ops/adaptation/training/status` currently exposes `latest_job.log_tail`, `log_file_path`, and `log_available`, but schema is loose and behavior is untested for 200-line bounds/missing logs/error traces.
9. Registry `active_model_id` is stored/read by `ModelRegistry` in `src/plume/services/convlstm_operations.py` and exposed via Ops registry routes.
10. ConvLSTM backend can resolve registry active checkpoint only if `use_model_registry=true` and `model_registry_path` is set; default `configs/backend.yaml` still uses static checkpoint mode, so Ops activation alone does not change serving.
11. Dataset playback overrides session/active forecast in `ForecastContextService.latest(source="auto")` when playback is enabled, and `ForecastPage.tsx` suppresses session overlays when playback is active.
12. Decision Support now partially includes provenance in compact context and prompt rules, but there are no tests for “who is doing predictions?” provenance answers.
13. `ridge_plume_baseline` appears in dataset playback `runtime.model_name`, `raw.model_inference.model_name`, overlay/raster metadata, and ConvLSTM ridge model config. These are debug/runtime details, not proof of active serving.
14. Active model activation changes registry state, but actual forecast-serving backend behavior changes only when runtime config enables registry resolution; default config does not.
15. Existing tests cover Ops API, worker, forecast context routes, ConvLSTM backend registry/static behavior, dataset scenarios, and decision support basics. Gaps: automatic enqueue edge cases, log tail/status runtime/checkpoint behavior, provenance truth tests, decision-support provenance answer tests, frontend modal/log facts, and required audit script.

## Incomplete/current issues found
- Auto enqueue diagnostics do not include the required `attempted` field and use inconsistent cooldown field names.
- Auto enqueue has no dedicated tests for all required active/cooldown/readiness/waiting scenarios.
- Training status does not include all required cooldown fields (`cooldown_remaining_seconds`, `next_automatic_training_eligible_at`, `cooldown_source`) or `trigger_source` consistently.
- ConvLSTM torch forecast metadata itself lacks standardized provenance fields; online service infers from session load metadata, but forecast metadata should also identify loaded checkpoint usage.
- Provenance lacks `active_registry_model_id` and `generated_at` fields.
- No `scripts/audit_ops_task_10b.py` exists.
- Frontend has no test script; build can be run, but UI-specific assertions need audit/static checks.
