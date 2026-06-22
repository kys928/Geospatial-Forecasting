# Service Modes

## Current deployment shape

This project currently runs as a **modular monolith with optional local worker process modes**:

- A control/API process (FastAPI) for request handling, job submission, artifact/status serving, forecast context, decision support, and ops routes.
- Worker process modes for queued forecast and retraining execution.

This keeps one codebase and one repository while making control-vs-execution boundaries explicit. It is not a fully split microservice deployment and does not require a broker.

## Control service mode

Run the control/API service:

```bash
python scripts/run_control_service.py
```

Equivalent direct command:

```bash
python -m uvicorn plume.api.main:app --host 0.0.0.0 --port 8000
```

Responsibilities:

- Serve batch forecast routes and persisted forecast artifacts.
- Submit forecast and retraining jobs.
- Serve job status and worker status snapshots.
- Serve session/runtime routes through the internal `ForecastRuntimeClient` seam.
- Serve forecast-context and decision-support routes.
- Own optional OpenRemote service-registration lifecycle where configured.

## Internal runtime seam

The control service uses `ForecastRuntimeClient` as the internal runtime boundary. The current in-process implementation is `LocalForecastRuntimeClient`, which delegates:

- batch forecast work to `ForecastService`, and
- online/session work to `OnlineForecastService`.

The current configured online backend is `convlstm_online`, with `gaussian_fallback` as fallback. `mock_online` remains available for development/test support.

## Execution worker mode

Run one-shot worker execution via the process-mode wrapper:

```bash
python scripts/run_execution_worker.py --kind forecast
python scripts/run_execution_worker.py --kind retraining
python scripts/run_execution_worker.py --kind all
```

Equivalent direct command:

```bash
python -m plume.workers.run --kind all
```

Notes:

- `--once` is accepted for compatibility, but one-shot behavior is already the default.
- `--kind all` runs forecast once, then retraining once.
- Forecast worker dependencies are composed in `plume.workers.deps` (service/runtime/storage), not via API route dependency wiring.
- Forecast stale-job recovery is optional and disabled by default. When enabled, stale `running` jobs are marked `failed` (not requeued) before claiming queued work.
- Configure stale-job recovery with `PLUME_FORECAST_JOB_STALE_RECOVERY_ENABLED` and `PLUME_FORECAST_JOB_STALE_AFTER_SECONDS`.
- Retraining stale-job recovery is optional and disabled by default. When enabled, stale `running` retraining jobs are marked `failed` (not requeued) before claiming queued work.
- Configure retraining stale-job recovery with `PLUME_RETRAINING_JOB_STALE_RECOVERY_ENABLED` and `PLUME_RETRAINING_JOB_STALE_AFTER_SECONDS` (default `7200`).

## Shared local boundaries

Control and execution modes coordinate through app-owned local stores/artifacts:

- forecast job store
- forecast artifact store
- retraining job store
- model registry
- operational state
- event log
- optional CSV session store
- optional app-owned SQLite-backed Ops stores where configured

No broker or external database is required for this proof-of-concept shape. Optional SQLite-backed Ops storage is app-owned local persistence and is not OpenRemote DB mirroring.

## Dataset playback and decision-support mode

Dataset playback is demo/context support exposed through forecast-context routes. It is not live sensor data, not OpenRemote data, and not active ConvLSTM inference.

Decision-support routes are backend service routes. They use `ForecastContextService` context plus deterministic logic and optional LLM interpretation. LLM explanation is grounded in supplied forecast context and is not autonomous forecasting.

## OpenRemote mode

OpenRemote support is optional and provisional:

- service registration and heartbeat lifecycle exist,
- optional HTTP publishing components exist,
- these paths are disabled by default unless configured,
- no live-validated OpenRemote schema/contract is claimed,
- no OpenRemote database mirroring is implemented.

## What this is not

- Not production atmospheric dispersion modeling.
- Not real online learning.
- Not two separately deployed services yet.
- No broker introduced.
- No external SQL database is required.
- No OpenRemote DB mirroring.
- Gaussian fallback is not active ConvLSTM serving.
- Dataset playback is not active ConvLSTM inference.

## Future path

- Continue to preserve the internal runtime seam.
- Run control API and execution worker as separate processes where useful.
- Later containerize process modes separately if needed.
- Later evaluate optional broker/remote inference client only if a concrete operational need appears.

## Local two-process run

Terminal 1 (control service):

```bash
python scripts/run_control_service.py
```

Terminal 2 (execution worker):

```bash
python scripts/run_execution_worker.py --kind all
```

Notes:

- This is still one repository with two local process modes, not a fully split microservice deployment.
- Shared state is coordinated through configured local artifact/job/state files.
- Worker execution is one-shot by default.
- Optional local supervision loop: add `--loop` (and optionally `--interval-seconds` / `--max-iterations`) to the unified worker command.
- Worker heartbeat/status is written to `artifacts/worker_status/worker_status.json` by default; override with `PLUME_WORKER_STATUS_PATH` or `--worker-status-path`.
- Control API exposes `GET /ops/workers/status` for the latest file-backed worker heartbeat/status snapshot.
- This status mechanism is local file-based visibility, not distributed service discovery.
- Existing specific scripts (`scripts/run_forecast_worker.py` and `scripts/run_retraining_worker.py`) remain available.

## Ops recommendation explanation context

- The Ops recommendation API includes `GET /ops/retraining/recommendation/context` for LLM-ready structured context.
- Ops also exposes `GET /ops/models/candidate/context` for deterministic candidate-vs-active review context used by UI/LLM explanation layers; it does not approve/promote models automatically.
- This context is deterministic, derived from current operational recommendation state, and does not call an LLM.
- It is intended for future explanation layers and Training/Ops UX surfaces.
