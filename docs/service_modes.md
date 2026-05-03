# Service Modes

## Current deployment shape

This project currently runs as a **modular monolith with an optional worker process**:

- A control/API process (FastAPI) for request handling, job submission, and artifact/status serving.
- A worker process for queued job execution (one-shot by default, optional local loop mode).

This keeps one codebase and one repository while making control-vs-execution boundaries explicit.

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
- Submit forecast and retraining jobs.
- Serve forecast artifacts and job status.
- Own OpenRemote registration/publishing behavior where currently implemented.

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

## Shared boundaries

Control and execution modes coordinate through shared local stores/artifacts:

- forecast job store
- forecast artifact store
- retraining job store
- model registry
- operational state
- event log
- optional CSV session store

## What this is not

- Not two separately deployed services yet.
- No broker introduced.
- No SQL/SQLite database introduced.
- No OpenRemote DB mirroring.

## Future path

- Run control API and execution worker as separate processes.
- Later containerize them separately.
- Later evaluate optional broker/remote inference client only if needed.


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
- No broker or SQL database is required for this shape.
- Shared state is coordinated through configured local artifact/job/state files.
- Worker execution is one-shot by default.
- Optional local supervision loop: add `--loop` (and optionally `--interval-seconds` / `--max-iterations`) to the unified worker command.
- Existing specific scripts (`scripts/run_forecast_worker.py` and `scripts/run_retraining_worker.py`) remain available.

## Ops recommendation explanation context

- The Ops recommendation API includes `GET /ops/retraining/recommendation/context` for LLM-ready structured context.
- Ops also exposes `GET /ops/models/candidate/context` for deterministic candidate-vs-active review context used by UI/LLM explanation layers; it does not approve/promote models automatically.
- This context is deterministic, derived from current operational recommendation state, and does not call an LLM.
- It is intended for future explanation layers and Training/Ops UX surfaces.
