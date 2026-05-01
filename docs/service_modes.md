# Service Modes

## Current deployment shape

This project currently runs as a **modular monolith with an optional worker process**:

- A control/API process (FastAPI) for request handling, job submission, and artifact/status serving.
- A worker process for one-shot queued job execution.

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
- Worker execution is one-shot by default. For repeated processing, run the worker repeatedly or use external process supervision later.
- Existing specific scripts (`scripts/run_forecast_worker.py` and `scripts/run_retraining_worker.py`) remain available.
