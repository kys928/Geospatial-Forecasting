# Service Modes

## Current deployment shape

This project currently runs as a **modular monolith with an optional worker process**:

- A control/API process (FastAPI) for request handling, job submission, and artifact/status serving.
- A worker process for one-shot queued job execution.

This keeps one codebase and one repository while making control-vs-execution boundaries explicit.

## Control service mode

Run the control/API service:

```bash
uvicorn plume.api.main:app --host 0.0.0.0 --port 8000
```

Responsibilities:
- Submit forecast and retraining jobs.
- Serve forecast artifacts and job status.
- Own OpenRemote registration/publishing behavior where currently implemented.

## Execution worker mode

Run one-shot worker execution via the unified runner:

```bash
python -m plume.workers.run --kind forecast
python -m plume.workers.run --kind retraining
python -m plume.workers.run --kind all
```

Notes:
- `--once` is accepted for compatibility, but one-shot behavior is already the default.
- `--kind all` runs forecast once, then retraining once.

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
