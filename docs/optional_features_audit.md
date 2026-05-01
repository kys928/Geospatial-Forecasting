# Optional Features Audit

This audit summarizes optional, opt-in, provisional, external-dependent, and test-only features in the current proof-of-concept.

| Feature | Default state | How to enable | Requires external dependency? | Persistence / side effects | Production readiness note |
|---|---|---|---|---|---|
| OpenRemote service registration | Disabled | `PLUME_OPENREMOTE_SERVICE_REGISTRATION_ENABLED=true` plus manager URL/token env | Yes — OpenRemote Manager API + token (`write:services`) | Registers service, heartbeat, deregisters on shutdown | Provisional integration path; not validated against live OpenRemote in this repo |
| OpenRemote forecast attribute publishing | Disabled (`enabled: false`, `sink_mode: disabled`) | Set `PLUME_OPENREMOTE_ENABLED=true`, `PLUME_OPENREMOTE_SINK_MODE=http`, base URL, asset ID, token env var | Yes — OpenRemote HTTP API + asset + token | Writes attribute payloads to configured OpenRemote asset; local forecast still persists first | Provisional generic payload translator; not a validated schema contract |
| CSV session persistence | Optional (default in-memory store) | Set state store mode to CSV (`PLUME_STATE_STORE=csv`) and store dir | No external system; local filesystem only | Persists session/state CSV files under configured store dir | Suitable for local durability only; CSV concurrency limits remain |
| Persisted batch explanation artifacts | Disabled | `PLUME_PERSIST_BATCH_EXPLANATION=true` | No external system for non-LLM mode | Writes `explanation.json` under forecast artifact directory | Explicitly optional; endpoint returns limitation when missing artifact |
| LLM-backed explanations | Disabled by default through explanation persistence flags/config | Enable persisted explanation and LLM usage flags/config | Yes when using hosted model/provider credentials (e.g., HF) | Adds generated explanation content to persisted artifact when enabled | Best-effort path with fallback; not positioned as hardened production NLP pipeline |
| Async forecast jobs | Optional (sync `POST /forecast` remains core) | Use `POST /forecast/jobs` + run forecast worker | No external broker; local file stores | Job status/artifacts persisted in local JSON/filesystem stores | Local process-mode decoupling only; no broker or distributed queue |
| Forecast worker loop mode | One-shot default | `python -m plume.workers.run --kind <...> --loop [--interval-seconds N] [--max-iterations N]` | No external system | Repeatedly claims/processes queued jobs and writes standard artifacts | Local supervision loop only; not daemon/orchestrator framework |
| Retraining worker execution | Optional/manual execution path | Run `--kind retraining` or retraining worker script | No external system by default | Updates retraining job store, model registry JSON, operational state, event log | Process-mode worker boundary; no long-running training service |
| Ops auth | Optional/hardening toggle | Configure backend ops auth settings and provide frontend `VITE_OPS_API_TOKEN` when required | Depends on token distribution model in deployment | Gatekeeps ops route access; may affect reads/writes in ops workspace | Intended as lightweight guardrail, not enterprise IAM |
| ConvLSTM online backend / registry model usage | ConvLSTM backend default; registry usage disabled (`use_model_registry: false`) | Enable registry mode and provide model registry path/checkpoints | Optional external model assets/checkpoints | Impacts online prediction backend selection/loading path | Proof-of-concept backend with fallback; production readiness depends on real checkpoint/registry ops evidence |
| Gaussian fallback/batch baseline | Enabled as fallback/default-safe baseline | Built-in default/fallback config | No external system | Deterministic forecast artifacts persisted like normal forecasts | Core baseline behavior; still PoC model scope |
| Test-only doubles/fakes | Test-only | N/A (kept under tests) | No | No runtime side effects in production code paths | Fake OpenRemote sink/test doubles are intentionally non-runtime |
| Frontend Ops workspace token behavior | Token may be required depending on backend ops auth config | Set `VITE_OPS_API_TOKEN` for frontend when ops auth requires it | Depends on backend auth mode | Frontend includes token for ops API requests where required | Optional integration detail; frontend behavior unchanged unless auth enforced |
| Hugging Face/model preload | Disabled | `--preload-models` or `PLUME_PRELOAD_HF_MODELS=true` + HF env vars | Yes — network/model repo access (and optional auth) | Preloads model artifacts locally before runtime | Optional startup optimization path; environment-dependent |

## Notes

- Core/default runtime remains local process-mode architecture with file/CSV/JSON boundaries.
- No broker, SQL/SQLite, or OpenRemote DB mirroring is required/implemented.
- OpenRemote integrations are optional and disabled by default.
