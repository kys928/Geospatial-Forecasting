import { useEffect, useMemo, useRef, useState } from "react";
import { opsClient } from "../api/opsClient";
import { useOpsJobs } from "../hooks/useOpsJobs";
import { useOpsStatus } from "../hooks/useOpsStatus";
import { useRetrainingRecommendation } from "../hooks/useRetrainingRecommendation";
import { useModelCandidateContext } from "../hooks/useModelCandidateContext";
import type {
  AdaptationTrainingStatus,
  OpsJobRecord,
  RetrainingTriggerRequest,
} from "../types/ops.types";

type PresetKey = "fast_refresh" | "balanced" | "high_accuracy" | "recovery";
type LearningRateMode = "conservative" | "default" | "aggressive";
type RuntimeOption = "15m" | "30m" | "1h" | "2h";
type ChecklistState = "met" | "not_met" | "unknown" | "checking";

type ManualControlState = {
  maxEpochs: number;
  maxRuntime: RuntimeOption;
  batchSize: "8" | "16" | "32" | "64";
  learningRate: LearningRateMode;
  earlyStoppingPatience: number;
  validationSplit: "10" | "15" | "20";
};
const presetDescriptions: Record<PresetKey, string> = {
  fast_refresh: "Quick lightweight update for small new data batches.",
  balanced: "Default training profile with a practical speed/quality tradeoff.",
  high_accuracy: "Longer run focused on better validation quality.",
  recovery:
    "Conservative run after failures, drift, or unstable model behavior.",
};
const presetDefaults: Record<PresetKey, ManualControlState> = {
  fast_refresh: {
    maxEpochs: 3,
    maxRuntime: "15m",
    batchSize: "16",
    learningRate: "conservative",
    earlyStoppingPatience: 3,
    validationSplit: "10",
  },
  balanced: {
    maxEpochs: 8,
    maxRuntime: "30m",
    batchSize: "32",
    learningRate: "default",
    earlyStoppingPatience: 4,
    validationSplit: "15",
  },
  high_accuracy: {
    maxEpochs: 20,
    maxRuntime: "2h",
    batchSize: "32",
    learningRate: "conservative",
    earlyStoppingPatience: 6,
    validationSplit: "20",
  },
  recovery: {
    maxEpochs: 5,
    maxRuntime: "1h",
    batchSize: "16",
    learningRate: "conservative",
    earlyStoppingPatience: 5,
    validationSplit: "15",
  },
};

const asObj = (v: unknown) =>
  v && typeof v === "object" ? (v as Record<string, unknown>) : {};
const asStr = (v: unknown) => (typeof v === "string" && v.trim() ? v : null);
const pick = (obj: Record<string, unknown>, keys: string[]) =>
  keys.map((k) => obj[k]).find((v) => v !== undefined && v !== null);

export function OpsTrainingTab() {
  const jobsState = useOpsJobs();
  const statusState = useOpsStatus();
  const recommendationState = useRetrainingRecommendation();
  const candidateState = useModelCandidateContext();
  const [manualOpen, setManualOpen] = useState(false);
  const [manualSubmitting, setManualSubmitting] = useState(false);
  const [manualNotice, setManualNotice] = useState<string | null>(null);
  const [followLogs, setFollowLogs] = useState(true);
  const [adaptationTraining, setAdaptationTraining] =
    useState<AdaptationTrainingStatus | null>(null);
  const [adaptationTrainingError, setAdaptationTrainingError] = useState<
    string | null
  >(null);
  const logRef = useRef<HTMLPreElement | null>(null);
  const jobs = jobsState.jobs?.jobs ?? [];
  const runningJobs = jobs.filter(
    (j) => j.status === "queued" || j.status === "running",
  );
  const recommendation = recommendationState.recommendation;
  const checklist = useMemo(
    () =>
      buildChecklist({
        recommendation,
        runningJobs,
        status: statusState.status,
        adaptationTraining,
      }),
    [recommendation, runningJobs, statusState.status, adaptationTraining],
  );

  const latestJob = useMemo(
    () =>
      (adaptationTraining?.latest_job as OpsJobRecord | null) ??
      statusState.status?.latest_retraining_job ??
      jobsState.jobs?.latest_job ??
      jobs[0] ??
      null,
    [
      adaptationTraining?.latest_job,
      jobs,
      jobsState.jobs?.latest_job,
      statusState.status?.latest_retraining_job,
    ],
  );
  const trainingView = useMemo(
    () => deriveTrainingView(statusState.status, latestJob, adaptationTraining),
    [statusState.status, latestJob, adaptationTraining],
  );
  const summaryText = useMemo(
    () => buildSummaryText(trainingView.state, checklist),
    [trainingView.state, checklist],
  );
  const logs = useMemo(
    () => collectLogs(statusState.status, latestJob),
    [statusState.status, latestJob],
  );
  const hasErrorLogs = logs.some((line) => line.startsWith("ERROR:"));

  useEffect(() => {
    if (followLogs && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs, followLogs]);
  useEffect(() => {
    void refreshAdaptationTraining();
    const timer = window.setInterval(() => {
      void refreshAll();
    }, 10000);
    return () => window.clearInterval(timer);
  }, []);

  async function refreshAdaptationTraining() {
    try {
      setAdaptationTraining(await opsClient.getAdaptationTrainingStatus());
      setAdaptationTrainingError(null);
    } catch (err) {
      setAdaptationTrainingError(
        err instanceof Error
          ? err.message
          : "Unable to load adaptation training status.",
      );
    }
  }
  async function refreshAll() {
    await Promise.all([
      jobsState.refresh(),
      statusState.refresh(),
      recommendationState.refresh(),
      candidateState.refresh(),
      refreshAdaptationTraining(),
    ]);
  }
  async function handleManualStart(payload: RetrainingTriggerRequest) {
    setManualSubmitting(true);
    setManualNotice(null);
    try {
      const r = await opsClient.triggerRetraining(payload);
      setManualNotice(
        r.submitted
          ? "Manual training job submitted."
          : "Submission was not accepted by backend policy.",
      );
      setManualOpen(false);
      await refreshAll();
    } catch (e) {
      setManualNotice(
        e instanceof Error
          ? e.message
          : "Unable to submit manual training job.",
      );
    } finally {
      setManualSubmitting(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <section className="panel">
        <h3 style={{ margin: 0 }}>Training Status</h3>
        {adaptationTrainingError ? (
          <p className="failure-text">
            Unable to load adaptation training status: {adaptationTrainingError}
          </p>
        ) : null}
        <div className="ops-training-status-panel">
          <div>
            <div className="ops-training-state">{trainingView.state}</div>
            <p style={{ margin: "8px 0 0" }}>{summaryText}</p>
          </div>
          <dl className="ops-training-facts">
            {trainingView.rows.map((row) => (
              <div key={row.label}>
                <dt>{row.label}</dt>
                <dd title={row.value}>{row.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>
      {trainingView.metrics.length > 0 ? (
        <section className="panel">
          <h3>Training Metrics</h3>
          <>
            <div className="ops-status-grid">
              {trainingView.metrics.map((row) => (
                <article key={row.label} className="ops-stat-card">
                  <p className="muted" style={{ margin: 0 }}>
                    {row.label}
                  </p>
                  <strong>{row.value}</strong>
                </article>
              ))}
            </div>
            {trainingView.progressPct !== null ? (
              <div style={{ marginTop: 10 }}>
                <div className="ops-progress">
                  <div style={{ width: `${trainingView.progressPct}%` }} />
                </div>
              </div>
            ) : null}
          </>
        </section>
      ) : null}
      <section className="panel">
        <div
          className="button-row"
          style={{ justifyContent: "space-between", alignItems: "center" }}
        >
          <h3 style={{ margin: 0 }}>Live Training Logs</h3>
          <div className="button-row">
            <label
              className="muted"
              style={{ display: "flex", gap: 6, alignItems: "center" }}
              title="Keep the log view pinned to the newest line."
            >
              <input
                type="checkbox"
                checked={followLogs}
                onChange={(e) => setFollowLogs(e.target.checked)}
              />
              Auto-scroll
            </label>
            <button
              className="secondary-button"
              onClick={() =>
                void navigator.clipboard?.writeText(logs.join("\n"))
              }
              disabled={!logs.length}
            >
              Copy logs
            </button>
          </div>
        </div>
        {!logs.length ? (
          <p className="muted" style={{ marginBottom: 0 }}>
            No training logs reported yet.
          </p>
        ) : (
          <pre
            ref={logRef}
            className={`ops-log-window${hasErrorLogs ? " ops-log-window-error" : ""}`}
          >
            {logs.join("\n")}
          </pre>
        )}
      </section>
      <section className="panel">
        <details className="advanced-section">
          <summary>Automatic Training Readiness</summary>
          {checklist.map((item) => (
            <ReadinessItem
              key={item.label}
              label={item.label}
              state={item.state}
              detail={item.detail}
            />
          ))}
        </details>
      </section>
      <section className="panel">
        <h3>Manual Training</h3>
        <p className="muted">
          Manual training is an advanced override. Automatic training is the
          normal workflow.
        </p>
        <button
          className="secondary-button"
          onClick={() => setManualOpen(true)}
        >
          Start manual training
        </button>
        {manualNotice ? <p className="muted">{manualNotice}</p> : null}
      </section>
      {manualOpen ? (
        <ManualTrainingModal
          onClose={() => setManualOpen(false)}
          onSubmit={handleManualStart}
          submitting={manualSubmitting}
        />
      ) : null}
    </div>
  );
}

function ReadinessItem({
  label,
  state,
  detail,
}: {
  label: string;
  state: ChecklistState;
  detail: string;
}) {
  return (
    <div className={`ops-readiness-item ops-readiness-${state}`}>
      <span className="ops-readiness-dot" />
      <div>
        <strong>{label}</strong>
        <p className="muted" style={{ margin: 0 }}>
          {detail}
        </p>
      </div>
    </div>
  );
}

function deriveTrainingView(
  status: any,
  latestJob: OpsJobRecord | null,
  adaptationTraining: AdaptationTrainingStatus | null,
) {
  const jobObj = asObj(latestJob);
  const runCfg = asObj(
    (() => {
      try {
        return JSON.parse(asStr(latestJob?.run_config_ref) ?? "{}");
      } catch {
        return {};
      }
    })(),
  );
  const metrics = asObj(
    pick(jobObj, ["metrics", "training_metrics", "progress_metrics"]),
  );
  const stateRaw =
    asStr(pick(jobObj, ["status"])) ??
    (Array.isArray(status?.retraining_job_statuses)
      ? asStr(status.retraining_job_statuses[0])
      : null);
  const mapState: Record<string, string> = {
    queued: "Queued",
    running: "Running",
    failed: "Failed",
    completed: "Completed",
    succeeded: "Completed",
    idle: "Idle",
  };
  const state = mapState[(stateRaw ?? "").toLowerCase()] ?? "Not reported";
  const started = asStr(pick(jobObj, ["started_at", "start_time"]));
  const completed = asStr(pick(jobObj, ["completed_at", "end_time"]));
  const elapsed = started
    ? `${Math.max(0, Math.floor(((completed ? Date.parse(completed) : Date.now()) - Date.parse(started)) / 1000))}s`
    : "Not reported";
  const progress = pick(metrics, [
    "progress",
    "progress_pct",
    "percent_complete",
  ]);
  const progressPct =
    typeof progress === "number"
      ? Math.max(0, Math.min(100, progress > 1 ? progress : progress * 100))
      : null;
  const latestStatus = asStr(pick(jobObj, ["status"])) ?? "Not reported";
  const rows = [
    { label: "Current state", value: state },
    {
      label: "Latest job",
      value:
        asStr(latestJob?.job_id) ??
        (adaptationTraining?.latest_job
          ? "Recorded without job id"
          : "No adaptation training job recorded"),
    },
    { label: "Latest status", value: latestStatus },
    {
      label: "Candidate",
      value:
        adaptationTraining?.candidate_model_id ??
        asStr(pick(jobObj, ["candidate_model_id"])) ??
        "Not reported",
    },
    {
      label: "Output dir",
      value:
        adaptationTraining?.output_dir ??
        asStr(pick(jobObj, ["output_dir"])) ??
        "Not reported",
    },
    {
      label: "Result run dir",
      value:
        adaptationTraining?.result_run_dir ??
        asStr(pick(jobObj, ["result_run_dir"])) ??
        "Not reported",
    },
    {
      label: "Best checkpoint",
      value: adaptationTraining?.best_overall_checkpoint ?? "Not reported",
    },
    {
      label: "Final checkpoint",
      value: adaptationTraining?.final_checkpoint ?? "Not reported",
    },
    { label: "Started", value: started ?? "Not reported" },
    { label: "Elapsed", value: elapsed },
  ];
  const jobCounts = adaptationTraining?.job_counts ?? {};
  const readinessSnapshot =
    adaptationTraining?.latest_readiness_snapshot ?? null;
  const readinessBlocking = Array.isArray(readinessSnapshot?.blocking_reasons)
    ? readinessSnapshot.blocking_reasons.join("; ")
    : null;
  const metricRows = [
    ["Queued jobs", jobCounts.queued],
    ["Running jobs", jobCounts.running],
    ["Completed jobs", jobCounts.completed ?? jobCounts.succeeded],
    ["Failed jobs", jobCounts.failed],
    [
      "Readiness",
      typeof readinessSnapshot?.status === "string"
        ? readinessSnapshot.status
        : null,
    ],
    ["Readiness blockers", readinessBlocking],
    ["Progress", progressPct !== null ? `${progressPct.toFixed(1)}%` : null],
    ["Current epoch", pick(metrics, ["current_epoch", "epoch"])],
    ["Total epochs", pick(metrics, ["total_epochs", "max_epochs"])],
    ["Training loss", pick(metrics, ["training_loss", "train_loss"])],
    ["Validation loss", pick(metrics, ["validation_loss", "val_loss"])],
    [
      "Best validation loss",
      pick(metrics, ["best_validation_loss", "best_val_loss"]),
    ],
    ["Steps completed", pick(metrics, ["steps_completed", "global_step"])],
    ["ETA", pick(metrics, ["eta", "eta_seconds"])],
    ["Elapsed time", pick(metrics, ["elapsed", "elapsed_time"])],
    [
      "Candidate model produced",
      pick(jobObj, ["candidate_model_id", "candidate_model_produced"]),
    ],
  ]
    .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== "")
    .map(([label, value]) => ({ label: String(label), value: String(value) }));
  return {
    state,
    rows,
    metrics: metricRows,
    progressPct,
    hasActiveJob: state === "Running" || state === "Queued",
  };
}

function collectLogs(status: any, latestJob: OpsJobRecord | null): string[] {
  const jobObj = asObj(latestJob);
  const lines: string[] = [];
  const logs = pick(jobObj, ["logs", "log_lines", "events"]);
  if (Array.isArray(logs))
    logs.forEach((l) =>
      lines.push(typeof l === "string" ? l : JSON.stringify(l)),
    );
  if (typeof logs === "string") lines.push(logs);
  const failure =
    asStr(pick(jobObj, ["failure_reason", "error_message"])) ??
    asStr(status?.last_retraining_job_failure_reason);
  if (failure) lines.push(`ERROR: ${failure}`);
  return lines;
}
function buildSummaryText(
  state: string,
  checklist: Array<{ label: string; state: ChecklistState }>,
) {
  const notReady =
    checklist.find((c) => c.label === "Automatic training enabled")?.state ===
    "not_met";
  if (state === "Running")
    return "Adaptation training is running. Metrics will update as the worker reports progress.";
  if (state === "Queued")
    return "An adaptation training job is queued and waiting for a worker.";
  if (state === "Failed")
    return "The latest adaptation training job failed. Review logs and job details before starting another run.";
  if (state === "Completed")
    return "The latest adaptation training job completed. Review the candidate model in Model Versions.";
  if (state === "Not reported")
    return "No adaptation training job has been recorded yet.";
  if (notReady)
    return "Automatic adaptation training is not ready yet. Open the readiness checklist for blocking conditions.";
  return "Automatic adaptation training is waiting for readiness conditions or new work.";
}

function ManualTrainingModal({
  onClose,
  onSubmit,
  submitting,
}: {
  onClose: () => void;
  submitting: boolean;
  onSubmit: (payload: RetrainingTriggerRequest) => Promise<void>;
}) {
  const [datasetMode, setDatasetMode] = useState("buffered");
  const [datasetValue, setDatasetValue] = useState("");
  const [checkpointMode, setCheckpointMode] = useState("latest");
  const [checkpointValue, setCheckpointValue] = useState("");
  const [preset, setPreset] = useState<PresetKey>("balanced");
  const [controls, setControls] = useState<ManualControlState>(
    presetDefaults.balanced,
  );
  useEffect(() => {
    setControls(presetDefaults[preset]);
  }, [preset]);
  const runConfigRef = JSON.stringify({
    preset,
    max_epochs: controls.maxEpochs,
    max_runtime: controls.maxRuntime,
    checkpoint_mode: checkpointMode,
    checkpoint_ref: checkpointValue || undefined,
    training_controls: {
      batch_size: Number(controls.batchSize),
      learning_rate_mode: controls.learningRate,
      early_stopping_patience: controls.earlyStoppingPatience,
      validation_split_pct: Number(controls.validationSplit),
    },
  });
  const datasetSnapshotRef =
    datasetMode === "custom" ? datasetValue : "buffered_internal_dataset";
  return (
    <div className="ops-modal-backdrop">
      <section
        className="panel ops-manual-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Start Manual Training"
      >
        <header>
          <h3>Start Manual Training</h3>
        </header>
        <div className="ops-modal-grid">
          <div className="ops-modal-section">
            <h4>Dataset</h4>
            <label className="field">
              <span>Dataset source</span>
              <select
                value={datasetMode}
                onChange={(e) => setDatasetMode(e.target.value)}
              >
                <option value="buffered">
                  Use buffered internal dataset default
                </option>
                <option value="custom">Custom dataset path/link</option>
              </select>
            </label>
            {datasetMode === "custom" ? (
              <label className="field">
                <span>Custom dataset path/link</span>
                <input
                  value={datasetValue}
                  onChange={(e) => setDatasetValue(e.target.value)}
                />
              </label>
            ) : null}
          </div>
          <div className="ops-modal-section">
            <h4>Base checkpoint</h4>
            <label className="field">
              <span>Base checkpoint</span>
              <select
                value={checkpointMode}
                onChange={(e) => setCheckpointMode(e.target.value)}
              >
                <option value="latest">Latest best model default</option>
                <option value="active">Active production model</option>
                <option value="custom">Custom checkpoint path</option>
              </select>
            </label>
            {checkpointMode === "custom" ? (
              <label className="field">
                <span>Custom checkpoint path</span>
                <input
                  value={checkpointValue}
                  onChange={(e) => setCheckpointValue(e.target.value)}
                />
              </label>
            ) : null}
          </div>
          <div className="ops-modal-section">
            <h4>Training preset</h4>
            <label className="field">
              <span>Training preset</span>
              <select
                value={preset}
                onChange={(e) => setPreset(e.target.value as PresetKey)}
              >
                <option value="fast_refresh">Fast refresh</option>
                <option value="balanced">Balanced</option>
                <option value="high_accuracy">High accuracy</option>
                <option value="recovery">Recovery</option>
              </select>
              <small className="muted">{presetDescriptions[preset]}</small>
            </label>
          </div>
          <div className="ops-modal-section">
            <h4>Training controls</h4>
            <div className="ops-control-grid">
              <label className="field">
                <span>Max runtime</span>
                <select
                  value={controls.maxRuntime}
                  onChange={(e) =>
                    setControls((p) => ({
                      ...p,
                      maxRuntime: e.target.value as RuntimeOption,
                    }))
                  }
                >
                  <option value="15m">15m</option>
                  <option value="30m">30m</option>
                  <option value="1h">1h</option>
                  <option value="2h">2h</option>
                </select>
              </label>
              <label className="field">
                <span>Batch size</span>
                <select
                  value={controls.batchSize}
                  onChange={(e) =>
                    setControls((p) => ({
                      ...p,
                      batchSize: e.target
                        .value as ManualControlState["batchSize"],
                    }))
                  }
                >
                  <option value="8">8</option>
                  <option value="16">16</option>
                  <option value="32">32</option>
                  <option value="64">64</option>
                </select>
              </label>
              <label className="field">
                <span>Learning rate mode</span>
                <select
                  value={controls.learningRate}
                  onChange={(e) =>
                    setControls((p) => ({
                      ...p,
                      learningRate: e.target.value as LearningRateMode,
                    }))
                  }
                >
                  <option value="conservative">Conservative</option>
                  <option value="default">Default</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </label>
              <label className="field">
                <span>Validation split</span>
                <select
                  value={controls.validationSplit}
                  onChange={(e) =>
                    setControls((p) => ({
                      ...p,
                      validationSplit: e.target
                        .value as ManualControlState["validationSplit"],
                    }))
                  }
                >
                  <option value="10">10%</option>
                  <option value="15">15%</option>
                  <option value="20">20%</option>
                </select>
              </label>
              <label className="field">
                <span>
                  Early stopping patience: {controls.earlyStoppingPatience}
                </span>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={controls.earlyStoppingPatience}
                  onChange={(e) =>
                    setControls((p) => ({
                      ...p,
                      earlyStoppingPatience: Number(e.target.value),
                    }))
                  }
                />
              </label>
              <label className="field" style={{ gridColumn: "1 / -1" }}>
                <span>Max epochs: {controls.maxEpochs}</span>
                <input
                  type="range"
                  min={1}
                  max={50}
                  value={controls.maxEpochs}
                  onChange={(e) =>
                    setControls((p) => ({
                      ...p,
                      maxEpochs: Number(e.target.value),
                    }))
                  }
                />
              </label>
            </div>
          </div>
        </div>
        <div className="button-row">
          <button
            className="secondary-button"
            onClick={onClose}
            disabled={submitting}
          >
            Cancel
          </button>
          <button
            className="primary-button"
            onClick={() =>
              void onSubmit({
                manual_override: true,
                dataset_snapshot_ref: datasetSnapshotRef,
                run_config_ref: runConfigRef,
              })
            }
            disabled={submitting}
          >
            {submitting ? "Starting training..." : "Start training"}
          </button>
        </div>
      </section>
    </div>
  );
}

function buildChecklist({
  recommendation,
  runningJobs,
  status,
  adaptationTraining,
}: {
  recommendation: any;
  runningJobs: OpsJobRecord[];
  status: any;
  adaptationTraining: AdaptationTrainingStatus | null;
}): Array<{ label: string; state: ChecklistState; detail: string }> {
  const evidence = recommendation?.evidence ?? {};
  const legacyReadiness = status?.retraining_readiness ?? {};
  const readinessSnapshot =
    adaptationTraining?.latest_readiness_snapshot ?? null;
  const readiness = asObj(readinessSnapshot) ?? {};
  const workerAvailable =
    typeof legacyReadiness.worker_available === "boolean"
      ? legacyReadiness.worker_available
      : null;
  const snapshotChecks = Array.isArray(readiness.checks)
    ? readiness.checks
    : [];
  const blockingReasons = Array.isArray(readiness.blocking_reasons)
    ? readiness.blocking_reasons.map(String).filter(Boolean)
    : [];
  const rows: Array<{ label: string; state: ChecklistState; detail: string }> =
    [];

  rows.push({
    label: "Adaptation readiness snapshot",
    state: readinessState(readiness),
    detail:
      asStr(readiness.status) ??
      (readinessSnapshot
        ? "Snapshot reported by backend"
        : "No readiness snapshot reported yet"),
  });

  if (blockingReasons.length) {
    rows.push({
      label: "Blocking reasons",
      state: "not_met",
      detail: blockingReasons.slice(0, 4).join("; "),
    });
  }

  for (const [index, rawCheck] of snapshotChecks.slice(0, 8).entries()) {
    const check = asObj(rawCheck);
    rows.push({
      label:
        asStr(check.label) ??
        asStr(check.name) ??
        `Readiness check ${index + 1}`,
      state: readinessState(check),
      detail:
        asStr(check.message) ??
        asStr(check.reason) ??
        asStr(check.status) ??
        "Reported by backend readiness snapshot",
    });
  }

  rows.push(
    {
      label: "Enough new validated data",
      state: typeof evidence.new_samples === "number" ? "met" : "unknown",
      detail:
        typeof evidence.new_samples === "number"
          ? `${evidence.new_samples} samples reported`
          : "Not reported",
    },
    {
      label: "No retraining job currently running",
      state: runningJobs.length ? "not_met" : "met",
      detail: runningJobs.length
        ? `${runningJobs.length} job(s) queued or running`
        : "No queued/running jobs",
    },
    {
      label: "Worker available",
      state:
        workerAvailable === null
          ? "unknown"
          : workerAvailable
            ? "met"
            : "not_met",
      detail:
        workerAvailable === null
          ? "Not reported"
          : workerAvailable
            ? "Available"
            : "Not available",
    },
    {
      label: "Compute resources available",
      state:
        typeof legacyReadiness.resource_pressure === "boolean"
          ? !legacyReadiness.resource_pressure
            ? "met"
            : "not_met"
          : "unknown",
      detail:
        typeof legacyReadiness.resource_pressure === "boolean"
          ? legacyReadiness.resource_pressure
            ? "Resource pressure reported"
            : "No resource pressure reported"
          : "Not reported",
    },
    {
      label: "Dataset source available",
      state: recommendation?.reason ? "met" : "unknown",
      detail: recommendation?.reason
        ? "Derived from recommendation payload"
        : "Not reported",
    },
    {
      label: "Base checkpoint available",
      state:
        typeof legacyReadiness.base_checkpoint_available === "boolean"
          ? legacyReadiness.base_checkpoint_available
            ? "met"
            : "not_met"
          : "unknown",
      detail:
        typeof legacyReadiness.base_checkpoint_available === "boolean"
          ? legacyReadiness.base_checkpoint_available
            ? "Available"
            : "Not available"
          : "Not reported",
    },
    {
      label: "Retraining is recommended",
      state:
        recommendation?.should_retrain === true
          ? "met"
          : recommendation?.should_retrain === false
            ? "not_met"
            : "unknown",
      detail: recommendation?.reason ?? "Not reported",
    },
    {
      label: "Automatic training enabled",
      state:
        typeof legacyReadiness.retraining_enabled === "boolean"
          ? legacyReadiness.retraining_enabled
            ? "met"
            : "not_met"
          : "unknown",
      detail:
        typeof legacyReadiness.retraining_enabled === "boolean"
          ? legacyReadiness.retraining_enabled
            ? "Enabled"
            : "Disabled"
          : "Not reported",
    },
  );

  return rows;
}

function readinessState(check: Record<string, unknown>): ChecklistState {
  const status = String(check.status ?? "").toLowerCase();
  if (
    check.ready === true ||
    check.passed === true ||
    status === "green" ||
    status === "ready" ||
    status === "passed"
  )
    return "met";
  if (
    check.blocking === true ||
    check.ready === false ||
    check.passed === false ||
    status === "red" ||
    status === "blocked" ||
    status === "failed"
  )
    return "not_met";
  if (status === "checking" || status === "running") return "checking";
  return "unknown";
}
