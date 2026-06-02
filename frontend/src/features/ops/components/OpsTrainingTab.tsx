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
type ChecklistRow = {
  label: string;
  state: ChecklistState;
  detail: string;
  warning?: string;
};

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

const TRAINING_METRIC_KEYS = [
  "training_loss",
  "train_loss",
  "validation_loss",
  "val_loss",
  "best_validation_loss",
  "best_val_loss",
  "epoch",
  "progress",
  "weighted_mse",
  "mae",
  "plume_iou",
  "mass_abs_error",
  "peak_location_error",
  "selection_score",
  "val_rollout_weighted_mse",
  "val_rollout_weighted_mse_t3",
  "val_rollout_weighted_mse_t4",
];

const asObj = (v: unknown) =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
const asStr = (v: unknown) => (typeof v === "string" && v.trim() ? v : null);
const pick = (obj: Record<string, unknown>, keys: string[]) =>
  keys.map((k) => obj[k]).find((v) => v !== undefined && v !== null);
const latestBySequence = (jobs: OpsJobRecord[]): OpsJobRecord | null =>
  jobs.length
    ? [...jobs].sort(
        (a, b) =>
          Number(b.created_sequence ?? -1) - Number(a.created_sequence ?? -1),
      )[0]
    : null;

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
  const checklist = useMemo(
    () =>
      buildChecklist({
        runningJobs,
        status: statusState.status,
        adaptationTraining,
      }),
    [runningJobs, statusState.status, adaptationTraining],
  );
  const latestJob = useMemo(() => {
    const retrainingJobs = jobs.filter((job) => asStr(job.job_id));
    const active = latestBySequence(
      retrainingJobs.filter((job) =>
        ["running", "queued", "waiting", "starting"].includes(
          String(job.status ?? "").toLowerCase(),
        ),
      ),
    );
    return (
      active ??
      latestBySequence(retrainingJobs) ??
      statusState.status?.latest_retraining_job ??
      (adaptationTraining?.latest_manual_job as OpsJobRecord | null) ??
      (adaptationTraining?.latest_job as OpsJobRecord | null) ??
      null
    );
  }, [
    adaptationTraining?.latest_job,
    adaptationTraining?.latest_manual_job,
    jobs,
    statusState.status?.latest_retraining_job,
  ]);
  const trainingView = useMemo(
    () => deriveTrainingView(statusState.status, latestJob, adaptationTraining),
    [statusState.status, latestJob, adaptationTraining],
  );
  const summaryText = useMemo(
    () => buildSummaryText(trainingView.state, checklist),
    [trainingView.state, checklist],
  );
  const logs = useMemo(
    () => collectLogs(statusState.status, latestJob, adaptationTraining),
    [statusState.status, latestJob, adaptationTraining],
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
              warning={item.warning}
            />
          ))}
        </details>
      </section>
      <section className="panel">
        <details className="advanced-section">
          <summary>Manual Training</summary>
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
        </details>
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

function ReadinessItem({ label, state, detail, warning }: ChecklistRow) {
  return (
    <div className={`ops-readiness-item ops-readiness-${state}`}>
      <span className="ops-readiness-dot" />
      <div>
        <strong>{label}</strong>
        <p className="muted" style={{ margin: 0 }}>
          {detail}
        </p>
        {warning ? (
          <p className="muted" style={{ margin: "6px 0 0" }}>
            {warning}
          </p>
        ) : null}
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
  const metricSources = collectTrainingMetricSources(jobObj);
  const metrics = metricSources[0] ?? {};
  const stateRaw =
    asStr(pick(jobObj, ["status"])) ??
    (Array.isArray(status?.retraining_job_statuses)
      ? asStr(status.retraining_job_statuses[0])
      : null);
  const mapState: Record<string, string> = {
    queued: "Queued",
    running: "Running",
    waiting: "Queued",
    failed: "Failed",
    completed: "Completed",
    succeeded: "Completed",
    idle: "Idle",
  };
  const state = mapState[(stateRaw ?? "").toLowerCase()] ?? "Not reported";
  const started = asStr(pick(jobObj, ["started_at", "start_time"]));
  const completed = asStr(pick(jobObj, ["completed_at", "end_time"]));
  const elapsed = started
    ? `${Math.max(
        0,
        Math.floor(
          ((completed ? Date.parse(completed) : Date.now()) -
            Date.parse(started)) /
            1000,
        ),
      )}s`
    : "Not reported";
  const progress = pick(metrics, ["progress"]);
  const progressPct =
    typeof progress === "number"
      ? Math.max(0, Math.min(100, progress > 1 ? progress : progress * 100))
      : null;
  const latestStatus = asStr(pick(jobObj, ["status"])) ?? "Not reported";
  const jobCounts = adaptationTraining?.job_counts ?? {};
  const jobCountSummary = formatJobCountSummary(jobCounts);
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
    { label: "Job counts", value: jobCountSummary },
  ];
  const metricRows = collectTrainingMetricRows(metricSources, progressPct);
  return {
    state,
    rows,
    metrics: metricRows,
    progressPct,
    hasActiveJob: state === "Running" || state === "Queued",
  };
}

function collectTrainingMetricSources(
  jobObj: Record<string, unknown>,
): Record<string, unknown>[] {
  return [
    pick(jobObj, ["metrics", "training_metrics", "progress_metrics"]),
    pick(jobObj, ["training_summary"]),
    pick(asObj(pick(jobObj, ["training_summary"])), ["metrics"]),
    pick(jobObj, ["best_metrics"]),
    jobObj,
  ]
    .map(asObj)
    .filter((source) => Object.keys(source).length > 0);
}

function collectTrainingMetricRows(
  sources: Record<string, unknown>[],
  progressPct: number | null,
): Array<{ label: string; value: string }> {
  const seen = new Set<string>();
  const rows: Array<{ label: string; value: string }> = [];
  for (const key of TRAINING_METRIC_KEYS) {
    for (const source of sources) {
      const value = source[key];
      if (value === null || value === undefined || String(value).trim() === "")
        continue;
      if (seen.has(key)) break;
      rows.push({
        label: formatMetricLabel(key),
        value: key === "progress" ? formatProgressValue(value) : String(value),
      });
      seen.add(key);
      break;
    }
  }
  if (progressPct !== null && !seen.has("progress")) {
    rows.push({ label: "Progress", value: `${progressPct.toFixed(1)}%` });
  }
  return rows;
}

function formatMetricLabel(key: string): string {
  return key
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatProgressValue(value: unknown): string {
  if (typeof value !== "number") return String(value);
  const pct = value > 1 ? value : value * 100;
  return `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
}

function formatJobCountSummary(jobCounts: Record<string, number>): string {
  const entries = [
    ["queued", jobCounts.queued],
    ["running", jobCounts.running],
    ["completed", jobCounts.completed ?? jobCounts.succeeded],
    ["failed", jobCounts.failed],
  ].filter(([, value]) => typeof value === "number");
  return entries.length
    ? entries.map(([label, value]) => `${label}: ${value}`).join("; ")
    : "Not reported";
}

function collectLogs(
  status: any,
  latestJob: OpsJobRecord | null,
  adaptationTraining: AdaptationTrainingStatus | null,
): string[] {
  const jobObj = asObj(latestJob);
  const lines: string[] = [];
  const logs = pick(jobObj, ["logs", "log_lines", "events"]);
  if (Array.isArray(logs))
    logs.forEach((l) =>
      lines.push(typeof l === "string" ? l : JSON.stringify(l)),
    );
  if (typeof logs === "string") lines.push(logs);
  const statusValue = String(jobObj.status ?? "").toLowerCase();
  const metadata = asObj(jobObj.metadata);
  const isManual =
    metadata.manual_trigger === true ||
    (adaptationTraining?.latest_manual_job as OpsJobRecord | null)?.job_id ===
      latestJob?.job_id;
  if (!lines.length && isManual && ["queued", "waiting"].includes(statusValue)) {
    lines.push("Manual training job submitted; waiting for worker pickup.");
    if (!jobObj.started_at && !jobObj.worker_pid)
      lines.push("Worker has not claimed this job yet.");
  }
  const failure =
    asStr(pick(jobObj, ["failure_reason", "error_message"])) ??
    (latestJob ? null : asStr(status?.last_retraining_job_failure_reason));
  if (failure)
    lines.push(isManual ? `Manual training job failed: ${failure}` : `ERROR: ${failure}`);
  return lines;
}
function buildSummaryText(
  state: string,
  checklist: Array<{ label: string; state: ChecklistState }>,
) {
  const notReady = checklist.some((c) => c.state === "not_met");
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
    return "Automatic adaptation training is not ready yet. Review the readiness rows before starting another run.";
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
  runningJobs,
  status,
  adaptationTraining,
}: {
  runningJobs: OpsJobRecord[];
  status: any;
  adaptationTraining: AdaptationTrainingStatus | null;
}): ChecklistRow[] {
  const legacyReadiness = status?.retraining_readiness ?? {};
  const readinessSnapshot =
    adaptationTraining?.latest_readiness_snapshot ?? null;
  const readiness = asObj(readinessSnapshot);
  const checks = Array.isArray(readiness.checks) ? readiness.checks : [];
  const adaptationEnabledCheck = findReadinessCheck(checks, [
    "adaptation_enabled",
    "adaptation enabled",
  ]);
  const bufferCheck = findReadinessCheck(checks, [
    "buffer_exists",
    "buffer exists",
  ]);
  const freshSamplesCheck = findReadinessCheck(checks, [
    "enough_fresh_samples",
    "fresh samples",
  ]);
  const reservePolicyCheck = findReadinessCheck(checks, [
    "reserve_policy",
    "reserve policy",
  ]);
  const fallbackCheck = findReadinessCheck(checks, [
    "fallback_training_dataset_available",
  ]);
  const checkpointCheck = findReadinessCheck(checks, [
    "checkpoint_available",
    "checkpoint available",
  ]);
  const noTrainingJobCheck = findReadinessCheck(checks, [
    "no_training_job_running",
    "no training job",
  ]);
  const gpuCheck = findReadinessCheck(checks, [
    "gpu_memory_ready",
    "gpu memory",
  ]);
  const retryCooldownCheck = findReadinessCheck(checks, [
    "retry_cooldown",
    "retry cooldown",
  ]);
  const storageWarningCheck = findReadinessCheck(checks, [
    "checkpoint_storage_warning",
    "storage warning",
  ]);

  const adaptationDisabled =
    adaptationEnabledCheck &&
    readinessState(adaptationEnabledCheck) === "not_met";
  const overallState = adaptationDisabled
    ? "not_met"
    : readinessState(readiness);
  const storageWarning = storageWarningCheck
    ? storageWarningText(storageWarningCheck)
    : null;

  return [
    {
      label: "Training readiness",
      state: overallState,
      detail: trainingReadinessDetail(overallState),
    },
    buildNewTrainingDataRow(freshSamplesCheck, fallbackCheck),
    buildBufferDataRow(bufferCheck, fallbackCheck),
    buildBaseModelRow(checkpointCheck, legacyReadiness),
    buildTrainingJobStateRow(
      runningJobs,
      noTrainingJobCheck,
      adaptationTraining,
    ),
    {
      label: "GPU / resources",
      ...buildGpuResourcesState(gpuCheck, retryCooldownCheck),
      warning: storageWarning ?? undefined,
    },
  ];
}

function trainingReadinessDetail(state: ChecklistState): string {
  if (state === "met") return "Training can start when scheduled.";
  if (state === "checking") return "Waiting for collected samples or cooldown.";
  if (state === "not_met") return "Training cannot start yet.";
  return "Training readiness is not reported.";
}

function buildNewTrainingDataRow(
  freshSamplesCheck: Record<string, unknown> | null,
  fallbackCheck: Record<string, unknown> | null,
): ChecklistRow {
  const freshState = freshSamplesCheck ? readinessState(freshSamplesCheck) : "unknown";
  const fallbackDetails = asObj(fallbackCheck?.details);
  const fallbackAvailable = Boolean(fallbackDetails.selected_dataset_path);
  if (freshState === "met") {
    return {
      label: "New training data",
      state: "met",
      detail: "Enough new training data is available.",
    };
  }
  if (fallbackAvailable && (freshState === "checking" || freshState === "not_met" || freshState === "unknown")) {
    return {
      label: "New training data",
      state: "checking",
      detail: "Waiting for enough collected training data.",
    };
  }
  if (freshState === "not_met") {
    return {
      label: "New training data",
      state: "not_met",
      detail: "No usable training data source is available.",
    };
  }
  if (freshState === "checking") {
    return {
      label: "New training data",
      state: "checking",
      detail: "Waiting for enough collected training data.",
    };
  }
  return {
    label: "New training data",
    state: "unknown",
    detail: "New training data status is not reported.",
  };
}

function buildBufferDataRow(
  bufferCheck: Record<string, unknown> | null,
  fallbackCheck: Record<string, unknown> | null,
): ChecklistRow {
  const bufferState = bufferCheck ? readinessState(bufferCheck) : "unknown";
  const fallbackDetails = asObj(fallbackCheck?.details);
  const fallbackAvailable = Boolean(fallbackDetails.selected_dataset_path);
  if (bufferState === "met") {
    return { label: "Buffer Data", state: "met", detail: "Buffer data is ready." };
  }
  if (fallbackAvailable && (bufferState === "checking" || bufferState === "not_met" || bufferState === "unknown")) {
    return {
      label: "Buffer Data",
      state: "checking",
      detail: "Waiting for collected samples.",
    };
  }
  if (bufferState === "not_met") {
    return {
      label: "Buffer Data",
      state: "not_met",
      detail: "No usable training data source is available.",
    };
  }
  if (bufferState === "checking") {
    return { label: "Buffer Data", state: "checking", detail: "Waiting for collected samples." };
  }
  return {
    label: "Buffer Data",
    state: "unknown",
    detail: "Buffer data status is not reported.",
  };
}

function buildBaseModelRow(
  checkpointCheck: Record<string, unknown> | null,
  legacyReadiness: Record<string, unknown>,
): ChecklistRow {
  const legacyAvailable = legacyReadiness.base_checkpoint_available;
  const state = checkpointCheck
    ? readinessState(checkpointCheck)
    : typeof legacyAvailable === "boolean"
      ? legacyAvailable
        ? "met"
        : "not_met"
      : "unknown";
  return {
    label: "Base model",
    state,
    detail:
      state === "met"
        ? "Base model is available."
        : state === "not_met"
          ? "Base model is not available."
          : "Base model status is not reported.",
  };
}

function buildTrainingJobStateRow(
  runningJobs: OpsJobRecord[],
  noTrainingJobCheck: Record<string, unknown> | null,
  adaptationTraining: AdaptationTrainingStatus | null,
): ChecklistRow {
  const latestJob = asObj(adaptationTraining?.latest_job ?? null);
  const latestStatus = String(latestJob.status ?? "").toLowerCase();
  const counts = adaptationTraining?.job_counts ?? {};
  const queuedCount = typeof counts.queued === "number" ? counts.queued : 0;
  const runningCount = typeof counts.running === "number" ? counts.running : 0;
  const hasRunning =
    runningCount > 0 || runningJobs.some((job) => job.status === "running");
  const hasQueued =
    queuedCount > 0 || runningJobs.some((job) => job.status === "queued");

  if (hasRunning || latestStatus === "running")
    return {
      label: "Training job",
      state: "checking",
      detail: "Running",
    };
  if (hasQueued || latestStatus === "queued" || latestStatus === "waiting")
    return {
      label: "Training job",
      state: "checking",
      detail: "Manual training job is waiting for worker pickup.",
    };
  if (latestStatus === "failed")
    return {
      label: "Training job",
      state: "not_met",
      detail: "Latest job failed",
    };
  if (noTrainingJobCheck) {
    const state = readinessState(noTrainingJobCheck);
    return {
      label: "Training job",
      state,
      detail:
        state === "met"
          ? "No queued/running jobs"
          : "Training job state is not reported.",
    };
  }
  if (runningJobs.length === 0)
    return {
      label: "Training job",
      state: "met",
      detail: "No queued/running jobs",
    };
  return {
    label: "Training job",
    state: "unknown",
    detail: "Training job state is not reported.",
  };
}

function buildGpuResourcesState(
  gpuCheck: Record<string, unknown> | null,
  retryCooldownCheck: Record<string, unknown> | null,
): Pick<ChecklistRow, "state" | "detail"> {
  const gpuState = gpuCheck ? readinessState(gpuCheck) : "unknown";
  const retryState = retryCooldownCheck
    ? readinessState(retryCooldownCheck)
    : "unknown";
  if (gpuState === "not_met")
    return { state: "not_met", detail: "GPU memory is not ready." };
  if (retryState === "not_met" || retryState === "checking")
    return {
      state: retryState === "checking" ? "checking" : "not_met",
      detail: "Waiting before retrying GPU training.",
    };
  if (gpuState === "met" && retryState === "met")
    return { state: "met", detail: "Ready" };
  if (gpuState === "met" && !retryCooldownCheck)
    return { state: "met", detail: "Ready" };
  return { state: "unknown", detail: "GPU/resource status is not reported." };
}

function storageWarningText(check: Record<string, unknown>): string | null {
  const state = readinessState(check);
  if (state === "met") return null;
  const detail = readinessDetail(check);
  return detail === "Reported by backend readiness snapshot"
    ? "Storage warning reported."
    : "Storage warning reported.";
}

function findReadinessCheck(
  checks: unknown[],
  terms: string[],
): Record<string, unknown> | null {
  for (const rawCheck of checks) {
    const check = asObj(rawCheck);
    const haystack = `${asStr(check.name) ?? ""} ${
      asStr(check.label) ?? ""
    } ${asStr(check.key) ?? ""} ${asStr(check.id) ?? ""}`.toLowerCase();
    if (terms.some((term) => haystack.includes(term))) return check;
  }
  return null;
}

function readinessDetail(check: Record<string, unknown>): string {
  return (
    asStr(check.message) ??
    asStr(check.reason) ??
    asStr(check.summary) ??
    asStr(check.status) ??
    "Reported by backend readiness snapshot"
  );
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
    status === "red" ||
    status === "blocked" ||
    status === "failed"
  )
    return "not_met";
  if (
    status === "checking" ||
    status === "running" ||
    status === "yellow" ||
    status === "waiting" ||
    status === "warning"
  )
    return "checking";
  if (check.ready === false || check.passed === false) return "not_met";
  return "unknown";
}
