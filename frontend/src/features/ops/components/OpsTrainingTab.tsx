import { useEffect, useMemo, useRef, useState } from "react";
import { opsClient } from "../api/opsClient";
import { useOpsJobs } from "../hooks/useOpsJobs";
import { useOpsStatus } from "../hooks/useOpsStatus";
import { useRetrainingRecommendation } from "../hooks/useRetrainingRecommendation";
import { useModelCandidateContext } from "../hooks/useModelCandidateContext";
import type {
  AdaptationBufferStatus,
  AdaptationReadiness,
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
  "stage",
  "stage_name",
  "global_epoch",
  "epoch_in_stage",
  "train_loss",
  "val_loss",
  "val_rollout_weighted_mse",
  "val_direct_weighted_mse",
  "val_rollout_mae",
  "val_rollout_mass_abs_error",
  "val_rollout_peak_location_error",
  "val_rollout_plume_iou",
  "val_free_rollout_gap",
  "selection_score",
  "learning_rate",
  "lr",
  "teacher_forcing_ratio",
  "teacher_forcing_prob",
  "best_score",
  "best_stage",
  "best_global_epoch",
  "best_checkpoint_path",
  "progress",
];

const asObj = (v: unknown) =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
const asStr = (v: unknown) => (typeof v === "string" && v.trim() ? v : null);
const pick = (obj: Record<string, unknown>, keys: string[]) =>
  keys.map((k) => obj[k]).find((v) => v !== undefined && v !== null);
const TRAINING_JOB_TIMESTAMP_KEYS = [
  "updated_at",
  "completed_at",
  "started_at",
  "created_at",
  "submitted_at",
] as const;

const REAL_ACTIVE_JOB_STATUSES = ["running", "claimed", "starting"] as const;
const TERMINAL_JOB_STATUSES = [
  "succeeded",
  "completed",
  "failed",
  "cancelled",
] as const;

const jobStatus = (job: OpsJobRecord | null | undefined): string =>
  String(job?.status ?? job?.effective_status ?? "").toLowerCase();

const isRealActiveJob = (job: OpsJobRecord | null | undefined): boolean =>
  REAL_ACTIVE_JOB_STATUSES.includes(
    jobStatus(job) as (typeof REAL_ACTIVE_JOB_STATUSES)[number],
  );

const isTerminalJob = (job: OpsJobRecord | null | undefined): boolean =>
  TERMINAL_JOB_STATUSES.includes(
    jobStatus(job) as (typeof TERMINAL_JOB_STATUSES)[number],
  );

const hasJobIdentity = (job: unknown): job is OpsJobRecord => {
  const obj = asObj(job);
  return Boolean(asStr(obj.job_id) && asStr(obj.status));
};

const timestampRank = (job: OpsJobRecord): number => {
  const dates = TRAINING_JOB_TIMESTAMP_KEYS.map((key) => {
    const value = job[key];
    if (typeof value !== "string") return Number.NaN;
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : Number.NaN;
  }).filter(Number.isFinite);
  return dates.length ? Math.max(...dates) : Number.NEGATIVE_INFINITY;
};

const jobIdSuffixRank = (job: OpsJobRecord): number => {
  const match = asStr(job.job_id)?.match(/(\d+)$/);
  return match ? Number(match[1]) : Number.NEGATIVE_INFINITY;
};

const compareJobsNewestFirst = (a: OpsJobRecord, b: OpsJobRecord): number => {
  const aTimestamp = timestampRank(a);
  const bTimestamp = timestampRank(b);
  if (aTimestamp !== bTimestamp) return bTimestamp - aTimestamp;

  const aSuffix = jobIdSuffixRank(a);
  const bSuffix = jobIdSuffixRank(b);
  if (aSuffix !== bSuffix) return bSuffix - aSuffix;

  return Number(b.created_sequence ?? -1) - Number(a.created_sequence ?? -1);
};

const latestByTimestampOrId = (jobs: OpsJobRecord[]): OpsJobRecord | null =>
  jobs.length ? [...jobs].sort(compareJobsNewestFirst)[0] : null;

export function OpsTrainingTab({ active = true }: { active?: boolean }) {
  const jobsState = useOpsJobs(active);
  const statusState = useOpsStatus(active);
  const recommendationState = useRetrainingRecommendation(active);
  const candidateState = useModelCandidateContext(active);
  const [manualOpen, setManualOpen] = useState(false);
  const [manualSubmitting, setManualSubmitting] = useState(false);
  const [stopSubmitting, setStopSubmitting] = useState(false);
  const [manualNotice, setManualNotice] = useState<string | null>(null);
  const [followLogs, setFollowLogs] = useState(true);
  const [pinnedLogJob, setPinnedLogJob] = useState<OpsJobRecord | null>(null);
  const [adaptationTraining, setAdaptationTraining] =
    useState<AdaptationTrainingStatus | null>(() =>
      opsClient.peekAdaptationTrainingStatus(),
    );
  const [adaptationReadiness, setAdaptationReadiness] =
    useState<AdaptationReadiness | null>(() =>
      opsClient.peekAdaptationReadiness(),
    );
  const [adaptationBuffer, setAdaptationBuffer] =
    useState<AdaptationBufferStatus | null>(() =>
      opsClient.peekAdaptationBufferStatus(),
    );
  const [adaptationTrainingError, setAdaptationTrainingError] = useState<
    string | null
  >(null);
  const logRef = useRef<HTMLPreElement | null>(null);
  const jobs = jobsState.jobs?.jobs ?? [];
  const runningJobs = jobs.filter(isRealActiveJob);
  const latestJob = useMemo(() => {
    const adaptationLatest = adaptationTraining?.latest_job;
    if (hasJobIdentity(adaptationLatest)) return adaptationLatest;

    const retrainingJobs = jobs.filter((job) => asStr(job.job_id));
    return (
      latestByTimestampOrId(retrainingJobs) ??
      statusState.status?.latest_retraining_job ??
      (hasJobIdentity(adaptationTraining?.latest_manual_job)
        ? adaptationTraining.latest_manual_job
        : null) ??
      null
    );
  }, [
    adaptationTraining?.latest_job,
    adaptationTraining?.latest_manual_job,
    jobs,
    statusState.status?.latest_retraining_job,
  ]);
  const activeJobForDisplay = useMemo(() => {
    const candidates = [
      ...(Array.isArray(statusState.status?.current_retraining_jobs)
        ? statusState.status.current_retraining_jobs
        : []),
      ...jobs,
      adaptationTraining?.latest_job,
      adaptationTraining?.latest_manual_job,
      statusState.status?.latest_retraining_job,
    ].filter(hasJobIdentity);
    return latestByTimestampOrId(candidates.filter(isRealActiveJob));
  }, [
    adaptationTraining?.latest_job,
    adaptationTraining?.latest_manual_job,
    jobs,
    statusState.status?.current_retraining_jobs,
    statusState.status?.latest_retraining_job,
  ]);
  const jobForLogs = pinnedLogJob ?? activeJobForDisplay ?? latestJob;
  const checklist = useMemo(
    () =>
      buildChecklist({
        runningJobs,
        status: statusState.status,
        latestJob,
        adaptationTraining,
        adaptationReadiness,
        adaptationBuffer,
      }),
    [
      runningJobs,
      statusState.status,
      latestJob,
      adaptationTraining,
      adaptationReadiness,
      adaptationBuffer,
    ],
  );
  const trainingView = useMemo(
    () =>
      deriveTrainingView(
        activeJobForDisplay,
        latestJob,
        adaptationTraining,
        adaptationReadiness,
      ),
    [activeJobForDisplay, latestJob, adaptationTraining, adaptationReadiness],
  );
  const summaryText = useMemo(
    () => buildSummaryText(trainingView.state, checklist, trainingView.detail),
    [trainingView.state, checklist, trainingView.detail],
  );
  const rawLogs = useMemo(
    () =>
      collectLogs(
        statusState.status,
        jobForLogs,
        adaptationTraining,
        candidateState.context,
      ),
    [
      statusState.status,
      jobForLogs,
      adaptationTraining,
      candidateState.context,
    ],
  );
  const metricLogLines = useMemo(
    () => formatTrainingMetricsAsLogLines(jobForLogs, adaptationTraining),
    [jobForLogs, adaptationTraining],
  );
  const logs = useMemo(
    () => combineTrainingLogs(rawLogs, metricLogLines, jobForLogs),
    [rawLogs, metricLogLines, jobForLogs],
  );
  const visibleLogs = logs.filter(
    (line) =>
      !/FutureWarning.*torch\.load|torch\.load.*FutureWarning/i.test(line),
  );
  const hiddenWarningCount = logs.length - visibleLogs.length;
  const hasErrorLogs = logs.some((line) => line.startsWith("ERROR:"));
  const canStopTraining = Boolean(activeJobForDisplay);

  useEffect(() => {
    if (followLogs && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs, followLogs]);
  useEffect(() => {
    if (!pinnedLogJob) return;
    const replacement = activeJobForDisplay ?? latestJob;
    if (
      replacement?.job_id &&
      replacement.job_id !== pinnedLogJob.job_id &&
      compareJobsNewestFirst(replacement, pinnedLogJob) < 0
    ) {
      setPinnedLogJob(null);
    }
  }, [activeJobForDisplay, latestJob, pinnedLogJob]);
  useEffect(() => {
    if (!active) return;
    void refreshAdaptationTraining(false);
    const timer = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void refreshAll(false);
    }, 10000);
    return () => window.clearInterval(timer);
  }, [active]);

  async function refreshAdaptationTraining(force = true) {
    try {
      const [training, readiness, buffer] = await Promise.all([
        opsClient.getAdaptationTrainingStatus({ force }),
        opsClient.getAdaptationReadiness({ force }),
        opsClient.getAdaptationBufferStatus({ force }),
      ]);
      setAdaptationTraining(training);
      setAdaptationReadiness(readiness);
      setAdaptationBuffer(buffer);
      setAdaptationTrainingError(null);
    } catch (err) {
      setAdaptationTrainingError(
        err instanceof Error
          ? err.message
          : "Unable to load adaptation training readiness.",
      );
    }
  }
  async function refreshAll(force = true) {
    await Promise.all([
      jobsState.refresh(force),
      statusState.refresh(force),
      recommendationState.refresh(force),
      candidateState.refresh(force),
      refreshAdaptationTraining(force),
    ]);
  }
  async function handleStopTraining() {
    setStopSubmitting(true);
    setManualNotice(null);
    try {
      const response = await opsClient.stopRetraining();
      setManualNotice(
        response.message ||
          (response.stopped
            ? "Training stop requested."
            : "No active training job to stop."),
      );
      await refreshAll();
    } catch (e) {
      setManualNotice(
        e instanceof Error ? e.message : "Unable to stop training job.",
      );
    } finally {
      setStopSubmitting(false);
    }
  }

  async function handleManualStart(payload: RetrainingTriggerRequest) {
    setManualSubmitting(true);
    setManualNotice(null);
    try {
      const r = await opsClient.triggerRetraining(payload);
      if (r.job?.job_id) setPinnedLogJob(r.job);
      setManualNotice(
        r.submitted
          ? "Manual training job submitted."
          : "Submission was not accepted by backend policy.",
      );
      setManualOpen(false);
      setManualSubmitting(false);
      void Promise.all([
        jobsState.refresh(false),
        statusState.refresh(false),
        refreshAdaptationTraining(false),
      ]).catch(() => {
      });
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
          {trainingView.technicalRows.length > 0 ? (
            <details
              className="advanced-section"
              style={{ gridColumn: "1 / -1" }}
            >
              <summary>Technical Details</summary>
              <dl className="ops-training-facts">
                {trainingView.technicalRows.map((row) => (
                  <div key={row.label}>
                    <dt>{row.label}</dt>
                    <dd title={row.value}>{row.value}</dd>
                  </div>
                ))}
              </dl>
            </details>
          ) : null}
        </div>
      </section>
      <section className="panel">
        <details
          className="advanced-section"
          open={Boolean(activeJobForDisplay)}
        >
          <summary>Live Training Logs</summary>
          <div
            className="button-row"
            style={{
              justifyContent: "space-between",
              alignItems: "center",
              marginTop: 10,
            }}
          >
            <h3 style={{ margin: 0 }}>Terminal training output</h3>
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
          {jobForLogs?.log_available === false ? (
            <p className="muted" style={{ margin: "8px 0 0" }}>
              Real training log file not available; showing generated metrics
              and summary.
            </p>
          ) : null}
          {hiddenWarningCount > 0 ? (
            <p className="muted" style={{ margin: "8px 0 0" }}>
              {hiddenWarningCount} known torch.load warning line(s) hidden from
              this view; Copy logs preserves raw output.
            </p>
          ) : null}
          {!visibleLogs.length ? (
            <p className="muted" style={{ marginBottom: 0 }}>
              {activeJobForDisplay
                ? "Active training job has no logs available."
                : "No active training job is producing logs."}
            </p>
          ) : (
            <pre
              ref={logRef}
              className={`ops-log-window${hasErrorLogs ? " ops-log-window-error" : ""}`}
            >
              {visibleLogs.join("\n")}
            </pre>
          )}
        </details>
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
          <div className="button-row">
            <button
              className="secondary-button"
              onClick={() => setManualOpen(true)}
            >
              Start manual training
            </button>
            <button
              className="secondary-button"
              style={{
                background: "#dc2626",
                color: "white",
                borderColor: "#dc2626",
              }}
              onClick={() => void handleStopTraining()}
              disabled={!canStopTraining || stopSubmitting}
            >
              Stop training
            </button>
          </div>
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
  activeJob: OpsJobRecord | null,
  latestJob: OpsJobRecord | null,
  adaptationTraining: AdaptationTrainingStatus | null,
  adaptationReadiness: AdaptationReadiness | null,
) {
  const displayJob = activeJob ?? latestJob;
  const jobObj = asObj(displayJob);
  const latestJobObj = asObj(latestJob);
  const runCfg = asObj(
    (() => {
      try {
        return JSON.parse(asStr(displayJob?.run_config_ref) ?? "{}");
      } catch {
        return {};
      }
    })(),
  );
  const activeStatusRaw = activeJob
    ? asStr(pick(asObj(activeJob), ["status", "effective_status"]))
    : null;
  const readinessSnapshot =
    adaptationReadiness ??
    adaptationTraining?.latest_readiness_snapshot ??
    null;
  const readiness = asObj(readinessSnapshot);
  const readinessDetail = Object.keys(readiness).length
    ? trainingReadinessDetail(readiness, readinessState(readiness))
    : null;
  const state = activeJob
    ? activeTrainingState(activeStatusRaw)
    : readinessTrainingState(readiness, adaptationTraining);
  const started = asStr(pick(jobObj, ["started_at", "start_time"]));
  const completed = asStr(
    pick(jobObj, ["finished_at", "completed_at", "end_time"]),
  );
  const runtimeSeconds =
    typeof jobObj.runtime_seconds === "number" ? jobObj.runtime_seconds : null;
  const elapsedSeconds =
    typeof jobObj.elapsed_seconds === "number" ? jobObj.elapsed_seconds : null;
  const elapsed = started
    ? formatDurationSeconds(
        elapsedSeconds ??
          Math.max(
            0,
            Math.floor(
              ((completed ? Date.parse(completed) : Date.now()) -
                Date.parse(started)) /
                1000,
            ),
          ),
      )
    : "Not reported";
  const runtime =
    runtimeSeconds !== null
      ? formatDurationSeconds(runtimeSeconds)
      : completed && started
        ? formatDurationSeconds(
            Math.max(
              0,
              Math.floor((Date.parse(completed) - Date.parse(started)) / 1000),
            ),
          )
        : null;
  const latestStatus =
    asStr(pick(latestJobObj, ["status", "effective_status"])) ?? "Not reported";
  const activeStatus = activeStatusRaw ?? "Not reported";
  const jobCounts = adaptationTraining?.job_counts ?? {};
  const jobCountSummary = formatJobCountSummary(jobCounts);
  const logJobObj = activeJob ? jobObj : latestJobObj;
  const logStatus =
    logJobObj.is_stale === true
      ? "stale"
      : logJobObj.log_available === true
        ? "available"
        : logJobObj.log_available === false
          ? "unavailable"
          : "initializing";
  const readinessStatus =
    readiness.ready === true
      ? "ready"
      : (asStr(readiness.status) ??
        (Object.keys(readiness).length ? readinessState(readiness) : null));
  const blockingReason = primaryReadinessReason(readiness);
  const rows = [
    { label: "Current state", value: state },
    ...(readinessStatus
      ? [{ label: "Current readiness", value: String(readinessStatus) }]
      : []),
    ...(blockingReason
      ? [{ label: "Readiness blocker", value: blockingReason }]
      : readinessDetail
        ? [{ label: "Readiness detail", value: readinessDetail }]
        : []),
    ...(activeJob
      ? [
          {
            label: "Active job",
            value: asStr(activeJob.job_id) ?? "Recorded without job id",
          },
          { label: "Active status", value: activeStatus },
        ]
      : []),
    {
      label: "Latest job",
      value:
        asStr(latestJob?.job_id) ??
        (adaptationTraining?.latest_job
          ? "Recorded without job id"
          : "No adaptation training job recorded"),
    },
    { label: "Latest status", value: latestStatus },
    ...(isTerminalJob(latestJob)
      ? [{ label: "Latest job role", value: "Historical" }]
      : []),
    { label: "Training log", value: `Training log: ${logStatus}` },
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
      value:
        (typeof jobObj.best_checkpoint === "string"
          ? jobObj.best_checkpoint
          : null) ??
        adaptationTraining?.best_overall_checkpoint ??
        "Not reported",
    },
    {
      label: "Final checkpoint",
      value:
        (typeof jobObj.final_checkpoint === "string"
          ? jobObj.final_checkpoint
          : null) ??
        adaptationTraining?.final_checkpoint ??
        "Not reported",
    },
    { label: "Started", value: started ?? "Not reported" },
    ...(runtime
      ? [{ label: "Runtime", value: runtime }]
      : [{ label: "Elapsed", value: elapsed }]),
    {
      label: "Retraining cooldown",
      value: formatDurationSeconds(
        adaptationTraining?.cooldown_seconds ?? 10800,
      ),
    },
    ...(adaptationTraining?.cooldown_remaining_seconds &&
    adaptationTraining.cooldown_remaining_seconds > 0
      ? [
          {
            label: "Next automatic training eligible in",
            value: formatDurationSeconds(
              adaptationTraining.cooldown_remaining_seconds,
            ),
          },
        ]
      : []),
    { label: "Job counts", value: jobCountSummary },
  ].filter(
    (row) =>
      row.value !== "Not reported" ||
      [
        "Current state",
        "Current readiness",
        "Latest job",
        "Latest status",
        "Retraining cooldown",
        "Job counts",
      ].includes(row.label),
  );
  const technicalLabels = new Set([
    "Latest job role",
    "Training log",
    "Candidate",
    "Output dir",
    "Result run dir",
    "Best checkpoint",
    "Final checkpoint",
    "Started",
    "Job counts",
  ]);
  const prominentRows = rows.filter((row) => !technicalLabels.has(row.label));
  const technicalRows = rows.filter((row) => technicalLabels.has(row.label));
  return {
    state,
    detail: readinessDetail,
    rows: prominentRows,
    technicalRows,
  };
}

function activeTrainingState(statusRaw: string | null): string {
  const mapState: Record<string, string> = {
    running: "Running",
    claimed: "Running",
    starting: "Starting",
  };
  return mapState[(statusRaw ?? "").toLowerCase()] ?? "Running";
}

function readinessTrainingState(
  readiness: Record<string, unknown>,
  adaptationTraining: AdaptationTrainingStatus | null,
): string {
  if ((adaptationTraining?.cooldown_remaining_seconds ?? 0) > 0)
    return "Cooling down";
  if (!Object.keys(readiness).length) return "Not reported";
  if (readiness.ready === true || readinessState(readiness) === "met")
    return "Ready for automatic training";
  if (
    primaryReadinessReason(readiness) ||
    readinessState(readiness) === "not_met"
  )
    return "Readiness blocked";
  return "Waiting for readiness";
}

function primaryReadinessReason(
  readiness: Record<string, unknown>,
): string | null {
  const blockingReasons = Array.isArray(readiness.blocking_reasons)
    ? readiness.blocking_reasons.filter((reason) => typeof reason === "string")
    : [];
  const warnings = Array.isArray(readiness.warnings)
    ? readiness.warnings.filter((warning) => typeof warning === "string")
    : [];
  return (
    (blockingReasons[0] as string | undefined) ??
    (warnings[0] as string | undefined) ??
    null
  );
}

function sourceJobId(source: Record<string, unknown>): string | null {
  return asStr(
    pick(source, ["job_id", "training_job_id", "retraining_job_id", "run_id"]),
  );
}

function sourceMatchesSelectedJob(
  source: Record<string, unknown>,
  selectedJobId: string | null,
  allowUnidentified = false,
): boolean {
  if (!selectedJobId) return true;
  const id = sourceJobId(source);
  return id ? id === selectedJobId : allowUnidentified;
}

function collectTrainingMetricSources(
  jobObj: Record<string, unknown>,
  adaptationTraining?: AdaptationTrainingStatus | null,
): Record<string, unknown>[] {
  return [
    adaptationTraining?.training_metrics,
    pick(jobObj, ["metrics", "training_metrics", "progress_metrics"]),
    pick(jobObj, ["training_summary"]),
    pick(asObj(pick(jobObj, ["training_summary"])), ["metrics"]),
    pick(jobObj, ["best_metrics"]),
    jobObj,
  ]
    .map(asObj)
    .filter((source) => Object.keys(source).length > 0);
}

function formatMetricValue(key: string, value: unknown): string {
  if (key.includes("checkpoint_path") && typeof value === "string") {
    return value.split(/[\\/]/).slice(-2).join("/");
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toPrecision(6);
  }
  return String(value);
}

function metricSourceKey(source: Record<string, unknown>): string {
  return [
    source.stage_name ?? source.stage ?? "stage",
    source.global_epoch ?? "epoch",
    source.epoch_in_stage ?? "stage_epoch",
  ]
    .map((value) => String(value))
    .join(":");
}

function firstMetricValue(
  source: Record<string, unknown>,
  keys: string[],
): unknown {
  for (const key of keys) {
    const value = source[key];
    if (value !== undefined && value !== null && String(value).trim() !== "")
      return value;
  }
  return undefined;
}

function formatMetricPair(key: string, value: unknown): string | null {
  if (value === undefined || value === null || String(value).trim() === "")
    return null;
  return `${key}=${formatMetricValue(key, value)}`;
}

function metricPairs(
  source: Record<string, unknown>,
  entries: Array<[string, string[]]>,
): string {
  return entries
    .map(([label, keys]) =>
      formatMetricPair(label, firstMetricValue(source, keys)),
    )
    .filter((value): value is string => Boolean(value))
    .join(" ");
}

function collectMetricLogSources(
  latestJob: OpsJobRecord | null,
  adaptationTraining: AdaptationTrainingStatus | null,
): Record<string, unknown>[] {
  const jobObj = asObj(latestJob);
  const selectedJobId = asStr(latestJob?.job_id);
  const sources = collectTrainingMetricSources(
    jobObj,
    adaptationTraining,
  ).filter(
    (source) =>
      source === jobObj || sourceMatchesSelectedJob(source, selectedJobId),
  );
  const deduped: Record<string, unknown>[] = [];
  const seen = new Set<string>();
  for (const source of sources) {
    if (!Object.keys(source).some((key) => TRAINING_METRIC_KEYS.includes(key)))
      continue;
    const key = metricSourceKey(source);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(source);
  }
  return deduped.slice(0, 3);
}

export function formatTrainingMetricsAsLogLines(
  latestJob: OpsJobRecord | null,
  adaptationTraining: AdaptationTrainingStatus | null,
): string[] {
  const lines: string[] = [];
  for (const source of collectMetricLogSources(latestJob, adaptationTraining)) {
    const identity = metricPairs(source, [
      ["stage", ["stage_name", "stage"]],
      ["global_epoch", ["global_epoch"]],
      ["epoch_in_stage", ["epoch_in_stage"]],
    ]);
    if (identity) lines.push(`[metrics] ${identity}`);
    const losses = metricPairs(source, [
      ["train_loss", ["train_loss"]],
      ["val_loss", ["val_loss"]],
      ["val_rollout_weighted_mse", ["val_rollout_weighted_mse"]],
      ["val_direct_weighted_mse", ["val_direct_weighted_mse"]],
    ]);
    if (losses) lines.push(`[metrics] ${losses}`);
    const rollout = metricPairs(source, [
      ["val_rollout_mae", ["val_rollout_mae"]],
      ["val_rollout_mass_abs_error", ["val_rollout_mass_abs_error"]],
      ["val_rollout_peak_location_error", ["val_rollout_peak_location_error"]],
    ]);
    if (rollout) lines.push(`[metrics] ${rollout}`);
    const selection = metricPairs(source, [
      ["val_rollout_plume_iou", ["val_rollout_plume_iou"]],
      ["val_free_rollout_gap", ["val_free_rollout_gap"]],
      ["selection_score", ["selection_score"]],
    ]);
    if (selection) lines.push(`[metrics] ${selection}`);
    const scheduler = metricPairs(source, [
      ["lr", ["learning_rate", "lr"]],
      [
        "teacher_forcing_prob",
        ["teacher_forcing_prob", "teacher_forcing_ratio"],
      ],
    ]);
    if (scheduler) lines.push(`[metrics] ${scheduler}`);
  }

  const jobObj = asObj(latestJob);
  const selectedJobId = asStr(latestJob?.job_id);
  const bestSource = collectTrainingMetricSources(jobObj, adaptationTraining)
    .filter(
      (source) =>
        source === jobObj || sourceMatchesSelectedJob(source, selectedJobId),
    )
    .find((source) =>
      [
        "best_score",
        "best_stage",
        "best_global_epoch",
        "best_checkpoint_path",
      ].some((key) => source[key] !== undefined && source[key] !== null),
    );
  if (bestSource) {
    const best = metricPairs(bestSource, [
      ["score", ["best_score"]],
      ["stage", ["best_stage"]],
      ["global_epoch", ["best_global_epoch"]],
      ["checkpoint", ["best_checkpoint_path"]],
    ]);
    if (best) lines.push(`[best] ${best}`);
  }
  return lines;
}

function combineTrainingLogs(
  rawLogs: string[],
  metricLogLines: string[],
  selectedJob: OpsJobRecord | null,
): string[] {
  const selectedJobId = asStr(selectedJob?.job_id);
  const waitingLine = selectedJobId
    ? `Training job ${selectedJobId} was submitted. Waiting for worker logs...`
    : null;
  const rawHasOnlyWaiting =
    Boolean(waitingLine) && rawLogs.length === 1 && rawLogs[0] === waitingLine;
  if (!metricLogLines.length) return rawLogs;
  const rawMetricKeys = new Set(
    rawLogs
      .filter(
        (line) => line.startsWith("[metrics]") || line.startsWith("[best]"),
      )
      .map((line) => line.trim()),
  );
  const generated = metricLogLines.filter(
    (line) => !rawMetricKeys.has(line.trim()),
  );
  if (!generated.length) return rawLogs;
  if (rawHasOnlyWaiting) return generated;
  return rawLogs.length ? [...rawLogs, "", ...generated] : generated;
}

function formatDurationSeconds(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  if (hours > 0 && minutes === 0 && secs === 0) return `${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function formatJobCountSummary(jobCounts: Record<string, number>): string {
  const entries = [
    ["queued", jobCounts.queued],
    ["running", jobCounts.running],
    ["waiting", jobCounts.waiting],
    ["failed", jobCounts.failed],
    ["succeeded", jobCounts.succeeded ?? jobCounts.completed],
  ].filter(([, value]) => typeof value === "number");
  return entries.length
    ? entries.map(([label, value]) => `${label}: ${value}`).join("; ")
    : "Not reported";
}

function collectLogs(
  status: any,
  latestJob: OpsJobRecord | null,
  adaptationTraining: AdaptationTrainingStatus | null,
  candidateContext: Record<string, unknown> | null,
): string[] {
  const jobObj = asObj(latestJob);
  const lines: string[] = [];
  const adaptationLatestJob = adaptationTraining?.latest_job as
    | OpsJobRecord
    | null
    | undefined;
  const selectedJobId = asStr(latestJob?.job_id);
  const adaptationLatestMatchesSelected =
    !selectedJobId || adaptationLatestJob?.job_id === selectedJobId;
  const latestLogTail =
    adaptationLatestMatchesSelected &&
    Array.isArray(adaptationLatestJob?.log_tail)
      ? adaptationLatestJob.log_tail
      : [];
  if (latestLogTail.length) return latestLogTail.map((line) => String(line));
  const logs = sourceMatchesSelectedJob(jobObj, selectedJobId, true)
    ? pick(jobObj, ["logs", "log_lines", "events"])
    : undefined;
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
  if (["queued", "waiting", "starting"].includes(statusValue)) {
    lines.unshift("Training job is queued and waiting for a worker.");
    if (isManual && !jobObj.started_at && !jobObj.worker_pid)
      lines.push("Worker has not claimed this manual job yet.");
  }
  if (jobObj.is_stale === true || asStr(jobObj.effective_status) === "stale") {
    lines.unshift(
      "Training job appears stale; no recent worker update was reported.",
    );
  } else if (statusValue === "running") {
    lines.unshift(
      `Training job ${asStr(jobObj.job_id) ?? "latest"} is running.`,
    );
  }
  if (["succeeded", "completed"].includes(statusValue)) {
    lines.unshift(
      `Training job ${asStr(jobObj.job_id) ?? "latest"} succeeded.`,
    );
    const candidateId =
      adaptationTraining?.candidate_model_id ??
      asStr(pick(jobObj, ["candidate_model_id"]));
    const candidate = asObj(candidateContext?.candidate_model);
    const active = asObj(candidateContext?.active_model);
    const candidateStatus =
      asStr(candidate.status) ??
      (asStr(active.model_id) === candidateId ? asStr(active.status) : null);
    const approvalStatus =
      asStr(candidate.approval_status) ??
      (asStr(active.model_id) === candidateId
        ? asStr(active.approval_status)
        : null);
    if (candidateId) {
      const details = [candidateStatus, approvalStatus]
        .filter(Boolean)
        .join("/");
      lines.push(
        `Candidate model ${candidateId}${details ? ` is ${details}.` : " is recorded."}`,
      );
    }
    const runDir =
      adaptationTraining?.result_run_dir ??
      asStr(pick(jobObj, ["result_run_dir", "output_dir"]));
    if (runDir) lines.push(`Run directory: ${runDir}`);
  }
  const failure = ["failed", "error"].includes(statusValue)
    ? asStr(pick(jobObj, ["failure_reason", "error_message"]))
    : latestJob
      ? null
      : asStr(status?.last_retraining_job_failure_reason);
  if (failure)
    lines.push(
      isManual ? `Manual training job failed: ${failure}` : `ERROR: ${failure}`,
    );
  if (!lines.length && selectedJobId)
    lines.push(
      `Training job ${selectedJobId} was submitted. Waiting for worker logs...`,
    );
  return lines;
}

function buildSummaryText(
  state: string,
  checklist: Array<{ label: string; state: ChecklistState }>,
  detail?: string | null,
) {
  const notReady = checklist.some((c) => c.state === "not_met");
  if (state === "Stale")
    return "Training job appears stale; no recent worker update was reported.";
  if (state === "Running")
    return "Adaptation training is running. Metrics will update as the worker reports progress.";
  if (state === "Starting")
    return "Adaptation training is starting. Logs and metrics will update once the worker reports progress.";
  if (state === "Queued")
    return "An adaptation training job is queued and waiting for a worker.";
  if (state === "Cooling down")
    return "Automatic adaptation training is cooling down before another run can start.";
  if (state === "Readiness blocked")
    return (
      detail ?? "Automatic adaptation training is blocked by readiness checks."
    );
  if (state === "Ready for automatic training")
    return "Automatic adaptation training is ready to start when new work is selected.";
  if (state === "Waiting for readiness")
    return (
      detail ??
      "Automatic adaptation training is waiting for readiness conditions or new work."
    );
  if (state === "Failed")
    return "The latest adaptation training job failed. Review logs and job details before starting another run.";
  if (state === "Completed")
    return "The latest adaptation training job completed. Review the candidate model in Model Versions.";
  if (state === "Not reported")
    return "No adaptation training readiness has been reported yet.";
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
  latestJob,
  adaptationTraining,
  adaptationReadiness,
  adaptationBuffer,
}: {
  runningJobs: OpsJobRecord[];
  status: any;
  latestJob: OpsJobRecord | null;
  adaptationTraining: AdaptationTrainingStatus | null;
  adaptationReadiness: AdaptationReadiness | null;
  adaptationBuffer: AdaptationBufferStatus | null;
}): ChecklistRow[] {
  const legacyReadiness = status?.retraining_readiness ?? {};
  const readinessSnapshot =
    adaptationReadiness ??
    adaptationTraining?.latest_readiness_snapshot ??
    null;
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
  const sampleSpanCheck = findReadinessCheck(checks, [
    "accepted_sample_time_span",
    "sample time span",
  ]);
  const sampleAgeCheck = findReadinessCheck(checks, [
    "accepted_sample_age",
    "sample age",
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
      detail: trainingReadinessDetail(readiness, overallState),
    },
    buildNewTrainingDataRow(freshSamplesCheck, fallbackCheck, adaptationBuffer),
    buildBufferDataRow(
      bufferCheck,
      sampleSpanCheck,
      sampleAgeCheck,
      fallbackCheck,
      adaptationBuffer,
    ),
    buildBaseModelRow(checkpointCheck, readiness, legacyReadiness),
    buildTrainingJobStateRow(
      runningJobs,
      latestJob,
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

function trainingReadinessDetail(
  readiness: Record<string, unknown>,
  state: ChecklistState,
): string {
  if (readiness.ready === true) return "Training can start.";
  const status = String(readiness.status ?? "").toLowerCase();
  const blockingReasons = Array.isArray(readiness.blocking_reasons)
    ? readiness.blocking_reasons.filter((reason) => typeof reason === "string")
    : [];
  const warnings = Array.isArray(readiness.warnings)
    ? readiness.warnings.filter((warning) => typeof warning === "string")
    : [];
  const reason =
    (blockingReasons[0] as string | undefined) ??
    (warnings[0] as string | undefined);
  if (state === "met") return "Training can start.";
  if (state === "checking" || status === "waiting" || status === "yellow") {
    return reason ? `Waiting: ${reason}` : "Waiting for readiness conditions.";
  }
  if (state === "not_met") {
    return reason
      ? `Training cannot start yet: ${reason}`
      : "Training cannot start yet.";
  }
  return Object.keys(readiness).length
    ? "Training readiness is being evaluated."
    : "Training readiness is not reported.";
}

function buildNewTrainingDataRow(
  freshSamplesCheck: Record<string, unknown> | null,
  fallbackCheck: Record<string, unknown> | null,
  adaptationBuffer: AdaptationBufferStatus | null,
): ChecklistRow {
  const freshState = freshSamplesCheck
    ? readinessState(freshSamplesCheck)
    : "unknown";
  const counts = adaptationBufferCounts(adaptationBuffer);
  const fallbackAvailable = hasFallbackDataset(fallbackCheck);
  if (freshState === "met") {
    return {
      label: "New training data",
      state: "met",
      detail:
        counts.acceptedTotal > 0
          ? `Enough accepted training samples are available (${formatAcceptedCounts(counts)}).`
          : "Enough new training data is available.",
    };
  }
  if (counts.acceptedTotal > 0) {
    return {
      label: "New training data",
      state: "checking",
      detail: `Accepted samples are available but readiness is still waiting (${formatAcceptedCounts(counts)}).`,
    };
  }
  if (fallbackAvailable) {
    return {
      label: "New training data",
      state: "checking",
      detail:
        "Waiting for enough collected training data; a historical fallback dataset is available.",
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
    detail: adaptationBuffer
      ? "No accepted training samples are available yet."
      : "New training data status is not reported.",
  };
}

function buildBufferDataRow(
  bufferCheck: Record<string, unknown> | null,
  sampleSpanCheck: Record<string, unknown> | null,
  sampleAgeCheck: Record<string, unknown> | null,
  fallbackCheck: Record<string, unknown> | null,
  adaptationBuffer: AdaptationBufferStatus | null,
): ChecklistRow {
  const bufferState = bufferCheck ? readinessState(bufferCheck) : "unknown";
  const spanState = sampleSpanCheck
    ? readinessState(sampleSpanCheck)
    : "unknown";
  const ageState = sampleAgeCheck ? readinessState(sampleAgeCheck) : "unknown";
  const counts = adaptationBufferCounts(adaptationBuffer);
  const fallbackAvailable = hasFallbackDataset(fallbackCheck);
  const countDetail =
    counts.acceptedTotal > 0 ? formatAcceptedCounts(counts) : null;

  if (counts.acceptedTotal > 0) {
    const dataChecksPass =
      bufferState === "met" &&
      [spanState, ageState].every(
        (state) => state === "met" || state === "unknown",
      );
    return {
      label: "Buffer Data",
      state: dataChecksPass ? "met" : "checking",
      detail: dataChecksPass
        ? `Buffer data is ready (${countDetail}).`
        : `Buffer has accepted samples but is still waiting on readiness checks (${countDetail}).`,
    };
  }
  if (fallbackAvailable) {
    return {
      label: "Buffer Data",
      state: "checking",
      detail:
        "No buffer samples yet; a historical fallback dataset is available.",
    };
  }
  if (bufferState === "not_met") {
    return {
      label: "Buffer Data",
      state: "not_met",
      detail: "No usable buffer samples are available.",
    };
  }
  if (bufferState === "met" || bufferState === "checking") {
    return {
      label: "Buffer Data",
      state: "checking",
      detail: "Waiting for collected samples.",
    };
  }
  return {
    label: "Buffer Data",
    state: "unknown",
    detail: adaptationBuffer
      ? "No buffer samples have been accepted yet."
      : "Buffer data status is not reported.",
  };
}

function buildBaseModelRow(
  checkpointCheck: Record<string, unknown> | null,
  readiness: Record<string, unknown>,
  legacyReadiness: Record<string, unknown>,
): ChecklistRow {
  const selectedCheckpointPath = selectedCheckpointFromReadiness(
    readiness,
    checkpointCheck,
  );
  if (selectedCheckpointPath) {
    return {
      label: "Base model",
      state: "met",
      detail: `Base model is available (${basename(selectedCheckpointPath)}).`,
      warning: selectedCheckpointPath,
    };
  }

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
  latestJob: OpsJobRecord | null,
  noTrainingJobCheck: Record<string, unknown> | null,
  adaptationTraining: AdaptationTrainingStatus | null,
): ChecklistRow {
  const latest = asObj(latestJob ?? adaptationTraining?.latest_job ?? null);
  const latestStatus = String(latest.status ?? "").toLowerCase();
  const hasRunningLatest = ["running", "claimed"].includes(latestStatus);
  const hasQueuedLatest = ["queued", "waiting", "starting"].includes(
    latestStatus,
  );

  if (runningJobs.length > 0) {
    const activeStatus = jobStatus(runningJobs[0]);
    return {
      label: "Training job",
      state: "checking",
      detail:
        activeStatus === "starting"
          ? "Training job is starting."
          : "Training job is running.",
    };
  }
  if (hasRunningLatest)
    return {
      label: "Training job",
      state: "checking",
      detail: "Training job is running.",
    };
  if (hasQueuedLatest)
    return {
      label: "Training job",
      state: "checking",
      detail: "Training job is queued and waiting for a worker.",
    };
  if (
    TERMINAL_JOB_STATUSES.includes(
      latestStatus as (typeof TERMINAL_JOB_STATUSES)[number],
    )
  )
    return {
      label: "Training job",
      state: "met",
      detail: `Latest job is ${latestStatus}; no active training job is selected.`,
    };
  if (noTrainingJobCheck) {
    const state = readinessState(noTrainingJobCheck);
    return {
      label: "Training job",
      state,
      detail:
        state === "met"
          ? "No active training jobs."
          : "Training job state is not reported.",
    };
  }
  if (runningJobs.length === 0)
    return {
      label: "Training job",
      state: "met",
      detail: "No active training jobs.",
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
    : detail;
}

function adaptationBufferCounts(buffer: AdaptationBufferStatus | null) {
  const acceptedTrain = Number(buffer?.accepted_train ?? 0);
  const acceptedVal = Number(buffer?.accepted_val ?? 0);
  return {
    acceptedTrain,
    acceptedVal,
    acceptedTotal: acceptedTrain + acceptedVal,
  };
}

function formatAcceptedCounts({
  acceptedTrain,
  acceptedVal,
}: {
  acceptedTrain: number;
  acceptedVal: number;
}): string {
  return `${acceptedTrain} train, ${acceptedVal} validation`;
}

function hasFallbackDataset(
  fallbackCheck: Record<string, unknown> | null,
): boolean {
  const fallbackDetails = asObj(fallbackCheck?.details);
  return Boolean(fallbackDetails.selected_dataset_path);
}

function selectedCheckpointFromReadiness(
  readiness: Record<string, unknown>,
  checkpointCheck: Record<string, unknown> | null,
): string | null {
  const summaryCheckpoint = asObj(asObj(readiness.summary).checkpoint);
  const checkDetails = asObj(checkpointCheck?.details);
  return (
    asStr(readiness.selected_checkpoint_path) ??
    asStr(summaryCheckpoint.selected_checkpoint_path) ??
    asStr(checkDetails.selected_checkpoint_path) ??
    asStr(checkpointCheck?.selected_checkpoint_path) ??
    null
  );
}

function basename(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? path;
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
