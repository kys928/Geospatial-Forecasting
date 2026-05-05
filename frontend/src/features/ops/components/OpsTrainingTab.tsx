import { useMemo, useState } from "react";
import { opsClient } from "../api/opsClient";
import { useOpsJobs } from "../hooks/useOpsJobs";
import { useOpsStatus } from "../hooks/useOpsStatus";
import { useRetrainingRecommendation } from "../hooks/useRetrainingRecommendation";
import { useModelCandidateContext } from "../hooks/useModelCandidateContext";
import type { OpsJobRecord, RetrainingTriggerRequest } from "../types/ops.types";

function isJobActive(status: string | undefined): boolean {
  return status === "queued" || status === "running";
}

function formatTime(value: unknown): string {
  return typeof value === "string" && value.trim() ? value : "Not reported yet";
}

export function OpsTrainingTab() {
  const jobsState = useOpsJobs();
  const statusState = useOpsStatus();
  const recommendationState = useRetrainingRecommendation();
  const candidateState = useModelCandidateContext();

  const [manualOpen, setManualOpen] = useState(false);
  const [manualSubmitting, setManualSubmitting] = useState(false);
  const [manualNotice, setManualNotice] = useState<string | null>(null);

  const jobs = jobsState.jobs?.jobs ?? [];
  const runningJobs = jobs.filter((job) => isJobActive(job.status));

  const readiness = statusState.status?.retraining_readiness ?? {};
  const shouldRetrain = recommendationState.recommendation?.should_retrain;
  const evidence = recommendationState.recommendation?.evidence ?? {};

  const automaticEnabled = useMemo(() => {
    const enabled = readiness.automatic_training_enabled;
    if (typeof enabled === "boolean") {
      return enabled ? "Enabled" : "Disabled";
    }
    return "Not configured";
  }, [readiness]);

  const currentState = useMemo(() => {
    if (statusState.loading || recommendationState.loading || jobsState.loading) return "Checking status";
    if (runningJobs.some((j) => j.status === "running")) return "Training running";
    if (runningJobs.some((j) => j.status === "queued")) return "Training queued";
    if (statusState.status?.latest_retraining_job?.status === "failed") return "Failed";
    if (shouldRetrain === true) return "Ready to train";
    if (shouldRetrain === false) return "Waiting for more data or minimum interval";
    return "Blocked or unavailable";
  }, [jobsState.loading, recommendationState.loading, runningJobs, shouldRetrain, statusState.loading, statusState.status?.latest_retraining_job?.status]);

  async function refreshAll() {
    await Promise.all([jobsState.refresh(), statusState.refresh(), recommendationState.refresh(), candidateState.refresh()]);
  }

  async function handleManualStart(payload: RetrainingTriggerRequest) {
    setManualSubmitting(true);
    setManualNotice(null);
    try {
      const result = await opsClient.triggerRetraining(payload);
      setManualNotice(result.submitted ? "Manual training job submitted." : "Submission was not accepted by backend policy.");
      await refreshAll();
    } catch (error) {
      setManualNotice(error instanceof Error ? error.message : "Unable to submit manual training job.");
    } finally {
      setManualSubmitting(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <section className="panel">
        <h3>Automatic Training Status</h3>
        <p className="muted">Automatic training is the normal workflow. Manual training is available as an advanced override.</p>
        <div className="button-row">
          <button className="secondary-button" onClick={() => void refreshAll()} disabled={manualSubmitting}>Refresh training status</button>
        </div>
        <p>{statusState.loading || recommendationState.loading || jobsState.loading ? "⏳ Checking training status…" : ""}</p>
        <dl className="detail-list">
          <div className="detail-list-row"><dt>Automatic training</dt><dd>{automaticEnabled}</dd></div>
          <div className="detail-list-row"><dt>Current state</dt><dd>{currentState}</dd></div>
          <div className="detail-list-row"><dt>Last training run</dt><dd>{formatTime(statusState.status?.latest_retraining_job?.completed_at ?? statusState.status?.latest_retraining_job?.started_at)}</dd></div>
          <div className="detail-list-row"><dt>Next eligibility check</dt><dd>{formatTime(readiness.next_eligibility_check_at)}</dd></div>
          <div className="detail-list-row"><dt>Data since last training</dt><dd>{typeof evidence.new_samples === "number" ? `${evidence.new_samples} samples` : "Not reported yet"}</dd></div>
          <div className="detail-list-row"><dt>Recommendation</dt><dd>{shouldRetrain === true ? "Retrain" : shouldRetrain === false ? "Wait" : "Investigate / unavailable"}</dd></div>
        </dl>
      </section>

      <section className="panel">
        <h3>Training Readiness Checklist</h3>
        <ReadinessItem label="Enough new validated data" state={typeof evidence.new_samples === "number" ? (shouldRetrain ? "met" : "not_met") : "unknown"} detail={typeof evidence.new_samples === "number" ? `${evidence.new_samples} samples reported` : "Threshold not reported"} loading={recommendationState.loading} />
        <ReadinessItem label="Minimum time since last training passed" state={typeof evidence.min_interval_passed === "boolean" ? (evidence.min_interval_passed ? "met" : "not_met") : "unknown"} detail={typeof evidence.min_interval_passed === "boolean" ? "Interval check reported" : "Not enough evidence available"} loading={recommendationState.loading} />
        <ReadinessItem label="No retraining job currently running" state={runningJobs.length > 0 ? "not_met" : "met"} detail={runningJobs.length > 0 ? `${runningJobs.length} queued/running job(s)` : "No queued/running jobs"} loading={jobsState.loading} />
        <ReadinessItem label="Worker/resources available" state={typeof readiness.worker_available === "boolean" ? (readiness.worker_available ? "met" : "not_met") : "unknown"} detail={typeof readiness.worker_available === "boolean" ? "Resource signal reported" : "Not reported yet"} loading={statusState.loading} />
        <ReadinessItem label="Data quality acceptable" state={typeof evidence.data_quality_ok === "boolean" ? (evidence.data_quality_ok ? "met" : "not_met") : "unknown"} detail={typeof evidence.data_quality_ok === "boolean" ? "Quality signal reported" : "Not reported yet"} loading={recommendationState.loading} />
        <ReadinessItem label="Evidence indicates retraining is useful" state={typeof shouldRetrain === "boolean" ? (shouldRetrain ? "met" : "not_met") : "unknown"} detail={recommendationState.recommendation?.reason ?? "Not enough evidence available"} loading={recommendationState.loading} />
      </section>

      <section className="panel">
        <h3>Recent / Running Training Jobs</h3>
        <JobList jobs={jobs} loading={jobsState.loading} />
      </section>

      <section className="panel">
        <h3>Candidate Model Summary</h3>
        <p className="muted">Review or activate model versions in Model Versions.</p>
        <dl className="detail-list">
          <div className="detail-list-row"><dt>Current active model</dt><dd>{String(candidateState.context?.active_model?.model_id ?? statusState.status?.active_model?.model_id ?? "Not reported yet")}</dd></div>
          <div className="detail-list-row"><dt>Candidate model</dt><dd>{String(candidateState.context?.candidate_model?.model_id ?? "None")}</dd></div>
          <div className="detail-list-row"><dt>Decision state</dt><dd>{String(candidateState.context?.decision_state ?? "none")}</dd></div>
          <div className="detail-list-row"><dt>Safe next action</dt><dd>{candidateState.context?.safe_user_actions?.[0]?.title ?? "Review candidate evidence"}</dd></div>
          <div className="detail-list-row"><dt>Evidence summary</dt><dd>{candidateState.context?.comparison?.comparison_summary ?? "Comparable evidence not reported yet"}</dd></div>
        </dl>
      </section>

      <section className="panel">
        <h3>Manual Training (Advanced Override)</h3>
        <p className="muted">Use only when you need to bypass normal automatic scheduling.</p>
        <button className="secondary-button" onClick={() => setManualOpen(true)}>Start manual training</button>
        {manualNotice ? <p className="muted">{manualNotice}</p> : null}
      </section>

      <section className="panel">
        <details>
          <summary>Automatic training policy status</summary>
          <p className="muted">Automatic training should run only when automatic training is enabled, enough validated data exists, enough time has passed, no active job exists, resources are available, data quality is acceptable, and evidence recommends retraining.</p>
        </details>
      </section>

      {manualOpen ? <ManualTrainingModal onClose={() => setManualOpen(false)} onSubmit={handleManualStart} submitting={manualSubmitting} /> : null}
    </div>
  );
}

type ChecklistState = "met" | "not_met" | "unknown";
function ReadinessItem({ label, state, detail, loading }: { label: string; state: ChecklistState; detail: string; loading?: boolean }) {
  const icon = loading ? "⏳" : state === "met" ? "✅" : state === "not_met" ? "❌" : "⚪";
  return <p>{icon} <strong>{label}:</strong> {detail}</p>;
}

function JobList({ jobs, loading }: { jobs: OpsJobRecord[]; loading: boolean }) {
  const visible = jobs.slice(0, 5);
  if (loading) return <p className="muted">⏳ Loading jobs…</p>;
  if (jobs.length === 0) return <p className="muted">No jobs yet.</p>;
  return <div style={{ display: "grid", gap: 8 }}>{visible.map((job) => <article key={String(job.job_id)} className="badge" style={{ borderRadius: 8 }}><strong>{String(job.job_id)} · {job.status ?? "unknown"} {isJobActive(job.status) ? "⏳" : ""}</strong><p style={{ margin: "6px 0 0" }}>Created: {formatTime(job.created_at)} · Started: {formatTime(job.started_at)} · Completed: {formatTime(job.completed_at)}</p><p style={{ margin: "6px 0 0" }}>Dataset: {job.dataset_snapshot_ref ?? "Not reported yet"} · Base/checkpoint: {job.run_config_ref ?? "Not reported yet"}</p>{job.error_message ? <p style={{ margin: "6px 0 0" }}>Failure reason: {job.error_message}</p> : null}</article>)}</div>;
}

function ManualTrainingModal({ onClose, onSubmit, submitting }: { onClose: () => void; submitting: boolean; onSubmit: (payload: RetrainingTriggerRequest) => Promise<void> }) {
  const [datasetMode, setDatasetMode] = useState("buffered");
  const [datasetValue, setDatasetValue] = useState("");
  const [checkpointMode, setCheckpointMode] = useState("latest");
  const [checkpointValue, setCheckpointValue] = useState("");
  const [preset, setPreset] = useState("balanced");
  const [maxEpochs, setMaxEpochs] = useState("1");
  const [maxRuntime, setMaxRuntime] = useState("30m");
  const [advancedConfig, setAdvancedConfig] = useState("");

  const runConfigRef = JSON.stringify({ preset, max_epochs: Number(maxEpochs), max_runtime: maxRuntime, checkpoint_mode: checkpointMode, checkpoint_ref: checkpointValue || undefined, advanced_config_json: advancedConfig || undefined }, null, 2);
  const datasetSnapshotRef = datasetMode === "custom" ? datasetValue : "buffered_internal_dataset";

  return <div style={{ position: "fixed", inset: 0, background: "rgba(15,23,42,0.35)", display: "grid", placeItems: "center", zIndex: 50 }}><section className="panel" style={{ width: "min(760px, 96vw)", maxHeight: "90vh", overflow: "auto" }}><h3>Start Manual Training</h3><p className="muted">Manual training bypasses normal automatic scheduling. Use it only when you know why.</p>
  <label>Dataset source<select value={datasetMode} onChange={(e) => setDatasetMode(e.target.value)}><option value="buffered">Use buffered internal dataset</option><option value="custom">Custom dataset path/link</option></select></label>
  {datasetMode === "custom" ? <input value={datasetValue} onChange={(e) => setDatasetValue(e.target.value)} placeholder="/path/to/dataset or s3://bucket/dataset" /> : null}
  <label>Base checkpoint<select value={checkpointMode} onChange={(e) => setCheckpointMode(e.target.value)}><option value="latest">Latest best model</option><option value="active">Active production model</option><option value="custom">Custom checkpoint path</option></select></label>
  {checkpointMode === "custom" ? <input value={checkpointValue} onChange={(e) => setCheckpointValue(e.target.value)} placeholder="/path/to/checkpoint" /> : null}
  <label>Training preset<select value={preset} onChange={(e) => setPreset(e.target.value)}><option value="fast_refresh">Fast refresh</option><option value="balanced">Balanced</option><option value="high_accuracy">High accuracy</option><option value="recovery">Recovery / stabilization</option></select></label>
  <label>Max epochs<input value={maxEpochs} onChange={(e) => setMaxEpochs(e.target.value)} /></label>
  <label>Max runtime<input value={maxRuntime} onChange={(e) => setMaxRuntime(e.target.value)} /></label>
  <details><summary>Advanced options</summary><textarea rows={4} value={advancedConfig} onChange={(e) => setAdvancedConfig(e.target.value)} placeholder='Custom config JSON (optional). Not all fields are wired by backend.' /></details>
  <p className="muted">Note: dataset/checkpoint/preset details are sent through existing run_config_ref and dataset_snapshot_ref fields. Backend support depends on current retraining worker configuration.</p>
  <div className="button-row"><button className="secondary-button" onClick={onClose} disabled={submitting}>Cancel</button><button className="primary-button" onClick={() => void onSubmit({ manual_override: true, dataset_snapshot_ref: datasetSnapshotRef, run_config_ref: runConfigRef })} disabled={submitting}>{submitting ? "Starting..." : "Start training"}</button></div>
  </section></div>;
}
