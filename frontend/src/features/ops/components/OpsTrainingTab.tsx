import { useEffect, useMemo, useState } from "react";
import { opsClient } from "../api/opsClient";
import { useOpsJobs } from "../hooks/useOpsJobs";
import { useOpsStatus } from "../hooks/useOpsStatus";
import { useRetrainingRecommendation } from "../hooks/useRetrainingRecommendation";
import { useModelCandidateContext } from "../hooks/useModelCandidateContext";
import type { OpsJobRecord, RetrainingTriggerRequest } from "../types/ops.types";

type PresetKey = "fast_refresh" | "balanced" | "high_accuracy" | "recovery";
type LearningRateMode = "conservative" | "default" | "aggressive";
type RuntimeOption = "15m" | "30m" | "1h" | "2h";

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
  recovery: "Conservative run after failures, drift, or unstable model behavior."
};

const presetDefaults: Record<PresetKey, ManualControlState> = {
  fast_refresh: { maxEpochs: 3, maxRuntime: "15m", batchSize: "16", learningRate: "conservative", earlyStoppingPatience: 3, validationSplit: "10" },
  balanced: { maxEpochs: 8, maxRuntime: "30m", batchSize: "32", learningRate: "default", earlyStoppingPatience: 4, validationSplit: "15" },
  high_accuracy: { maxEpochs: 20, maxRuntime: "2h", batchSize: "32", learningRate: "conservative", earlyStoppingPatience: 6, validationSplit: "20" },
  recovery: { maxEpochs: 5, maxRuntime: "1h", batchSize: "16", learningRate: "conservative", earlyStoppingPatience: 5, validationSplit: "15" }
};

export function OpsTrainingTab() {
  const jobsState = useOpsJobs();
  const statusState = useOpsStatus();
  const recommendationState = useRetrainingRecommendation();
  const candidateState = useModelCandidateContext();
  const [manualOpen, setManualOpen] = useState(false);
  const [manualSubmitting, setManualSubmitting] = useState(false);
  const [manualNotice, setManualNotice] = useState<string | null>(null);
  const jobs = jobsState.jobs?.jobs ?? [];
  const runningJobs = jobs.filter((j) => j.status === "queued" || j.status === "running");
  const recommendation = recommendationState.recommendation;
  const checklist = useMemo(() => buildChecklist({ recommendation, runningJobs, status: statusState.status }), [recommendation, runningJobs, statusState.status]);

  async function refreshAll() {
    await Promise.all([jobsState.refresh(), statusState.refresh(), recommendationState.refresh(), candidateState.refresh()]);
  }

  async function handleManualStart(payload: RetrainingTriggerRequest) {
    setManualSubmitting(true);
    setManualNotice(null);
    try {
      const r = await opsClient.triggerRetraining(payload);
      setManualNotice(r.submitted ? "Manual training job submitted." : "Submission was not accepted by backend policy.");
      setManualOpen(false);
      await refreshAll();
    } catch (e) {
      setManualNotice(e instanceof Error ? e.message : "Unable to submit manual training job.");
    } finally {
      setManualSubmitting(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <section className="panel">
        <h3>Automatic Training Status</h3>
        <div className="button-row">
          <button className="secondary-button" onClick={() => void refreshAll()} disabled={manualSubmitting}>Refresh training status</button>
        </div>
      </section>
      <section className="panel">
        <h3>Training Readiness Checklist</h3>
        {checklist.map((item) => <ReadinessItem key={item.label} label={item.label} state={item.state} detail={item.detail} />)}
      </section>
      <section className="panel">
        <h3>Recent / Running Training Jobs</h3>
        <JobList jobs={jobs} loading={jobsState.loading} />
      </section>
      <section className="panel">
        <h3>Manual Training</h3>
        <button className="secondary-button" onClick={() => setManualOpen(true)}>Start manual training</button>
        {manualNotice ? <p className="muted">{manualNotice}</p> : null}
      </section>
      {manualOpen ? <ManualTrainingModal onClose={() => setManualOpen(false)} onSubmit={handleManualStart} submitting={manualSubmitting} /> : null}
    </div>
  );
}

type ChecklistState = "met" | "not_met" | "unknown" | "checking";
function ReadinessItem({ label, state, detail }: { label: string; state: ChecklistState; detail: string }) {
  return <div className={`ops-readiness-item ops-readiness-${state}`}><span className="ops-readiness-dot" /><div><strong>{label}</strong><p className="muted" style={{ margin: 0 }}>{detail}</p></div></div>;
}
function JobList({ jobs, loading }: { jobs: OpsJobRecord[]; loading: boolean }) {
  if (loading) return <p className="muted">Loading jobs...</p>;
  if (!jobs.length) return <p className="muted">No jobs yet.</p>;
  return <div style={{ display: "grid", gap: 8 }}>{jobs.slice(0, 5).map((job) => <article key={String(job.job_id)} className="ops-job-card"><strong>{String(job.job_id)} | {job.status ?? "unknown"}</strong></article>)}</div>;
}

function ManualTrainingModal({ onClose, onSubmit, submitting }: { onClose: () => void; submitting: boolean; onSubmit: (payload: RetrainingTriggerRequest) => Promise<void> }) {
  const [datasetMode, setDatasetMode] = useState("buffered");
  const [datasetValue, setDatasetValue] = useState("");
  const [checkpointMode, setCheckpointMode] = useState("latest");
  const [checkpointValue, setCheckpointValue] = useState("");
  const [preset, setPreset] = useState<PresetKey>("balanced");
  const [controls, setControls] = useState<ManualControlState>(presetDefaults.balanced);

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
      validation_split_pct: Number(controls.validationSplit)
    }
  });

  const datasetSnapshotRef = datasetMode === "custom" ? datasetValue : "buffered_internal_dataset";

  return (
    <div className="ops-modal-backdrop">
      <section className="panel ops-manual-modal" role="dialog" aria-modal="true" aria-label="Start Manual Training">
        <header>
          <h3>Start Manual Training</h3>
          <p className="muted" style={{ margin: "4px 0 0" }}>Manual training bypasses normal automatic scheduling. Use it only when you know why.</p>
        </header>
        <div className="ops-modal-grid">
          <div className="ops-modal-section">
            <h4>Dataset</h4>
            <label className="field"><span>Dataset source</span><select value={datasetMode} onChange={(e) => setDatasetMode(e.target.value)}><option value="buffered">Use buffered internal dataset default</option><option value="custom">Custom dataset path/link</option></select></label>
            {datasetMode === "custom" ? <label className="field"><span>Custom dataset path/link</span><input value={datasetValue} placeholder="/path/to/dataset, s3://bucket/dataset, https://..." onChange={(e) => setDatasetValue(e.target.value)} /><small className="muted">Custom dataset path is passed to retraining metadata/config if supported by the worker.</small></label> : null}
          </div>

          <div className="ops-modal-section">
            <h4>Base checkpoint</h4>
            <label className="field"><span>Base checkpoint</span><select value={checkpointMode} onChange={(e) => setCheckpointMode(e.target.value)}><option value="latest">Latest best model default</option><option value="active">Active production model</option><option value="custom">Custom checkpoint path</option></select></label>
            {checkpointMode === "custom" ? <label className="field"><span>Custom checkpoint path</span><input value={checkpointValue} placeholder="/path/to/checkpoint.pt or s3://bucket/model.pt" onChange={(e) => setCheckpointValue(e.target.value)} /></label> : null}
          </div>

          <div className="ops-modal-section">
            <h4>Training preset</h4>
            <label className="field">
              <span>Training preset</span>
              <select value={preset} onChange={(e) => setPreset(e.target.value as PresetKey)}>
                <option value="fast_refresh">Fast refresh</option>
                <option value="balanced">Balanced</option>
                <option value="high_accuracy">High accuracy</option>
                <option value="recovery">Recovery / stabilization</option>
              </select>
              <small className="muted">{presetDescriptions[preset]}</small>
            </label>
          </div>

          <div className="ops-modal-section">
            <h4>Training controls</h4>
            <div className="ops-control-grid">
              <label className="field">
                <span>Max runtime</span>
                <select value={controls.maxRuntime} onChange={(e) => setControls((p) => ({ ...p, maxRuntime: e.target.value as RuntimeOption }))}>
                  <option value="15m">15m</option><option value="30m">30m</option><option value="1h">1h</option><option value="2h">2h</option>
                </select>
              </label>
              <label className="field"><span>Batch size</span><select value={controls.batchSize} onChange={(e) => setControls((p) => ({ ...p, batchSize: e.target.value as ManualControlState["batchSize"] }))}><option value="8">8</option><option value="16">16</option><option value="32">32</option><option value="64">64</option></select></label>
              <label className="field"><span>Learning rate</span><select value={controls.learningRate} onChange={(e) => setControls((p) => ({ ...p, learningRate: e.target.value as LearningRateMode }))}><option value="conservative">Conservative</option><option value="default">Default</option><option value="aggressive">Aggressive</option></select></label>
              <label className="field"><span>Validation split</span><select value={controls.validationSplit} onChange={(e) => setControls((p) => ({ ...p, validationSplit: e.target.value as ManualControlState["validationSplit"] }))}><option value="10">10%</option><option value="15">15%</option><option value="20">20%</option></select></label>
              <label className="field"><span>Early stopping patience: {controls.earlyStoppingPatience}</span><input type="range" min={1} max={10} step={1} value={controls.earlyStoppingPatience} onChange={(e) => setControls((p) => ({ ...p, earlyStoppingPatience: Number(e.target.value) }))} /></label>
              <label className="field" style={{ gridColumn: "1 / -1" }}><span>Max epochs: {controls.maxEpochs}</span><input type="range" min={1} max={50} step={1} value={controls.maxEpochs} onChange={(e) => setControls((p) => ({ ...p, maxEpochs: Number(e.target.value) }))} /></label>
            </div>
            <small className="muted">Only backend-supported fields are enforced; additional controls are passed in run configuration metadata for workers that support them.</small>
          </div>
        </div>

        <div className="button-row">
          <button className="secondary-button" onClick={onClose} disabled={submitting}>Cancel</button>
          <button className="primary-button" onClick={() => void onSubmit({ manual_override: true, dataset_snapshot_ref: datasetSnapshotRef, run_config_ref: runConfigRef })} disabled={submitting}>{submitting ? "Starting training..." : "Start training"}</button>
        </div>
      </section>
    </div>
  );
}

function buildChecklist({ recommendation, runningJobs, status }: { recommendation: any; runningJobs: OpsJobRecord[]; status: any }): Array<{ label: string; state: ChecklistState; detail: string }> {
  const evidence = recommendation?.evidence ?? {};
  const readiness = status?.retraining_readiness ?? {};
  const workerAvailable = typeof readiness.worker_available === "boolean" ? readiness.worker_available : null;

  return [
    { label: "Enough new validated data", state: typeof evidence.new_samples === "number" ? "met" : "unknown", detail: typeof evidence.new_samples === "number" ? `${evidence.new_samples} samples reported` : "Not reported" },
    { label: "No retraining job currently running", state: runningJobs.length ? "not_met" : "met", detail: runningJobs.length ? `${runningJobs.length} job(s) queued or running` : "No queued/running jobs" },
    { label: "Worker available", state: workerAvailable === null ? "unknown" : workerAvailable ? "met" : "not_met", detail: workerAvailable === null ? "Not reported" : workerAvailable ? "Available" : "Not available" },
    { label: "Compute resources available", state: typeof readiness.resource_pressure === "boolean" ? (!readiness.resource_pressure ? "met" : "not_met") : "unknown", detail: typeof readiness.resource_pressure === "boolean" ? (readiness.resource_pressure ? "Resource pressure reported" : "No resource pressure reported") : "Not reported" },
    { label: "Dataset source available", state: recommendation?.reason ? "met" : "unknown", detail: recommendation?.reason ? "Derived from recommendation payload" : "Not reported" },
    { label: "Base checkpoint available", state: typeof readiness.base_checkpoint_available === "boolean" ? (readiness.base_checkpoint_available ? "met" : "not_met") : "unknown", detail: typeof readiness.base_checkpoint_available === "boolean" ? (readiness.base_checkpoint_available ? "Available" : "Not available") : "Not reported" },
    { label: "Retraining is recommended", state: recommendation?.should_retrain === true ? "met" : recommendation?.should_retrain === false ? "not_met" : "unknown", detail: recommendation?.reason ?? "Not reported" },
    { label: "Automatic training enabled", state: typeof readiness.retraining_enabled === "boolean" ? (readiness.retraining_enabled ? "met" : "not_met") : "unknown", detail: typeof readiness.retraining_enabled === "boolean" ? (readiness.retraining_enabled ? "Enabled" : "Disabled") : "Not reported" }
  ];
}
