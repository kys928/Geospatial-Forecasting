import { useMemo, useState } from "react";
import { opsClient } from "../api/opsClient";
import { useOpsJobs } from "../hooks/useOpsJobs";
import { useOpsStatus } from "../hooks/useOpsStatus";
import { useRetrainingRecommendation } from "../hooks/useRetrainingRecommendation";
import { useModelCandidateContext } from "../hooks/useModelCandidateContext";
import type { OpsJobRecord, RetrainingTriggerRequest } from "../types/ops.types";

const presetDescriptions: Record<string, string> = {
  fast_refresh: "Quick lightweight update for small new data batches.",
  balanced: "Default training profile with a practical speed/quality tradeoff.",
  high_accuracy: "Longer run focused on better validation quality.",
  recovery: "Conservative run after failures, drift, or unstable model behavior."
};

export function OpsTrainingTab() {
  const jobsState = useOpsJobs(); const statusState = useOpsStatus(); const recommendationState = useRetrainingRecommendation(); const candidateState = useModelCandidateContext();
  const [manualOpen,setManualOpen]=useState(false); const [manualSubmitting,setManualSubmitting]=useState(false); const [manualNotice,setManualNotice]=useState<string|null>(null);
  const jobs=jobsState.jobs?.jobs??[]; const runningJobs=jobs.filter((j)=>j.status==="queued"||j.status==="running"); const recommendation=recommendationState.recommendation;
  const checklist = useMemo(() => buildChecklist({ recommendation, runningJobs, status: statusState.status }), [recommendation, runningJobs, statusState.status]);
  async function refreshAll(){await Promise.all([jobsState.refresh(),statusState.refresh(),recommendationState.refresh(),candidateState.refresh()]);}
  async function handleManualStart(payload: RetrainingTriggerRequest){setManualSubmitting(true);setManualNotice(null);try{const r=await opsClient.triggerRetraining(payload);setManualNotice(r.submitted?"Manual training job submitted.":"Submission was not accepted by backend policy.");setManualOpen(false);await refreshAll();}catch(e){setManualNotice(e instanceof Error?e.message:"Unable to submit manual training job.");}finally{setManualSubmitting(false);}}
  return <div style={{display:"grid",gap:12}}><section className="panel"><h3>Automatic Training Status</h3><div className="button-row"><button className="secondary-button" onClick={()=>void refreshAll()} disabled={manualSubmitting}>Refresh training status</button></div></section>
  <section className="panel"><h3>Training Readiness Checklist</h3>{checklist.map((item)=><ReadinessItem key={item.label} label={item.label} state={item.state} detail={item.detail}/>)}</section>
  <section className="panel"><h3>Recent / Running Training Jobs</h3><JobList jobs={jobs} loading={jobsState.loading}/></section>
  <section className="panel"><h3>Manual Training</h3><button className="secondary-button" onClick={()=>setManualOpen(true)}>Start manual training</button>{manualNotice?<p className="muted">{manualNotice}</p>:null}</section>
  {manualOpen?<ManualTrainingModal onClose={()=>setManualOpen(false)} onSubmit={handleManualStart} submitting={manualSubmitting}/>:null}</div>;
}

type ChecklistState="met"|"not_met"|"unknown"|"checking";
function ReadinessItem({label,state,detail}:{label:string;state:ChecklistState;detail:string}){return <div className={`ops-readiness-item ops-readiness-${state}`}><span className="ops-readiness-dot"/><div><strong>{label}</strong><p className="muted" style={{margin:0}}>{detail}</p></div></div>;}
function JobList({jobs,loading}:{jobs:OpsJobRecord[];loading:boolean}){if(loading)return <p className="muted">Loading jobs...</p>; if(!jobs.length) return <p className="muted">No jobs yet.</p>; return <div style={{display:"grid",gap:8}}>{jobs.slice(0,5).map((job)=><article key={String(job.job_id)} className="ops-job-card"><strong>{String(job.job_id)} | {job.status??"unknown"}</strong></article>)}</div>;}

function ManualTrainingModal({ onClose, onSubmit, submitting }: { onClose: () => void; submitting: boolean; onSubmit: (payload: RetrainingTriggerRequest) => Promise<void> }) { const [datasetMode,setDatasetMode]=useState("buffered"); const [datasetValue,setDatasetValue]=useState(""); const [checkpointMode,setCheckpointMode]=useState("latest"); const [checkpointValue,setCheckpointValue]=useState(""); const [preset,setPreset]=useState("balanced"); const [maxEpochs,setMaxEpochs]=useState(8); const [maxRuntime,setMaxRuntime]=useState("30m");
const runConfigRef=JSON.stringify({preset,max_epochs:maxEpochs,max_runtime:maxRuntime,checkpoint_mode:checkpointMode,checkpoint_ref:checkpointValue||undefined}); const datasetSnapshotRef=datasetMode==="custom"?datasetValue:"buffered_internal_dataset";
return <div className="ops-modal-backdrop"><section className="panel ops-manual-modal"><h3>Start Manual Training</h3><div className="ops-modal-grid"><label className="field"><span>Dataset source</span><select value={datasetMode} onChange={(e)=>setDatasetMode(e.target.value)}><option value="buffered">Use buffered internal dataset</option><option value="custom">Custom dataset</option></select></label><label className="field"><span>Base checkpoint</span><select value={checkpointMode} onChange={(e)=>setCheckpointMode(e.target.value)}><option value="latest">Latest best model</option><option value="active">Active production model</option><option value="custom">Custom checkpoint</option></select></label>{datasetMode==="custom"?<label className="field"><span>Custom dataset path/link</span><input value={datasetValue} onChange={(e)=>setDatasetValue(e.target.value)} /></label>:null}{checkpointMode==="custom"?<label className="field"><span>Custom checkpoint path/link</span><input value={checkpointValue} onChange={(e)=>setCheckpointValue(e.target.value)} /></label>:null}<label className="field"><span>Training preset</span><select value={preset} onChange={(e)=>setPreset(e.target.value)}><option value="fast_refresh">Fast refresh</option><option value="balanced">Balanced</option><option value="high_accuracy">High accuracy</option><option value="recovery">Recovery / stabilization</option></select><small className="muted">{presetDescriptions[preset]}</small></label><label className="field"><span>Max runtime</span><input type="range" min={0} max={3} step={1} value={["15m","30m","1h","2h"].indexOf(maxRuntime)} onChange={(e)=>setMaxRuntime(["15m","30m","1h","2h"][Number(e.target.value)])} /><small>{maxRuntime}</small></label><label className="field" style={{gridColumn:"1 / -1"}}><span>Max epochs: {maxEpochs}</span><input type="range" min={1} max={40} step={1} value={maxEpochs} onChange={(e)=>setMaxEpochs(Number(e.target.value))}/></label></div><div className="button-row"><button className="secondary-button" onClick={onClose} disabled={submitting}>Cancel</button><button className="primary-button" onClick={()=>void onSubmit({manual_override:true,dataset_snapshot_ref:datasetSnapshotRef,run_config_ref:runConfigRef})} disabled={submitting}>{submitting?"Starting training...":"Start training"}</button></div></section></div>; }

function buildChecklist({ recommendation, runningJobs, status }: { recommendation: any; runningJobs: OpsJobRecord[]; status: any }): Array<{label:string;state:ChecklistState;detail:string}> {
  const evidence = recommendation?.evidence ?? {};
  const readiness = status?.retraining_readiness ?? {};
  const workerAvailable = typeof readiness.worker_available === "boolean" ? readiness.worker_available : null;
  return [
    { label: "Enough new validated data", state: typeof evidence.new_samples === "number" ? "met" : "unknown", detail: typeof evidence.new_samples === "number" ? `${evidence.new_samples} samples reported` : "Not reported" },
    { label: "Minimum time since last training passed", state: typeof evidence.seconds_since_last_training === "number" ? "met" : "unknown", detail: typeof evidence.seconds_since_last_training === "number" ? `${Math.round(evidence.seconds_since_last_training / 3600)}h since last run` : "Not reported" },
    { label: "No retraining job currently running", state: runningJobs.length ? "not_met" : "met", detail: runningJobs.length ? `${runningJobs.length} job(s) queued or running` : "No queued/running jobs" },
    { label: "Worker available", state: workerAvailable === null ? "unknown" : workerAvailable ? "met" : "not_met", detail: workerAvailable === null ? "Not reported" : workerAvailable ? "Available" : "Not available" },
    { label: "Compute resources available", state: typeof readiness.resource_pressure === "boolean" ? (!readiness.resource_pressure ? "met" : "not_met") : "unknown", detail: "Not reported" },
    { label: "Dataset source available", state: recommendation?.reason ? "met" : "unknown", detail: "Derived from recommendation payload" },
    { label: "Base checkpoint available", state: "unknown", detail: "Not reported" },
    { label: "Data quality acceptable", state: typeof evidence.data_quality_ok === "boolean" ? (evidence.data_quality_ok ? "met" : "not_met") : "unknown", detail: "Not reported" },
    { label: "Retraining is recommended", state: recommendation?.should_retrain === true ? "met" : recommendation?.should_retrain === false ? "not_met" : "unknown", detail: recommendation?.reason ?? "Not reported" },
    { label: "Automatic training enabled", state: typeof readiness.retraining_enabled === "boolean" ? (readiness.retraining_enabled ? "met" : "not_met") : "unknown", detail: "Not reported" }
  ];
}
