import { useState } from "react";
import type { RetrainingTriggerRequest } from "../types/ops.types";

interface RetrainingTriggerFormProps {
  disabled: boolean;
  onSubmit: (payload: RetrainingTriggerRequest) => Promise<void>;
}

export function RetrainingTriggerForm({ disabled, onSubmit }: RetrainingTriggerFormProps) {
  const [manualOverride, setManualOverride] = useState(true);
  const [datasetSnapshotRef, setDatasetSnapshotRef] = useState("");
  const [runConfigRef, setRunConfigRef] = useState("");
  const [outputDir, setOutputDir] = useState("");

  return (
    <section className="panel">
      <h3>Trigger retraining</h3>
      <p className="muted">dataset_snapshot_ref and run_config_ref are string fields; JSON can be provided as encoded strings if your backend expects it.</p>
      <label><input type="checkbox" checked={manualOverride} onChange={(e) => setManualOverride(e.target.checked)} /> manual_override</label>
      <div className="field"><span>dataset_snapshot_ref</span><input value={datasetSnapshotRef} onChange={(e) => setDatasetSnapshotRef(e.target.value)} /></div>
      <div className="field"><span>run_config_ref</span><input value={runConfigRef} onChange={(e) => setRunConfigRef(e.target.value)} /></div>
      <div className="field"><span>output_dir</span><input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} /></div>
      <button
        className="primary-button"
        disabled={disabled}
        onClick={() => void onSubmit({ manual_override: manualOverride, dataset_snapshot_ref: datasetSnapshotRef || undefined, run_config_ref: runConfigRef || undefined, output_dir: outputDir || undefined })}
      >
        Submit retraining job
      </button>
    </section>
  );
}
