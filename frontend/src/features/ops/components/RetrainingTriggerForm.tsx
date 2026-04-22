import { useMemo, useState } from "react";
import type { RetrainingTriggerRequest } from "../types/ops.types";

interface RetrainingTriggerFormProps {
  disabled: boolean;
  onSubmit: (payload: RetrainingTriggerRequest) => Promise<void>;
}

const LOCAL_TRAIN_PATH = "/workspace/datasets/online_learning_subset/train";
const LOCAL_VAL_PATH = "/workspace/datasets/online_learning_subset/val";
const DEFAULT_OUTPUT_DIR = "/workspace/ops_artifacts/online_subset_test";

const LOCAL_DATASET_PRESET = JSON.stringify(
  {
    train_data_path: LOCAL_TRAIN_PATH,
    val_data_path: LOCAL_VAL_PATH
  },
  null,
  2
);

const LOCAL_RUN_CONFIG_PRESET = JSON.stringify(
  {
    run_name: "online_subset_test",
    num_epochs: 1
  },
  null,
  2
);

export function RetrainingTriggerForm({ disabled, onSubmit }: RetrainingTriggerFormProps) {
  const [manualOverride, setManualOverride] = useState(true);
  const [datasetSnapshotRef, setDatasetSnapshotRef] = useState(LOCAL_DATASET_PRESET);
  const [runConfigRef, setRunConfigRef] = useState(LOCAL_RUN_CONFIG_PRESET);
  const [outputDir, setOutputDir] = useState(DEFAULT_OUTPUT_DIR);
  const [useGuidedMode, setUseGuidedMode] = useState(true);
  const [notice, setNotice] = useState<string | null>("Local subset preset loaded.");

  const payload = useMemo<RetrainingTriggerRequest>(() => ({
    manual_override: manualOverride,
    dataset_snapshot_ref: datasetSnapshotRef || undefined,
    run_config_ref: runConfigRef || undefined,
    output_dir: outputDir || undefined
  }), [manualOverride, datasetSnapshotRef, runConfigRef, outputDir]);

  async function handleCopyPayload() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setNotice("Copied payload JSON.");
    } catch {
      setNotice("Unable to copy payload in this browser context.");
    }
  }

  function applyLocalPreset() {
    setManualOverride(true);
    setDatasetSnapshotRef(LOCAL_DATASET_PRESET);
    setRunConfigRef(LOCAL_RUN_CONFIG_PRESET);
    setOutputDir(DEFAULT_OUTPUT_DIR);
    setNotice("Local subset preset loaded.");
  }

  function resetForRawMode() {
    setUseGuidedMode(false);
    setManualOverride(true);
    setDatasetSnapshotRef("");
    setRunConfigRef("");
    setOutputDir("");
    setNotice("Raw mode ready. Enter values exactly as backend expects.");
  }

  return (
    <section className="panel retraining-panel">
      <h3>Trigger retraining</h3>
      <p className="muted">
        Guided mode is for local pod testing. These paths are local filesystem locations, not remote registry references.
      </p>
      <p className="muted retraining-help">
        Sequence: load preset → inspect/edit payload → submit retraining job → refresh status, jobs, and events.
      </p>
      <p className="muted retraining-help">
        Worker must be running for queued jobs to progress. Auth may be required based on backend settings and configured ops token.
      </p>

      <div className="button-row mode-toggle-row">
        <button className={useGuidedMode ? "primary-button" : "secondary-button"} onClick={() => setUseGuidedMode(true)} type="button">Guided local subset</button>
        <button className={!useGuidedMode ? "primary-button" : "secondary-button"} onClick={() => setUseGuidedMode(false)} type="button">Raw payload mode</button>
      </div>

      {notice ? <p className="muted retraining-notice">{notice}</p> : null}

      {useGuidedMode ? (
        <div className="button-row retraining-guided-actions">
          <button className="secondary-button" type="button" onClick={applyLocalPreset}>Use local subset preset</button>
          <button className="secondary-button" type="button" onClick={handleCopyPayload}>Copy JSON payload</button>
          <button className="secondary-button" type="button" onClick={resetForRawMode}>Reset to raw mode</button>
        </div>
      ) : null}

      <label><input type="checkbox" checked={manualOverride} onChange={(e) => setManualOverride(e.target.checked)} /> manual_override</label>

      <div className="field">
        <span>dataset_snapshot_ref (string)</span>
        <textarea
          rows={useGuidedMode ? 5 : 3}
          value={datasetSnapshotRef}
          onChange={(e) => setDatasetSnapshotRef(e.target.value)}
          placeholder='Example: {"train_data_path":"...","val_data_path":"..."}'
        />
      </div>

      <div className="field">
        <span>run_config_ref (string)</span>
        <textarea
          rows={useGuidedMode ? 4 : 3}
          value={runConfigRef}
          onChange={(e) => setRunConfigRef(e.target.value)}
          placeholder='Example: {"run_name":"online_subset_test","num_epochs":1}'
        />
      </div>

      <div className="field">
        <span>output_dir (string)</span>
        <input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} placeholder={DEFAULT_OUTPUT_DIR} />
      </div>

      <div className="panel retraining-preview">
        <h4>Payload preview (submitted as-is)</h4>
        <pre>{JSON.stringify(payload, null, 2)}</pre>
      </div>

      <button
        className="primary-button"
        disabled={disabled}
        onClick={() => void onSubmit(payload)}
      >
        Submit retraining job
      </button>
    </section>
  );
}
