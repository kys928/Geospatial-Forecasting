import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { opsClient } from "../api/opsClient";
import { useRegistry } from "../../registry/hooks/useRegistry";
import type { RegistryModelRecord } from "../../registry/types/registry.types";
import type { CandidateDecisionRequest } from "../types/ops.types";

type RowActionMenu = { id: string; top: number; left: number } | null;
type MenuAnchorRect = Pick<DOMRect, "top" | "right" | "bottom">;
type ViewportMenuPositionOptions = {
  rect: MenuAnchorRect;
  viewportWidth: number;
  viewportHeight: number;
  menuWidth?: number;
  menuHeight?: number;
  gap?: number;
};
type DisplayModel = RegistryModelRecord;

const ROW_MENU_WIDTH = 190;
const ROW_MENU_HEIGHT = 120;
const ROW_MENU_GAP = 8;

export function computeViewportMenuPosition({
  rect,
  viewportWidth,
  viewportHeight,
  menuWidth = ROW_MENU_WIDTH,
  menuHeight = ROW_MENU_HEIGHT,
  gap = ROW_MENU_GAP,
}: ViewportMenuPositionOptions): { top: number; left: number } {
  const maxLeft = Math.max(gap, viewportWidth - menuWidth - gap);
  const preferredTop = rect.bottom + gap;
  const upwardTop = rect.top - menuHeight - gap;
  const rawTop =
    preferredTop + menuHeight > viewportHeight ? upwardTop : preferredTop;
  return {
    top: Math.max(
      gap,
      Math.min(rawTop, Math.max(gap, viewportHeight - menuHeight - gap)),
    ),
    left: Math.max(gap, Math.min(rect.right - menuWidth, maxLeft)),
  };
}
type DetailRow = { label: string; value: string };
type MetricRow = { label: string; value: string };

const ADAPTATION_CONTRACT_VERSION = "robust_convlstm_adaptation_v1";
const MODEL_METRIC_KEYS = [
  "selection_score",
  "val_rollout_weighted_mse",
  "val_rollout_mae",
  "val_rollout_mass_abs_error",
  "val_rollout_peak_location_error",
  "plume_iou",
  "val_loss",
  "train_loss",
];

const METRIC_LABELS: Record<string, string> = {
  selection_score: "selection_score",
  val_rollout_weighted_mse: "val_rollout_weighted_mse",
  val_rollout_mae: "Rollout MAE",
  plume_iou: "Plume IoU",
  weighted_mse: "Weighted MSE",
  mae: "MAE",
  mass_abs_error: "Mass absolute error",
  peak_location_error: "Peak location error",
  val_loss: "Validation loss",
  train_loss: "Training loss",
};

function metricLabel(key: string): string {
  return (
    METRIC_LABELS[key] ??
    key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase())
  );
}

function formatNumber(value: number): string {
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(3);
  return Number(value.toPrecision(4)).toString();
}

function formatCellValue(value: unknown, fallback = "Not reported") {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string" && value.trim().length === 0) return fallback;
  return String(value);
}

function formatStructuredValue(value: unknown): string {
  if (value === null || value === undefined || value === "")
    return "Not reported";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "string" || typeof value === "boolean")
    return String(value);
  if (Array.isArray(value)) {
    const lines = value
      .map((item) => formatStructuredValue(item))
      .filter((item) => item !== "Not reported");
    return lines.length ? lines.join("; ") : "Not reported";
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .map(([key, val]) => `${key}: ${formatStructuredValue(val)}`)
      .filter((entry) => !entry.endsWith("Not reported"));
    return entries.length ? entries.join("; ") : "Not reported";
  }
  return "Not reported";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function pickValue(model: RegistryModelRecord, paths: string[][]): unknown {
  for (const path of paths) {
    let current: unknown = model;
    for (const key of path) {
      const record = asRecord(current);
      if (!record || !(key in record)) {
        current = undefined;
        break;
      }
      current = record[key];
    }
    if (current !== undefined && current !== null && current !== "")
      return current;
  }
  return undefined;
}

function fieldRow(label: string, value: unknown): DetailRow {
  return { label, value: formatCellValue(value) };
}

function structuredRow(label: string, value: unknown): DetailRow {
  return { label, value: formatStructuredValue(value) };
}

export function formatCheckpointMetric(value: unknown): string {
  const record = asRecord(value);
  if (record) {
    const name =
      typeof record.name === "string" && record.name.trim()
        ? record.name.trim()
        : undefined;
    const metricValue = record.value;
    if (
      name &&
      metricValue !== null &&
      metricValue !== undefined &&
      metricValue !== ""
    ) {
      return `${name} = ${formatStructuredValue(metricValue)}`;
    }
  }
  return formatStructuredValue(value);
}

function isAdaptationRecord(model: RegistryModelRecord): boolean {
  return Boolean(
    (model.adaptation_run && typeof model.adaptation_run === "object") ||
    model.contract_version === ADAPTATION_CONTRACT_VERSION,
  );
}

function checkpointFileExists(model: RegistryModelRecord): boolean | null {
  const deleted = pickValue(model, [
    ["checkpoint_file_deleted"],
    ["metadata", "checkpoint_file_deleted"],
    ["adaptation_run", "checkpoint_file_deleted"],
  ]);
  if (deleted === true) return false;

  const value = pickValue(model, [
    ["checkpoint_file_exists"],
    ["metadata", "checkpoint_file_exists"],
    ["adaptation_run", "checkpoint_file_exists"],
  ]);
  return typeof value === "boolean" ? value : null;
}

export function compactPathLabel(path: unknown): string {
  if (typeof path !== "string" || !path.trim()) return "Not reported";
  const cleaned = path.trim();
  const parts = cleaned.split(/[\\/]+/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : cleaned;
}

function selectedResumeCheckpoint(
  model: RegistryModelRecord,
): Record<string, unknown> | null {
  return asRecord(
    pickValue(model, [
      ["selected_resume_checkpoint"],
      ["adaptation_run", "selected_resume_checkpoint"],
      ["metadata", "selected_resume_checkpoint"],
    ]),
  );
}

export function modelParentLabel(model: RegistryModelRecord): string {
  const parentModelId = pickValue(model, [
    ["parent_active_model_id"],
    ["adaptation_run", "parent_active_model_id"],
    ["metadata", "parent_active_model_id"],
    ["trained_from_model_id"],
    ["metadata", "trained_from_model_id"],
  ]);
  if (typeof parentModelId === "string" && parentModelId.trim())
    return parentModelId.trim();

  const resume = selectedResumeCheckpoint(model);
  const source = resume?.source;
  if (typeof source === "string" && source.trim()) return source.trim();
  const checkpointPath = resume?.checkpoint_path ?? resume?.path;
  const compactPath = compactPathLabel(checkpointPath);
  return compactPath !== "Not reported" ? compactPath : "Not reported";
}

function selectionGateOutcome(
  model: RegistryModelRecord,
): Record<string, unknown> | null {
  return asRecord(
    pickValue(model, [
      ["selection_gate_outcome"],
      ["adaptation_run", "selection_gate_outcome"],
      ["metadata", "selection_gate_outcome"],
      ["last_adaptation_promotion_decision", "selection_gate_outcome"],
      ["last_promotion_result", "selection_gate_outcome"],
    ]),
  );
}

export function modelGateLabel(model: RegistryModelRecord): string {
  const outcome = selectionGateOutcome(model);
  if (!outcome) return "Not reported";
  const gatesEnabled =
    outcome.enabled === true || outcome.gates_enabled === true;
  if (outcome.stage3_rejected_by_gates === true)
    return "Stage 3 rejected; promoted Stage 2";
  if (gatesEnabled) return "Passed";
  if (outcome.stage3_rejected_by_gates === false) return "No gate issue";
  return "Not reported";
}

function checkpointFileDeletedMetadataPresent(
  model: RegistryModelRecord,
): boolean {
  return (
    pickValue(model, [
      ["checkpoint_file_deleted"],
      ["metadata", "checkpoint_file_deleted"],
      ["adaptation_run", "checkpoint_file_deleted"],
    ]) === true
  );
}

export function modelCheckpointHealthLabel(model: RegistryModelRecord): string {
  const exists = checkpointFileExists(model);
  if (exists === null) return "Not reported";
  return exists ? "Exists" : "Missing";
}

function isModelActive(
  model: DisplayModel,
  activeModelId: string | null,
): boolean {
  const modelId = typeof model.model_id === "string" ? model.model_id : "";
  return Boolean(
    (activeModelId && modelId === activeModelId) || model.status === "active",
  );
}

const ARCHIVED_ACTIVATION_APPROVAL_STATUSES = new Set([
  "approved_for_activation",
  "approved",
  "not_required",
]);

function canActivateModel(
  model: DisplayModel,
  activeModelId: string | null,
): boolean {
  if (isModelActive(model, activeModelId)) return false;
  const modelId = typeof model.model_id === "string" ? model.model_id : "";
  if (!modelId) return false;
  const approval = String(model.approval_status ?? "").toLowerCase();
  const status = String(model.status ?? "").toLowerCase();
  return (
    status === "candidate" ||
    status === "approved" ||
    (status === "archived" &&
      ARCHIVED_ACTIVATION_APPROVAL_STATUSES.has(approval))
  );
}

function checkpointDeleteDisabledReason(
  model: DisplayModel,
  activeModelId: string | null,
): string | null {
  const modelId = typeof model.model_id === "string" ? model.model_id : "";
  if (!modelId) return "Model ID is missing.";
  if (!isAdaptationRecord(model))
    return "Only eligible adaptation checkpoint records can be deleted.";
  if (isModelActive(model, activeModelId))
    return "Active model checkpoints cannot be deleted.";
  if (checkpointFileDeletedMetadataPresent(model))
    return "Checkpoint file is already deleted.";
  return null;
}

function canDeleteCheckpointFile(
  model: DisplayModel,
  activeModelId: string | null,
): boolean {
  return checkpointDeleteDisabledReason(model, activeModelId) === null;
}

function decisionPayload(comment: string): CandidateDecisionRequest {
  return { actor: "ops-ui", comment };
}

function trainingDataSummary(model: DisplayModel): string {
  const counts = asRecord(
    pickValue(model, [
      ["dataset_counts"],
      ["adaptation_run", "dataset_counts"],
      ["metadata", "dataset_counts"],
      ["latest_readiness_snapshot", "summary"],
      ["adaptation_run", "readiness_snapshot", "summary"],
    ]),
  );
  const train =
    counts?.train_total ??
    counts?.train ??
    counts?.train_count ??
    counts?.training ??
    counts?.training_count;
  const validation =
    counts?.val_total ??
    counts?.validation ??
    counts?.val ??
    counts?.validation_count ??
    counts?.val_count;
  if (train !== undefined && validation !== undefined)
    return `${formatStructuredValue(train)} train / ${formatStructuredValue(validation)} validation`;
  return "Not reported";
}

function bestScoreLabel(model: DisplayModel): string {
  const metric = pickValue(model, [
    ["checkpoint_metric"],
    ["metrics", "selection_score"],
    ["best_metrics", "selection_score"],
    ["metadata", "checkpoint_metric"],
    ["metadata", "metrics", "selection_score"],
    ["metadata", "best_metrics", "selection_score"],
    ["adaptation_run", "checkpoint_metric"],
    ["adaptation_run", "best_metrics", "selection_score"],
  ]);
  if (metric === undefined || metric === null || metric === "")
    return "Not reported";
  if (typeof metric === "number") {
    const name = pickValue(model, [
      ["checkpoint_metric_name"],
      ["metadata", "checkpoint_metric_name"],
      ["adaptation_run", "checkpoint_metric_name"],
    ]);
    return `${typeof name === "string" && name.trim() ? name.trim() : "selection_score"} = ${formatNumber(metric)}`;
  }
  return formatCheckpointMetric(metric);
}

function modelSummaryRows(
  model: DisplayModel,
  activeModelId: string | null,
): DetailRow[] {
  return [
    fieldRow("Status", model.status),
    fieldRow(
      "Active model",
      isModelActive(model, activeModelId) ? "Yes" : "No",
    ),
    fieldRow("Parent model", modelParentLabel(model)),
    fieldRow("Promotion result", modelGateLabel(model)),
    fieldRow(
      "Training status",
      pickValue(model, [
        ["training_summary_status"],
        ["training_summary", "status"],
        ["adaptation_run", "training_summary", "status"],
        ["adaptation_run", "status"],
        ["metadata", "training_summary", "status"],
      ]),
    ),
    fieldRow("Training data", trainingDataSummary(model)),
    fieldRow("Best score", bestScoreLabel(model)),
  ];
}

function resumeCheckpointSummary(model: DisplayModel): DetailRow[] {
  const resume = selectedResumeCheckpoint(model);
  return [
    fieldRow(
      "Run ID",
      pickValue(model, [
        ["run_id"],
        ["adaptation_run", "run_id"],
        ["metadata", "run_id"],
      ]),
    ),
    fieldRow(
      "Resume mode",
      resume?.resume_mode ??
        pickValue(model, [
          ["resume_mode"],
          ["adaptation_run", "resume_mode"],
          ["metadata", "resume_mode"],
        ]),
    ),
    fieldRow(
      "Resume source",
      resume?.source ??
        pickValue(model, [
          ["resume_source"],
          ["adaptation_run", "resume_source"],
          ["metadata", "resume_source"],
        ]),
    ),
    fieldRow(
      "Base checkpoint",
      compactPathLabel(
        resume?.checkpoint_path ??
          resume?.path ??
          pickValue(model, [
            ["base_checkpoint"],
            ["adaptation_run", "base_checkpoint"],
            ["metadata", "base_checkpoint"],
          ]),
      ),
    ),
    fieldRow(
      "Best checkpoint",
      compactPathLabel(
        pickValue(model, [
          ["best_overall_checkpoint"],
          ["adaptation_run", "best_overall_checkpoint"],
          ["metadata", "best_overall_checkpoint"],
        ]),
      ),
    ),
    fieldRow(
      "Final checkpoint",
      compactPathLabel(
        pickValue(model, [
          ["final_checkpoint"],
          ["adaptation_run", "final_checkpoint"],
          ["metadata", "final_checkpoint"],
        ]),
      ),
    ),
  ];
}

function technicalDetailRows(
  model: DisplayModel,
  activeModelId: string | null,
): DetailRow[] {
  return [
    fieldRow("Model ID", model.model_id),
    fieldRow(
      "Version / contract version",
      model.version ?? model.contract_version,
    ),
    fieldRow("Status", model.status),
    fieldRow("Approval", model.approval_status),
    fieldRow(
      "Active / current",
      isModelActive(model, activeModelId) ? "Yes" : "No",
    ),
    fieldRow("Raw path", model.path),
    structuredRow(
      "Metrics / evidence",
      model.metrics ?? model.checkpoint_metric ?? model.checkpoint_metric_name,
    ),
    fieldRow("Training log path", model.training_log_path),
    fieldRow(
      "Training log available",
      typeof model.training_log_available === "boolean"
        ? model.training_log_available
          ? "Yes"
          : "No"
        : undefined,
    ),
    fieldRow("Checkpoint status", modelCheckpointHealthLabel(model)),
    structuredRow("Checkpoint metric", model.checkpoint_metric),
    structuredRow(
      "Selected resume checkpoint",
      selectedResumeCheckpoint(model),
    ),
    structuredRow(
      "Dataset counts",
      pickValue(model, [
        ["dataset_counts"],
        ["adaptation_run", "dataset_counts"],
        ["metadata", "dataset_counts"],
      ]),
    ),
    fieldRow(
      "Output dir / result run dir",
      pickValue(model, [
        ["output_dir"],
        ["result_run_dir"],
        ["adaptation_run", "output_dir"],
        ["adaptation_run", "result_run_dir"],
        ["metadata", "output_dir"],
        ["metadata", "result_run_dir"],
      ]),
    ),
    fieldRow(
      "Best overall checkpoint path",
      pickValue(model, [
        ["best_overall_checkpoint"],
        ["adaptation_run", "best_overall_checkpoint"],
        ["metadata", "best_overall_checkpoint"],
      ]),
    ),
    fieldRow(
      "Final checkpoint path",
      pickValue(model, [
        ["final_checkpoint"],
        ["adaptation_run", "final_checkpoint"],
        ["metadata", "final_checkpoint"],
      ]),
    ),
    structuredRow(
      "Last adaptation promotion decision",
      model.last_adaptation_promotion_decision,
    ),
    structuredRow("Last promotion result", model.last_promotion_result),
    structuredRow("Notes", model.notes),
  ];
}

function collectModelMetricRows(model: DisplayModel): MetricRow[] {
  const sources = [
    model.metrics,
    model,
    pickValue(model, [["metadata", "metrics"]]),
    pickValue(model, [["metadata", "best_metrics"]]),
    pickValue(model, [["metadata", "promotion_metrics"]]),
    pickValue(model, [["metadata", "training_summary"]]),
    pickValue(model, [["metadata", "training_summary", "metrics"]]),
    model.adaptation_run,
    pickValue(model, [["adaptation_run", "training_summary"]]),
    pickValue(model, [["adaptation_run", "training_summary", "metrics"]]),
    model.last_adaptation_promotion_decision,
    model.last_promotion_result,
  ];
  const rows: MetricRow[] = [];
  const seen = new Set<string>();
  for (const key of MODEL_METRIC_KEYS) {
    for (const source of sources) {
      const record = asRecord(source);
      if (!record || !(key in record)) continue;
      const value = record[key];
      if (value === null || value === undefined || value === "") continue;
      if (seen.has(key)) break;
      rows.push({
        label: metricLabel(key),
        value: formatCheckpointMetric(value),
      });
      seen.add(key);
      break;
    }
  }
  return rows;
}

export function OpsRegistryTab() {
  const registryState = useRegistry(300000);
  const [inspectModelId, setInspectModelId] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState<RowActionMenu>(null);
  const [runningAction, setRunningAction] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const models = registryState.registry?.models ?? [];
  const inspectModel = useMemo(
    () => models.find((model) => model.model_id === inspectModelId) ?? null,
    [models, inspectModelId],
  );
  const activeModelId = registryState.registry?.active_model_id ?? null;

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setMenuOpen(null);
    }
    function closeOnOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node))
        setMenuOpen(null);
    }
    window.addEventListener("keydown", closeOnEscape);
    window.addEventListener("mousedown", closeOnOutside);
    return () => {
      window.removeEventListener("keydown", closeOnEscape);
      window.removeEventListener("mousedown", closeOnOutside);
    };
  }, []);

  async function runAction(
    modelId: string,
    label: string,
    action: () => Promise<unknown>,
  ) {
    setRunningAction(`${modelId}:${label}`);
    setActionError(null);
    setActionNotice(null);
    try {
      await action();
      setActionNotice(`${label} completed for ${modelId}.`);
      await registryState.refresh(false);
      setMenuOpen(null);
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : `${label} failed for ${modelId}.`,
      );
    } finally {
      setRunningAction(null);
    }
  }

  async function handleActivateModel(model: DisplayModel) {
    if (!canActivateModel(model, activeModelId)) return;
    const modelId = String(model.model_id);
    const status = String(model.status ?? "").toLowerCase();
    const activation =
      isAdaptationRecord(model) && status === "candidate"
        ? () =>
            opsClient.approveAdaptationCandidate(
              modelId,
              decisionPayload("Approved and activated from Ops UI."),
            )
        : () => opsClient.activateModel(modelId);
    await runAction(modelId, "Activate model", activation);
  }

  async function handleDeleteCheckpointFile(model: DisplayModel) {
    const modelId = typeof model.model_id === "string" ? model.model_id : "";
    const disabledReason = checkpointDeleteDisabledReason(model, activeModelId);
    if (disabledReason || !modelId) {
      setActionError(
        disabledReason ??
          "Checkpoint deletion is not available for this model.",
      );
      return;
    }
    const confirmed = window.confirm(
      `Delete checkpoint record for ${modelId}? The checkpoint file will be removed if present, and this model version will disappear from the table. Active model checkpoints cannot be deleted. This action cannot be undone.`,
    );
    if (!confirmed) return;
    await runAction(modelId, "Delete checkpoint record", () =>
      opsClient.deleteAdaptationCheckpointFile(
        modelId,
        decisionPayload(
          "Checkpoint file deleted from Ops UI row actions menu.",
        ),
      ),
    );
  }

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <section className="panel">
        <h3 style={{ marginBottom: 4 }}>Model Versions</h3>
        <p className="muted" style={{ margin: 0 }}>
          Auto-refresh every 5 minutes.
        </p>
      </section>

      <section className="panel" style={{ overflow: "visible" }}>
        {registryState.loading ? (
          <p className="muted">Loading model versions...</p>
        ) : null}
        {registryState.refreshing && !registryState.loading ? (
          <p className="muted">Refreshing model versions...</p>
        ) : null}
        {registryState.error ? (
          <p className="failure-text">
            Unable to load model versions: {registryState.error}
          </p>
        ) : null}
        {actionError ? <p className="failure-text">{actionError}</p> : null}
        {actionNotice ? <p className="muted">{actionNotice}</p> : null}

        {!registryState.loading &&
        !registryState.error &&
        models.length === 0 ? (
          <p className="muted">No model records are currently registered.</p>
        ) : null}
        {!registryState.loading && !registryState.error && models.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table className="ops-model-table">
              <thead>
                <tr>
                  <th>Model ID</th>
                  <th>Status</th>
                  <th>Approval</th>
                  <th>Parent / Trained from</th>
                  <th>Gate / Promotion</th>
                  <th>Checkpoint</th>
                  <th>Updated</th>
                  <th>Active</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => {
                  const id = formatCellValue(model.model_id, "Not reported");
                  const modelId =
                    typeof model.model_id === "string" ? model.model_id : "";
                  const active = isModelActive(model, activeModelId);
                  const canActivate = canActivateModel(model, activeModelId);
                  const busy =
                    runningAction?.startsWith(`${modelId}:`) ?? false;

                  return (
                    <tr key={`${id}-${formatCellValue(model.path, "na")}`}>
                      <td>{id}</td>
                      <td>{formatCellValue(model.status)}</td>
                      <td>{formatCellValue(model.approval_status)}</td>
                      <td title={modelParentLabel(model)}>
                        {modelParentLabel(model)}
                      </td>
                      <td>{modelGateLabel(model)}</td>
                      <td
                        title={
                          active
                            ? "Active model checkpoint deletion is not available from the row menu."
                            : undefined
                        }
                      >
                        {modelCheckpointHealthLabel(model)}
                      </td>
                      <td>
                        {formatCellValue(model.updated_at ?? model.created_at)}
                      </td>
                      <td>{active ? "Yes" : "No"}</td>
                      <td>
                        <button
                          className="ops-actions-button"
                          onClick={(event) => {
                            event.stopPropagation();
                            if (menuOpen?.id === modelId) {
                              setMenuOpen(null);
                              return;
                            }
                            const rect = (
                              event.currentTarget as HTMLButtonElement
                            ).getBoundingClientRect();
                            const position = computeViewportMenuPosition({
                              rect,
                              viewportWidth: window.innerWidth,
                              viewportHeight: window.innerHeight,
                            });
                            setMenuOpen({
                              id: modelId,
                              top: position.top,
                              left: position.left,
                            });
                          }}
                          disabled={!modelId}
                          aria-label="Model actions"
                        >
                          ⋮
                        </button>
                        {menuOpen?.id === modelId
                          ? createPortal(
                              <div
                                ref={menuRef}
                                className="ops-row-menu"
                                style={{
                                  position: "fixed",
                                  top: menuOpen.top,
                                  left: menuOpen.left,
                                }}
                                onClick={(event) => event.stopPropagation()}
                              >
                                <button
                                  onClick={() =>
                                    void handleActivateModel(model)
                                  }
                                  disabled={!canActivate || busy}
                                  title={
                                    active
                                      ? "This model is already active."
                                      : canActivate
                                        ? "Activate model using backend validation."
                                        : "Only candidate, approved, or archived previously-approved models can be activated."
                                  }
                                >
                                  {busy &&
                                  runningAction?.endsWith("Activate model")
                                    ? "Activating..."
                                    : "Activate model"}
                                </button>
                                <button
                                  onClick={() => {
                                    setInspectModelId(modelId);
                                    setMenuOpen(null);
                                  }}
                                >
                                  Inspect
                                </button>
                                <button
                                  onClick={() =>
                                    void handleDeleteCheckpointFile(model)
                                  }
                                  disabled={
                                    !canDeleteCheckpointFile(
                                      model,
                                      activeModelId,
                                    ) ||
                                    (busy &&
                                      runningAction?.endsWith(
                                        "Delete checkpoint record",
                                      ))
                                  }
                                  title={
                                    checkpointDeleteDisabledReason(
                                      model,
                                      activeModelId,
                                    ) ??
                                    (checkpointFileExists(model) === false
                                      ? "Checkpoint file is already missing; backend will remove the registry record."
                                      : checkpointFileExists(model) === null
                                        ? "Checkpoint status is not reported; backend will verify before deletion."
                                        : "Delete checkpoint record; backend will remove the file if present.")
                                  }
                                >
                                  {busy &&
                                  runningAction?.endsWith(
                                    "Delete checkpoint record",
                                  )
                                    ? "Deleting..."
                                    : "Delete"}
                                </button>
                              </div>,
                              document.body,
                            )
                          : null}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      {inspectModel ? (
        <div
          className="ops-modal-backdrop"
          onClick={() => setInspectModelId(null)}
        >
          <section
            className="panel ops-model-details-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Model Details"
            onClick={(event) => event.stopPropagation()}
          >
            <div>
              <h3 style={{ margin: 0 }}>Model inspection</h3>
              <p className="muted" style={{ margin: "4px 0 0" }}>
                {formatCellValue(inspectModel.model_id)}
              </p>
            </div>
            {actionError ? <p className="failure-text">{actionError}</p> : null}
            {actionNotice ? <p className="muted">{actionNotice}</p> : null}
            <ModelDetailSection
              title="Model summary"
              rows={modelSummaryRows(inspectModel, activeModelId)}
            />
            {collectModelMetricRows(inspectModel).length > 1 ? (
              <ModelDetailSection
                title="Evaluation metrics"
                rows={collectModelMetricRows(inspectModel)}
              />
            ) : null}
            <ModelDetailSection
              title="Training provenance"
              rows={resumeCheckpointSummary(inspectModel)}
            />
            <details className="advanced-section">
              <summary>Technical details</summary>
              <ModelDetailSection
                title="Raw registry details"
                rows={technicalDetailRows(inspectModel, activeModelId)}
              />
            </details>
            <div className="button-row" style={{ justifyContent: "flex-end" }}>
              <button
                className="secondary-button"
                onClick={() => setInspectModelId(null)}
              >
                Close
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

function ModelDetailSection({
  title,
  rows,
  preserveLabels,
}: {
  title: string;
  rows: DetailRow[];
  preserveLabels?: Set<string>;
}) {
  const visibleRows = rows.filter(
    (row) => row.value !== "Not reported" || preserveLabels?.has(row.label),
  );
  if (!visibleRows.length) return null;
  return (
    <div className="ops-modal-section">
      <h4>{title}</h4>
      <dl className="ops-model-details-list">
        {visibleRows.map((row) => (
          <div key={row.label}>
            <dt>{row.label}</dt>
            <dd>{row.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
