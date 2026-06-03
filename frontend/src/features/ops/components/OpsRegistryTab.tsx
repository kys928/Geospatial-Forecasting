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
type DisplayModel = RegistryModelRecord & { isDemo?: boolean };

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
  const rawTop = preferredTop + menuHeight > viewportHeight ? upwardTop : preferredTop;
  return {
    top: Math.max(gap, Math.min(rawTop, Math.max(gap, viewportHeight - menuHeight - gap))),
    left: Math.max(gap, Math.min(rect.right - menuWidth, maxLeft)),
  };
}
type DetailRow = { label: string; value: string };
type MetricRow = { label: string; value: string };

const ADAPTATION_CONTRACT_VERSION = "robust_convlstm_adaptation_v1";
const CORE_DETAIL_LABELS = new Set([
  "Model ID",
  "Status",
  "Approval",
  "Active / current",
  "Path",
  "Updated time",
]);
const MODEL_METRIC_KEYS = [
  "selection_score",
  "val_rollout_weighted_mse",
  "val_rollout_weighted_mse_t3",
  "val_rollout_weighted_mse_t4",
  "val_rollout_mae",
  "val_rollout_mass_abs_error",
  "val_rollout_peak_location_error",
  "plume_iou",
  "weighted_mse",
  "mae",
  "mass_abs_error",
  "peak_location_error",
  "checkpoint_metric",
  "checkpoint_metric_name",
  "best_validation_loss",
  "validation_loss",
  "val_loss",
  "train_loss",
  "training_loss",
  "best_overall_score",
  "final_score",
  "score",
];

const METRIC_LABELS: Record<string, string> = {
  selection_score: "Checkpoint selection score",
  val_rollout_weighted_mse: "Rollout weighted MSE",
  val_rollout_weighted_mse_t3: "Rollout weighted MSE T+3",
  val_rollout_weighted_mse_t4: "Rollout weighted MSE T+4",
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
  return METRIC_LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatNumber(value: number): string {
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(3);
  return Number(value.toPrecision(4)).toString();
}

const demoRow: DisplayModel = {
  model_id: "demo_convlstm_v0_1",
  status: "ready",
  approval_status: "approved",
  path: "/models/demo_convlstm_v0_1.pt",
  updated_at: "Demo",
  metadata: { source: "Demo row" },
  isDemo: true,
};

function formatCellValue(value: unknown, fallback = "Not reported") {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string" && value.trim().length === 0) return fallback;
  return String(value);
}

function formatStructuredValue(value: unknown): string {
  if (value === null || value === undefined || value === "")
    return "Not reported";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "string" || typeof value === "boolean") return String(value);
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

function isAdaptationRecord(model: RegistryModelRecord): boolean {
  return Boolean(
    (model.adaptation_run && typeof model.adaptation_run === "object") ||
    model.contract_version === ADAPTATION_CONTRACT_VERSION,
  );
}

function checkpointFileExists(model: RegistryModelRecord): boolean | null {
  const value = pickValue(model, [
    ["checkpoint_file_exists"],
    ["metadata", "checkpoint_file_exists"],
    ["adaptation_run", "checkpoint_file_exists"],
  ]);
  return typeof value === "boolean" ? value : null;
}

function isModelActive(
  model: DisplayModel,
  activeModelId: string | null,
): boolean {
  if (model.isDemo) return false;
  const modelId = typeof model.model_id === "string" ? model.model_id : "";
  return Boolean(
    (activeModelId && modelId === activeModelId) || model.status === "active",
  );
}

function canActivateModel(
  model: DisplayModel,
  activeModelId: string | null,
): boolean {
  if (model.isDemo || isModelActive(model, activeModelId)) return false;
  const modelId = typeof model.model_id === "string" ? model.model_id : "";
  if (!modelId) return false;
  const approval = String(model.approval_status ?? "").toLowerCase();
  const status = String(model.status ?? "").toLowerCase();
  return (
    approval === "approved" || status === "candidate" || status === "approved"
  );
}

function decisionPayload(comment: string): CandidateDecisionRequest {
  return { actor: "ops-ui", comment };
}

function coreRows(
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
    fieldRow("Path", model.path),
    fieldRow("Created time", model.created_at),
    fieldRow("Updated time", model.updated_at),
  ];
}

function formatResumeCheckpoint(value: unknown): unknown {
  const record = asRecord(value);
  if (!record) return value;
  const parts = ["checkpoint_path", "source", "resume_mode"]
    .map((key) => (record[key] ? `${key}: ${formatStructuredValue(record[key])}` : null))
    .filter(Boolean);
  return parts.length ? parts.join("; ") : undefined;
}

function adaptationRows(model: DisplayModel): DetailRow[] {
  if (!isAdaptationRecord(model)) return [];
  const exists = checkpointFileExists(model);
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
      "Best overall checkpoint",
      pickValue(model, [
        ["best_overall_checkpoint"],
        ["adaptation_run", "best_overall_checkpoint"],
        ["metadata", "best_overall_checkpoint"],
      ]),
    ),
    fieldRow(
      "Final checkpoint",
      pickValue(model, [
        ["final_checkpoint"],
        ["adaptation_run", "final_checkpoint"],
        ["metadata", "final_checkpoint"],
      ]),
    ),
    structuredRow(
      "Selected resume checkpoint",
      formatResumeCheckpoint(pickValue(model, [
        ["selected_resume_checkpoint"],
        ["adaptation_run", "selected_resume_checkpoint"],
        ["metadata", "selected_resume_checkpoint"],
      ])),
    ),
    structuredRow(
      "Dataset counts",
      pickValue(model, [
        ["dataset_counts"],
        ["adaptation_run", "dataset_counts"],
        ["metadata", "dataset_counts"],
        ["latest_readiness_snapshot", "summary"],
        ["adaptation_run", "readiness_snapshot", "summary"],
      ]),
    ),
    structuredRow(
      "Readiness snapshot summary",
      pickValue(model, [
        ["latest_readiness_snapshot", "summary"],
        ["readiness_snapshot", "summary"],
        ["adaptation_run", "readiness_snapshot", "summary"],
        ["metadata", "latest_readiness_snapshot"],
      ]),
    ),
    fieldRow(
      "Training summary status",
      pickValue(model, [
        ["training_summary_status"],
        ["training_summary", "status"],
        ["adaptation_run", "training_summary", "status"],
        ["adaptation_run", "status"],
      ]),
    ),
    structuredRow(
      "Best metrics / checkpoint metric",
      pickValue(model, [
        ["best_metrics"],
        ["metrics"],
        ["checkpoint_metric"],
        ["adaptation_run", "best_metrics"],
        ["metadata", "best_metrics"],
      ]),
    ),
    structuredRow(
      "Last adaptation promotion decision",
      model.last_adaptation_promotion_decision,
    ),
    structuredRow("Last promotion result", model.last_promotion_result),
    fieldRow(
      "Checkpoint file exists",
      exists === null ? undefined : exists ? "Yes" : "No",
    ),
  ];
}

function supplementalRows(model: DisplayModel): DetailRow[] {
  return [
    structuredRow(
      "Metrics / evidence",
      model.metrics ?? model.checkpoint_metric ?? model.checkpoint_metric_name,
    ),
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
      rows.push({ label: metricLabel(key), value: formatStructuredValue(value) });
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
  const displayModels = models.length ? models : [demoRow];
  const inspectModel = useMemo(
    () =>
      displayModels.find((model) => model.model_id === inspectModelId) ?? null,
    [displayModels, inspectModelId],
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
    const activation = isAdaptationRecord(model)
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
    if (
      !modelId ||
      !isAdaptationRecord(model) ||
      isModelActive(model, activeModelId)
    )
      return;
    const confirmed = window.confirm(
      `Delete checkpoint file for ${modelId}? The model record and history stay visible. Only the .pt file is removed. This cannot be undone.`,
    );
    if (!confirmed) return;
    await runAction(modelId, "Delete checkpoint file", () =>
      opsClient.deleteAdaptationCheckpointFile(
        modelId,
        decisionPayload("Checkpoint file deleted from Ops UI."),
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
        {models.length === 0 ? (
          <p className="muted" style={{ marginBottom: 0 }}>
            Showing a demo row because the registry is empty.
          </p>
        ) : null}
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

        {!registryState.loading && !registryState.error ? (
          <div style={{ overflowX: "auto" }}>
            <table className="ops-model-table">
              <thead>
                <tr>
                  <th>Model ID</th>
                  <th>Status</th>
                  <th>Approval</th>
                  <th>Path</th>
                  <th>Updated</th>
                  <th>Active</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {displayModels.map((model) => {
                  const id = formatCellValue(model.model_id, "Not reported");
                  const modelId =
                    typeof model.model_id === "string" ? model.model_id : "";
                  const isDemo = Boolean(model.isDemo);
                  const active = isModelActive(model, activeModelId);
                  const canActivate = canActivateModel(model, activeModelId);
                  const adaptation = isAdaptationRecord(model);
                  const checkpointExists = checkpointFileExists(model);
                  const canDeleteCheckpoint =
                    adaptation &&
                    !active &&
                    !isDemo &&
                    checkpointExists !== false &&
                    modelId.length > 0;
                  const busy =
                    runningAction?.startsWith(`${modelId}:`) ?? false;

                  return (
                    <tr key={`${id}-${formatCellValue(model.path, "na")}`}>
                      <td>{id}</td>
                      <td>{formatCellValue(model.status)}</td>
                      <td>{formatCellValue(model.approval_status)}</td>
                      <td title={formatCellValue(model.path)}>
                        {formatCellValue(model.path)}
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
                                    isDemo
                                      ? "Demo row cannot be activated."
                                      : active
                                        ? "This model is already active."
                                        : canActivate
                                          ? "Activate model using backend validation."
                                          : "Only approved or candidate models can be activated."
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
                                  Inspect model
                                </button>
                                {adaptation ? (
                                  <button
                                    onClick={() =>
                                      void handleDeleteCheckpointFile(model)
                                    }
                                    disabled={!canDeleteCheckpoint || busy}
                                    title={
                                      active
                                        ? "Active model checkpoint files cannot be deleted from this UI."
                                        : canDeleteCheckpoint
                                          ? "Delete checkpoint file only; registry metadata remains."
                                          : "Checkpoint deletion is not available for this record."
                                    }
                                  >
                                    {busy &&
                                    runningAction?.endsWith(
                                      "Delete checkpoint file",
                                    )
                                      ? "Deleting..."
                                      : "Delete checkpoint file"}
                                  </button>
                                ) : (
                                  <button
                                    disabled
                                    title="Full model deletion is not available from this UI."
                                  >
                                    Delete not available
                                  </button>
                                )}
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
            <h3 style={{ margin: 0 }}>Model Details</h3>
            {inspectModel.isDemo ? (
              <p className="muted" style={{ margin: 0 }}>
                This is a demo row for UI verification and is not sent to
                backend actions.
              </p>
            ) : null}
            <ModelDetailSection
              title="Core"
              rows={coreRows(inspectModel, activeModelId)}
              preserveLabels={CORE_DETAIL_LABELS}
            />
            {isAdaptationRecord(inspectModel) ? (
              <ModelDetailSection
                title="Adaptation"
                rows={adaptationRows(inspectModel)}
              />
            ) : null}
            <details className="advanced-section">
              <summary>Model Metrics</summary>
              {collectModelMetricRows(inspectModel).length ? (
                <dl className="ops-model-details-list">
                  {collectModelMetricRows(inspectModel).map((row) => (
                    <div key={row.label}>
                      <dt>{row.label}</dt>
                      <dd>{row.value}</dd>
                    </div>
                  ))}
                  {collectModelMetricRows(inspectModel).some((row) => row.label === "Checkpoint selection score") ? (
                    <div>
                      <dt>Checkpoint selection score note</dt>
                      <dd>Internal score used to choose the best checkpoint for this adaptation run.</dd>
                    </div>
                  ) : null}
                </dl>
              ) : (
                <p className="muted">No model metrics reported.</p>
              )}
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
