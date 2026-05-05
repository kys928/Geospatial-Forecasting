import { useEffect, useMemo, useState } from "react";
import { useRegistry } from "../../registry/hooks/useRegistry";
import type { RegistryModelRecord } from "../../registry/types/registry.types";
import { useOpsActions } from "../hooks/useOpsActions";

type RowActionMenu = string | null;
type DisplayModel = RegistryModelRecord & { isDemo?: boolean };

function formatCellValue(value: unknown, fallback = "Not reported") { if (value === null || value === undefined) return fallback; if (typeof value === "string" && value.trim().length === 0) return fallback; return String(value); }
function formatJson(value: unknown) { try { return JSON.stringify(value, null, 2); } catch { return "Not reported"; } }
function fieldRows(model: DisplayModel) { return [ { label: "Model ID", value: formatCellValue(model.model_id) }, { label: "Version", value: formatCellValue(model.version ?? model.contract_version) }, { label: "Status", value: formatCellValue(model.status) }, { label: "Approval status", value: formatCellValue(model.approval_status) }, { label: "Path", value: formatCellValue(model.path) }, { label: "Created time", value: formatCellValue(model.created_at) }, { label: "Updated time", value: formatCellValue(model.updated_at) }, { label: "Metrics / evidence", value: formatCellValue(model.metrics ?? model.checkpoint_metric ?? model.checkpoint_metric_name) }, { label: "Notes / metadata", value: formatCellValue(model.notes ?? model.metadata) } ]; }
const demoRow: DisplayModel = { model_id: "demo_convlstm_v0_1", status: "ready", approval_status: "approved", path: "/models/demo_convlstm_v0_1.pt", updated_at: "Demo", metadata: { source: "Demo row" }, isDemo: true };

export function OpsRegistryTab() {
  const registryState = useRegistry(10000); const [inspectModelId, setInspectModelId] = useState<string | null>(null); const [menuOpen, setMenuOpen] = useState<RowActionMenu>(null);
  const actions = useOpsActions(async () => { await registryState.refresh(false); });
  const models = registryState.registry?.models ?? [];
  const displayModels = models.length ? models : [demoRow];
  const inspectModel = useMemo(() => displayModels.find((model) => model.model_id === inspectModelId) ?? null, [displayModels, inspectModelId]);
  const activeModelId = registryState.registry?.active_model_id ?? null;

  useEffect(() => { const closeMenu = () => setMenuOpen(null); window.addEventListener("click", closeMenu); return () => window.removeEventListener("click", closeMenu); }, []);
  async function handleActivateModel(model: DisplayModel) { if (model.isDemo) return; const modelId = typeof model.model_id === "string" ? model.model_id : ""; if (!modelId || model.approval_status !== "approved") return; await actions.activateModel(modelId); setMenuOpen(null); }

  return <div style={{ display: "grid", gap: 12 }}>
    <section className="panel"><h3 style={{ marginBottom: 4 }}>Model Versions</h3><p className="muted" style={{ margin: 0 }}>Auto-refresh every 10 seconds.</p>{models.length === 0 ? <p className="muted" style={{ marginBottom: 0 }}>Showing a demo row because the registry is empty.</p> : null}</section>
    <section className="panel" style={{ overflow: "visible" }}>{registryState.loading ? <p className="muted">Loading model versions...</p> : null}{registryState.error ? <p className="failure-text">Unable to load model versions: {registryState.error}</p> : null}
      {!registryState.loading && !registryState.error ? <div style={{ overflowX: "auto" }}><table className="ops-model-table"><thead><tr><th>Actions</th><th>Model ID</th><th>Status</th><th>Approval</th><th>Path</th><th>Updated</th><th>Active</th></tr></thead><tbody>{displayModels.map((model) => { const id = formatCellValue(model.model_id, "Not reported"); const modelId = typeof model.model_id === "string" ? model.model_id : ""; const isDemo = Boolean((model as DisplayModel).isDemo); const canActivate = !isDemo && model.approval_status === "approved" && modelId.length > 0; const isActive = !isDemo && ((activeModelId && modelId === activeModelId) || model.status === "active"); return <tr key={`${id}-${formatCellValue(model.path, "na")}`}><td style={{ position: "relative" }}><button className="ops-actions-button" onClick={(event) => { event.stopPropagation(); setMenuOpen(menuOpen === modelId ? null : modelId); }} disabled={!modelId} aria-label="Model actions">⋮</button>{menuOpen === modelId ? <div className="ops-row-menu" onClick={(event) => event.stopPropagation()}><button onClick={() => void handleActivateModel(model)} disabled={!canActivate} title={isDemo ? "Demo row cannot be activated." : canActivate ? "Activate model" : "Only approved models can be activated"}>Activate model</button><button onClick={() => { setInspectModelId(modelId); setMenuOpen(null); }}>Inspect model</button><button disabled title="Delete is not wired yet">Delete model</button></div> : null}</td><td>{id}</td><td>{formatCellValue(model.status)}</td><td>{formatCellValue(model.approval_status)}</td><td>{formatCellValue(model.path)}</td><td>{formatCellValue(model.updated_at ?? model.created_at)}</td><td>{isActive ? "Yes" : "No"}</td></tr>; })}</tbody></table></div> : null}
    </section>
    {inspectModel ? <div className="ops-modal-backdrop" onClick={() => setInspectModelId(null)}><section className="panel ops-model-details-modal" role="dialog" aria-modal="true" aria-label="Model Details" onClick={(event) => event.stopPropagation()}><h3 style={{ margin: 0 }}>Model Details</h3>{(inspectModel as DisplayModel).isDemo ? <p className="muted" style={{ margin: 0 }}>This is a demo row for UI verification and is not sent to backend actions.</p> : null}<dl className="ops-model-details-list">{fieldRows(inspectModel).map((row) => <div key={row.label}><dt>{row.label}</dt><dd>{row.value}</dd></div>)}<div><dt>Active / current</dt><dd>{(inspectModel as DisplayModel).isDemo ? "No" : inspectModel.model_id === activeModelId || inspectModel.status === "active" ? "Yes" : "No"}</dd></div></dl><details><summary>Technical details</summary><pre style={{ margin: "8px 0 0", maxHeight: 260, overflow: "auto" }}>{formatJson(inspectModel)}</pre></details><div className="button-row" style={{ justifyContent: "flex-end" }}><button className="secondary-button" onClick={() => setInspectModelId(null)}>Close</button></div></section></div> : null}
  </div>;
}
