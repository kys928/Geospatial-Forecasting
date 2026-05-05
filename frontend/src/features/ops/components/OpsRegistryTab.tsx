import { useMemo, useState } from "react";
import { useRegistry } from "../../registry/hooks/useRegistry";
import type { RegistryModelRecord } from "../../registry/types/registry.types";
import { useOpsActions } from "../hooks/useOpsActions";

type RowActionMenu = string | null;

function formatCellValue(value: unknown, fallback = "Not reported") {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string" && value.trim().length === 0) return fallback;
  return String(value);
}

export function OpsRegistryTab() {
  const registryState = useRegistry(10000);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState<RowActionMenu>(null);
  const actions = useOpsActions(async () => {
    await registryState.refresh(false);
  });

  const models = registryState.registry?.models ?? [];
  const selectedModel = useMemo(() => models.find((model) => model.model_id === selectedModelId) ?? null, [models, selectedModelId]);
  const approvedModelIds = useMemo(
    () => models.filter((model) => model.approval_status === "approved" && typeof model.model_id === "string").map((model) => model.model_id as string),
    [models]
  );

  const activeModelId = registryState.registry?.active_model_id ?? null;

  async function handleActivateModel(model: RegistryModelRecord) {
    const modelId = typeof model.model_id === "string" ? model.model_id : "";
    if (!modelId) return;
    if (model.approval_status !== "approved") return;
    await actions.activateModel(modelId);
    setMenuOpen(null);
  }

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <section className="panel">
        <div className="button-row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h3 style={{ marginBottom: 4 }}>Model Versions</h3>
            <p className="muted" style={{ margin: 0 }}>Auto-refresh every 10 seconds.</p>
          </div>
          <div className="button-row">
            <button className="secondary-button" onClick={() => void registryState.refresh(false)} disabled={actions.runningAction !== null || registryState.refreshing}>
              {registryState.refreshing ? "Refreshing..." : "Refresh registry"}
            </button>
            <button className="secondary-button" onClick={() => void actions.rollbackModel()} disabled={actions.runningAction !== null || approvedModelIds.length === 0}>
              Roll back active model
            </button>
          </div>
        </div>
        {approvedModelIds.length === 0 ? <p className="muted" style={{ marginBottom: 0 }}>No approved models are available for activation or rollback.</p> : null}
      </section>

      <section className="panel" style={{ overflowX: "auto" }}>
        {registryState.loading ? <p className="muted">Loading model versions...</p> : null}
        {registryState.error ? <p className="failure-text">Unable to load model versions: {registryState.error}</p> : null}
        {!registryState.loading && !registryState.error && models.length === 0 ? <p className="muted">No model versions are registered yet.</p> : null}
        {!registryState.loading && !registryState.error && models.length > 0 ? (
          <table className="ops-model-table">
            <thead>
              <tr>
                <th>Actions</th><th>Model ID</th><th>Status</th><th>Approval</th><th>Path</th><th>Updated / created</th><th>Active</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => {
                const id = formatCellValue(model.model_id, "Not reported");
                const modelId = typeof model.model_id === "string" ? model.model_id : "";
                const isSelected = selectedModelId === modelId;
                const canActivate = model.approval_status === "approved" && modelId.length > 0;
                const isActive = (activeModelId && modelId === activeModelId) || model.status === "active";
                return (
                  <tr key={`${id}-${formatCellValue(model.path, "na")}`} className={isSelected ? "selected" : ""}>
                    <td style={{ position: "relative" }}>
                      <button className="secondary-button" onClick={() => setMenuOpen(menuOpen === modelId ? null : modelId)} disabled={!modelId} aria-label="Model actions">⋮</button>
                      {menuOpen === modelId ? (
                        <div className="ops-row-menu">
                          <button onClick={() => void handleActivateModel(model)} disabled={!canActivate} title={canActivate ? "Activate model" : "Only approved models can be activated"}>Activate model</button>
                          <button onClick={() => { setSelectedModelId(modelId); setMenuOpen(null); }}>Inspect model</button>
                          <button disabled title="Delete model is not supported by the backend endpoint">Delete model</button>
                        </div>
                      ) : null}
                    </td>
                    <td>{id}</td>
                    <td>{formatCellValue(model.status)}</td>
                    <td>{formatCellValue(model.approval_status)}</td>
                    <td>{formatCellValue(model.path)}</td>
                    <td>{formatCellValue(model.updated_at ?? model.created_at)}</td>
                    <td>{isActive ? "Current" : "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : null}
      </section>

      <section className="panel">
        <h3>Model inspector</h3>
        {!selectedModel ? <p className="muted">Select a model to inspect its details.</p> : <pre style={{ margin: 0, maxHeight: 360, overflow: "auto" }}>{JSON.stringify(selectedModel, null, 2)}</pre>}
      </section>
    </div>
  );
}
