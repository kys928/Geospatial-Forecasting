import { useMemo, useState } from "react";
import { useRegistry } from "../../registry/hooks/useRegistry";
import { useOpsActions } from "../hooks/useOpsActions";
import { ActivationPanel } from "../../registry/components/ActivationPanel";
import { RollbackPanel } from "../../registry/components/RollbackPanel";
import { RegistryModelsTable } from "../../registry/components/RegistryModelsTable";
import { ModelVersionInspector } from "../../registry/components/ModelVersionInspector";

export function OpsRegistryTab() {
  const registryState = useRegistry(10000);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const actions = useOpsActions(async () => { await registryState.refresh(false); });
  const selectedModel = useMemo(() => registryState.registry?.models.find((model) => model.model_id === selectedModelId) ?? null, [registryState.registry, selectedModelId]);
  const approvedModelIds = useMemo(() => (registryState.registry?.models ?? []).filter((model) => model.approval_status === "approved" && typeof model.model_id === "string").map((model) => model.model_id as string), [registryState.registry]);

  return (<div className="workspace-grid" style={{ gridTemplateColumns: "1fr 2fr 1fr" }}><div className="workspace-column"><button className="primary-button" onClick={() => void registryState.refresh(false)}>{registryState.refreshing ? "Refreshing..." : "Refresh registry"}</button><p className="muted">Auto-refresh every 10 seconds.</p>
  <ActivationPanel approvedModelIds={approvedModelIds} disabled={actions.runningAction !== null} onActivate={async (modelId) => { await actions.activateModel(modelId); }} />
  <RollbackPanel disabled={actions.runningAction !== null} onRollback={async () => { await actions.rollbackModel(); }} /></div>
  <div className="workspace-column"><RegistryModelsTable models={registryState.registry?.models ?? []} selectedModelId={selectedModelId} onSelectModel={setSelectedModelId} onActivate={(modelId)=>void actions.activateModel(modelId)} /></div>
  <div className="workspace-column"><ModelVersionInspector model={selectedModel} /></div></div>);
}
