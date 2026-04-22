import { useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { useRegistry } from "../features/registry/hooks/useRegistry";
import { useOpsActions } from "../features/ops/hooks/useOpsActions";
import { RegistryModelsTable } from "../features/registry/components/RegistryModelsTable";
import { ModelVersionInspector } from "../features/registry/components/ModelVersionInspector";
import { ActivationPanel } from "../features/registry/components/ActivationPanel";
import { RollbackPanel } from "../features/registry/components/RollbackPanel";
import { RegistryEventStream } from "../features/registry/components/RegistryEventStream";
import { BackendCapabilitiesPanel } from "../features/registry/components/BackendCapabilitiesPanel";

export function RegistryPage() {
  const { registry, loading, error, refresh } = useRegistry();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const actions = useOpsActions(refresh);

  const selectedModel = useMemo(
    () => registry?.models.find((model) => model.model_id === selectedModelId) ?? null,
    [registry, selectedModelId]
  );

  const approvedModelIds = useMemo(
    () =>
      (registry?.models ?? [])
        .filter((model) => model.approval_status === "approved" && typeof model.model_id === "string")
        .map((model) => model.model_id as string),
    [registry]
  );

  return (
    <AppShell
      title="Registry workspace"
      subtitle="Inspect model versions, approvals, activation, and rollback actions."
      statusText={error ?? actions.error ?? "Ready"}
      metaItems={[{ label: registry?.active_model_id ? `Active: ${registry.active_model_id}` : "Model registry" }]}
    >
      <div className="workspace-grid">
        <div className="workspace-column">
          <BackendCapabilitiesPanel registry={registry} />
          <button className="primary-button" onClick={() => void refresh()}>{loading ? "Refreshing..." : "Refresh registry"}</button>
          <ActivationPanel
            approvedModelIds={approvedModelIds}
            disabled={actions.runningAction !== null}
            onActivate={async (modelId) => {
              await actions.activateModel(modelId);
            }}
          />
          <RollbackPanel
            disabled={actions.runningAction !== null}
            onRollback={async () => {
              await actions.rollbackModel();
            }}
          />
        </div>

        <div className="workspace-column">
          <RegistryModelsTable models={registry?.models ?? []} selectedModelId={selectedModelId} onSelectModel={setSelectedModelId} />
          <RegistryEventStream events={registry?.events ?? []} approvalAudit={registry?.approval_audit ?? []} />
        </div>

        <div className="workspace-column">
          <ModelVersionInspector model={selectedModel} />
          <section className="panel">
            <h3>Action output</h3>
            <pre style={{ margin: 0, maxHeight: 360, overflow: "auto" }}>{JSON.stringify(actions.lastResult, null, 2)}</pre>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
