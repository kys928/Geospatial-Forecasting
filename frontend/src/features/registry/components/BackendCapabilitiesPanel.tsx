import type { OpsRegistryResponse } from "../types/registry.types";

interface BackendCapabilitiesPanelProps {
  registry: OpsRegistryResponse | null;
}

export function BackendCapabilitiesPanel({ registry }: BackendCapabilitiesPanelProps) {
  return (
    <section className="panel">
      <h3>Registry summary</h3>
      <p><strong>active_model_id:</strong> {registry?.active_model_id ?? "n/a"}</p>
      <p><strong>previous_active_model_id:</strong> {registry?.previous_active_model_id ?? "n/a"}</p>
      <p><strong>revision:</strong> {String(registry?.revision ?? "n/a")}</p>
      <p><strong>next_event_index:</strong> {String(registry?.next_event_index ?? "n/a")}</p>
    </section>
  );
}
