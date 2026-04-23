import type { RegistryModelRecord } from "../types/registry.types";

interface ModelVersionInspectorProps {
  model: RegistryModelRecord | null;
}

export function ModelVersionInspector({ model }: ModelVersionInspectorProps) {
  return (
    <section className="panel">
      <h3>Model inspector</h3>
      <pre style={{ margin: 0, maxHeight: 360, overflow: "auto" }}>{JSON.stringify(model, null, 2)}</pre>
    </section>
  );
}
