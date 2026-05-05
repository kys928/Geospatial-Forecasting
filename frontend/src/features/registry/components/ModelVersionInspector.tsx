import type { RegistryModelRecord } from "../types/registry.types";

interface ModelVersionInspectorProps {
  model: RegistryModelRecord | null;
}

export function ModelVersionInspector({ model }: ModelVersionInspectorProps) {
  return <section className="panel"><h3>Model inspector</h3>{!model?<p className="muted">Select a model to inspect its details.</p>:<pre style={{ margin: 0, maxHeight: 360, overflow: "auto" }}>{JSON.stringify(model, null, 2)}</pre>}</section>;
}
