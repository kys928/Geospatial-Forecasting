interface ActiveModelPanelProps {
  activeModel: Record<string, unknown> | null;
}

export function ActiveModelPanel({ activeModel }: ActiveModelPanelProps) {
  return (
    <section className="panel">
      <h3>Active model</h3>
      <pre style={{ margin: 0, maxHeight: 200, overflow: "auto" }}>{JSON.stringify(activeModel, null, 2)}</pre>
    </section>
  );
}
