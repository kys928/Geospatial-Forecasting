interface ActiveModelPanelProps {
  activeModel: Record<string, unknown> | null;
}

export function ActiveModelPanel({ activeModel }: ActiveModelPanelProps) {
  return (
    <section className="panel">
      <h3>Active model</h3>
      {activeModel ? (
        <dl style={{ margin: 0, display: "grid", gap: 6 }}>
          <div><strong>ID:</strong> {String(activeModel.model_id ?? "n/a")}</div>
          <div><strong>Status:</strong> {String(activeModel.status ?? "n/a")}</div>
          <div><strong>Approval:</strong> {String(activeModel.approval_status ?? "n/a")}</div>
          <details>
            <summary>Technical details</summary>
            <pre style={{ margin: 0, maxHeight: 200, overflow: "auto" }}>{JSON.stringify(activeModel, null, 2)}</pre>
          </details>
        </dl>
      ) : (
        <p className="muted">No active model found.</p>
      )}
    </section>
  );
}
