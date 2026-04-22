interface CandidateModelPanelProps {
  candidateModel: Record<string, unknown> | null;
}

export function CandidateModelPanel({ candidateModel }: CandidateModelPanelProps) {
  return (
    <section className="panel">
      <h3>Candidate model</h3>
      {candidateModel ? (
        <dl style={{ margin: 0, display: "grid", gap: 6 }}>
          <div><strong>ID:</strong> {String(candidateModel.model_id ?? "n/a")}</div>
          <div><strong>Status:</strong> {String(candidateModel.status ?? "n/a")}</div>
          <div><strong>Approval:</strong> {String(candidateModel.approval_status ?? "n/a")}</div>
          <details>
            <summary>Technical details</summary>
            <pre style={{ margin: 0, maxHeight: 200, overflow: "auto" }}>{JSON.stringify(candidateModel, null, 2)}</pre>
          </details>
        </dl>
      ) : (
        <p className="muted">No candidate model found.</p>
      )}
    </section>
  );
}
