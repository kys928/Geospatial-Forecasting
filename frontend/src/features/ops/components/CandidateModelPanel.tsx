interface CandidateModelPanelProps {
  candidateModel: Record<string, unknown> | null;
}

export function CandidateModelPanel({ candidateModel }: CandidateModelPanelProps) {
  return (
    <section className="panel">
      <h3>Candidate model</h3>
      <pre style={{ margin: 0, maxHeight: 200, overflow: "auto" }}>{JSON.stringify(candidateModel, null, 2)}</pre>
    </section>
  );
}
