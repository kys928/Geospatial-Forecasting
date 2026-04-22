interface ActivationPanelProps {
  approvedModelIds: string[];
  disabled: boolean;
  onActivate: (modelId: string) => Promise<void>;
}

export function ActivationPanel({ approvedModelIds, disabled, onActivate }: ActivationPanelProps) {
  return (
    <section className="panel">
      <h3>Activate approved model</h3>
      <div style={{ display: "grid", gap: 8 }}>
        {approvedModelIds.map((id) => (
          <button key={id} className="primary-button" disabled={disabled} onClick={() => void onActivate(id)}>{id}</button>
        ))}
        {approvedModelIds.length === 0 ? <p className="muted">No approved models available.</p> : null}
      </div>
    </section>
  );
}
