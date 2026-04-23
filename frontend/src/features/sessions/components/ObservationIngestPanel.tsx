import { useState } from "react";

interface ObservationIngestPanelProps {
  disabled: boolean;
  onIngest: (observations: Array<Record<string, unknown>>) => Promise<void>;
}

export function ObservationIngestPanel({ disabled, onIngest }: ObservationIngestPanelProps) {
  const [text, setText] = useState('[{"timestamp":"2026-01-01T00:00:00Z","value":0.42}]');
  const [error, setError] = useState<string | null>(null);

  async function handleIngest() {
    setError(null);
    try {
      const observations = JSON.parse(text) as Array<Record<string, unknown>>;
      await onIngest(observations);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid observation payload");
    }
  }

  return (
    <section className="panel">
      <details className="advanced-section">
        <summary>Advanced observation ingest (JSON)</summary>
        <p className="muted">Manual observation payloads can be posted here for diagnostics or custom runs.</p>
        <textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} style={{ width: "100%" }} />
        {error ? <p className="muted">{error}</p> : null}
        <button className="secondary-button" disabled={disabled} onClick={() => void handleIngest()}>Post observations</button>
      </details>
    </section>
  );
}
