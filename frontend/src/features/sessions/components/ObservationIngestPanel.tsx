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
      <h3>Observation ingest</h3>
      <textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} style={{ width: "100%" }} />
      {error ? <p className="muted">{error}</p> : null}
      <button className="primary-button" disabled={disabled} onClick={() => void handleIngest()}>Post observations</button>
    </section>
  );
}
