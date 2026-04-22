import { useState } from "react";

interface PredictionRequestFormProps {
  disabled: boolean;
  onPredict: (payload: Record<string, unknown>) => Promise<void>;
}

export function PredictionRequestForm({ disabled, onPredict }: PredictionRequestFormProps) {
  const [text, setText] = useState("{}");
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    setError(null);
    try {
      const payload = JSON.parse(text) as Record<string, unknown>;
      await onPredict(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid prediction request payload");
    }
  }

  return (
    <section className="panel">
      <h3>Prediction request</h3>
      <textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} style={{ width: "100%" }} />
      {error ? <p className="muted">{error}</p> : null}
      <button className="primary-button" disabled={disabled} onClick={() => void handlePredict()}>Run prediction</button>
    </section>
  );
}
