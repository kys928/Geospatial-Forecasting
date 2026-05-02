import { useState } from "react";

interface PredictionRequestFormProps {
  disabled: boolean;
  onPredict: (payload: Record<string, unknown>) => Promise<void>;
  showAdvancedOptions?: boolean;
}

export function PredictionRequestForm({ disabled, onPredict, showAdvancedOptions = true }: PredictionRequestFormProps) {
  const [text, setText] = useState("{}");
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    setError(null);
    try {
      const trimmed = text.trim();
      const payload = trimmed ? (JSON.parse(trimmed) as Record<string, unknown>) : {};
      await onPredict(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid forecast request payload");
    }
  }

  return (
    <section className="panel">
      <h3>Run forecast</h3>
      <p className="muted">Generate the latest forecast for the selected session.</p>
      <button className="primary-button" disabled={disabled} onClick={() => void handlePredict()}>Run forecast</button>
      {showAdvancedOptions ? (
        <details className="advanced-section">
          <summary>Advanced forecast options (JSON)</summary>
          <textarea
            rows={6}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="{}"
            style={{ width: "100%" }}
          />
        </details>
      ) : null}
      {error ? <p className="muted">{error}</p> : null}
    </section>
  );
}
