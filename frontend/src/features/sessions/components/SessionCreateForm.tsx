import { useState } from "react";
import type { CreateSessionRequest } from "../types/session.types";

interface SessionCreateFormProps {
  onCreate: (payload: CreateSessionRequest) => Promise<void>;
}

export function SessionCreateForm({ onCreate }: SessionCreateFormProps) {
  const [backendName, setBackendName] = useState("convlstm_online");
  const [modelName, setModelName] = useState("");
  const [metadataText, setMetadataText] = useState("{}");
  const [error, setError] = useState<string | null>(null);

  async function handleCreate() {
    setError(null);
    try {
      const metadata = metadataText.trim() ? (JSON.parse(metadataText) as Record<string, unknown>) : {};
      await onCreate({
        backend_name: backendName.trim(),
        model_name: modelName.trim() || undefined,
        metadata
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create session");
    }
  }

  return (
    <section className="panel">
      <h3>Create Session</h3>
      <div className="field"><span>backend_name</span><input value={backendName} onChange={(e) => setBackendName(e.target.value)} /></div>
      <div className="field"><span>model_name (optional)</span><input value={modelName} onChange={(e) => setModelName(e.target.value)} /></div>
      <div className="field"><span>metadata JSON</span><textarea value={metadataText} onChange={(e) => setMetadataText(e.target.value)} rows={4} /></div>
      {error ? <p className="muted">{error}</p> : null}
      <button className="primary-button" onClick={() => void handleCreate()}>Create session</button>
    </section>
  );
}
