import { useState } from "react";
import type { CreateSessionRequest } from "../types/session.types";

interface SessionCreateFormProps {
  onCreate: (payload: CreateSessionRequest) => Promise<void>;
}

const DEFAULT_BACKEND_NAME = "convlstm_online";

export function SessionCreateForm({ onCreate }: SessionCreateFormProps) {
  const [error, setError] = useState<string | null>(null);

  async function handleCreate() {
    setError(null);
    try {
      await onCreate({
        backend_name: DEFAULT_BACKEND_NAME
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create session");
    }
  }

  return (
    <section className="panel">
      <h3>Create session</h3>
      <p className="muted">Creates a new forecasting session using the default online backend.</p>
      {error ? <p className="muted">{error}</p> : null}
      <button className="primary-button" onClick={() => void handleCreate()}>Create session</button>
    </section>
  );
}
