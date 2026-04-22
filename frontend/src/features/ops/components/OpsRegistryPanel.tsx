import { useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsRegistryResponse } from "../../registry/types/registry.types";
import { ActiveModelPanel } from "./ActiveModelPanel";
import { CandidateModelPanel } from "./CandidateModelPanel";

interface OpsRegistryPanelProps {
  enabled: boolean;
}

export function OpsRegistryPanel({ enabled }: OpsRegistryPanelProps) {
  const [registry, setRegistry] = useState<OpsRegistryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    async function fetchRegistry() {
      setLoading(true);
      setError(null);
      try {
        setRegistry(await opsClient.getRegistry());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load registry.");
      } finally {
        setLoading(false);
      }
    }

    void fetchRegistry();
  }, [enabled]);

  return (
    <div className="workspace-grid">
      <div className="workspace-column">
        <ActiveModelPanel activeModel={registry?.models.find((model) => model.model_id === registry.active_model_id) ?? null} />
        <CandidateModelPanel
          candidateModel={registry?.models.find((model) => model.status === "candidate") ?? null}
        />
      </div>
      <div className="workspace-column" style={{ gridColumn: "span 2" }}>
        <section className="panel">
          <h3>Registry detail</h3>
          {loading ? <p className="muted">Loading registry...</p> : null}
          {error ? <p className="failure-text">Unable to load registry: {error}</p> : null}
          {!loading && !error && (registry?.models.length ?? 0) === 0 ? <p className="muted">No models in registry.</p> : null}
          {!loading && !error && (registry?.models.length ?? 0) > 0 ? (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th align="left">Model</th>
                  <th align="left">Status</th>
                  <th align="left">Approval</th>
                  <th align="left">Metric</th>
                </tr>
              </thead>
              <tbody>
                {registry?.models.map((model, index) => (
                  <tr key={`${String(model.model_id ?? model.path ?? "model")}-${index}`}>
                    <td>{String(model.model_id ?? "n/a")}</td>
                    <td>{String(model.status ?? "n/a")}</td>
                    <td>{String(model.approval_status ?? "n/a")}</td>
                    <td>{typeof model.checkpoint_metric === "number" ? model.checkpoint_metric.toFixed(4) : "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </section>
      </div>
    </div>
  );
}
