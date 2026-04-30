import type { ForecastArtifactMetadata } from "../types/forecast.types";

interface RecentForecastsPanelProps {
  forecasts: ForecastArtifactMetadata[];
  loading: boolean;
  error: string | null;
  loadingForecastId: string | null;
  onRefresh: () => void;
  onLoad: (forecast: ForecastArtifactMetadata) => void;
}

function shortId(value: string): string {
  if (value.length <= 14) {
    return value;
  }
  return `${value.slice(0, 8)}…${value.slice(-4)}`;
}

export function RecentForecastsPanel({
  forecasts,
  loading,
  error,
  loadingForecastId,
  onRefresh,
  onLoad
}: RecentForecastsPanelProps) {
  return (
    <section className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.5rem" }}>
        <h3 style={{ margin: 0 }}>Recent persisted forecasts</h3>
        <button type="button" className="secondary-button" onClick={onRefresh} disabled={loading}>
          Refresh
        </button>
      </div>

      {loading ? <p className="muted">Loading persisted forecasts…</p> : null}
      {error ? <p className="muted">Unable to load persisted forecasts: {error}</p> : null}

      {!loading && !error && forecasts.length === 0 ? (
        <p className="muted">No persisted forecasts are available yet.</p>
      ) : null}

      {!error && forecasts.length > 0 ? (
        <div style={{ display: "grid", gap: "0.75rem", marginTop: "0.75rem" }}>
          {forecasts.map((forecast) => {
            const artifacts = forecast.available_artifacts ?? [];
            return (
              <article key={forecast.forecast_id} className="panel" style={{ margin: 0 }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
                  <div>
                    <strong>{shortId(forecast.forecast_id)}</strong>
                    <p className="muted" style={{ margin: "0.25rem 0" }}>
                      Issued: {forecast.issued_at}
                    </p>
                    <p className="muted" style={{ margin: "0.25rem 0" }}>
                      Model: {forecast.model ?? "unknown"}
                      {forecast.runtime?.model_family ? ` · family ${forecast.runtime.model_family}` : ""}
                      {forecast.runtime?.prediction_trust ? ` · trust ${forecast.runtime.prediction_trust}` : ""}
                    </p>
                    <p className="muted" style={{ margin: "0.25rem 0" }}>
                      Artifacts ({artifacts.length}): {artifacts.length > 0 ? artifacts.join(", ") : "none listed"}
                    </p>
                  </div>
                  <div>
                    <button
                      type="button"
                      className="primary-button"
                      disabled={loadingForecastId === forecast.forecast_id}
                      onClick={() => onLoad(forecast)}
                    >
                      {loadingForecastId === forecast.forecast_id ? "Loading…" : "Load on map"}
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}
