import type { SessionPredictionResponse } from "../types/session.types";
import { normalizeSessionPrediction } from "../utils/normalizeSessionPrediction";

interface SessionResultSummaryProps {
  lastPrediction: SessionPredictionResponse | null;
  loading: boolean;
  error: string | null;
}

function formatSource(source: string | null): string {
  if (!source) {
    return "session prediction";
  }

  return source;
}

export function SessionResultSummary({ lastPrediction, loading, error }: SessionResultSummaryProps) {
  const normalized = normalizeSessionPrediction(lastPrediction);

  return (
    <section className="detail-drawer panel">
      <div className="panel-header">
        <h2>Explanation</h2>
      </div>

      <div className="panel-body">
        <p className="muted explanation-source">Source: {formatSource(normalized?.source ?? null)}</p>

        {loading ? <p className="muted">Loading session details...</p> : null}
        {error ? <p className="muted">{error}</p> : null}

        <p className="detail-explanation-text">
          {normalized?.explanation ?? "Run a prediction to generate a readable result summary."}
        </p>

        {normalized?.riskLevel ? (
          <div className="detail-inline-meta">
            <span className="detail-chip">Risk: {normalized.riskLevel}</span>
          </div>
        ) : null}

        {normalized?.recommendation ? (
          <>
            <div className="section-spacer" />
            <h3>Recommendation</h3>
            <p className="muted">{normalized.recommendation}</p>
          </>
        ) : null}

        {normalized?.uncertaintyNote ? (
          <>
            <div className="section-spacer" />
            <h3>Uncertainty note</h3>
            <p className="muted">{normalized.uncertaintyNote}</p>
          </>
        ) : null}

        <div className="section-spacer" />

        <h3>Inspect map</h3>
        <p className="muted">Open the Forecast workspace to inspect plume layers and source details on the map.</p>
      </div>
    </section>
  );
}
