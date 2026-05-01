import { useRetrainingRecommendation } from "../hooks/useRetrainingRecommendation";
import type { SafeUserAction } from "../types/ops.types";

function getActionTitle(action: SafeUserAction, index: number): string {
  if (typeof action.title === "string" && action.title.trim()) {
    return action.title;
  }

  return `Action ${index + 1}`;
}

function getActionDescription(action: SafeUserAction): string {
  if (typeof action.description === "string" && action.description.trim()) {
    return action.description;
  }

  return "No additional details provided.";
}

export function RetrainingRecommendationPanel() {
  const { recommendation, context, loading, error, refresh } = useRetrainingRecommendation();
  const shouldRetrain = recommendation?.should_retrain;
  const statusLabel = shouldRetrain ? "Retraining is recommended" : "Retraining is not currently recommended";

  return (
    <section className="panel">
      <h3>Retraining recommendation</h3>
      <p className="muted">Read-only guidance only. This panel does not start retraining or change any active model.</p>
      <div className="button-row">
        <button className="secondary-button" onClick={() => void refresh()} disabled={loading}>{loading ? "Refreshing..." : "Refresh recommendation"}</button>
      </div>

      {loading ? <p className="muted">Loading retraining recommendation...</p> : null}
      {error ? <p role="alert">{error}</p> : null}

      {!loading && !error && !recommendation ? <p className="muted">No recommendation is currently available.</p> : null}

      {!loading && !error && recommendation ? (
        <>
          {context?.summary_seed ? <p>{context.summary_seed}</p> : null}
          <p className="muted" style={{ marginTop: 8 }}>Recommendation status only; no retraining job has been created.</p>
          <p className="badge" style={{ display: "inline-block", borderRadius: 8 }}>{statusLabel}</p>
          <dl className="detail-list" style={{ marginTop: 12 }}>
            <div className="detail-list-row"><dt>Severity</dt><dd>{recommendation.severity ?? "n/a"}</dd></div>
            <div className="detail-list-row"><dt>Reason</dt><dd>{recommendation.reason ?? "n/a"}</dd></div>
          </dl>

          <h4 style={{ marginTop: 16 }}>Safe user actions</h4>
          {context?.safe_user_actions && context.safe_user_actions.length > 0 ? (
            <div style={{ display: "grid", gap: 8 }}>
              {context.safe_user_actions.map((action, index) => (
                <article className="badge" style={{ borderRadius: 8 }} key={`${getActionTitle(action, index)}-${index}`}>
                  <strong>{getActionTitle(action, index)}</strong>
                  <p style={{ margin: "8px 0 0" }}>{getActionDescription(action)}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="muted">No safe user actions were provided.</p>
          )}

          {context?.system_boundaries && context.system_boundaries.length > 0 ? (
            <details style={{ marginTop: 12 }}>
              <summary className="muted">System boundaries</summary>
              <ul style={{ margin: "8px 0 0 20px" }}>
                {context.system_boundaries.map((boundary, index) => (
                  <li key={`${boundary}-${index}`}>{boundary}</li>
                ))}
              </ul>
            </details>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
