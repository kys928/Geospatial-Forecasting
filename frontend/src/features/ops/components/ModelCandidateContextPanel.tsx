import { useModelCandidateContext } from "../hooks/useModelCandidateContext";
import type { SafeUserAction } from "../types/ops.types";

function decisionMessage(decisionState: string | undefined): string {
  if (decisionState === "pending_review") {
    return "Candidate model is waiting for review";
  }

  if (decisionState === "rejected") {
    return "Candidate model was rejected or not promoted";
  }

  if (decisionState === "no_candidate") {
    return "No candidate model is waiting for review";
  }

  return "Candidate model context is available";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

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

function renderKeyValueTable(data: Record<string, unknown>) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return <p className="muted">No details available.</p>;
  }

  return (
    <dl className="detail-list" style={{ marginTop: 8 }}>
      {entries.map(([key, value]) => (
        <div className="detail-list-row" key={key}>
          <dt>{key}</dt>
          <dd>{typeof value === "string" || typeof value === "number" || typeof value === "boolean" ? String(value) : JSON.stringify(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ModelCandidateContextPanel() {
  const { context, loading, error, refresh } = useModelCandidateContext();
  const decisionState = typeof context?.decision_state === "string" ? context.decision_state : undefined;

  const activeModel = asRecord(context?.active_model);
  const candidateModel = asRecord(context?.candidate_model);
  const comparison = asRecord(context?.comparison);
  const metricValues = asRecord(comparison?.metrics ?? comparison?.metric_values);
  const missingMetrics = Array.isArray(comparison?.missing_metrics) ? comparison?.missing_metrics : [];
  const safeUserActions = Array.isArray(context?.safe_user_actions) ? (context.safe_user_actions as SafeUserAction[]) : [];
  const systemBoundaries = Array.isArray(context?.system_boundaries) ? context.system_boundaries : [];

  return (
    <section className="panel">
      <h3>Model candidate context</h3>
      <p className="muted">Read-only context only. This panel does not approve, reject, activate, or roll back models.</p>
      <div className="button-row">
        <button className="secondary-button" onClick={() => void refresh()} disabled={loading}>{loading ? "Refreshing..." : "Refresh context"}</button>
      </div>

      {loading ? <p className="muted">Loading candidate context...</p> : null}
      {error ? <p role="alert">{error}</p> : null}
      {!loading && !error && !context ? <p className="muted">Candidate context is currently unavailable.</p> : null}

      {!loading && !error && context ? (
        <>
          <p className="badge" style={{ display: "inline-block", borderRadius: 8 }}>{decisionMessage(decisionState)}</p>
          <dl className="detail-list" style={{ marginTop: 12 }}>
            <div className="detail-list-row"><dt>Decision state</dt><dd>{decisionState ?? "unknown"}</dd></div>
            <div className="detail-list-row"><dt>Comparison available</dt><dd>{comparison && comparison.can_compare !== undefined ? String(comparison.can_compare) : "unknown"}</dd></div>
            <div className="detail-list-row"><dt>Comparison summary</dt><dd>{typeof comparison?.comparison_summary === "string" ? comparison.comparison_summary : "n/a"}</dd></div>
          </dl>

          <h4 style={{ marginTop: 16 }}>Active model summary</h4>
          {activeModel ? renderKeyValueTable(activeModel) : <p className="muted">No active model summary provided.</p>}

          <h4 style={{ marginTop: 16 }}>Candidate model summary</h4>
          {candidateModel ? renderKeyValueTable(candidateModel) : <p className="muted">No candidate model summary provided.</p>}

          <h4 style={{ marginTop: 16 }}>Metrics snapshot</h4>
          {metricValues ? renderKeyValueTable(metricValues) : <p className="muted">No metric values were provided.</p>}
          {missingMetrics.length > 0 ? (
            <p className="muted" style={{ marginTop: 8 }}>Missing metrics: {missingMetrics.map(String).join(", ")}</p>
          ) : null}

          <h4 style={{ marginTop: 16 }}>Safe user actions</h4>
          {safeUserActions.length > 0 ? (
            <div style={{ display: "grid", gap: 8 }}>
              {safeUserActions.map((action, index) => (
                <article className="badge" style={{ borderRadius: 8 }} key={`${getActionTitle(action, index)}-${index}`}>
                  <strong>{getActionTitle(action, index)}</strong>
                  <p style={{ margin: "8px 0 0" }}>{getActionDescription(action)}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="muted">No safe user actions were provided.</p>
          )}

          {systemBoundaries.length > 0 ? (
            <details style={{ marginTop: 12 }}>
              <summary className="muted">System boundaries</summary>
              <ul style={{ margin: "8px 0 0 20px" }}>
                {systemBoundaries.map((boundary, index) => (
                  <li key={`${String(boundary)}-${index}`}>{String(boundary)}</li>
                ))}
              </ul>
            </details>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
