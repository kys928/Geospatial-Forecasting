import type { SessionStateSummary } from "../types/session.types";

interface RecentObservationsTableProps {
  state: SessionStateSummary | null;
}

export function RecentObservationsTable({ state }: RecentObservationsTableProps) {
  const recent = Array.isArray(state?.recent_observations) ? state.recent_observations : [];

  return (
    <section className="panel">
      <h3>Recent observations</h3>
      {recent.length === 0 ? (
        <p className="muted">No recent observations available in current state payload.</p>
      ) : (
        <pre style={{ margin: 0, maxHeight: 220, overflow: "auto" }}>{JSON.stringify(recent, null, 2)}</pre>
      )}
    </section>
  );
}
