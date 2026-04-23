import type { OpsStatusResponse } from "../types/ops.types";

interface OpsControlTowerProps {
  status: OpsStatusResponse | null;
  loading: boolean;
  onRefreshStatus: () => void;
  onRefreshJobs: () => void;
  onRefreshEvents: () => void;
}

export function OpsControlTower({ status, loading, onRefreshStatus, onRefreshJobs, onRefreshEvents }: OpsControlTowerProps) {
  return (
    <section className="panel">
      <h3>Ops control tower</h3>
      <p className="muted">
        After submitting retraining, use these refresh controls to verify progress across status, jobs, and events.
      </p>
      <div className="button-row">
        <button className="primary-button" onClick={onRefreshStatus}>{loading ? "Refreshing..." : "Refresh status"}</button>
        <button className="secondary-button" onClick={onRefreshJobs}>Refresh jobs</button>
        <button className="secondary-button" onClick={onRefreshEvents}>Refresh events</button>
      </div>
      <dl className="detail-list">
        <div className="detail-list-row"><dt>Phase</dt><dd>{status?.phase ?? "n/a"}</dd></div>
        <div className="detail-list-row"><dt>Pending approval</dt><dd>{String(status?.has_pending_manual_approval ?? false)}</dd></div>
        <div className="detail-list-row"><dt>Candidate approval</dt><dd>{status?.candidate_approval_status ?? "n/a"}</dd></div>
        <div className="detail-list-row"><dt>Latest warning/error</dt><dd>{status?.latest_warning_or_error ?? "none"}</dd></div>
      </dl>
    </section>
  );
}
