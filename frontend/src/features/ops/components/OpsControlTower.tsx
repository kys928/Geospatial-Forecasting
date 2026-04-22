import type { OpsStatusResponse } from "../types/ops.types";

interface OpsControlTowerProps {
  status: OpsStatusResponse | null;
  loading: boolean;
  onRefresh: () => void;
}

export function OpsControlTower({ status, loading, onRefresh }: OpsControlTowerProps) {
  return (
    <section className="panel">
      <h3>Ops control tower</h3>
      <button className="primary-button" onClick={onRefresh}>{loading ? "Refreshing..." : "Refresh ops status"}</button>
      <dl className="detail-list">
        <div className="detail-list-row"><dt>Phase</dt><dd>{status?.phase ?? "n/a"}</dd></div>
        <div className="detail-list-row"><dt>Pending approval</dt><dd>{String(status?.has_pending_manual_approval ?? false)}</dd></div>
        <div className="detail-list-row"><dt>Candidate approval</dt><dd>{status?.candidate_approval_status ?? "n/a"}</dd></div>
        <div className="detail-list-row"><dt>Latest warning/error</dt><dd>{status?.latest_warning_or_error ?? "none"}</dd></div>
      </dl>
    </section>
  );
}
