import type { SessionDetail } from "../types/session.types";

interface SessionBackendPanelProps {
  detail: SessionDetail | null;
}

export function SessionBackendPanel({ detail }: SessionBackendPanelProps) {
  return (
    <section className="panel">
      <h3>Session backend</h3>
      {!detail ? (
        <p className="muted">Select a session to inspect backend details.</p>
      ) : (
        <dl className="detail-list">
          <div className="detail-list-row"><dt>Session ID</dt><dd>{detail.session_id}</dd></div>
          <div className="detail-list-row"><dt>Backend</dt><dd>{detail.backend_name}</dd></div>
          <div className="detail-list-row"><dt>Model</dt><dd>{detail.model_name}</dd></div>
          <div className="detail-list-row"><dt>Status</dt><dd>{detail.status}</dd></div>
        </dl>
      )}
    </section>
  );
}
