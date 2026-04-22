import type { SessionSummary } from "../types/session.types";

interface SessionListPanelProps {
  sessions: SessionSummary[];
  selectedSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onRefresh: () => void;
  loading: boolean;
}

export function SessionListPanel({ sessions, selectedSessionId, onSelectSession, onRefresh, loading }: SessionListPanelProps) {
  return (
    <section className="panel">
      <h3>Sessions</h3>
      <button className="primary-button" onClick={onRefresh} style={{ marginBottom: 10 }}>
        {loading ? "Refreshing..." : "Refresh sessions"}
      </button>
      <div style={{ display: "grid", gap: 8, maxHeight: 320, overflow: "auto" }}>
        {sessions.map((session) => (
          <button
            key={session.session_id}
            className="badge"
            style={{
              textAlign: "left",
              borderColor: selectedSessionId === session.session_id ? "#7aa2f7" : undefined,
              width: "100%"
            }}
            onClick={() => onSelectSession(session.session_id)}
          >
            {session.session_id} · {session.backend_name}
          </button>
        ))}
        {sessions.length === 0 ? <p className="muted">No sessions available.</p> : null}
      </div>
    </section>
  );
}
