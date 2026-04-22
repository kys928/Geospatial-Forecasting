import type { SessionDetail, SessionStateSummary } from "../types/session.types";

interface SessionStateInspectorProps {
  detail: SessionDetail | null;
  state: SessionStateSummary | null;
}

export function SessionStateInspector({ detail, state }: SessionStateInspectorProps) {
  return (
    <section className="panel">
      <h3>State inspector</h3>
      <pre style={{ margin: 0, maxHeight: 360, overflow: "auto" }}>
        {JSON.stringify({ detail, state }, null, 2)}
      </pre>
    </section>
  );
}
