import type { SessionDetail, SessionStateSummary } from "../types/session.types";

interface SessionStateRibbonProps {
  detail: SessionDetail | null;
  state: SessionStateSummary | null;
}

export function SessionStateRibbon({ detail, state }: SessionStateRibbonProps) {
  return (
    <section className="panel" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      <span className="badge">status: {detail?.status ?? "n/a"}</span>
      <span className="badge">state_version: {String(state?.state_version ?? "n/a")}</span>
      <span className="badge">observation_count: {String(state?.observation_count ?? "n/a")}</span>
      <span className="badge">updated: {state?.last_update_time ?? detail?.updated_at ?? "n/a"}</span>
    </section>
  );
}
