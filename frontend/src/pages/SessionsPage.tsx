import { useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { SessionListPanel } from "../features/sessions/components/SessionListPanel";
import { SessionCreateForm } from "../features/sessions/components/SessionCreateForm";
import { SessionStateRibbon } from "../features/sessions/components/SessionStateRibbon";
import { SessionBackendPanel } from "../features/sessions/components/SessionBackendPanel";
import { SessionStateInspector } from "../features/sessions/components/SessionStateInspector";
import { SessionActionBar } from "../features/sessions/components/SessionActionBar";
import { ObservationIngestPanel } from "../features/sessions/components/ObservationIngestPanel";
import { PredictionRequestForm } from "../features/sessions/components/PredictionRequestForm";
import { RecentObservationsTable } from "../features/sessions/components/RecentObservationsTable";
import { useSessions } from "../features/sessions/hooks/useSessions";
import { useSessionState } from "../features/sessions/hooks/useSessionState";
import { useSessionActions } from "../features/sessions/hooks/useSessionActions";

export function SessionsPage() {
  const { sessions, loading, error, refresh, createSession } = useSessions();
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  const effectiveSessionId = useMemo(() => selectedSessionId ?? sessions[0]?.session_id ?? null, [selectedSessionId, sessions]);
  const sessionState = useSessionState(effectiveSessionId);
  const actions = useSessionActions(effectiveSessionId, async () => {
    await Promise.all([refresh(), sessionState.refresh()]);
  });

  return (
    <AppShell
      title="Sessions workspace"
      subtitle="Create and operate forecasting sessions with ingest, update, and prediction actions."
      metaItems={[{ label: sessionState.detail?.model_name ?? "Session management" }]}
    >
      <div className="workspace-grid">
        <div style={{ display: "grid", gap: 12 }}>
          <SessionListPanel
            sessions={sessions}
            selectedSessionId={effectiveSessionId}
            onSelectSession={setSelectedSessionId}
            onRefresh={() => void refresh()}
            loading={loading}
          />
          <SessionCreateForm
            onCreate={async (payload) => {
              const created = await createSession(payload);
              setSelectedSessionId(created.session_id);
            }}
          />
          {error ? <section className="panel muted">{error}</section> : null}
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          <SessionStateRibbon detail={sessionState.detail} state={sessionState.state} />
          <SessionBackendPanel detail={sessionState.detail} />
          <SessionActionBar
            disabled={!effectiveSessionId}
            runningAction={actions.runningAction}
            onUpdate={async () => {
              await actions.update();
            }}
          />
          <RecentObservationsTable state={sessionState.state} />
          <SessionStateInspector detail={sessionState.detail} state={sessionState.state} />
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          <ObservationIngestPanel
            disabled={!effectiveSessionId || actions.runningAction !== null}
            onIngest={async (observations) => {
              await actions.ingest(observations);
            }}
          />
          <PredictionRequestForm
            disabled={!effectiveSessionId || actions.runningAction !== null}
            onPredict={async (payload) => {
              await actions.predict(payload);
            }}
          />
          <section className="panel">
            <h3>Action output</h3>
            {sessionState.loading ? <p className="muted">Loading session details...</p> : null}
            {sessionState.error ? <p className="muted">{sessionState.error}</p> : null}
            {actions.error ? <p className="muted">{actions.error}</p> : null}
            <pre style={{ margin: 0, maxHeight: 360, overflow: "auto" }}>
              {JSON.stringify(
                {
                  lastIngestResult: actions.lastIngestResult,
                  lastUpdateResult: actions.lastUpdateResult,
                  lastPrediction: actions.lastPrediction
                },
                null,
                2
              )}
            </pre>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
