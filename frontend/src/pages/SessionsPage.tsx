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

  const lastActionType = actions.lastPrediction
    ? "predict"
    : actions.lastIngestResult
      ? "ingest"
      : actions.lastUpdateResult
        ? "update"
        : null;

  const lastActionPayload = actions.lastPrediction ?? actions.lastIngestResult ?? actions.lastUpdateResult;

  return (
    <AppShell
      title="Sessions workspace"
      subtitle="Create or select a session, refresh state when needed, then run predictions."
      metaItems={sessionState.detail?.model_name ? [{ label: sessionState.detail.model_name }] : undefined}
    >
      <div className="workspace-grid" style={{ gridTemplateColumns: "0.9fr 1.35fr 1fr" }}>
        <div className="workspace-column">
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

        <div className="workspace-column">
          <SessionStateRibbon detail={sessionState.detail} state={sessionState.state} />
          <PredictionRequestForm
            disabled={!effectiveSessionId || actions.runningAction !== null}
            onPredict={async (payload) => {
              await actions.predict(payload);
            }}
          />
          <SessionActionBar
            disabled={!effectiveSessionId}
            runningAction={actions.runningAction}
            onUpdate={async () => {
              await actions.update();
            }}
          />
          <ObservationIngestPanel
            disabled={!effectiveSessionId || actions.runningAction !== null}
            onIngest={async (observations) => {
              await actions.ingest(observations);
            }}
          />
        </div>

        <div className="workspace-column">
          <details className="panel advanced-section">
            <summary>Operational details</summary>
            <div className="advanced-content">
              <SessionBackendPanel detail={sessionState.detail} />
              <RecentObservationsTable state={sessionState.state} />
              <SessionStateInspector detail={sessionState.detail} state={sessionState.state} />
            </div>
          </details>
          <section className="panel">
            <h3>Recent action status</h3>
            {sessionState.loading ? <p className="muted">Loading session details...</p> : null}
            {sessionState.error ? <p className="muted">{sessionState.error}</p> : null}
            {!lastActionType && !actions.error ? <p className="muted">No recent session action.</p> : null}
            {lastActionType ? <p><strong>Last action:</strong> {lastActionType}</p> : null}
            <p><strong>Result:</strong> {actions.error ? "Failure" : lastActionPayload ? "Success" : "Idle"}</p>
            <p>{actions.error ?? "No action failures reported."}</p>
            {lastActionPayload ? (
              <details>
                <summary>Technical details</summary>
                <pre style={{ margin: 0, maxHeight: 240, overflow: "auto" }}>{JSON.stringify(lastActionPayload, null, 2)}</pre>
              </details>
            ) : null}
          </section>
        </div>
      </div>
    </AppShell>
  );
}
