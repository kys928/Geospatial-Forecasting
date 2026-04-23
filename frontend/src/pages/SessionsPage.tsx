import { useEffect, useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { SessionListPanel } from "../features/sessions/components/SessionListPanel";
import { SessionCreateForm } from "../features/sessions/components/SessionCreateForm";
import { SessionStateRibbon } from "../features/sessions/components/SessionStateRibbon";
import { SessionBackendPanel } from "../features/sessions/components/SessionBackendPanel";
import { SessionStateInspector } from "../features/sessions/components/SessionStateInspector";
import { SessionActionBar } from "../features/sessions/components/SessionActionBar";
import { PredictionRequestForm } from "../features/sessions/components/PredictionRequestForm";
import { SessionResultSummary } from "../features/sessions/components/SessionResultSummary";
import { useSessions } from "../features/sessions/hooks/useSessions";
import { useSessionState } from "../features/sessions/hooks/useSessionState";
import { useSessionActions } from "../features/sessions/hooks/useSessionActions";
import { useSessionForecastBundle } from "../features/sessions/hooks/useSessionForecastBundle";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";

type SessionWorkspaceMode = "basic" | "operator";

export function SessionsPage() {
  const { sessions, loading, error, refresh, createSession } = useSessions();
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [mode, setMode] = useState<SessionWorkspaceMode>("basic");

  const effectiveSessionId = useMemo(() => selectedSessionId ?? sessions[0]?.session_id ?? null, [selectedSessionId, sessions]);
  const sessionState = useSessionState(effectiveSessionId);
  const latestForecast = useSessionForecastBundle(effectiveSessionId);
  const {
    selectedFeature,
    setActiveSessionId,
    setLatestForecastBundle,
    clearSelectedFeature
  } = useSessionForecastView();

  const actions = useSessionActions(effectiveSessionId, async () => {
    await Promise.all([refresh(), sessionState.refresh()]);
  });

  const showOperatorPanels = mode === "operator";

  useEffect(() => {
    setActiveSessionId(effectiveSessionId);
  }, [effectiveSessionId, setActiveSessionId]);

  useEffect(() => {
    if (!effectiveSessionId) {
      setLatestForecastBundle(null, null);
      return;
    }

    if (latestForecast.bundle) {
      setLatestForecastBundle(effectiveSessionId, latestForecast.bundle);
      return;
    }

    if (!latestForecast.loading) {
      setLatestForecastBundle(effectiveSessionId, null);
    }
  }, [effectiveSessionId, latestForecast.bundle, latestForecast.loading, setLatestForecastBundle]);

  return (
    <AppShell
      title="Sessions workspace"
      subtitle="Start a session, choose it, run forecast, and review the result summary."
      metaItems={sessionState.detail?.model_name ? [{ label: sessionState.detail.model_name }] : undefined}
    >
      <section className="panel">
        <div className="button-row">
          <button
            className={mode === "basic" ? "primary-button" : "secondary-button"}
            type="button"
            onClick={() => setMode("basic")}
          >
            Basic
          </button>
          <button
            className={mode === "operator" ? "primary-button" : "secondary-button"}
            type="button"
            onClick={() => setMode("operator")}
          >
            Operator
          </button>
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          {mode === "basic"
            ? "Basic mode focuses on the core session workflow and hides technical controls."
            : "Operator mode shows ingest, manual update, backend details, and raw state tools."}
        </p>
      </section>

      <div className="workspace-grid" style={{ gridTemplateColumns: "0.95fr 1.3fr" }}>
        <div className="workspace-column">
          <SessionListPanel
            sessions={sessions}
            selectedSessionId={effectiveSessionId}
            onSelectSession={setSelectedSessionId}
            onRefresh={() => void refresh()}
            loading={loading}
          />
          <SessionCreateForm
            heading="Start session"
            actionLabel="Start session"
            onCreate={async (payload) => {
              const created = await createSession(payload);
              setSelectedSessionId(created.session_id);
            }}
          />
          {error ? <section className="panel muted">{error}</section> : null}
        </div>

        <div className="workspace-column">
          {showOperatorPanels ? <SessionStateRibbon detail={sessionState.detail} state={sessionState.state} /> : null}
          <PredictionRequestForm
            disabled={!effectiveSessionId || actions.runningAction !== null}
            showAdvancedOptions={showOperatorPanels}
            onPredict={async (payload) => {
              await actions.predict(payload);
              clearSelectedFeature();
              await latestForecast.refresh();
            }}
          />
          <SessionResultSummary
            loading={sessionState.loading || latestForecast.loading}
            error={actions.error ?? latestForecast.error ?? sessionState.error}
            lastPrediction={actions.lastPrediction ?? latestForecast.bundle?.explanation ?? latestForecast.bundle?.summary ?? null}
            selectedFeature={selectedFeature}
          />
        </div>
      </div>

      {showOperatorPanels ? (
        <details className="panel advanced-section" open>
          <summary>Operator controls</summary>
          <div className="advanced-content workspace-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="workspace-column">
              <SessionActionBar
                disabled={!effectiveSessionId}
                runningAction={actions.runningAction}
                onUpdate={async () => {
                  await actions.update();
                }}
              />
            </div>
            <div className="workspace-column">
              <SessionBackendPanel detail={sessionState.detail} />
            </div>
            <div className="workspace-column">
              <SessionStateInspector detail={sessionState.detail} state={sessionState.state} />
            </div>
          </div>
        </details>
      ) : null}
    </AppShell>
  );
}
