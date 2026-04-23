import { useEffect, useMemo, useState } from "react";
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
import { SessionResultSummary } from "../features/sessions/components/SessionResultSummary";
import { useSessions } from "../features/sessions/hooks/useSessions";
import { useSessionState } from "../features/sessions/hooks/useSessionState";
import { useSessionActions } from "../features/sessions/hooks/useSessionActions";
import { useSessionForecastBundle } from "../features/sessions/hooks/useSessionForecastBundle";
import { ForecastMap } from "../features/map/components/ForecastMap";
import type { GeoJsonFeatureCollection, SelectedFeatureState } from "../features/forecast/types/forecast.types";

type SessionWorkspaceMode = "basic" | "operator";

export function SessionsPage() {
  const { sessions, loading, error, refresh, createSession } = useSessions();
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [mode, setMode] = useState<SessionWorkspaceMode>("basic");

  const effectiveSessionId = useMemo(() => selectedSessionId ?? sessions[0]?.session_id ?? null, [selectedSessionId, sessions]);
  const sessionState = useSessionState(effectiveSessionId);
  const latestForecast = useSessionForecastBundle(effectiveSessionId);
  const [selectedFeature, setSelectedFeature] = useState<SelectedFeatureState | null>(null);
  const actions = useSessionActions(effectiveSessionId, async () => {
    await Promise.all([refresh(), sessionState.refresh()]);
  });

  const latestGeojson = (latestForecast.bundle?.geojson ?? null) as GeoJsonFeatureCollection | null;

  const showOperatorPanels = mode === "operator";

  useEffect(() => {
    setSelectedFeature(null);
  }, [effectiveSessionId]);

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
          <SessionStateRibbon detail={sessionState.detail} state={sessionState.state} />
          <PredictionRequestForm
            disabled={!effectiveSessionId || actions.runningAction !== null}
            onPredict={async (payload) => {
              await actions.predict(payload);
              setSelectedFeature(null);
              await latestForecast.refresh();
            }}
          />
          <section className="panel">
            <div className="panel-header">
              <h2>Forecast map</h2>
            </div>
            <div className="panel-body" style={{ gap: "0.75rem", display: "grid" }}>
              <p className="muted" style={{ margin: 0 }}>
                {latestForecast.loading
                  ? "Loading latest session forecast map..."
                  : latestForecast.error
                    ? "Run a forecast to load map artifacts for this session."
                    : "Inspect the selected session forecast plume and source directly in Sessions."}
              </p>
              <ForecastMap
                geojson={latestGeojson}
                selectedFeature={selectedFeature}
                onSelectFeature={setSelectedFeature}
              />
            </div>
          </section>
          <SessionResultSummary
            loading={sessionState.loading || latestForecast.loading}
            error={actions.error ?? latestForecast.error ?? sessionState.error}
            lastPrediction={actions.lastPrediction ?? latestForecast.bundle?.explanation ?? latestForecast.bundle?.summary ?? null}
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
              <ObservationIngestPanel
                disabled={!effectiveSessionId || actions.runningAction !== null}
                onIngest={async (observations) => {
                  await actions.ingest(observations);
                }}
              />
            </div>
            <div className="workspace-column">
              <SessionBackendPanel detail={sessionState.detail} />
              <RecentObservationsTable state={sessionState.state} />
            </div>
            <div className="workspace-column">
              <SessionStateInspector detail={sessionState.detail} state={sessionState.state} />
              <section className="panel">
                <h3>Technical action dump</h3>
                {!actions.lastPrediction && !actions.lastIngestResult && !actions.lastUpdateResult && !actions.error ? (
                  <p className="muted">No recent session action.</p>
                ) : null}
                <p><strong>Result:</strong> {actions.error ? "Failure" : actions.lastPrediction ?? actions.lastIngestResult ?? actions.lastUpdateResult ? "Success" : "Idle"}</p>
                <p>{actions.error ?? "No action failures reported."}</p>
                {actions.lastPrediction ?? actions.lastIngestResult ?? actions.lastUpdateResult ? (
                  <pre style={{ margin: 0, maxHeight: 240, overflow: "auto" }}>
                    {JSON.stringify(actions.lastPrediction ?? actions.lastIngestResult ?? actions.lastUpdateResult, null, 2)}
                  </pre>
                ) : null}
              </section>
            </div>
          </div>
        </details>
      ) : null}
    </AppShell>
  );
}
