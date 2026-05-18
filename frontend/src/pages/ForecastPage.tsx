import { useEffect, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";
import type { AdapterMetadata, SessionStateSummary } from "../features/sessions/types/session.types";

export function ForecastPage() {
  const {
    activeSessionId,
    latestForecastBundle,
    selectedFeature,
    setSelectedFeature,
    setActiveSessionId,
    setLatestForecastBundle
  } = useSessionForecastView();

  const [forecastRunning, setForecastRunning] = useState(false);
  const [adapterMetadata, setAdapterMetadata] = useState<AdapterMetadata | null>(null);
  const {
    framesMetadata,
    selectedFrameIndex,
    selectedFrameGeoJson,
    selectedFrameSummary,
    frameLoading,
    frameError,
    refreshFrames,
    refreshFramesForSession,
    setSelectedFrameIndex,
  } = useSessionForecastFrames(activeSessionId);

  const readAdapterMetadata = (sessionState: SessionStateSummary): AdapterMetadata | null => {
    const internalState = sessionState.internal_state as { last_input_adapter_metadata?: unknown } | undefined;
    const nested = internalState?.last_input_adapter_metadata;
    const flattened = sessionState.last_input_adapter_metadata;
    const metadata = nested ?? flattened;
    if (!metadata || typeof metadata !== "object") {
      return null;
    }
    return metadata as AdapterMetadata;
  };

  const runLatestForecast = async () => {
    setForecastRunning(true);
    try {
      const runResult = await sessionClient.runSessionForecast({});
      setActiveSessionId(runResult.sessionId);
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
      await refreshFramesForSession(runResult.sessionId);
      const state = await sessionClient.getSessionState(runResult.sessionId);
      setAdapterMetadata(readAdapterMetadata(state));
    } catch {
      // Keep map page map-first; runtime status details live on Forecast Overview.
    } finally {
      setForecastRunning(false);
    }
  };

  useEffect(() => {
    void runLatestForecast();
  }, []);

  useEffect(() => {
    if (latestForecastBundle) {
      void refreshFrames();
    }
  }, [latestForecastBundle, refreshFrames]);

  const predictionEngine = typeof framesMetadata?.metadata?.prediction_engine === "string"
    ? framesMetadata.metadata.prediction_engine
    : undefined;
  const forecastFitKey = framesMetadata?.forecast_id ?? activeSessionId ?? null;
  const geojson = ((selectedFrameGeoJson ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;
  const timelineDisabled = !framesMetadata || framesMetadata.frame_count === 0;
  const timelineStatus = frameError
    ? "Frame sequence unavailable"
    : frameLoading && !framesMetadata
      ? "Loading ConvLSTM forecast frames..."
      : !latestForecastBundle
        ? "Run forecast to enable timeline"
        : null;


  return (
    <AppShell
      title="Map / Forecast"
      subtitle="Current forecast map and plume overlay."
    >
      <main className="map-column">
        <div className="forecast-controls">
          <span className={`status-dot ${forecastRunning ? "is-running" : "is-ready"}`} aria-hidden="true" />
          <div className="forecast-controls-labels">
            <strong>ConvLSTM multi-step</strong>
            <span className="timeline-note">Operational forecast player</span>
          </div>
          <button type="button" className="primary-button run-forecast-button" onClick={() => void runLatestForecast()} disabled={forecastRunning}>
            {latestForecastBundle ? "Refresh forecast" : "Run forecast"}
          </button>
          {import.meta.env.DEV ? (
            <span className="timeline-note">frame: {selectedFrameIndex + 1} / {framesMetadata?.frame_count ?? 0}</span>
          ) : null}
        </div>
        <ForecastMap
          geojson={geojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
          center={null}
          autoFitKey={forecastFitKey}
        />
        <ForecastFrameTimeline
          frameCount={framesMetadata?.frame_count ?? 0}
          frameIndices={framesMetadata?.frame_indices ?? []}
          selectedFrameIndex={selectedFrameIndex}
          onSelectFrame={setSelectedFrameIndex}
          loading={frameLoading}
          disabled={timelineDisabled}
          metadata={framesMetadata?.metadata}
          selectedFrameSummary={selectedFrameSummary}
          predictionTrust={adapterMetadata?.prediction_trust ?? null}
          inputMode={adapterMetadata?.input_mode ?? null}
          missingChannelsCount={adapterMetadata?.input_completeness?.missing_channels?.length ?? null}
          observedFrameCount={adapterMetadata?.input_completeness?.observed_frame_count ?? null}
          requiredFrameCount={adapterMetadata?.input_completeness?.required_frame_count ?? null}
          meteorologySourceKind={adapterMetadata?.meteorology_source_kind ?? null}
          modelName={framesMetadata?.model ?? "ConvLSTM multi-step"}
          predictionEngine={predictionEngine ?? "torch_multistep"}
          frameDurationSeconds={typeof framesMetadata?.metadata?.frame_duration_seconds === "number" ? framesMetadata.metadata.frame_duration_seconds : null}
          errorMessage={timelineStatus ?? frameError}
        />
      </main>
    </AppShell>
  );
}
