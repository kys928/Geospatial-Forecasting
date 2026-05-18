import { useEffect, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";

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
  const {
    framesMetadata,
    selectedFrameIndex,
    selectedFrameGeoJson,
    selectedFrameSummary,
    frameLoading,
    frameError,
    refreshFrames,
    setSelectedFrameIndex,
  } = useSessionForecastFrames(activeSessionId);

  const runLatestForecast = async () => {
    setForecastRunning(true);
    try {
      const runResult = await sessionClient.runSessionForecast({});
      setActiveSessionId(runResult.sessionId);
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
      await refreshFrames();
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

  const summaryMetadata = latestForecastBundle?.summary?.metadata as Record<string, unknown> | undefined;
  const adapterMetadata = summaryMetadata?.input_adapter_metadata as Record<string, unknown> | undefined;
  const predictionEngine = typeof framesMetadata?.metadata?.prediction_engine === "string"
    ? framesMetadata.metadata.prediction_engine
    : undefined;
  const geojson = ((selectedFrameGeoJson ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;
  const timelineDisabled = !framesMetadata || framesMetadata.frame_count <= 1;
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
          predictionTrust={typeof adapterMetadata?.prediction_trust === "string" ? adapterMetadata.prediction_trust : null}
          inputMode={typeof adapterMetadata?.input_mode === "string" ? adapterMetadata.input_mode : null}
          modelName={framesMetadata?.model ?? "ConvLSTM multi-step"}
          predictionEngine={predictionEngine ?? "torch_multistep"}
          errorMessage={timelineStatus ?? frameError}
        />
      </main>
    </AppShell>
  );
}
