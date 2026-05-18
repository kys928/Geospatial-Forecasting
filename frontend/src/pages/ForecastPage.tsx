import { useEffect } from "react";
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

  const {
    framesMetadata,
    selectedFrameIndex,
    selectedFrameGeoJson,
    frameLoading,
    frameError,
    refreshFrames,
    refreshFramesForSession,
    setSelectedFrameIndex,
  } = useSessionForecastFrames(activeSessionId);

  const runLatestForecast = async () => {
    try {
      const runResult = await sessionClient.runSessionForecast({});
      setActiveSessionId(runResult.sessionId);
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
      await refreshFramesForSession(runResult.sessionId);
    } catch {
      // Keep map page map-first; runtime status details live on Forecast Overview.
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

  const forecastFitKey = framesMetadata?.forecast_id ?? activeSessionId ?? null;
  const geojson = ((selectedFrameGeoJson ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;
  const timelineDisabled = !framesMetadata || framesMetadata.frame_count === 0;
  const timelineStatus = frameError
    ? "Frame sequence unavailable"
    : frameLoading && !framesMetadata
      ? "Loading forecast…"
      : null;


  return (
    <AppShell
      title="Map / Forecast"
      subtitle="Current forecast map and plume overlay."
    >
      <main className="map-column">
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
          frameDurationSeconds={typeof framesMetadata?.metadata?.frame_duration_seconds === "number" ? framesMetadata.metadata.frame_duration_seconds : null}
          errorMessage={timelineStatus ?? frameError}
        />
      </main>
    </AppShell>
  );
}
