import { useEffect } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";

export function ForecastPage() {
  const {
    activeSessionId,
    latestForecastBundle,
    selectedFeature,
    setSelectedFeature,
  } = useSessionForecastView();

  const {
    framesMetadata,
    selectedFrameIndex,
    selectedFrameGeoJson,
    frameLoading,
    frameError,
    refreshFrames,
    setSelectedFrameIndex,
  } = useSessionForecastFrames(activeSessionId);

  useEffect(() => {
    if (!activeSessionId) return;
    if (!latestForecastBundle) return;
    void refreshFrames();
  }, [activeSessionId, latestForecastBundle, refreshFrames]);

  useEffect(() => {
    if (!import.meta.env.DEV || !selectedFrameGeoJson) {
      return;
    }

    const features = Array.isArray((selectedFrameGeoJson as { features?: unknown }).features)
      ? ((selectedFrameGeoJson as { features: Array<{ properties?: unknown }> }).features)
      : [];

    console.debug("[ForecastPage] selected frame GeoJSON", {
      featureCount: features.length,
      sampleProperties: features.slice(0, 3).map((feature) => feature?.properties ?? null),
    });
  }, [selectedFrameGeoJson]);

  const forecastFitKey = framesMetadata?.forecast_id ?? activeSessionId ?? null;
  const geojson = ((selectedFrameGeoJson ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;
  const timelineDisabled = !framesMetadata || framesMetadata.frame_count === 0;
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
        />
        {frameError ? <span className="sr-only">{frameError}</span> : null}
      </main>
    </AppShell>
  );
}
