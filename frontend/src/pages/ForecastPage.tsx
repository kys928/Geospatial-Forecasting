import { useEffect, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { useActiveForecast } from "../features/forecast-selection/context/ActiveForecastContext";
import { sessionClient } from "../features/sessions/api/sessionClient";

export function ForecastPage() {
  const { activeForecastBundle, activeFramesMetadata, activeForecastKind, activeSessionId, selectedFeature, setSelectedFeature, selectedFrameIndex, setSelectedFrameIndex, runActiveForecast, status } = useActiveForecast();
  const bootRef = useRef(false);
  const [frameGeoJson, setFrameGeoJson] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (bootRef.current || activeForecastBundle) return;
    bootRef.current = true;
    void runActiveForecast();
  }, [activeForecastBundle, runActiveForecast]);

  useEffect(() => {
    if (activeForecastKind !== "session_convlstm" || !activeSessionId || !activeFramesMetadata || activeFramesMetadata.frame_count <= 1) {
      setFrameGeoJson(null);
      return;
    }
    void sessionClient.getLatestForecastFrameGeoJson(activeSessionId, selectedFrameIndex).then(setFrameGeoJson).catch(() => setFrameGeoJson(null));
  }, [activeForecastKind, activeSessionId, activeFramesMetadata, selectedFrameIndex]);

  const geojson = ((frameGeoJson ?? activeForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;
  const frameCount = activeFramesMetadata?.frame_count ?? 1;
  const disabled = frameCount <= 1;

  return <AppShell title="Map / Forecast" subtitle="Current forecast map and plume overlay.">
    <main className="map-column">
      <ForecastMap geojson={geojson} selectedFeature={selectedFeature} onSelectFeature={setSelectedFeature} center={null} autoFitKey={activeFramesMetadata?.forecast_id ?? activeSessionId ?? "none"} />
      <ForecastFrameTimeline frameCount={frameCount} frameIndices={activeFramesMetadata?.frame_indices ?? [0]} selectedFrameIndex={selectedFrameIndex} onSelectFrame={setSelectedFrameIndex} loading={status === "running" || status === "loading"} disabled={disabled} />
    </main>
  </AppShell>;
}
