import { useEffect, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { countGeojsonKinds } from "../features/map/utils/geojsonDiagnostics";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";

type MapPipelineStatus = "idle" | "creating_session" | "predicting" | "loading_bundle" | "loading_frames" | "ready" | "no_plume_cells" | "error";

export function ForecastPage() {
  const {
    activeSessionId,
    latestForecastBundle,
    selectedFeature,
    setSelectedFeature,
    setActiveSessionId,
    setLatestForecastBundle,
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

  const inFlightRef = useRef(false);
  const hasAutoBootstrappedRef = useRef(false);
  const [mapPipelineStatus, setMapPipelineStatus] = useState<MapPipelineStatus>("idle");
  const [mapPipelineError, setMapPipelineError] = useState<string | null>(null);
  const [lastFrameFeatureCount, setLastFrameFeatureCount] = useState<number | null>(null);
  const [lastFramePlumeCellCount, setLastFramePlumeCellCount] = useState<number | null>(null);
  const [lastFrameKinds, setLastFrameKinds] = useState<string[]>([]);

  const inspectFrameGeoJson = (geojson: Record<string, unknown> | null) => {
    const counts = countGeojsonKinds(geojson);
    setLastFrameFeatureCount(counts.featureCount);
    setLastFramePlumeCellCount(counts.plumeCellCount);
    setLastFrameKinds(counts.kinds);
    if (import.meta.env.DEV) {
      console.debug("[forecast-map] frame geojson", counts);
    }
    return counts;
  };

  useEffect(() => {
    if (inFlightRef.current || hasAutoBootstrappedRef.current) {
      return;
    }

    inFlightRef.current = true;
    setMapPipelineError(null);

    const ensureForecast = async () => {
      if (activeSessionId && latestForecastBundle) {
        setMapPipelineStatus("loading_frames");
        const frameResult = await refreshFrames(activeSessionId);
        const counts = inspectFrameGeoJson(frameResult?.selectedFrameGeoJson ?? null);
        setMapPipelineStatus(counts.plumeCellCount > 0 ? "ready" : "no_plume_cells");
        hasAutoBootstrappedRef.current = true;
        return;
      }

      setMapPipelineStatus("creating_session");
      if (import.meta.env.DEV) {
        console.debug("[forecast-map] auto-running session forecast");
      }

      setMapPipelineStatus("predicting");
      const runResult = await sessionClient.runSessionForecast({});
      setMapPipelineStatus("loading_bundle");
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId);

      setActiveSessionId(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);

      const forecastId = runResult.prediction?.forecast_id ?? bundle.summary?.forecast_id ?? null;
      if (import.meta.env.DEV) {
        console.debug("[forecast-map] session forecast ready", {
          sessionId: runResult.sessionId,
          forecastId,
        });
      }

      setMapPipelineStatus("loading_frames");
      const frameResult = await refreshFramesForSession(runResult.sessionId);
      const counts = inspectFrameGeoJson(frameResult?.selectedFrameGeoJson ?? null);
      setMapPipelineStatus(counts.plumeCellCount > 0 ? "ready" : "no_plume_cells");
      hasAutoBootstrappedRef.current = true;
    };

    void ensureForecast().catch((error: unknown) => {
      setMapPipelineStatus("error");
      setMapPipelineError(error instanceof Error ? error.message : String(error));
      if (import.meta.env.DEV) {
        console.debug("[forecast-map] auto-run failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }).finally(() => {
      inFlightRef.current = false;
    });
  }, [
    activeSessionId,
    latestForecastBundle,
    refreshFrames,
    refreshFramesForSession,
    setActiveSessionId,
    setLatestForecastBundle,
  ]);

  useEffect(() => {
    if (!selectedFrameGeoJson) return;
    const counts = inspectFrameGeoJson(selectedFrameGeoJson);
    if (mapPipelineStatus === "loading_frames" || mapPipelineStatus === "idle") {
      setMapPipelineStatus(counts.plumeCellCount > 0 ? "ready" : "no_plume_cells");
    }
  }, [mapPipelineStatus, selectedFrameGeoJson]);

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
        {import.meta.env.DEV ? (
          <div style={{ position: "absolute", right: 12, top: 12, zIndex: 10, background: "rgba(15,23,42,0.85)", color: "#e2e8f0", fontSize: 11, lineHeight: 1.3, padding: "6px 8px", borderRadius: 6, maxWidth: 320 }}>
            <div><strong>status:</strong> {mapPipelineStatus}</div>
            <div><strong>session:</strong> {activeSessionId ? `${activeSessionId.slice(0, 8)}…` : "none"}</div>
            <div><strong>features:</strong> {lastFrameFeatureCount ?? "-"}</div>
            <div><strong>plume_cells:</strong> {lastFramePlumeCellCount ?? "-"}</div>
            <div><strong>kinds:</strong> {lastFrameKinds.join(", ") || "-"}</div>
            <div><strong>frameError:</strong> {frameError ?? "-"}</div>
            <div><strong>error:</strong> {mapPipelineError ?? "-"}</div>
          </div>
        ) : null}
      </main>
    </AppShell>
  );
}
