import { useEffect, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { countGeojsonKinds } from "../features/map/utils/geojsonDiagnostics";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";
import { httpGet } from "../services/api/http";

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
  } = useSessionForecastFrames(activeSessionId, { includeFrameSummary: false });

  const inFlightRef = useRef(false);
  const hasAutoBootstrappedRef = useRef(false);
  const [mapPipelineStatus, setMapPipelineStatus] = useState("idle");
  const [datasetOverlayGeoJson, setDatasetOverlayGeoJson] = useState<GeoJsonFeatureCollection | null>(null);
  const [datasetSourceCenter, setDatasetSourceCenter] = useState<[number, number] | null>(null);

  const inspectGeoJson = (geojson: Record<string, unknown> | null, mode: "dataset" | "session") => {
    const baseCounts = countGeojsonKinds(geojson);
    const plumePointCount = Array.isArray((geojson as { features?: unknown[] } | null)?.features)
      ? (geojson as { features: Array<{ properties?: { kind?: unknown } }> }).features.filter((feature) => feature?.properties?.kind === "plume_point").length
      : 0;
    const plumeBandCount = Array.isArray((geojson as { features?: unknown[] } | null)?.features)
      ? (geojson as { features: Array<{ properties?: { kind?: unknown } }> }).features.filter((feature) => feature?.properties?.kind === "plume_band").length
      : 0;
    if (import.meta.env.DEV) {
      console.debug("[forecast-map] source", {
        sourceMode: mode,
        featureCount: baseCounts.featureCount,
        plumePointCount,
        plumeBandCount,
        plumeCellCount: baseCounts.plumeCellCount,
      });
    }
  };

  useEffect(() => {
    void (async () => {
      try {
        const [activeScenario, context] = await Promise.all([
          httpGet<{ enabled: boolean; available: boolean; active_scenario_id?: string | null; selected_scenario_id?: string | null }>("/forecast-context/dataset-scenarios/active"),
          httpGet<Record<string, unknown>>("/forecast-context/latest?source=dataset"),
        ]);
        const scenarioId = activeScenario.active_scenario_id ?? activeScenario.selected_scenario_id ?? null;
        const inputSource = typeof (context.forecast as Record<string, unknown> | undefined)?.input_source === "string"
          ? String((context.forecast as Record<string, unknown>).input_source)
          : "";
        const shouldUseDataset = Boolean(activeScenario.enabled && scenarioId && inputSource === "dataset_playback");
        if (!shouldUseDataset) {
          setDatasetOverlayGeoJson(null);
          return;
        }
        const overlay = await httpGet<GeoJsonFeatureCollection>("/forecast-context/dataset-scenarios/active/overlay");
        setDatasetOverlayGeoJson(overlay);
        const src = (context.source as Record<string, unknown> | undefined) ?? {};
        const lat = typeof src.latitude === "number" ? src.latitude : null;
        const lon = typeof src.longitude === "number" ? src.longitude : null;
        setDatasetSourceCenter(lat != null && lon != null ? [lon, lat] : null);
      } catch {
        setDatasetOverlayGeoJson(null);
      }
    })();
  }, []);

  useEffect(() => {
    if (inFlightRef.current || hasAutoBootstrappedRef.current) {
      return;
    }
    inFlightRef.current = true;
    const ensureForecast = async () => {
      if (activeSessionId && latestForecastBundle) {
        setMapPipelineStatus("loading_frames");
        await refreshFrames(activeSessionId);
        setMapPipelineStatus("ready");
        hasAutoBootstrappedRef.current = true;
        return;
      }
      setMapPipelineStatus("creating_session");
      setMapPipelineStatus("predicting");
      const runResult = await sessionClient.runSessionForecast({});
      setMapPipelineStatus("loading_bundle");
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId, { includeExplanation: false });
      setActiveSessionId(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
      setMapPipelineStatus("loading_frames");
      await refreshFramesForSession(runResult.sessionId);
      setMapPipelineStatus("ready");
      hasAutoBootstrappedRef.current = true;
    };
    void ensureForecast().catch(() => {
      setMapPipelineStatus("error");
    }).finally(() => {
      inFlightRef.current = false;
    });
  }, [activeSessionId, latestForecastBundle, refreshFrames, refreshFramesForSession, setActiveSessionId, setLatestForecastBundle]);

  useEffect(() => {
    if (!selectedFrameGeoJson) return;
    if (mapPipelineStatus === "loading_frames" || mapPipelineStatus === "idle") {
      setMapPipelineStatus("ready");
    }
  }, [mapPipelineStatus, selectedFrameGeoJson]);

  const hasMultiFrameSession = Boolean(framesMetadata && framesMetadata.frame_count > 1);
  const sessionGeojson = ((selectedFrameGeoJson ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;
  const sourceMode: "dataset" | "session" = datasetOverlayGeoJson ? "dataset" : "session";
  const mapGeojson = sourceMode === "dataset" ? datasetOverlayGeoJson : sessionGeojson;

  useEffect(() => {
    inspectGeoJson(mapGeojson as unknown as Record<string, unknown>, sourceMode);
  }, [mapGeojson, sourceMode]);

  const forecastFitKey = `${sourceMode}:${datasetOverlayGeoJson ? "dataset" : (framesMetadata?.forecast_id ?? activeSessionId ?? "none")}`;
  const timelineDisabled = !hasMultiFrameSession;

  return (
    <AppShell title="Map / Forecast" subtitle="Current forecast map and plume overlay.">
      <main className="map-column">
        <ForecastMap
          geojson={mapGeojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
          center={sourceMode === "dataset" ? datasetSourceCenter : null}
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
