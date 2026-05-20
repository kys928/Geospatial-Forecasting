import { useEffect, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { buildDatasetOverlayIdentity, countGeojsonKinds } from "../features/map/utils/geojsonDiagnostics";
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
  const refreshDatasetOverlay = async () => {
    try {
      const context = await httpGet<Record<string, unknown>>("/forecast-context/latest?source=dataset");
      const forecast = (context.forecast as Record<string, unknown> | undefined) ?? {};
      const source = (context.source as Record<string, unknown> | undefined) ?? {};
      const inputSource = typeof forecast.input_source === "string" ? forecast.input_source : "";
      const plumeMetrics = (context.plume_metrics as Record<string, unknown> | undefined) ?? {}
      const maxPlumeScore = typeof plumeMetrics.max_plume_score === "number" ? plumeMetrics.max_plume_score : null;
      const lat = typeof source.latitude === "number" ? source.latitude : null;
      const lon = typeof source.longitude === "number" ? source.longitude : null;
      if (import.meta.env.DEV) {
        console.debug("[forecast-map] dataset latest", { inputSource, maxPlumeScore, latitude: lat, longitude: lon });
      }
      if (inputSource !== "dataset_playback") {
        setDatasetOverlayGeoJson(null);
        setDatasetSourceCenter(lat != null && lon != null ? [lon, lat] : null);
        return;
      }
      const overlay = await httpGet<GeoJsonFeatureCollection>("/forecast-context/dataset-scenarios/active/overlay");
      const features = Array.isArray(overlay.features) ? overlay.features : [];
      const kinds = Array.from(new Set(features.map((f) => (typeof f?.properties?.kind === "string" ? f.properties.kind : "unknown"))));
      if (import.meta.env.DEV) {
        console.debug("[forecast-map] dataset overlay fetched", {
          featureCount: features.length,
          metadata: (overlay as unknown as { metadata?: unknown }).metadata ?? null,
          kinds,
        });
      }
      setDatasetOverlayGeoJson(overlay);
      setDatasetSourceCenter(lat != null && lon != null ? [lon, lat] : null);
    } catch {
      setDatasetOverlayGeoJson(null);
    }
  };

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
    void refreshDatasetOverlay();
  }, []);

  useEffect(() => {
    const refreshIfVisible = () => {
      if (document.visibilityState === "visible") {
        void refreshDatasetOverlay();
      }
    };
    const handleFocus = () => {
      void refreshDatasetOverlay();
    };
    window.addEventListener("focus", handleFocus);
    document.addEventListener("visibilitychange", refreshIfVisible);
    return () => {
      window.removeEventListener("focus", handleFocus);
      document.removeEventListener("visibilitychange", refreshIfVisible);
    };
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
      await refreshDatasetOverlay();
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
  const hasDatasetOverlay = Boolean(datasetOverlayGeoJson?.features?.length);
  const sourceMode: "dataset" | "session" = hasDatasetOverlay ? "dataset" : "session";
  const mapGeojson = hasDatasetOverlay ? datasetOverlayGeoJson : sessionGeojson;

  useEffect(() => {
    inspectGeoJson(mapGeojson as unknown as Record<string, unknown>, sourceMode);
  }, [mapGeojson, sourceMode]);

  const datasetOverlayIdentity = hasDatasetOverlay
    ? buildDatasetOverlayIdentity(datasetOverlayGeoJson as unknown as { features?: unknown[]; metadata?: Record<string, unknown> })
    : "dataset";
  const forecastFitKey = `${sourceMode}:${sourceMode === "dataset" ? datasetOverlayIdentity : (framesMetadata?.forecast_id ?? activeSessionId ?? "none")}`;
  const timelineDisabled = !hasMultiFrameSession;
  const datasetKinds = countGeojsonKinds(datasetOverlayGeoJson as unknown as Record<string, unknown> | null).kinds;
  const mapKinds = countGeojsonKinds(mapGeojson as unknown as Record<string, unknown> | null).kinds;

  if (import.meta.env.DEV) {
    console.debug("[forecast-map] render source", {
      sourceMode,
      datasetFeatures: datasetOverlayGeoJson?.features?.length ?? 0,
      sessionFeatures: sessionGeojson?.features?.length ?? 0,
      mapFeatures: mapGeojson?.features?.length ?? 0,
      datasetKinds,
      mapKinds,
      firstDatasetFeature: datasetOverlayGeoJson?.features?.[0] ?? null,
      autoFitKey: forecastFitKey,
      datasetSourceCenter
    });
  }


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
