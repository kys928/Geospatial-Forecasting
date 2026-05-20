import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { buildDatasetOverlayIdentity, countGeojsonKinds } from "../features/map/utils/geojsonDiagnostics";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";
import { httpGet } from "../services/api/http";
import { buildPlumeGridRasterOverlay } from "../features/map/utils/plumeGridRaster";

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
    selectedFrameSummary,
    selectedFrameGeoJson,
    selectedFrameRaster,
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

  const inspectGeoJson = (geojson: Record<string, unknown> | null, mode: "dataset" | "session-frame" | "session-bundle" | "none") => {
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
  const selectedFrameFeatures = Array.isArray((selectedFrameGeoJson as { features?: unknown[] } | null)?.features)
    ? (selectedFrameGeoJson as { features: unknown[] }).features.length
    : 0;
  const sessionBundleGeojson = (latestForecastBundle?.geojson ?? null) as GeoJsonFeatureCollection | null;
  const sessionBundleFeatures = Array.isArray((sessionBundleGeojson as { features?: unknown[] } | null)?.features)
    ? (sessionBundleGeojson as { features: unknown[] }).features.length
    : 0;
  const hasUsableSelectedFrame = hasMultiFrameSession && selectedFrameFeatures > 0;
  const hasUsableSessionBundle = sessionBundleFeatures > 0;
  const hasDatasetOverlay = Boolean(datasetOverlayGeoJson?.features?.length);
  const sourceMode: "dataset" | "session-frame" | "session-bundle" | "none" =
    hasUsableSelectedFrame ? "session-frame" : hasUsableSessionBundle ? "session-bundle" : hasDatasetOverlay ? "dataset" : "none";
  const rasterOverlay = useMemo(
    () => (sourceMode === "session-frame" ? buildPlumeGridRasterOverlay(selectedFrameRaster) : null),
    [sourceMode, selectedFrameRaster]
  );

  const mapGeojson =
    sourceMode === "session-frame"
      ? (selectedFrameGeoJson as unknown as GeoJsonFeatureCollection)
      : sourceMode === "session-bundle"
        ? sessionBundleGeojson
        : hasDatasetOverlay
          ? datasetOverlayGeoJson
          : null;

  useEffect(() => {
    inspectGeoJson(mapGeojson as unknown as Record<string, unknown>, sourceMode);
  }, [mapGeojson, sourceMode]);

  const datasetOverlayIdentity = hasDatasetOverlay
    ? buildDatasetOverlayIdentity(datasetOverlayGeoJson as unknown as { features?: unknown[]; metadata?: Record<string, unknown> })
    : "dataset";
  const sessionSourceIdentity = `${activeSessionId ?? "none"}:${framesMetadata?.forecast_id ?? "none"}`;
  const forecastFitKey = `${sourceMode}:${sourceMode === "dataset" ? datasetOverlayIdentity : sessionSourceIdentity}`;
  const timelineDisabled = !hasMultiFrameSession;
  const datasetKinds = countGeojsonKinds(datasetOverlayGeoJson as unknown as Record<string, unknown> | null).kinds;
  const selectedFrameKinds = countGeojsonKinds(selectedFrameGeoJson as unknown as Record<string, unknown> | null).kinds;
  const mapKinds = countGeojsonKinds(mapGeojson as unknown as Record<string, unknown> | null).kinds;

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    console.debug("[forecast-map] selected frame changed", {
      selectedFrameIndex,
      featureCount: selectedFrameFeatures,
      kinds: selectedFrameKinds
    });
    console.debug("[forecast-map] raster grid", {
      frameIndex: selectedFrameRaster?.frame_index ?? selectedFrameIndex,
      shape: selectedFrameRaster?.shape ?? null,
      min: selectedFrameRaster?.min ?? null,
      max: selectedFrameRaster?.max ?? null,
      threshold: selectedFrameRaster?.threshold ?? null,
      bounds: selectedFrameRaster?.bounds ?? null,
      hasImage: Boolean(rasterOverlay?.imageDataUrl)
    });
  }, [selectedFrameFeatures, selectedFrameIndex, selectedFrameKinds, selectedFrameRaster, rasterOverlay]);

  useEffect(() => {
    if (!import.meta.env.DEV || !hasUsableSelectedFrame) return;
    const metadataSummary = (framesMetadata?.metadata as Record<string, unknown> | undefined)?.frame_summaries;
    const frameSummaries = Array.isArray(metadataSummary) ? metadataSummary : [];
    const frameSummary = selectedFrameSummary ?? {};
    const maxConcentration =
      typeof (frameSummary as { max_concentration?: unknown }).max_concentration === "number"
        ? (frameSummary as { max_concentration: number }).max_concentration
        : null;
    const plumeCellCount =
      typeof (frameSummary as { plume_cell_count?: unknown }).plume_cell_count === "number"
        ? (frameSummary as { plume_cell_count: number }).plume_cell_count
        : null;
    console.debug("[forecast-map] selected frame summary", {
      selectedFrameIndex,
      maxConcentration,
      plumeCellCount,
      metadataFrameSummaries: frameSummaries.length
    });
  }, [framesMetadata?.metadata, hasUsableSelectedFrame, selectedFrameIndex, selectedFrameSummary]);

  if (import.meta.env.DEV) {
    console.debug("[forecast-map] render source", {
      sourceMode,
      selectedFrameIndex,
      frameCount: framesMetadata?.frame_count ?? 0,
      selectedFrameFeatures,
      sessionBundleFeatures,
      datasetFeatures: datasetOverlayGeoJson?.features?.length ?? 0,
      mapFeatures: mapGeojson?.features?.length ?? 0,
      datasetKinds,
      mapKinds,
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
          rasterOverlay={rasterOverlay}
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
