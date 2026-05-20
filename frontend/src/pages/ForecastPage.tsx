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
import type { ForecastFrameRasterPayload } from "../features/sessions/types/session.types";
import { isDatasetNormalScenario, resolveForecastMapDemoSource } from "../features/map/utils/resolveForecastMapDemoSource";

interface ActiveDatasetScenarioResponse {
  enabled: boolean;
  available: boolean;
  active_scenario_id?: string | null;
  selected_scenario_id?: string | null;
}
interface DatasetPlaybackState { enabled: boolean }
interface DatasetRasterPayload extends Omit<ForecastFrameRasterPayload, "session_id" | "frame_index"> {
  scenario_id: string;
  positive_count?: number;
}

export function ForecastPage() {
  const {
    activeSessionId,
    latestForecastBundle,
    selectedFeature,
    setSelectedFeature,
    setActiveSessionId,
    setLatestForecastBundle,
  } = useSessionForecastView();

  const inFlightRef = useRef(false);
  const hasAutoBootstrappedRef = useRef(false);
  const [mapPipelineStatus, setMapPipelineStatus] = useState("idle");
  const [datasetOverlayGeoJson, setDatasetOverlayGeoJson] = useState<GeoJsonFeatureCollection | null>(null);
  const [selectedDatasetRaster, setSelectedDatasetRaster] = useState<DatasetRasterPayload | null>(null);
  const [activeDataset, setActiveDataset] = useState<ActiveDatasetScenarioResponse | null>(null);
  const [datasetPlaybackState, setDatasetPlaybackState] = useState<DatasetPlaybackState | null>(null);
  const [datasetSourceCenter, setDatasetSourceCenter] = useState<[number, number] | null>(null);
  const datasetStateResolved = datasetPlaybackState !== null && activeDataset !== null;
  const datasetActive = datasetPlaybackState?.enabled === true
    && activeDataset?.enabled === true
    && Boolean(activeDataset?.selected_scenario_id || activeDataset?.active_scenario_id);
  const effectiveSessionId = datasetActive ? null : activeSessionId;
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
  } = useSessionForecastFrames(effectiveSessionId, { includeFrameSummary: false });
  const getDatasetPlaybackState = () => httpGet<DatasetPlaybackState>("/forecast-context/dataset-playback/state");
  const getActiveDatasetScenario = () => httpGet<ActiveDatasetScenarioResponse>("/forecast-context/dataset-scenarios/active");
  const getActiveDatasetOverlay = () => httpGet<GeoJsonFeatureCollection>("/forecast-context/dataset-scenarios/active/overlay");
  const getActiveDatasetRaster = () => httpGet<DatasetRasterPayload>("/forecast-context/dataset-scenarios/active/raster");

  const refreshDatasetOverlay = async () => {
    try {
      const [playback, active] = await Promise.all([getDatasetPlaybackState(), getActiveDatasetScenario()]);
      setDatasetPlaybackState(playback);
      setActiveDataset(active);
      const selectedDatasetScenarioId = active.selected_scenario_id ?? active.active_scenario_id ?? null;
      const datasetActive = playback.enabled === true && active.enabled === true && Boolean(selectedDatasetScenarioId);
      if (!datasetActive) {
        setDatasetOverlayGeoJson(null);
        setSelectedDatasetRaster(null);
        setDatasetSourceCenter(null);
        return;
      }
      const overlay = await getActiveDatasetOverlay();
      const features = Array.isArray(overlay.features) ? overlay.features : [];
      const sourceFeature = features.find((f) => f?.properties?.kind === "source");
      const geometryCoordinates = (sourceFeature?.geometry as { coordinates?: unknown } | undefined)?.coordinates;
      const coordinates = Array.isArray(geometryCoordinates) ? geometryCoordinates : null;
      const lon = typeof coordinates?.[0] === "number" ? coordinates[0] : null;
      const lat = typeof coordinates?.[1] === "number" ? coordinates[1] : null;
      setDatasetOverlayGeoJson(overlay);
      setDatasetSourceCenter(lat != null && lon != null ? [lon, lat] : null);
    } catch {
      setDatasetOverlayGeoJson(null);
      setSelectedDatasetRaster(null);
    }
  };
  useEffect(() => {
    if (!datasetActive) return;
    void (async () => {
      try {
        const raster = await getActiveDatasetRaster();
        setSelectedDatasetRaster(raster);
      } catch {
        setSelectedDatasetRaster(null);
      }
    })();
  }, [datasetActive]);

  const inspectGeoJson = (geojson: Record<string, unknown> | null, mode: "dataset-snapshot" | "session-frame" | "session-bundle" | "none") => {
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
    if (!datasetStateResolved || datasetActive || inFlightRef.current || hasAutoBootstrappedRef.current) {
      console.debug("[forecast-map] session bootstrap skipped", {
        reason: !datasetStateResolved ? "dataset_state_loading" : datasetActive ? "dataset_active" : null
      });
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
  }, [activeSessionId, latestForecastBundle, refreshFrames, refreshFramesForSession, setActiveSessionId, setLatestForecastBundle, datasetStateResolved, datasetActive]);

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
  const sourceMode = resolveForecastMapDemoSource({ datasetActive, hasUsableSelectedFrame, hasUsableSessionBundle });
  const activeScenarioId = activeDataset?.active_scenario_id ?? null;
  const selectedScenarioId = activeDataset?.selected_scenario_id ?? null;
  const currentScenarioId = selectedScenarioId ?? activeScenarioId;
  const normalScenarioActive = isDatasetNormalScenario(currentScenarioId);
  const datasetRasterHasPlume = !normalScenarioActive && (selectedDatasetRaster?.max ?? 0) > 0 && (selectedDatasetRaster?.positive_count ?? 0) > 0;
  const rasterOverlay = useMemo(
    () => (sourceMode === "dataset-snapshot" ? (datasetRasterHasPlume ? buildPlumeGridRasterOverlay(selectedDatasetRaster as unknown as ForecastFrameRasterPayload) : null) : sourceMode === "session-frame" ? buildPlumeGridRasterOverlay(selectedFrameRaster) : null),
    [sourceMode, selectedFrameRaster, selectedDatasetRaster, datasetRasterHasPlume]
  );

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    console.debug("[forecast-map] raster overlay ready", {
      hasRasterPayload: Boolean(selectedFrameRaster),
      hasRasterOverlay: Boolean(rasterOverlay),
      frameIndex: selectedFrameRaster?.frame_index,
      min: selectedFrameRaster?.min,
      max: selectedFrameRaster?.max,
      threshold: selectedFrameRaster?.threshold
    });
  }, [rasterOverlay, selectedFrameRaster]);

  const mapGeojson =
    sourceMode === "dataset-snapshot"
      ? (datasetOverlayGeoJson ?? null)
      : sourceMode === "session-frame"
      ? (selectedFrameGeoJson as unknown as GeoJsonFeatureCollection)
      : sourceMode === "session-bundle"
        ? sessionBundleGeojson
        : null;

  useEffect(() => {
    inspectGeoJson(mapGeojson as unknown as Record<string, unknown>, sourceMode);
  }, [mapGeojson, sourceMode]);

  const datasetOverlayIdentity = hasDatasetOverlay
    ? buildDatasetOverlayIdentity(datasetOverlayGeoJson as unknown as { features?: unknown[]; metadata?: Record<string, unknown> })
    : "dataset";
  const sessionSourceIdentity = `${activeSessionId ?? "none"}:${framesMetadata?.forecast_id ?? "none"}`;
  const forecastFitKey = `${sourceMode}:${sourceMode === "dataset-snapshot" ? datasetOverlayIdentity : sessionSourceIdentity}`;
  const selectedFrameKinds = countGeojsonKinds(selectedFrameGeoJson as unknown as Record<string, unknown> | null).kinds;

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
    console.debug("[forecast-map] dataset ownership decision", {
      datasetStateResolved,
      datasetActive,
      sourceMode,
      selectedDatasetScenarioId: selectedScenarioId,
      activeDatasetScenarioId: activeScenarioId,
      hasDatasetOverlay,
      hasDatasetRaster: Boolean(selectedDatasetRaster),
      datasetRasterMax: selectedDatasetRaster?.max ?? null,
      datasetRasterPositiveCount: selectedDatasetRaster?.positive_count ?? null,
      ignoredSessionFrame: datasetActive && Boolean(selectedFrameGeoJson),
      ignoredSessionRaster: datasetActive && Boolean(selectedFrameRaster)
    });
    console.debug("[forecast-map-demo] snapshot source", {
      sourceMode,
      datasetActive,
      selectedScenarioId,
      activeScenarioId,
      overlayFeatures: datasetOverlayGeoJson?.features?.length ?? 0,
      rasterAvailable: Boolean(selectedDatasetRaster),
      rasterMax: selectedDatasetRaster?.max ?? null,
      sessionIgnored: datasetActive && Boolean(activeSessionId)
    });
  }


  return (
    <AppShell title="Map / Forecast" subtitle="Current forecast map and plume overlay.">
      <main className="map-column">
        <ForecastMap
          geojson={mapGeojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
          center={sourceMode === "dataset-snapshot" ? datasetSourceCenter : null}
          autoFitKey={forecastFitKey}
          rasterOverlay={rasterOverlay}
          sourceMode={sourceMode}
          frameIndex={sourceMode === "dataset-snapshot" ? 0 : selectedFrameIndex}
        />
        {sourceMode === "dataset-snapshot" ? <p className="muted">Selected scenario snapshot</p> : <ForecastFrameTimeline
          frameCount={framesMetadata?.frame_count ?? 0}
          frameIndices={framesMetadata?.frame_indices ?? []}
          selectedFrameIndex={selectedFrameIndex}
          onSelectFrame={setSelectedFrameIndex}
          loading={frameLoading}
          disabled={!hasMultiFrameSession}
        />}
        {frameError ? <span className="sr-only">{frameError}</span> : null}
      </main>
    </AppShell>
  );
}
