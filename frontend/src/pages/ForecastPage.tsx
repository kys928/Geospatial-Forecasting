import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { countGeojsonKinds } from "../features/map/utils/geojsonDiagnostics";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";
import { httpPost } from "../services/api/http";
import { buildPlumeGridRasterOverlay } from "../features/map/utils/plumeGridRaster";

interface DatasetPlaybackState { enabled: boolean; active_scenario_id?: string | null; playback_running?: boolean }

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message;
  return "Active ConvLSTM forecast request failed.";
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
  const [activeForecastError, setActiveForecastError] = useState<string | null>(null);

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

  const disableDatasetPlayback = () => httpPost<DatasetPlaybackState, { enabled: boolean; playback_running: boolean }>(
    "/forecast-context/dataset-playback/state",
    { enabled: false, playback_running: false }
  );

  const runActiveForecast = async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      setActiveForecastError(null);
      setMapPipelineStatus("predicting");
      const runResult = await sessionClient.runSessionForecast({ metadata: { requested_forecast_mode: "active_model" } });
      setMapPipelineStatus("loading_bundle");
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId, { includeExplanation: false });
      setActiveSessionId(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
      setMapPipelineStatus("loading_frames");
      await refreshFramesForSession(runResult.sessionId);
      setMapPipelineStatus("ready");
      hasAutoBootstrappedRef.current = true;
    } catch (error) {
      setActiveForecastError(errorMessage(error));
      setMapPipelineStatus("error");
      hasAutoBootstrappedRef.current = true;
    } finally {
      inFlightRef.current = false;
    }
  };

  useEffect(() => {
    void disableDatasetPlayback().catch(() => undefined);
  }, []);

  useEffect(() => {
    if (inFlightRef.current || hasAutoBootstrappedRef.current) return;
    inFlightRef.current = true;
    const ensureForecast = async () => {
      try {
        setActiveForecastError(null);
        if (activeSessionId && latestForecastBundle) {
          setMapPipelineStatus("loading_frames");
          await refreshFrames(activeSessionId);
          setMapPipelineStatus("ready");
          hasAutoBootstrappedRef.current = true;
          return;
        }
        setMapPipelineStatus("creating_session");
        const runResult = await sessionClient.runSessionForecast({ metadata: { requested_forecast_mode: "active_model" } });
        setMapPipelineStatus("loading_bundle");
        const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId, { includeExplanation: false });
        setActiveSessionId(runResult.sessionId);
        setLatestForecastBundle(runResult.sessionId, bundle);
        setMapPipelineStatus("loading_frames");
        await refreshFramesForSession(runResult.sessionId);
        setMapPipelineStatus("ready");
        hasAutoBootstrappedRef.current = true;
      } catch (error) {
        setActiveForecastError(errorMessage(error));
        setMapPipelineStatus("error");
        hasAutoBootstrappedRef.current = true;
      } finally {
        inFlightRef.current = false;
      }
    };
    void ensureForecast();
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
  const sourceMode: "session-frame" | "session-bundle" | "none" = hasUsableSelectedFrame ? "session-frame" : hasUsableSessionBundle ? "session-bundle" : "none";
  const rasterOverlay = useMemo(
    () => (sourceMode === "session-frame" ? buildPlumeGridRasterOverlay(selectedFrameRaster) : null),
    [sourceMode, selectedFrameRaster]
  );

  const mapGeojson = sourceMode === "session-frame"
    ? (selectedFrameGeoJson as unknown as GeoJsonFeatureCollection)
    : sourceMode === "session-bundle"
      ? sessionBundleGeojson
      : null;

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const baseCounts = countGeojsonKinds(mapGeojson as unknown as Record<string, unknown> | null);
    console.debug("[forecast-map] active session source", {
      sourceMode,
      featureCount: baseCounts.featureCount,
      plumeCellCount: baseCounts.plumeCellCount,
      rasterFrameIndex: selectedFrameRaster?.frame_index ?? selectedFrameIndex,
      hasRasterOverlay: Boolean(rasterOverlay?.imageDataUrl)
    });
  }, [mapGeojson, sourceMode, selectedFrameRaster, selectedFrameIndex, rasterOverlay]);

  useEffect(() => {
    if (!import.meta.env.DEV || !hasUsableSelectedFrame) return;
    const frameSummary = selectedFrameSummary ?? {};
    console.debug("[forecast-map] selected active frame summary", {
      selectedFrameIndex,
      maxConcentration: typeof (frameSummary as { max_concentration?: unknown }).max_concentration === "number" ? (frameSummary as { max_concentration: number }).max_concentration : null,
      plumeCellCount: typeof (frameSummary as { plume_cell_count?: unknown }).plume_cell_count === "number" ? (frameSummary as { plume_cell_count: number }).plume_cell_count : null
    });
  }, [hasUsableSelectedFrame, selectedFrameIndex, selectedFrameSummary]);

  const sessionFrameMetadata = (framesMetadata?.metadata as Record<string, unknown> | undefined) ?? {};
  const sessionRasterMetadata = (selectedFrameRaster?.metadata as Record<string, unknown> | undefined) ?? {};
  const sessionProvenance = (sessionFrameMetadata.provenance as Record<string, unknown> | undefined) ?? sessionFrameMetadata;
  const provenanceError = activeForecastError ?? (frameError ? `Active ConvLSTM frame unavailable: ${frameError}` : null);
  const provenanceLabel = sessionProvenance.forecast_source === "active_model_inference" && sessionProvenance.model_family === "ConvLSTM" && sessionProvenance.fallback_used !== true
    ? `Active ConvLSTM forecast: ${String(sessionProvenance.model_id ?? sessionRasterMetadata.model ?? "unknown")}`
    : provenanceError
      ? `Active ConvLSTM unavailable: ${provenanceError}`
      : mapPipelineStatus === "error"
        ? "Active ConvLSTM unavailable: forecast request failed."
        : "Active ConvLSTM forecast starting.";
  const timelineDisabled = !hasMultiFrameSession;

  return (
    <AppShell title="Map / Forecast" subtitle="Current forecast map and plume overlay.">
      <main className="map-column">
        <div className="panel" style={{ padding: "8px 12px", display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between", flexWrap: "wrap" }}>
          <div>
            <div className="muted">Forecast mode: Live / active ConvLSTM forecast</div>
            <strong>{provenanceLabel}</strong>
          </div>
        </div>
        <ForecastMap
          geojson={mapGeojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
          center={null}
          autoFitKey={`active:${activeSessionId ?? "none"}:${framesMetadata?.forecast_id ?? "none"}`}
          rasterOverlay={rasterOverlay}
          sourceMode={sourceMode}
          frameIndex={selectedFrameIndex}
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
