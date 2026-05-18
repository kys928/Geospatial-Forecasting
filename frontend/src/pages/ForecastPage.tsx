import { useEffect, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import { ForecastFrameTimeline } from "../features/map/components/ForecastFrameTimeline";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { useSessionForecastFrames } from "../features/sessions/hooks/useSessionForecastFrames";
import { httpGet } from "../services/api/http";

export function ForecastPage() {
  const {
    activeSessionId,
    latestForecastBundle,
    selectedFeature,
    setSelectedFeature,
    setActiveSessionId,
    setLatestForecastBundle
  } = useSessionForecastView();

  const [datasetOverlay, setDatasetOverlay] = useState<GeoJsonFeatureCollection | null>(null);
  const [datasetCenter, setDatasetCenter] = useState<[number, number] | null>(null);
  const [datasetModeEnabled, setDatasetModeEnabled] = useState(false);
  const [forecastSourceMode, setForecastSourceMode] = useState<"convlstm" | "dataset">("convlstm");
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
    clearFrames
  } = useSessionForecastFrames(forecastSourceMode === "convlstm" ? activeSessionId : null);

  type ActiveDatasetScenarioResponse = {
    enabled: boolean;
    available: boolean;
    active_scenario_id?: string | null;
    selected_scenario_id?: string | null;
    scenario?: {
      source?: {
        latitude?: number;
        longitude?: number;
      };
    } | null;
  };

  const runLatestForecast = async () => {
    setForecastRunning(true);
    try {
      const runResult = await sessionClient.runSessionForecast({});
      setActiveSessionId(runResult.sessionId);
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
      setForecastSourceMode("convlstm");
      await refreshFrames();
    } catch {
      // Keep map page map-first; runtime status details live on Forecast Overview.
    } finally {
      setForecastRunning(false);
    }
  };

  const loadDatasetOverlay = async () => {
    try {
      const active = await httpGet<ActiveDatasetScenarioResponse>("/forecast-context/dataset-scenarios/active");
      const lat = active?.scenario?.source?.latitude;
      const lon = active?.scenario?.source?.longitude;
      if (typeof lat === "number" && typeof lon === "number") {
        setDatasetCenter([lon, lat]);
      }
      const overlay = await httpGet<GeoJsonFeatureCollection>("/forecast-context/dataset-scenarios/active/overlay");
      setDatasetOverlay(overlay);
    } catch {
      setDatasetOverlay(null);
      setDatasetCenter(null);
    }
  };

  useEffect(() => {
    httpGet<{ enabled: boolean }>("/forecast-context/dataset-playback/state")
      .then((playback) => {
        setDatasetModeEnabled(playback.enabled);
        if (playback.enabled) {
          void loadDatasetOverlay();
        } else {
          setDatasetOverlay(null);
          setDatasetCenter(null);
        }
        void runLatestForecast();
      })
      .catch(() => {
        setDatasetOverlay(null);
        setDatasetCenter(null);
        setDatasetModeEnabled(false);
        void runLatestForecast();
      });
  }, []);

  useEffect(() => {
    if (forecastSourceMode === "convlstm" && latestForecastBundle) {
      void refreshFrames();
    }
  }, [forecastSourceMode, latestForecastBundle, refreshFrames]);

  const switchToConvLstm = () => {
    setForecastSourceMode("convlstm");
    if (!latestForecastBundle) {
      void runLatestForecast();
      return;
    }
    void refreshFrames();
  };

  const switchToDataset = () => {
    if (!datasetModeEnabled) {
      return;
    }
    setForecastSourceMode("dataset");
    void loadDatasetOverlay();
  };

  const summaryMetadata = latestForecastBundle?.summary?.metadata as Record<string, unknown> | undefined;
  const adapterMetadata = summaryMetadata?.input_adapter_metadata as Record<string, unknown> | undefined;
  const predictionEngine = typeof framesMetadata?.metadata?.prediction_engine === "string"
    ? framesMetadata.metadata.prediction_engine
    : undefined;
  const geojson = forecastSourceMode === "dataset"
    ? (datasetOverlay ?? null)
    : ((selectedFrameGeoJson ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;
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
        <div className="forecast-source-toggle">
          <button type="button" className={forecastSourceMode === "convlstm" ? "is-active" : ""} onClick={switchToConvLstm}>ConvLSTM forecast</button>
          <button type="button" className={forecastSourceMode === "dataset" ? "is-active" : ""} onClick={switchToDataset} disabled={!datasetModeEnabled}>Dataset playback</button>
          <button type="button" className="primary-button run-forecast-button" onClick={() => void runLatestForecast()} disabled={forecastRunning}>
            {latestForecastBundle ? "Refresh ConvLSTM forecast" : "Run ConvLSTM forecast"}
          </button>
          {import.meta.env.DEV ? (
            <span className="timeline-note">source: {forecastSourceMode} · frame: {selectedFrameIndex + 1} / {framesMetadata?.frame_count ?? 0}</span>
          ) : null}
        </div>
        <ForecastMap
          geojson={geojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
          center={forecastSourceMode === "dataset" ? datasetCenter : null}
        />
        {forecastSourceMode === "convlstm" ? (
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
        ) : null}
      </main>
    </AppShell>
  );
}
