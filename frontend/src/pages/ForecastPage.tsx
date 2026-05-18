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
  } = useSessionForecastFrames(datasetModeEnabled ? null : activeSessionId);
  const geojson = ((datasetOverlay ?? selectedFrameGeoJson ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;

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
    try {
      const runResult = await sessionClient.runSessionForecast({});
      setActiveSessionId(runResult.sessionId);
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
    } catch {
      // Keep map page map-first; runtime status details live on Forecast Overview.
    }
  };

  useEffect(() => {
    httpGet<{ enabled: boolean }>("/forecast-context/dataset-playback/state")
      .then((playback) => {
        if (!playback.enabled) {
          setDatasetModeEnabled(false);
          setDatasetOverlay(null);
          setDatasetCenter(null);
          void runLatestForecast();
          return;
        }
        setDatasetModeEnabled(true);
        setLatestForecastBundle(null, null);
        clearFrames();
        return httpGet<ActiveDatasetScenarioResponse>("/forecast-context/dataset-scenarios/active")
          .then((active) => {
            const lat = active?.scenario?.source?.latitude;
            const lon = active?.scenario?.source?.longitude;
            if (typeof lat === "number" && typeof lon === "number") {
              setDatasetCenter([lon, lat]);
            }
            return httpGet<GeoJsonFeatureCollection>("/forecast-context/dataset-scenarios/active/overlay")
              .then((overlay) => setDatasetOverlay(overlay))
              .catch(() => setDatasetOverlay(null));
          }).catch(() => {
            setDatasetOverlay(null);
            setDatasetCenter(null);
          });
      })
      .catch(() => {
        setDatasetOverlay(null);
        setDatasetCenter(null);
        setDatasetModeEnabled(false);
        void runLatestForecast();
      });
  }, []);

  useEffect(() => {
    if (!datasetModeEnabled && latestForecastBundle) {
      void refreshFrames();
    }
  }, [datasetModeEnabled, latestForecastBundle, refreshFrames]);

  const summaryMetadata = latestForecastBundle?.summary?.metadata as Record<string, unknown> | undefined;
  const adapterMetadata = summaryMetadata?.input_adapter_metadata as Record<string, unknown> | undefined;
  const predictionEngine = typeof framesMetadata?.metadata?.prediction_engine === "string"
    ? framesMetadata.metadata.prediction_engine
    : undefined;


  return (
    <AppShell
      title="Map / Forecast"
      subtitle="Current forecast map and plume overlay."
    >
      <main className="map-column">
        <ForecastMap
          geojson={datasetModeEnabled ? datasetOverlay : geojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
          center={datasetCenter}
        />
        {!datasetModeEnabled ? (
          <ForecastFrameTimeline
            frameCount={framesMetadata?.frame_count ?? 0}
            frameIndices={framesMetadata?.frame_indices ?? []}
            selectedFrameIndex={selectedFrameIndex}
            onSelectFrame={setSelectedFrameIndex}
            loading={frameLoading}
            disabled={!framesMetadata || (framesMetadata.frame_count <= 1)}
            metadata={framesMetadata?.metadata}
            selectedFrameSummary={selectedFrameSummary}
            predictionTrust={typeof adapterMetadata?.prediction_trust === "string" ? adapterMetadata.prediction_trust : null}
            inputMode={typeof adapterMetadata?.input_mode === "string" ? adapterMetadata.input_mode : null}
            modelName={framesMetadata?.model ?? "ConvLSTM multi-step"}
            predictionEngine={predictionEngine ?? "torch_multistep"}
            errorMessage={frameError}
          />
        ) : null}
      </main>
    </AppShell>
  );
}
