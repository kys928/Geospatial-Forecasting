import { useEffect, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { sessionClient } from "../features/sessions/api/sessionClient";
import { httpGet } from "../services/api/http";

export function ForecastPage() {
  const {
    latestForecastBundle,
    selectedFeature,
    setSelectedFeature,
    setActiveSessionId,
    setLatestForecastBundle
  } = useSessionForecastView();

  const [datasetOverlay, setDatasetOverlay] = useState<GeoJsonFeatureCollection | null>(null);
  const [datasetCenter, setDatasetCenter] = useState<[number, number] | null>(null);
  const geojson = ((datasetOverlay ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;

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
    void runLatestForecast();
  }, []);

  useEffect(() => {
    httpGet<ActiveDatasetScenarioResponse>("/forecast-context/dataset-scenarios/active")
      .then((active) => {
        const hasActiveScenario = Boolean(active.active_scenario_id ?? active.selected_scenario_id);
        if (!active.enabled || !active.available || !hasActiveScenario) {
          setDatasetOverlay(null);
          setDatasetCenter(null);
          void runLatestForecast();
          return;
        }

        const lat = active?.scenario?.source?.latitude;
        const lon = active?.scenario?.source?.longitude;
        if (typeof lat === "number" && typeof lon === "number") {
          setDatasetCenter([lon, lat]);
        }

        return httpGet<GeoJsonFeatureCollection>("/forecast-context/dataset-scenarios/active/overlay")
          .then((overlay) => {
            setDatasetOverlay(overlay);
          })
          .catch(() => {
            setDatasetOverlay(null);
            void runLatestForecast();
          });
      })
      .catch(() => {
        setDatasetOverlay(null);
        setDatasetCenter(null);
        void runLatestForecast();
      });
  }, []);


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
          center={datasetCenter}
        />
      </main>
    </AppShell>
  );
}
