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
  const [datasetOverlayNote, setDatasetOverlayNote] = useState<string>("");
  const [datasetCenter, setDatasetCenter] = useState<[number, number] | null>(null);
  const geojson = ((datasetOverlay ?? latestForecastBundle?.geojson) ?? null) as GeoJsonFeatureCollection | null;

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
    httpGet<{ enabled: boolean; available: boolean }>("/forecast-context/dataset-scenarios/active")
      .then((active) => {
        if (!active.enabled || !active.available) {
          setDatasetOverlay(null);
          setDatasetOverlayNote("");
          return;
        }
        return httpGet<GeoJsonFeatureCollection>("/forecast-context/dataset-scenarios/active/overlay")
          .then(async (overlay) => {
            setDatasetOverlay(overlay);
            const activeCtx = await httpGet<any>("/forecast-context/dataset-scenarios/active");
            const lat = activeCtx?.scenario?.source?.latitude;
            const lon = activeCtx?.scenario?.source?.longitude;
            if (typeof lat === "number" && typeof lon === "number") setDatasetCenter([lon, lat]);
            setDatasetOverlayNote("Dataset playback plume · Approximate source-centered grid · Not live data");
          })
          .catch(() => {
            setDatasetOverlay(null);
            setDatasetOverlayNote("Dataset plume overlay unavailable.");
          });
      })
      .catch(() => {
        setDatasetOverlay(null);
        setDatasetOverlayNote("");
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
        {datasetOverlayNote ? <p className="muted" style={{ marginTop: 8 }}>{datasetOverlayNote}</p> : null}
      </main>
    </AppShell>
  );
}
