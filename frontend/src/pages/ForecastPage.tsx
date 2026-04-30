import { useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import {
  getForecastGeoJson,
  getForecastRasterMetadata,
  getForecastSummary
} from "../features/forecast/api/forecastArtifactsClient";
import { RecentForecastsPanel } from "../features/forecast/components/RecentForecastsPanel";
import { usePersistedForecasts } from "../features/forecast/hooks/usePersistedForecasts";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";

export function ForecastPage() {
  const { activeSessionId, latestForecastBundle, selectedFeature, setSelectedFeature, setLatestForecastBundle, clearSelectedFeature } = useSessionForecastView();
  const { forecasts, loading, error, refresh } = usePersistedForecasts(50);
  const [loadingForecastId, setLoadingForecastId] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const geojson = (latestForecastBundle?.geojson ?? null) as GeoJsonFeatureCollection | null;
  const statusText = useMemo(() => {
    if (loadError) {
      return `Could not load persisted artifacts: ${loadError}`;
    }
    if (activeSessionId) {
      return latestForecastBundle
        ? `Showing latest forecast map for session ${activeSessionId}`
        : "No forecast artifacts loaded for the active session yet";
    }
    if (latestForecastBundle) {
      return "Showing persisted forecast artifacts on the map";
    }
    return "Select and run a session forecast in Sessions, or load a persisted forecast below";
  }, [activeSessionId, latestForecastBundle, loadError]);

  const handleLoadPersistedForecast = async (forecastId: string) => {
    setLoadingForecastId(forecastId);
    setLoadError(null);
    try {
      const [summary, geojson, rasterMetadata] = await Promise.all([
        getForecastSummary(forecastId),
        getForecastGeoJson(forecastId),
        getForecastRasterMetadata(forecastId)
      ]);
      setLatestForecastBundle(activeSessionId, {
        summary,
        geojson: geojson as unknown as Record<string, unknown>,
        rasterMetadata: rasterMetadata as unknown as Record<string, unknown>,
        explanation: {}
      });
      clearSelectedFeature();
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load persisted forecast artifacts.");
    } finally {
      setLoadingForecastId(null);
    }
  };

  return (
    <AppShell
      title="Map workspace"
      subtitle="Interactive forecast map for the active session."
      statusText={statusText}
      metaItems={[{ label: activeSessionId ? `Session ${activeSessionId}` : "No active session" }]}
    >
      <main className="map-column">
        <RecentForecastsPanel
          forecasts={forecasts}
          loading={loading}
          error={error}
          loadingForecastId={loadingForecastId}
          onRefresh={() => void refresh()}
          onLoad={(forecast) => {
            void handleLoadPersistedForecast(forecast.forecast_id);
          }}
        />
        <ForecastMap
          geojson={geojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
        />
      </main>
    </AppShell>
  );
}
