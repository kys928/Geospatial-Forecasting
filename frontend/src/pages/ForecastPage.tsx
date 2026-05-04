import { useEffect, useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { getForecastGeoJson, getForecastRasterMetadata, getForecastSummary } from "../features/forecast/api/forecastArtifactsClient";
import { RecentForecastsPanel } from "../features/forecast/components/RecentForecastsPanel";
import { usePersistedForecasts } from "../features/forecast/hooks/usePersistedForecasts";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { sessionClient } from "../features/sessions/api/sessionClient";

export function ForecastPage() {
  const { activeSessionId, latestForecastBundle, forecastViewSource, activePersistedForecastId, selectedFeature, setSelectedFeature, setPersistedForecastBundle, clearSelectedFeature, setActiveSessionId, setLatestForecastBundle } = useSessionForecastView();
  const { forecasts, loading, error, refresh } = usePersistedForecasts(50);
  const [loadingForecastId, setLoadingForecastId] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runtimeNote, setRuntimeNote] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  const geojson = (latestForecastBundle?.geojson ?? null) as GeoJsonFeatureCollection | null;
  const statusText = useMemo(() => {
    if (loadError) return `Could not load persisted artifacts: ${loadError}`;
    if (runtimeNote) return runtimeNote;
    if (forecastViewSource === "persisted" && activePersistedForecastId) return `Showing persisted forecast artifact ${activePersistedForecastId.slice(0, 8)}`;
    if (latestForecastBundle) return "Current forecast loaded.";
    return "Loading latest forecast";
  }, [activePersistedForecastId, forecastViewSource, latestForecastBundle, loadError, runtimeNote]);

  const runLatestForecast = async () => {
    setIsRunning(true);
    setRuntimeNote(null);
    try {
      const runResult = await sessionClient.runSessionForecast({});
      setActiveSessionId(runResult.sessionId);
      const bundle = await sessionClient.getLatestForecastBundle(runResult.sessionId);
      setLatestForecastBundle(runResult.sessionId, bundle);
      const summary = (bundle.summary ?? {}) as Record<string, unknown>;
      if (Number(summary.affected_cells_above_threshold ?? 0) <= 0) setRuntimeNote("No meaningful plume detected above threshold.");
      if (runResult.recreatedSession) setRuntimeNote(`Session reset: ${runResult.resetReason ?? "runtime session was recreated"}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Forecast load failed.";
      if (message.includes("Failed to fetch")) setRuntimeNote("Backend unavailable.");
      else setRuntimeNote(`Forecast failed: ${message}`);
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => { void runLatestForecast(); }, []);

  const handleLoadPersistedForecast = async (forecastId: string) => {
    setLoadingForecastId(forecastId); setLoadError(null);
    try {
      const [summary, geojson, rasterMetadata] = await Promise.all([getForecastSummary(forecastId), getForecastGeoJson(forecastId), getForecastRasterMetadata(forecastId)]);
      setPersistedForecastBundle(forecastId, { summary, geojson: geojson as unknown as Record<string, unknown>, rasterMetadata: rasterMetadata as unknown as Record<string, unknown>, explanation: {} });
      clearSelectedFeature();
    } catch (err) { setLoadError(err instanceof Error ? err.message : "Failed to load persisted forecast artifacts."); }
    finally { setLoadingForecastId(null); }
  };

  return <AppShell title="Map / Forecast" subtitle="Current forecast and persisted forecast artifacts." statusText={statusText} metaItems={[{ label: activeSessionId ? "Session active" : "Session unavailable" }, { label: isRunning ? "Forecast running" : "Forecast idle" }]}>
    <main className="map-column">
      <div className="panel"><button className="primary-button" disabled={isRunning} onClick={() => void runLatestForecast()}>{isRunning ? "Refreshing…" : "Refresh forecast"}</button></div>
      <RecentForecastsPanel forecasts={forecasts} loading={loading} error={error} loadingForecastId={loadingForecastId} onRefresh={() => void refresh()} onLoad={(forecast) => { void handleLoadPersistedForecast(forecast.forecast_id); }} />
      <ForecastMap geojson={geojson} selectedFeature={selectedFeature} onSelectFeature={setSelectedFeature} />
    </main>
  </AppShell>;
}
