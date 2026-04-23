import { AppShell } from "../app/AppShell";
import { ForecastMap } from "../features/map/components/ForecastMap";
import type { GeoJsonFeatureCollection } from "../features/forecast/types/forecast.types";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";

export function ForecastPage() {
  const { activeSessionId, latestForecastBundle, selectedFeature, setSelectedFeature } = useSessionForecastView();

  const geojson = (latestForecastBundle?.geojson ?? null) as GeoJsonFeatureCollection | null;
  const statusText = activeSessionId
    ? latestForecastBundle
      ? `Showing latest forecast map for session ${activeSessionId}`
      : "No forecast artifacts loaded for the active session yet"
    : "Select and run a session forecast in Sessions to populate the map";

  return (
    <AppShell
      title="Map workspace"
      subtitle="Interactive forecast map for the active session."
      statusText={statusText}
      metaItems={[{ label: activeSessionId ? `Session ${activeSessionId}` : "No active session" }]}
    >
      <main className="map-column">
        <ForecastMap
          geojson={geojson}
          selectedFeature={selectedFeature}
          onSelectFeature={setSelectedFeature}
        />
      </main>
    </AppShell>
  );
}
