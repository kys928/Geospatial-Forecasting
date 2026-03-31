import { useEffect, useMemo, useState } from "react";
import { TopBar } from "../components/TopBar";
import { Sidebar } from "../components/Sidebar";
import { DetailDrawer } from "../components/DetailDrawer";
import { SummaryCards } from "../components/SummaryCards";
import { StatusBar } from "../components/StatusBar";
import { ScenarioControls } from "../features/scenario/ScenarioControls";
import { TimelineSlider } from "../features/timeline/TimelineSlider";
import { ForecastMap } from "../features/map/ForecastMap";
import { apiClient } from "../services/api/client";
import { loadCapabilities, loadForecastBundle, runForecast } from "../features/forecast/forecastApi";
import type {
  ApiMode,
  CapabilitiesResponse,
  ForecastSummary,
  GeoJsonFeatureCollection,
  SelectedFeatureState
} from "../features/forecast/forecast.types";

export function MapPage() {
  const [apiMode] = useState<ApiMode>("mock");
  const [apiHealthy, setApiHealthy] = useState(true);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [summary, setSummary] = useState<ForecastSummary | null>(null);
  const [geojson, setGeojson] = useState<GeoJsonFeatureCollection | null>(null);
  const [selected, setSelected] = useState<SelectedFeatureState | null>(null);
  const [statusText, setStatusText] = useState("Loading dashboard...");
  const [explanation, setExplanation] = useState(
    "Forecast explanation placeholder. This will later reflect backend explanation output."
  );

  useEffect(() => {
    async function bootstrap() {
      try {
        const health = await apiClient.getHealth(apiMode);
        setApiHealthy(health.status === "ok");

        const capabilitiesResponse = await loadCapabilities(apiMode);
        setCapabilities(capabilitiesResponse);

        const created = await runForecast(apiMode);
        const bundle = await loadForecastBundle(apiMode, created.forecast_id);

        setSummary(bundle.summary);
        setGeojson(bundle.geojson);
        setStatusText(`Loaded forecast ${created.forecast_id}`);
      } catch (error) {
        console.error(error);
        setApiHealthy(false);
        setStatusText("Failed to load forecast data");
      }
    }

    void bootstrap();
  }, [apiMode]);

  const modelLabel = useMemo(() => {
    return capabilities?.model?.[0] ?? "Gaussian Baseline";
  }, [capabilities]);

  async function handleRunForecast() {
    try {
      setStatusText("Running forecast...");
      const created = await runForecast(apiMode);
      const bundle = await loadForecastBundle(apiMode, created.forecast_id);

      setSummary(bundle.summary);
      setGeojson(bundle.geojson);
      setSelected(null);
      setStatusText(`Loaded forecast ${created.forecast_id}`);
    } catch (error) {
      console.error(error);
      setStatusText("Forecast request failed");
    }
  }

  return (
    <div className="app-shell">
      <TopBar
        apiMode={apiMode}
        apiHealthy={apiHealthy}
        modelLabel={modelLabel}
        scenarioName="Default Scenario"
      />

      <SummaryCards summary={summary} />

      <div className="main-layout">
        <Sidebar onRunForecast={handleRunForecast}>
          <ScenarioControls />
        </Sidebar>

        <main className="map-column">
          <ForecastMap geojson={geojson} onSelectFeature={setSelected} />
          <TimelineSlider />
        </main>

        <DetailDrawer selected={selected} explanation={explanation} />
      </div>

      <StatusBar statusText={statusText} />
    </div>
  );
}