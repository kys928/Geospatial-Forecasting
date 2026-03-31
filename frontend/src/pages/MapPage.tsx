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
import {
  loadCapabilities,
  loadForecastBundle,
  runForecast
} from "../features/forecast/forecastApi";
import type {
  ApiMode,
  CapabilitiesResponse,
  ForecastExplanation,
  ForecastSummary,
  GeoJsonFeatureCollection,
  ScenarioPreset,
  SelectedFeatureState,
  ThresholdPreset
} from "../features/forecast/forecast.types";

function buildExplanationText(
  explanationPayload: ForecastExplanation | null
): string {
  if (!explanationPayload) {
    return "No explanation loaded.";
  }

  const parts = [
    explanationPayload.explanation.summary,
    explanationPayload.explanation.recommendation,
    explanationPayload.explanation.uncertainty_note
  ].filter((value): value is string => Boolean(value && value.trim()));

  return parts.length > 0 ? parts.join(" ") : "No explanation text returned.";
}

export function MapPage() {
  const [apiMode] = useState<ApiMode>("live"); // switch to "live" when backend is running
  const [apiHealthy, setApiHealthy] = useState(true);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [summary, setSummary] = useState<ForecastSummary | null>(null);
  const [geojson, setGeojson] = useState<GeoJsonFeatureCollection | null>(null);
  const [explanationPayload, setExplanationPayload] =
    useState<ForecastExplanation | null>(null);
  const [selected, setSelected] = useState<SelectedFeatureState | null>(null);
  const [statusText, setStatusText] = useState("Loading dashboard...");
  const [scenario, setScenario] = useState<ScenarioPreset>("default");
  const [threshold, setThreshold] = useState<ThresholdPreset>("1e-5");

  useEffect(() => {
    async function bootstrap() {
      try {
        const health = await apiClient.getHealth(apiMode);
        setApiHealthy(health.status === "ok");

        const capabilitiesResponse = await loadCapabilities(apiMode);
        setCapabilities(capabilitiesResponse);

        const created = await runForecast(apiMode, { scenario, threshold });
        const bundle = await loadForecastBundle(apiMode, created.forecast_id, {
          threshold: Number(threshold),
          useLlm: true
        });

        setSummary(bundle.summary);
        setGeojson(bundle.geojson);
        setExplanationPayload(bundle.explanation);
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

  const explanationText = useMemo(() => {
    return buildExplanationText(explanationPayload);
  }, [explanationPayload]);

  async function handleRunForecast() {
    try {
      setStatusText("Running forecast...");

      const created = await runForecast(apiMode, { scenario, threshold });
      const bundle = await loadForecastBundle(apiMode, created.forecast_id, {
        threshold: Number(threshold),
        useLlm: true
      });

      setSummary(bundle.summary);
      setGeojson(bundle.geojson);
      setExplanationPayload(bundle.explanation);
      setSelected(null);

      const explanationMode = bundle.explanation.used_llm ? "LLM" : "fallback";
      setStatusText(`Loaded forecast ${created.forecast_id} · explanation: ${explanationMode}`);
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
        scenarioName={`${scenario} scenario`}
      />

      <SummaryCards summary={summary} />

      <div className="main-layout">
        <Sidebar onRunForecast={handleRunForecast}>
          <ScenarioControls
            scenario={scenario}
            threshold={threshold}
            onScenarioChange={setScenario}
            onThresholdChange={setThreshold}
          />
        </Sidebar>

        <main className="map-column">
          <ForecastMap geojson={geojson} onSelectFeature={setSelected} />
          <TimelineSlider />
        </main>

        <DetailDrawer
          selected={selected}
          explanation={explanationText}
          explanationSource={
            explanationPayload
              ? explanationPayload.used_llm
                ? "llm"
                : "fallback"
              : undefined
          }
        />
      </div>

      <StatusBar statusText={statusText} />
    </div>
  );
}