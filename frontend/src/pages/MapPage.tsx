import { useEffect, useMemo, useRef, useState } from "react";
import { TopBar } from "../components/TopBar";
import { Sidebar } from "../components/Sidebar";
import { DetailDrawer } from "../components/DetailDrawer";
import { StatusBar } from "../components/StatusBar";
import { ScenarioControls } from "../features/scenario/ScenarioControls";
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
  DemoScenario,
  ForecastExplanation,
  ForecastSummary,
  GeoJsonFeatureCollection,
  SelectedFeatureState,
  ThresholdPreset
} from "../features/forecast/forecast.types";

type ScenarioSeed = {
  id: string;
  label: string;
  notes: string;
  centerLat: number;
  centerLon: number;
  latJitter: number;
  lonJitter: number;
  emissionMin: number;
  emissionMax: number;
  severity: "low" | "moderate" | "high";
  mockVariant?: "default" | "urban" | "industrial";
};

const SCENARIO_SEEDS: ScenarioSeed[] = [
  {
    id: "dense-urban-core",
    label: "Dense urban core",
    notes: "Urban release around a dense city-center footprint.",
    centerLat: 52.0907,
    centerLon: 5.1214,
    latJitter: 0.010,
    lonJitter: 0.014,
    emissionMin: 110,
    emissionMax: 180,
    severity: "moderate",
    mockVariant: "default"
  },
  {
    id: "industrial-corridor",
    label: "Industrial corridor",
    notes: "Higher-emission release near industrial infrastructure.",
    centerLat: 51.9244,
    centerLon: 4.4777,
    latJitter: 0.012,
    lonJitter: 0.016,
    emissionMin: 160,
    emissionMax: 260,
    severity: "high",
    mockVariant: "industrial"
  },
  {
    id: "transport-corridor",
    label: "Transport corridor",
    notes: "Release near a busy movement corridor with strong local context.",
    centerLat: 52.3702,
    centerLon: 4.8952,
    latJitter: 0.010,
    lonJitter: 0.016,
    emissionMin: 120,
    emissionMax: 190,
    severity: "moderate",
    mockVariant: "urban"
  },
  {
    id: "southern-urban-edge",
    label: "Southern urban edge",
    notes: "Lower-density release near an urban edge condition.",
    centerLat: 51.4416,
    centerLon: 5.4697,
    latJitter: 0.010,
    lonJitter: 0.014,
    emissionMin: 90,
    emissionMax: 140,
    severity: "low",
    mockVariant: "default"
  }
];

const MAX_RANDOM_ATTEMPTS = 6;

function randomInRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function chooseRandomSeed(): ScenarioSeed {
  const index = Math.floor(Math.random() * SCENARIO_SEEDS.length);
  return SCENARIO_SEEDS[index];
}

function buildRandomScenario(threshold: ThresholdPreset): DemoScenario {
  const seed = chooseRandomSeed();
  const latitude = seed.centerLat + randomInRange(-seed.latJitter, seed.latJitter);
  const longitude = seed.centerLon + randomInRange(-seed.lonJitter, seed.lonJitter);
  const emissionsRate = Math.round(randomInRange(seed.emissionMin, seed.emissionMax));

  return {
    id: `${seed.id}-${Date.now()}-${Math.floor(Math.random() * 100000)}`,
    label: seed.label,
    latitude: Number(latitude.toFixed(6)),
    longitude: Number(longitude.toFixed(6)),
    emissionsRate,
    threshold,
    severity: seed.severity,
    notes: seed.notes,
    mockVariant: seed.mockVariant
  };
}

function hasStrongVisiblePlume(geojson: GeoJsonFeatureCollection | null): boolean {
  if (!geojson) {
    return false;
  }

  return geojson.features.some((feature) => {
    const kind = feature.properties?.kind;
    return kind === "plume_band_medium" || kind === "plume_band_high";
  });
}

export function MapPage() {
  const [apiMode] = useState<ApiMode>("live");
  const [apiHealthy, setApiHealthy] = useState(true);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [summary, setSummary] = useState<ForecastSummary | null>(null);
  const [geojson, setGeojson] = useState<GeoJsonFeatureCollection | null>(null);
  const [explanationPayload, setExplanationPayload] =
    useState<ForecastExplanation | null>(null);
  const [selected, setSelected] = useState<SelectedFeatureState | null>(null);
  const [statusText, setStatusText] = useState("Loading dashboard...");
  const [threshold, setThreshold] = useState<ThresholdPreset>("1e-6");
  const [activeScenario, setActiveScenario] = useState<DemoScenario | null>(null);

  const latestRequestIdRef = useRef(0);
  const didBootstrapRef = useRef(false);

  useEffect(() => {
    if (didBootstrapRef.current) {
      return;
    }
    didBootstrapRef.current = true;

    async function bootstrap() {
      const requestId = ++latestRequestIdRef.current;

      try {
        const health = await apiClient.getHealth(apiMode);
        if (requestId !== latestRequestIdRef.current) return;
        setApiHealthy(health.status === "ok");

        const capabilitiesResponse = await loadCapabilities(apiMode);
        if (requestId !== latestRequestIdRef.current) return;
        setCapabilities(capabilitiesResponse);

        await runRandomForecast("initial", requestId);
      } catch (error) {
        console.error(error);
        if (requestId !== latestRequestIdRef.current) return;
        setApiHealthy(false);
        setStatusText("Failed to load forecast data");
      }
    }

    void bootstrap();
  }, [apiMode]);

  const modelLabel = useMemo(() => {
    return capabilities?.model?.[0] ?? "Gaussian Baseline";
  }, [capabilities]);

  async function runRandomForecast(
    reason: "initial" | "manual",
    existingRequestId?: number
  ) {
    const requestId = existingRequestId ?? ++latestRequestIdRef.current;

    setStatusText(
      reason === "initial"
        ? "Loading initial forecast..."
        : "Searching for a strong visible plume..."
    );

    let bestScenario: DemoScenario | null = null;
    let bestBundle:
      | {
          summary: ForecastSummary;
          geojson: GeoJsonFeatureCollection;
          explanation: ForecastExplanation;
        }
      | null = null;
    let bestForecastId: string | null = null;

    for (let attempt = 0; attempt < MAX_RANDOM_ATTEMPTS; attempt += 1) {
      const scenario = buildRandomScenario(threshold);

      const created = await runForecast(apiMode, {
        scenario,
        threshold
      });

      if (requestId !== latestRequestIdRef.current) {
        return;
      }

      const bundle = await loadForecastBundle(apiMode, created.forecast_id, {
        threshold: Number(threshold),
        useLlm: true
      });

      if (requestId !== latestRequestIdRef.current) {
        return;
      }

      if (!bestBundle) {
        bestScenario = scenario;
        bestBundle = {
          summary: bundle.summary,
          geojson: bundle.geojson,
          explanation: bundle.explanation
        };
        bestForecastId = created.forecast_id;
      }

      if (hasStrongVisiblePlume(bundle.geojson)) {
        setSummary(bundle.summary);
        setGeojson(bundle.geojson);
        setExplanationPayload(bundle.explanation);
        setSelected(null);
        setActiveScenario(scenario);

        const explanationMode = bundle.explanation.used_llm ? "LLM" : "fallback";
        setStatusText(
          `Loaded ${scenario.label} · forecast ${created.forecast_id} · explanation: ${explanationMode}`
        );
        return;
      }
    }

    if (bestBundle && bestScenario && bestForecastId) {
      setSummary(bestBundle.summary);
      setGeojson(bestBundle.geojson);
      setExplanationPayload(bestBundle.explanation);
      setSelected(null);
      setActiveScenario(bestScenario);

      const explanationMode = bestBundle.explanation.used_llm ? "LLM" : "fallback";
      setStatusText(
        `Loaded ${bestScenario.label} · forecast ${bestForecastId} · explanation: ${explanationMode} · no strong plume found after retries`
      );
      return;
    }

    setStatusText("Forecast request failed");
  }

  async function handleRunForecast() {
    try {
      await runRandomForecast("manual");
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
        scenarioName={activeScenario?.label ?? "Scenario"}
      />

      <div className="main-layout">
        <Sidebar onRunForecast={handleRunForecast}>
          {activeScenario ? (
            <ScenarioControls
              activeScenario={activeScenario}
              threshold={threshold}
              onThresholdChange={setThreshold}
            />
          ) : null}
        </Sidebar>

        <main className="map-column">
          <ForecastMap
            geojson={geojson}
            selectedFeature={selected}
            onSelectFeature={setSelected}
          />
        </main>

        <DetailDrawer
          selected={selected}
          explanationPayload={explanationPayload}
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