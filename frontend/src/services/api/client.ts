import type {
  ApiMode,
  CapabilitiesResponse,
  DemoScenario,
  ForecastCreateResponse,
  ForecastExplanation,
  ForecastRunRequest,
  ForecastSummary,
  GeoJsonFeatureCollection,
  RasterMetadata
} from "../../features/forecast/forecast.types";

import capabilitiesMock from "../../mocks/capabilities.json";
import forecastMock from "../../mocks/forecast.json";
import rasterMetadataMock from "../../mocks/raster-metadata.json";

import summaryDefaultMock from "../../mocks/summary-default.json";
import summaryUrbanMock from "../../mocks/summary-urban.json";
import summaryIndustrialMock from "../../mocks/summary-industrial.json";

import geojsonDefaultMock from "../../mocks/geojson-default.json";
import geojsonUrbanMock from "../../mocks/geojson-urban.json";
import geojsonIndustrialMock from "../../mocks/geojson-industrial.json";

import explanationDefaultMock from "../../mocks/explanation-default.json";
import explanationUrbanMock from "../../mocks/explanation-urban.json";
import explanationIndustrialMock from "../../mocks/explanation-industrial.json";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function resolveApiBaseUrl(envValue: string | undefined): string {
  const configuredBaseUrl = envValue?.trim();
  return configuredBaseUrl ? configuredBaseUrl : DEFAULT_API_BASE_URL;
}

const API_BASE_URL = resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL);
const FIXED_EXPLANATION_THRESHOLD = 1e-5;

const DEFAULT_SCENARIO: DemoScenario = {
  id: "dense-urban-core",
  label: "Dense urban core",
  latitude: 52.0907,
  longitude: 5.1214,
  emissionsRate: 100,
  severity: "moderate",
  notes: "Compact city-center release with strong local context.",
  mockVariant: "default"
};

let lastMockRequest: ForecastRunRequest = {
  scenario: DEFAULT_SCENARIO
};

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

function getMockVariant(request: ForecastRunRequest): "default" | "urban" | "industrial" {
  return request.scenario.mockVariant ?? "default";
}

function getMockSummary(): ForecastSummary {
  switch (getMockVariant(lastMockRequest)) {
    case "urban":
      return summaryUrbanMock as ForecastSummary;
    case "industrial":
      return summaryIndustrialMock as ForecastSummary;
    default:
      return summaryDefaultMock as ForecastSummary;
  }
}

function getMockGeojson(): GeoJsonFeatureCollection {
  switch (getMockVariant(lastMockRequest)) {
    case "urban":
      return geojsonUrbanMock as GeoJsonFeatureCollection;
    case "industrial":
      return geojsonIndustrialMock as GeoJsonFeatureCollection;
    default:
      return geojsonDefaultMock as GeoJsonFeatureCollection;
  }
}

function getMockExplanation(): ForecastExplanation {
  switch (getMockVariant(lastMockRequest)) {
    case "urban":
      return explanationUrbanMock as ForecastExplanation;
    case "industrial":
      return explanationIndustrialMock as ForecastExplanation;
    default:
      return explanationDefaultMock as ForecastExplanation;
  }
}

export const apiClient = {
  async getHealth(mode: ApiMode): Promise<{ status: string }> {
    if (mode === "mock") {
      return { status: "ok" };
    }
    return getJson(`${API_BASE_URL}/health`);
  },

  async getCapabilities(mode: ApiMode): Promise<CapabilitiesResponse> {
    if (mode === "mock") {
      return capabilitiesMock as CapabilitiesResponse;
    }
    return getJson(`${API_BASE_URL}/capabilities`);
  },

  async createForecast(
    mode: ApiMode,
    request: ForecastRunRequest
  ): Promise<ForecastCreateResponse> {
    if (mode === "mock") {
      lastMockRequest = request;

      return {
        ...(forecastMock as ForecastCreateResponse),
        forecast_id: `demo-${request.scenario.id}`
      };
    }

    return postJson(`${API_BASE_URL}/forecast`, {
      run_name: request.scenario.id,
      latitude: request.scenario.latitude,
      longitude: request.scenario.longitude,
      emissions_rate: request.scenario.emissionsRate
    });
  },

  async getForecastSummary(mode: ApiMode, forecastId: string): Promise<ForecastSummary> {
    if (mode === "mock") {
      return getMockSummary();
    }
    return getJson(`${API_BASE_URL}/forecast/${forecastId}/summary`);
  },

  async getGeoJson(mode: ApiMode, forecastId: string): Promise<GeoJsonFeatureCollection> {
    if (mode === "mock") {
      return getMockGeojson();
    }
    return getJson(`${API_BASE_URL}/forecast/${forecastId}/geojson`);
  },

  async getRasterMetadata(mode: ApiMode, forecastId: string): Promise<RasterMetadata> {
    if (mode === "mock") {
      return rasterMetadataMock as RasterMetadata;
    }
    return getJson(`${API_BASE_URL}/forecast/${forecastId}/raster-metadata`);
  },

  async getExplanation(
    mode: ApiMode,
    forecastId: string,
    options?: { useLlm?: boolean }
  ): Promise<ForecastExplanation> {
    if (mode === "mock") {
      return getMockExplanation();
    }

    const useLlm = options?.useLlm ?? true;

    const params = new URLSearchParams({
      threshold: String(FIXED_EXPLANATION_THRESHOLD),
      use_llm: String(useLlm)
    });

    return getJson(`${API_BASE_URL}/forecast/${forecastId}/explanation?${params.toString()}`);
  }
};