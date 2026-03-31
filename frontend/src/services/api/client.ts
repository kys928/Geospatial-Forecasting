import type {
  ApiMode,
  CapabilitiesResponse,
  ForecastCreateResponse,
  ForecastExplanation,
  ForecastSummary,
  GeoJsonFeatureCollection,
  MockForecastRequest,
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

const API_BASE_URL = "http://localhost:8000";

let lastMockRequest: MockForecastRequest = {
  scenario: "default",
  threshold: "1e-5"
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

function getMockSummary(): ForecastSummary {
  switch (lastMockRequest.scenario) {
    case "urban":
      return summaryUrbanMock as ForecastSummary;
    case "industrial":
      return summaryIndustrialMock as ForecastSummary;
    default:
      return summaryDefaultMock as ForecastSummary;
  }
}

function getMockGeojson(): GeoJsonFeatureCollection {
  switch (lastMockRequest.scenario) {
    case "urban":
      return geojsonUrbanMock as GeoJsonFeatureCollection;
    case "industrial":
      return geojsonIndustrialMock as GeoJsonFeatureCollection;
    default:
      return geojsonDefaultMock as GeoJsonFeatureCollection;
  }
}

function getMockExplanation(): ForecastExplanation {
  switch (lastMockRequest.scenario) {
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
    request?: MockForecastRequest
  ): Promise<ForecastCreateResponse> {
    if (mode === "mock") {
      if (request) {
        lastMockRequest = request;
      }
      return {
        ...(forecastMock as ForecastCreateResponse),
        forecast_id: `demo-${lastMockRequest.scenario}-${lastMockRequest.threshold}`
      };
    }

    return postJson(`${API_BASE_URL}/forecast`, {
      run_name: request ? `${request.scenario}-${request.threshold}` : "frontend-run"
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
    options?: { threshold?: number; useLlm?: boolean }
  ): Promise<ForecastExplanation> {
    if (mode === "mock") {
      return getMockExplanation();
    }

    const threshold = options?.threshold ?? 1e-6;
    const useLlm = options?.useLlm ?? true;

    const params = new URLSearchParams({
      threshold: String(threshold),
      use_llm: String(useLlm)
    });

    return getJson(`${API_BASE_URL}/forecast/${forecastId}/explanation?${params.toString()}`);
  }
};