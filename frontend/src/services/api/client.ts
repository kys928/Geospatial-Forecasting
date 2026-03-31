import type {
  ApiMode,
  CapabilitiesResponse,
  ForecastCreateResponse,
  ForecastSummary,
  GeoJsonFeatureCollection,
  RasterMetadata
} from "../../features/forecast/forecast.types";

import capabilitiesMock from "../../mocks/capabilities.json";
import forecastMock from "../../mocks/forecast.json";
import summaryMock from "../../mocks/summary.json";
import geojsonMock from "../../mocks/geojson.json";
import rasterMetadataMock from "../../mocks/raster-metadata.json";

const API_BASE_URL = "http://localhost:8000";

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

  async createForecast(mode: ApiMode): Promise<ForecastCreateResponse> {
    if (mode === "mock") {
      return forecastMock as ForecastCreateResponse;
    }
    return postJson(`${API_BASE_URL}/forecast`, { run_name: "frontend-run" });
  },

  async getForecastSummary(mode: ApiMode, forecastId: string): Promise<ForecastSummary> {
    if (mode === "mock") {
      return summaryMock as ForecastSummary;
    }
    return getJson(`${API_BASE_URL}/forecast/${forecastId}/summary`);
  },

  async getGeoJson(mode: ApiMode, forecastId: string): Promise<GeoJsonFeatureCollection> {
    if (mode === "mock") {
      return geojsonMock as GeoJsonFeatureCollection;
    }
    return getJson(`${API_BASE_URL}/forecast/${forecastId}/geojson`);
  },

  async getRasterMetadata(mode: ApiMode, forecastId: string): Promise<RasterMetadata> {
    if (mode === "mock") {
      return rasterMetadataMock as RasterMetadata;
    }
    return getJson(`${API_BASE_URL}/forecast/${forecastId}/raster-metadata`);
  }
};