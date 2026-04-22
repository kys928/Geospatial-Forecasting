import { apiClient } from "../../../services/api/client";
import type {
  ApiMode,
  CapabilitiesResponse,
  ForecastCreateResponse,
  ForecastExplanation,
  ForecastRunRequest,
  ForecastSummary,
  GeoJsonFeatureCollection,
  RasterMetadata
} from "../types/forecast.types";

export async function loadCapabilities(mode: ApiMode): Promise<CapabilitiesResponse> {
  return apiClient.getCapabilities(mode);
}

export async function runForecast(
  mode: ApiMode,
  request: ForecastRunRequest
): Promise<ForecastCreateResponse> {
  return apiClient.createForecast(mode, request);
}

export async function loadForecastBundle(
  mode: ApiMode,
  forecastId: string,
  options?: { useLlm?: boolean }
): Promise<{
  summary: ForecastSummary;
  geojson: GeoJsonFeatureCollection;
  rasterMetadata: RasterMetadata;
  explanation: ForecastExplanation;
}> {
  const [summary, geojson, rasterMetadata, explanation] = await Promise.all([
    apiClient.getForecastSummary(mode, forecastId),
    apiClient.getGeoJson(mode, forecastId),
    apiClient.getRasterMetadata(mode, forecastId),
    apiClient.getExplanation(mode, forecastId, options)
  ]);

  return { summary, geojson, rasterMetadata, explanation };
}