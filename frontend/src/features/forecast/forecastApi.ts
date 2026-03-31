import { apiClient } from "../../services/api/client";
import type {
  ApiMode,
  CapabilitiesResponse,
  ForecastCreateResponse,
  ForecastSummary,
  GeoJsonFeatureCollection,
  RasterMetadata
} from "./forecast.types";

export async function loadCapabilities(mode: ApiMode): Promise<CapabilitiesResponse> {
  return apiClient.getCapabilities(mode);
}

export async function runForecast(mode: ApiMode): Promise<ForecastCreateResponse> {
  return apiClient.createForecast(mode);
}

export async function loadForecastBundle(
  mode: ApiMode,
  forecastId: string
): Promise<{
  summary: ForecastSummary;
  geojson: GeoJsonFeatureCollection;
  rasterMetadata: RasterMetadata;
}> {
  const [summary, geojson, rasterMetadata] = await Promise.all([
    apiClient.getForecastSummary(mode, forecastId),
    apiClient.getGeoJson(mode, forecastId),
    apiClient.getRasterMetadata(mode, forecastId)
  ]);

  return { summary, geojson, rasterMetadata };
}