import { httpGet } from "../../../services/api/http";
import type {
  ForecastListResponse,
  GeoJsonFeatureCollection,
  RasterMetadata
} from "../types/forecast.types";

export function listForecasts(limit = 50): Promise<ForecastListResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  return httpGet<ForecastListResponse>(`/forecasts?${params.toString()}`);
}

export function getForecastSummary(forecastId: string): Promise<Record<string, unknown>> {
  return httpGet<Record<string, unknown>>(`/forecast/${forecastId}/summary`);
}

export function getForecastGeoJson(forecastId: string): Promise<GeoJsonFeatureCollection> {
  return httpGet<GeoJsonFeatureCollection>(`/forecast/${forecastId}/geojson`);
}

export function getForecastRasterMetadata(forecastId: string): Promise<RasterMetadata> {
  return httpGet<RasterMetadata>(`/forecast/${forecastId}/raster-metadata`);
}
