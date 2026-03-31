import type { GeoJsonFeatureCollection } from "../forecast/forecast.types";

export function isValidFeatureCollection(
  value: GeoJsonFeatureCollection | null
): value is GeoJsonFeatureCollection {
  return Boolean(value && value.type === "FeatureCollection" && Array.isArray(value.features));
}