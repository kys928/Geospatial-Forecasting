import type { GeoJsonFeatureCollection } from "../../forecast/types/forecast.types";

export function isValidFeatureCollection(
  value: GeoJsonFeatureCollection | null
): value is GeoJsonFeatureCollection {
  return Boolean(value && value.type === "FeatureCollection" && Array.isArray(value.features));
}