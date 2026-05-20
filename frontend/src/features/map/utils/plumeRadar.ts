import type { GeoJsonFeature, GeoJsonFeatureCollection } from "../../forecast/types/forecast.types";

type NumericCandidateKey = "intensity" | "concentration" | "max_concentration" | "plume_score";

const NUMERIC_CANDIDATE_KEYS: NumericCandidateKey[] = [
  "intensity",
  "concentration",
  "max_concentration",
  "plume_score"
];

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function toNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function getFeatureKind(feature: GeoJsonFeature): string {
  return typeof feature.properties?.kind === "string" ? feature.properties.kind : "";
}

function getNormalizedIntensity(feature: GeoJsonFeature): number | null {
  return clamp01(toNumber(feature.properties?.normalized_intensity) ?? Number.NaN);
}

function extractIntensity(feature: GeoJsonFeature): number | null {
  for (const key of NUMERIC_CANDIDATE_KEYS) {
    const value = toNumber(feature.properties?.[key]);
    if (value !== null) return value;
  }
  return null;
}

function inferWeightFromKind(feature: GeoJsonFeature): number {
  const kind = getFeatureKind(feature);
  if (kind === "plume_band_high") return 0.9;
  if (kind === "plume_band_medium") return 0.55;
  if (kind === "plume_band_low") return 0.25;
  if (kind === "plume_cell") return 0.6;
  if (kind === "plume_point") return 0.7;
  if (kind === "plume_band") {
    const band = typeof feature.properties?.band === "string" ? feature.properties.band : "";
    if (band === "high") return 0.9;
    if (band === "medium") return 0.55;
    if (band === "low") return 0.25;
  }
  return 0.5;
}

function flattenCoordinates(coordinates: unknown, acc: [number, number][]): void {
  if (!Array.isArray(coordinates)) return;
  if (
    coordinates.length >= 2 &&
    typeof coordinates[0] === "number" &&
    typeof coordinates[1] === "number"
  ) {
    acc.push([coordinates[0], coordinates[1]]);
    return;
  }
  for (const item of coordinates) flattenCoordinates(item, acc);
}

function centroidFromFeature(feature: GeoJsonFeature): [number, number] | null {
  if (!feature.geometry || !("coordinates" in feature.geometry)) return null;

  if (feature.geometry.type === "Point") {
    const [lon, lat] = feature.geometry.coordinates;
    if (typeof lon === "number" && typeof lat === "number") return [lon, lat];
    return null;
  }

  const pairs: [number, number][] = [];
  flattenCoordinates(feature.geometry.coordinates, pairs);
  if (!pairs.length) return null;

  const sums = pairs.reduce(
    (acc, [lon, lat]) => {
      acc.lon += lon;
      acc.lat += lat;
      return acc;
    },
    { lon: 0, lat: 0 }
  );

  return [sums.lon / pairs.length, sums.lat / pairs.length];
}

export function extractRadarPointsFromGeojson(
  geojson: GeoJsonFeatureCollection | null
): GeoJsonFeatureCollection {
  const empty: GeoJsonFeatureCollection = { type: "FeatureCollection", features: [] };
  if (!geojson || geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) {
    return empty;
  }

  const plumeFeatures = geojson.features.filter((feature) => {
    const kind = getFeatureKind(feature);
    return kind !== "source" && kind !== "forecast_extent";
  });

  const intensityValues = plumeFeatures
    .map((feature) => extractIntensity(feature))
    .filter((value): value is number => value !== null && value > 0);
  const maxIntensity = intensityValues.length ? Math.max(...intensityValues) : null;

  const features: GeoJsonFeature[] = [];

  for (const feature of plumeFeatures) {
    const center = centroidFromFeature(feature);
    if (!center) continue;

    const normalizedIntensityValue = toNumber(feature.properties?.normalized_intensity);
    const normalizedIntensity = normalizedIntensityValue === null ? null : clamp01(normalizedIntensityValue);
    const intensity = extractIntensity(feature);

    let weight = normalizedIntensity;
    if (weight === null && intensity !== null && maxIntensity && maxIntensity > 0) {
      weight = clamp01(intensity / maxIntensity);
    }
    if (weight === null) {
      weight = inferWeightFromKind(feature);
    }

    features.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: center },
      properties: {
        kind: "plume_radar_point",
        weight: clamp01(weight),
        source_kind: getFeatureKind(feature),
        intensity: intensity ?? normalizedIntensityValue ?? null
      }
    });
  }

  return { type: "FeatureCollection", features };
}
