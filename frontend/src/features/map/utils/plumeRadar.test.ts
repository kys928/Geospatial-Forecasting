import { extractRadarPointsFromGeojson } from "./plumeRadar";
import type { GeoJsonFeatureCollection } from "../../forecast/types/forecast.types";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function getWeightBySourceKind(collection: GeoJsonFeatureCollection, sourceKind: string): number {
  const feature = collection.features.find(
    (f) => f.properties?.source_kind === sourceKind && f.properties?.kind === "plume_radar_point"
  );
  return typeof feature?.properties?.weight === "number" ? feature.properties.weight : -1;
}

const input: GeoJsonFeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]]
      },
      properties: { kind: "plume_band_high" }
    },
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[[1, 1], [2, 1], [2, 2], [1, 1]]]
      },
      properties: { kind: "plume_band_medium" }
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [2, 2] },
      properties: { kind: "plume_point", normalized_intensity: 2.2 }
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [3, 3] },
      properties: { kind: "source" }
    },
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[[0, 0], [4, 0], [4, 4], [0, 0]]]
      },
      properties: { kind: "forecast_extent" }
    }
  ]
};

const output = extractRadarPointsFromGeojson(input);
assert(output.features.length === 3, "output point count should match plume features only");
assert(Math.abs(getWeightBySourceKind(output, "plume_band_high") - 0.9) < 0.0001, "high plume weight should be ~0.9");
assert(Math.abs(getWeightBySourceKind(output, "plume_band_medium") - 0.55) < 0.0001, "medium plume weight should be ~0.55");
assert(getWeightBySourceKind(output, "source") === -1, "source features should be excluded");
assert(
  output.features.some((f) => f.properties?.source_kind === "plume_point" && f.properties?.weight === 1),
  "normalized_intensity should be respected and clamped"
);
