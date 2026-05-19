export interface GeoJsonKindCounts {
  featureCount: number;
  plumeCellCount: number;
  kinds: string[];
}

export function countGeojsonKinds(geojson: Record<string, unknown> | null): GeoJsonKindCounts {
  const features = Array.isArray((geojson as { features?: unknown } | null)?.features)
    ? ((geojson as { features: Array<{ properties?: unknown }> }).features)
    : [];
  const kindCounts = new Map<string, number>();
  for (const feature of features) {
    const properties = feature?.properties as { kind?: unknown } | undefined;
    if (typeof properties?.kind !== "string" || properties.kind.length === 0) {
      continue;
    }
    kindCounts.set(properties.kind, (kindCounts.get(properties.kind) ?? 0) + 1);
  }
  return {
    featureCount: features.length,
    plumeCellCount: kindCounts.get("plume_cell") ?? 0,
    kinds: [...kindCounts.keys()]
  };
}
