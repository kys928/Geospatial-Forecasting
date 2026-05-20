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


export function buildDatasetOverlayIdentity(
  geojson: { features?: unknown[]; metadata?: Record<string, unknown> } | null
): string {
  const metadata = geojson?.metadata ?? {};
  const scenarioId = metadata.active_scenario_id ?? "dataset";
  const windowId = metadata.active_window_id ?? "window";
  const featureCount = typeof metadata.feature_count === "number"
    ? metadata.feature_count
    : Array.isArray(geojson?.features) ? geojson.features.length : 0;
  return `${String(scenarioId)}:${String(windowId)}:${featureCount}`;
}
