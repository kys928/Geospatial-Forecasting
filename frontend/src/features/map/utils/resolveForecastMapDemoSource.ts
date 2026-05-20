export type ForecastMapSourceMode = "dataset-snapshot" | "session-frame" | "session-bundle" | "none";

interface ResolveForecastMapDemoSourceInput {
  datasetActive: boolean;
  hasUsableSelectedFrame: boolean;
  hasUsableSessionBundle: boolean;
}

export function resolveForecastMapDemoSource({
  datasetActive,
  hasUsableSelectedFrame,
  hasUsableSessionBundle,
}: ResolveForecastMapDemoSourceInput): ForecastMapSourceMode {
  if (datasetActive) return "dataset-snapshot";
  if (hasUsableSelectedFrame) return "session-frame";
  if (hasUsableSessionBundle) return "session-bundle";
  return "none";
}

export function isDatasetNormalScenario(scenarioId: string | null | undefined): boolean {
  return typeof scenarioId === "string" && scenarioId.trim().toLowerCase() === "dataset_normal";
}
