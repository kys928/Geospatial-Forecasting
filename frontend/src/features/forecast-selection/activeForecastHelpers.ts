import { MODEL_BY_ID, type ActiveForecastKind, type ActiveModelId } from "./modelRegistry";

export function isModelIdentityQuestion(message: string): boolean {
  return /\b(what\s+model\s+are\s+you|which\s+model|model\s+source)\b/i.test(message);
}

export function getActiveModelDisplayName(modelId: ActiveModelId): string {
  return MODEL_BY_ID[modelId].label;
}

export function getActiveForecastTechnicalDetails(input: {
  activeModelId: ActiveModelId;
  activeForecastKind: ActiveForecastKind;
  activeSessionId: string | null;
  activePersistedForecastId: string | null;
  summary: Record<string, unknown>;
  rasterMetadata?: Record<string, unknown> | null;
  framesMetadata?: Record<string, unknown> | null;
  forecastContextRuntime?: Record<string, unknown> | null;
}): Array<[string, string]> {
  const metadata = (input.summary.metadata as Record<string, unknown> | undefined) ?? {};
  const merged = { ...input.rasterMetadata, ...input.framesMetadata, ...input.forecastContextRuntime, ...input.summary, ...metadata } as Record<string, unknown>;
  const read = (...keys: string[]) => keys.map((k) => merged[k]).find((v) => v !== undefined && v !== null && String(v).trim() !== "");
  return [
    ["selected_model", getActiveModelDisplayName(input.activeModelId)],
    ["forecast_kind", input.activeForecastKind],
    ["session_id", input.activeSessionId ?? "n/a"],
    ["persisted_forecast_id", input.activePersistedForecastId ?? "n/a"],
    ["model_name", String(read("model_name", "model") ?? "n/a")],
    ["prediction_engine", String(read("prediction_engine", "engine") ?? "n/a")],
    ["frame_count", String(read("frame_count") ?? (input.activeModelId === "convlstm_multistep" ? 4 : 1))],
    ["input_source", String(read("input_source", "source") ?? "n/a")],
    ["georeferencing_status", String(read("georeferencing_status") ?? "n/a")],
    ["prediction_trust", String(read("prediction_trust") ?? "n/a")],
    ["input_mode", String(read("input_mode") ?? "n/a")],
    ["scenario_usage", String(read("scenario_usage") ?? "n/a")],
    ["scenario_note", String(read("scenario_note") ?? "n/a")],
  ];
}
