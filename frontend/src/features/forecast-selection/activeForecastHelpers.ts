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
}): Array<[string, string]> {
  const metadata = (input.summary.metadata as Record<string, unknown> | undefined) ?? {};
  return [
    ["selected_model", getActiveModelDisplayName(input.activeModelId)],
    ["forecast_kind", input.activeForecastKind],
    ["session_id", input.activeSessionId ?? "n/a"],
    ["persisted_forecast_id", input.activePersistedForecastId ?? "n/a"],
    ["model_name", String(metadata.model_name ?? input.summary.model_name ?? "n/a")],
    ["prediction_engine", String(metadata.prediction_engine ?? input.summary.prediction_engine ?? "n/a")],
    ["frame_count", String(metadata.frame_count ?? input.summary.frame_count ?? (input.activeModelId === "convlstm_multistep" ? 4 : 1))],
    ["input_source", String(metadata.input_source ?? input.summary.input_source ?? "n/a")],
    ["georeferencing_status", String(metadata.georeferencing_status ?? input.summary.georeferencing_status ?? "n/a")],
    ["prediction_trust", String(metadata.prediction_trust ?? input.summary.prediction_trust ?? "n/a")],
    ["input_mode", String(metadata.input_mode ?? input.summary.input_mode ?? "n/a")],
    ["scenario_usage", String(metadata.scenario_usage ?? "n/a")],
    ["scenario_note", String(metadata.scenario_note ?? "n/a")],
  ];
}
