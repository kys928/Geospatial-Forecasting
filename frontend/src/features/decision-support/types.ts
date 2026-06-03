export type DecisionSupportLatest = {
  mode?: "stub" | "llm" | string;
  used_llm?: boolean;
  briefing?: string;
  situation_summary?: string;
  risk_level?: string;
  recommended_action?: string;
  uncertainty_limitations?: string;
  forecast_evidence?: unknown;
  system_honesty?: string;
  limitations?: string[];
  live_inputs?: Record<string, unknown>;
  runtime_mode?: string;
  forecast_backend?: string;
  last_forecast_time?: string;
};

export type DatasetScenarioPreview = { scenario_id: string; label: string; status?: string; risk_level?: string };

export type ActiveDatasetScenarioResponse = {
  enabled: boolean;
  available: boolean;
  active_scenario_id?: string | null;
  selected_scenario_id?: string | null;
  scenario?: ForecastContextResponse | null;
};

export type DatasetPlaybackState = { enabled: boolean; active_scenario_id?: string | null; mode?: string };

export type ForecastContextResponse = {
  forecast: Record<string, unknown>;
  conditions: Record<string, unknown>;
  source: Record<string, unknown>;
  plume_metrics: Record<string, unknown>;
  runtime: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  raw: Record<string, unknown>;
};

export type ChatMessage = { role: "assistant" | "user"; content: string };
export type DisplayRow = [string, string];
