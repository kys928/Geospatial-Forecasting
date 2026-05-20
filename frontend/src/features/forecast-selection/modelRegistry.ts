export type ActiveModelId = "gaussian_baseline" | "ridge_baseline" | "convlstm_multistep";
export type ActiveForecastKind = "batch_gaussian" | "dataset_ridge" | "session_convlstm" | "none";

export interface ForecastModelDefinition {
  id: ActiveModelId;
  label: string;
  description: string;
  forecastKind: Exclude<ActiveForecastKind, "none">;
  supportsFrames: boolean;
  expectedFrameCount: number;
}

export const FORECAST_MODELS: ForecastModelDefinition[] = [
  {
    id: "gaussian_baseline",
    label: "Gaussian baseline",
    description: "Analytical/simple baseline",
    forecastKind: "batch_gaussian",
    supportsFrames: false,
    expectedFrameCount: 1,
  },
  {
    id: "ridge_baseline",
    label: "Ridge baseline",
    description: "Dataset playback learned baseline",
    forecastKind: "dataset_ridge",
    supportsFrames: false,
    expectedFrameCount: 1,
  },
  {
    id: "convlstm_multistep",
    label: "ConvLSTM multi-step",
    description: "Temporal neural forecast",
    forecastKind: "session_convlstm",
    supportsFrames: true,
    expectedFrameCount: 4,
  },
];

export const MODEL_BY_ID: Record<ActiveModelId, ForecastModelDefinition> = FORECAST_MODELS.reduce(
  (acc, model) => ({ ...acc, [model.id]: model }),
  {} as Record<ActiveModelId, ForecastModelDefinition>
);
