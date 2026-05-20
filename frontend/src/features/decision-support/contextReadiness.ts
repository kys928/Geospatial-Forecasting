import type { ForecastContextResponse } from "./types";

type ReadinessInput = Partial<ForecastContextResponse> | null | undefined;

export type ForecastContextReadiness = {
  ready: boolean;
  reason: string;
  forecastStatus?: string;
  riskLevel?: string;
  inputSource?: string;
  hasPlumeMetric: boolean;
  hasConditions: boolean;
  scenarioId?: string;
};

const UNAVAILABLE_VALUES = new Set(["", "unavailable", "forecast unavailable", "unknown", "null", "none"]);

const normalized = (value: unknown): string => String(value ?? "").trim().toLowerCase();
const hasUsableString = (value: unknown): boolean => !UNAVAILABLE_VALUES.has(normalized(value));
const isFiniteNumber = (value: unknown): boolean => typeof value === "number" && Number.isFinite(value);

export function isUsableForecastContext(context: ReadinessInput): ForecastContextReadiness {
  const forecast = (context?.forecast ?? {}) as Record<string, unknown>;
  const plume = (context?.plume_metrics ?? {}) as Record<string, unknown>;
  const conditions = (context?.conditions ?? {}) as Record<string, unknown>;
  const source = (context?.source ?? {}) as Record<string, unknown>;

  const forecastStatus = String(forecast.status ?? "");
  const riskLevel = String(forecast.risk_level ?? "");
  const inputSource = String(forecast.input_source ?? "");
  const scenarioId = String(forecast.scenario_id ?? "");

  const hasStatus = hasUsableString(forecastStatus);
  const hasRisk = hasUsableString(riskLevel);
  const hasInputSource = hasUsableString(inputSource);
  const hasPlumeMetric = isFiniteNumber(plume.max_plume_score) || isFiniteNumber(plume.max_concentration);
  const hasConditions = isFiniteNumber(conditions.wind_speed_ms)
    || (isFiniteNumber(source.latitude) && isFiniteNumber(source.longitude));

  const ready = hasStatus || hasRisk || hasInputSource || hasPlumeMetric || hasConditions;
  const reason = hasStatus ? "forecast_status"
    : hasRisk ? "forecast_risk"
      : hasInputSource ? "input_source"
        : hasPlumeMetric ? "plume_metric"
          : hasConditions ? "conditions_or_source"
            : "no_usable_fields";

  return { ready, reason, forecastStatus, riskLevel, inputSource, hasPlumeMetric, hasConditions, scenarioId };
}
