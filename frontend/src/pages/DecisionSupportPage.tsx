import { useEffect, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { sessionClient } from "../features/sessions/api/sessionClient";
import type { SessionDetail, SessionStateSummary } from "../features/sessions/types/session.types";
import { httpGet, httpPost } from "../services/api/http";
import { DecisionChatPanel } from "../features/decision-support/components/DecisionChatPanel";
import { ConditionsPanel } from "../features/decision-support/components/ConditionsPanel";
import { CHAT_STORAGE_KEY } from "../features/decision-support/constants";
import {
  cleanAssistantText,
  formatArea,
  formatCoordinate,
  formatDirection,
  formatDurationMinutes,
  formatGridSize,
  formatNumber,
  formatPercent,
  formatPressure,
  formatRiskLevel,
  formatSpeed,
  formatTemperature,
  formatTimestamp,
  formatUnknown,
  getNestedValue,
  safeText
} from "../features/decision-support/formatters";
import { hasMeaningfulPlume } from "../features/decision-support/plumeLogic";
import { isUsableForecastContext } from "../features/decision-support/contextReadiness";
import type {
  ChatMessage,
  DecisionSupportLatest,
  ForecastContextResponse
} from "../features/decision-support/types";

export function DecisionSupportPage() {
  const { activeSessionId, latestForecastBundle } = useSessionForecastView();
  const [data, setData] = useState<DecisionSupportLatest | null>(null);
  const [context, setContext] = useState<ForecastContextResponse | null>(null);
  const [, setSession] = useState<SessionDetail | null>(null);
  const [activeScenario, setActiveScenario] = useState<string>("");
  const [, setSessionState] = useState<SessionStateSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [llmWarning, setLlmWarning] = useState<string | null>(null);
  const [isContextLoading, setIsContextLoading] = useState(true);
  const [chatQuestion, setChatQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const raw = sessionStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return [];
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((item): item is ChatMessage =>
          Boolean(
            item
            && typeof item === "object"
            && ((item as ChatMessage).role === "assistant" || (item as ChatMessage).role === "user")
            && typeof (item as ChatMessage).content === "string"
          )
        )
        .slice(-50);
    } catch {
      return [];
    }
  });
  const threadRef = useRef<HTMLDivElement>(null);
  const lastBriefingKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const contextUrl = activeSessionId
      ? `/forecast-context/latest?source=session&session_id=${encodeURIComponent(activeSessionId)}`
      : "/forecast-context/latest?source=session";
    void httpGet<ForecastContextResponse>(contextUrl)
      .then((latestContext) => {
        setContext(latestContext);
        const scenarioId = safeText(latestContext.forecast?.scenario_id, "");
        setActiveScenario(scenarioId);
      })
      .catch(() => setContext(null))
      .finally(() => setIsContextLoading(false));
  }, [activeSessionId, latestForecastBundle]);

  useEffect(() => {
    if (!activeSessionId) {
      setSession(null);
      setSessionState(null);
      return;
    }
    Promise.all([sessionClient.getSession(activeSessionId), sessionClient.getSessionState(activeSessionId)])
      .then(([sessionDetail, state]) => {
        setSession(sessionDetail);
        setSessionState(state);
      })
      .catch(() => {
        setSession(null);
        setSessionState(null);
      });
  }, [activeSessionId]);

  const explanation = latestForecastBundle?.explanation ?? {};
  const summary = latestForecastBundle?.summary ?? {};

  useEffect(() => {
    try {
      sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages.slice(-50)));
    } catch {
      // ignore storage failures
    }
    const thread = threadRef.current;
    if (!thread) return;
    thread.scrollTop = thread.scrollHeight;
  }, [messages]);

  const hasContext = Boolean(latestForecastBundle || data || context || activeScenario);
  const values = summary as Record<string, unknown>;
  const ctxForecast = context?.forecast ?? {};
  const ctxConditions = context?.conditions ?? {};
  const ctxSource = context?.source ?? {};
  const ctxPlume = context?.plume_metrics ?? {};

  const contextReadiness = isUsableForecastContext(context);
  const isContextReady = contextReadiness.ready;

  useEffect(() => {
    console.debug("[decision-support] context readiness", {
      ready: contextReadiness.ready,
      reason: contextReadiness.reason,
      forecastStatus: contextReadiness.forecastStatus,
      riskLevel: contextReadiness.riskLevel,
      inputSource: contextReadiness.inputSource,
      hasPlumeMetric: contextReadiness.hasPlumeMetric,
      hasConditions: contextReadiness.hasConditions,
      scenarioId: contextReadiness.scenarioId || activeScenario
    });
  }, [contextReadiness, activeScenario]);

  const riskLevel = formatRiskLevel(ctxForecast.risk_level ?? data?.risk_level ?? explanation.risk_level ?? values.risk_level);
  const forecastEvidence = getNestedValue(data, "forecast_evidence") as Record<string, unknown> | undefined;
  const forecastEvidenceStats = getNestedValue(forecastEvidence, "summary_statistics") as Record<string, unknown> | undefined;
  const affectedAreaM2 = ctxPlume.affected_area_m2 ?? getNestedValue(summary, "affected_area_m2", "affected_area", "summary_statistics.affected_area_m2", "summary_statistics.affected_area") ?? getNestedValue(forecastEvidence, "affected_area_m2", "affected_area") ?? getNestedValue(forecastEvidenceStats, "affected_area_m2", "affected_area");
  const affectedCellsRaw = ctxPlume.affected_cells_above_threshold ?? getNestedValue(summary, "affected_cells_above_threshold", "summary_statistics.affected_cells_above_threshold") ?? getNestedValue(forecastEvidence, "affected_cells_above_threshold") ?? getNestedValue(forecastEvidenceStats, "affected_cells_above_threshold");
  const maxConcentration = ctxPlume.max_concentration ?? getNestedValue(summary, "max_concentration", "summary_statistics.max_concentration") ?? getNestedValue(forecastEvidence, "max_concentration") ?? getNestedValue(forecastEvidenceStats, "max_concentration");
  const meanConcentration = ctxPlume.mean_concentration ?? getNestedValue(summary, "mean_concentration", "summary_statistics.mean_concentration") ?? getNestedValue(forecastEvidence, "mean_concentration") ?? getNestedValue(forecastEvidenceStats, "mean_concentration");
  const dominantSpreadDirection = ctxPlume.dominant_spread_direction ?? getNestedValue(summary, "dominant_spread_direction", "summary_statistics.dominant_spread_direction", "wind_direction", "direction") ?? getNestedValue(forecastEvidence, "dominant_spread_direction") ?? getNestedValue(forecastEvidenceStats, "dominant_spread_direction");
  const thresholdUsed = ctxPlume.threshold_used ?? getNestedValue(summary, "threshold_used", "threshold", "summary_statistics.threshold_used") ?? getNestedValue(forecastEvidence, "threshold_used") ?? getNestedValue(forecastEvidenceStats, "threshold_used");
  const forecastTime = ctxForecast.timestamp ?? ctxForecast.issued_at ?? getNestedValue(data, "last_forecast_time") ?? getNestedValue(summary, "timestamp", "issued_at");
  const plumePresent = hasMeaningfulPlume({ affectedAreaM2, affectedCellsAboveThreshold: affectedCellsRaw, maxConcentration, explanationSummary: explanation.summary, riskLevel });
  const hasThreatSignal = safeText(data?.situation_summary ?? explanation.summary, "").toLowerCase().includes("threat");
  const explicitNoPlumeSignal = Number(affectedAreaM2) === 0 || Number(affectedCellsRaw) === 0 || Number(maxConcentration) === 0 || safeText(explanation.summary, "").toLowerCase().includes("no meaningful plume");
  const plumeStatus = plumePresent ? (hasThreatSignal ? "Threat detected" : "Plume detected above threshold") : (explicitNoPlumeSignal ? "No meaningful plume above threshold" : "Forecast unavailable");

  const windSpeedValue = ctxConditions.wind_speed_ms;
  const windDirectionValue = ctxConditions.wind_direction_label ?? ctxConditions.wind_direction_deg;
  const uWindValue = ctxConditions.u10m_ms;
  const vWindValue = ctxConditions.v10m_ms;
  const meteorologyRows = [
    ["Wind speed", formatSpeed(windSpeedValue)],
    ["Wind direction", formatDirection(windDirectionValue)],
    ["U wind", formatSpeed(uWindValue)],
    ["V wind", formatSpeed(vWindValue)],
    ["Temperature", formatTemperature(ctxConditions.temperature_c)],
    ["Relative humidity", formatPercent(ctxConditions.humidity_pct)],
    ["Surface pressure", formatPressure(ctxConditions.surface_pressure_hpa)],
    ["PBL height", `${formatNumber(ctxConditions.pbl_height_m, 1)} m`],
    ["Meteorology timestamp", formatTimestamp(ctxConditions.meteorology_timestamp)],
    ["Meteorology source", formatUnknown(ctxConditions.meteorology_source)]
  ] as Array<[string, string]>;

  const windSpeed = formatSpeed(windSpeedValue);
  const windDirection = formatDirection(windDirectionValue);
  const uWind = formatSpeed(uWindValue);
  const vWind = formatSpeed(vWindValue);
  const windSummary = windSpeed !== "Unavailable" && windDirection !== "Unavailable" ? `${windSpeed} ${windDirection}` : (uWind !== "Unavailable" || vWind !== "Unavailable" ? `U ${uWind}, V ${vWind}`.replace("U Unavailable, ", "").replace(", V Unavailable", "") : "Unavailable");
  const displayValue = (value: string, fallback = "Not available") => value === "Unavailable" ? fallback : value;

  const weatherCompactRows = [
    ["Wind", displayValue(windSummary)],
    ["Temperature", displayValue(formatTemperature(ctxConditions.temperature_c))],
    ["Humidity", displayValue(formatPercent(ctxConditions.humidity_pct))],
    ["Pressure / PBL", (() => {
      const pressure = formatPressure(ctxConditions.surface_pressure_hpa);
      const pbl = `${formatNumber(ctxConditions.pbl_height_m, 1)} m`;
      if (pressure === "Unavailable" && pbl === "Unavailable m") return "Not available";
      if (pressure !== "Unavailable" && pbl !== "Unavailable m") return `${pressure} / ${pbl}`;
      return pressure !== "Unavailable" ? pressure : pbl;
    })()]
  ] as Array<[string, string]>;
  const sourceLatitude = formatCoordinate(ctxSource.latitude);
  const sourceLongitude = formatCoordinate(ctxSource.longitude);
  const sourceLocation = sourceLatitude !== "Unavailable" && sourceLongitude !== "Unavailable" ? `${sourceLatitude}, ${sourceLongitude}` : null;
  const currentConditionsRows = [...weatherCompactRows, ["Source", sourceLocation ?? "Not configured"]] as Array<[string, string]>;

  const lastForecastLabel = formatTimestamp(forecastTime);
  const currentForecastRows = [["Status", formatUnknown(ctxForecast.status) || plumeStatus], ["Risk", riskLevel], ["Input source", formatUnknown(ctxForecast.input_source)]] as Array<[string, string]>;
  const plumeDetailRows: Array<[string, string]> = plumePresent
    ? [["Impact extent", formatArea(affectedAreaM2) === "Unavailable" ? "Estimated from plume grid" : formatArea(affectedAreaM2)], ["Peak plume score", formatNumber(maxConcentration)], ["Predicted spread", formatDirection(dominantSpreadDirection)], ...(lastForecastLabel !== "Unavailable" ? [["Forecast time", lastForecastLabel] as [string, string]] : [])]
    : [];

  const forecastHorizon = getNestedValue(summary, "forecast_horizon_minutes", "horizon_minutes", "summary_statistics.forecast_horizon_minutes") ?? getNestedValue(forecastEvidence, "forecast_horizon_minutes", "horizon_minutes");
  const gridSizeValue = getNestedValue(summary, "grid", "grid_size", "grid_shape", "summary_statistics.grid_size") ?? getNestedValue(forecastEvidence, "grid", "grid_size", "grid_shape");

  const detailsRows = [["Forecast horizon", formatDurationMinutes(forecastHorizon)], ["Mean plume score", formatNumber(meanConcentration)], ["Detection threshold", formatUnknown(thresholdUsed)], ["Grid size", formatGridSize([ctxPlume.grid_rows, ctxPlume.grid_columns]) === "Unavailable" ? formatGridSize(gridSizeValue) : formatGridSize([ctxPlume.grid_rows, ctxPlume.grid_columns])]] as Array<[string, string]>;

  const filterAvailableRows = (rows: Array<[string, string]>, { allowZero = true } = {}) =>
    rows.filter(([, value]) => {
      const normalized = value.trim().toLowerCase();
      if (!allowZero && normalized === "0") return false;
      return normalized !== "unavailable" && normalized !== "not available" && normalized !== "unavailable m";
    });
  const weatherContext = Object.fromEntries(meteorologyRows.filter(([, value]) => value !== "Unavailable"));
  const overlayMetadata = (context?.raw?.overlay_metadata as Record<string, unknown> | undefined) ?? {};
  const overlayFeatures = (context?.raw?.overlay_features as Array<Record<string, unknown>> | undefined) ?? [];
  const rawContext: Record<string, unknown> = {
    selected_scenario: activeScenario || ctxForecast.scenario_id,
    forecast: ctxForecast,
    conditions: ctxConditions,
    source: ctxSource,
    plume_metrics: ctxPlume,
    weather_context: weatherContext,
    model_inference: getNestedValue(context, "raw.model_inference", "raw.model_inference") ?? getNestedValue(context, "raw.model_inference"),
    overlay_summary: {
      endpoint_path: "/forecast-context/dataset-scenarios/active/overlay",
      feature_count: overlayMetadata.feature_count,
      plume_polygon_count: overlayMetadata.plume_polygon_count,
      source_point_count: overlayMetadata.source_point_count,
      bbox: overlayMetadata.bbox,
      first_3_feature_properties: overlayFeatures.slice(0, 3).map((feature) => feature.properties ?? {})
    },
    raw_reference: {
      source_file: getNestedValue(context, "raw.source_file"),
      scenario_id: ctxForecast.scenario_id,
      window_id: getNestedValue(context, "raw.window_row.window_id"),
      target_usage: getNestedValue(context, "raw.target_usage")
    }
  };

  const buildOperatorBriefing = (briefingText?: string): string => {
    const candidateBriefing = safeText(briefingText, "");
    if (candidateBriefing && candidateBriefing.length > 40 && !/\b\d+[\d,]*\s+grid cells?\b/i.test(candidateBriefing)) return cleanAssistantText(candidateBriefing);
    const status = formatUnknown(ctxForecast.status) || plumeStatus;
    const scenarioName = safeText(ctxForecast.scenario_id, activeScenario || "current scenario");
    const windLine = windSpeed !== "Unavailable" && windDirection !== "Unavailable" ? `Wind is ${windDirection} at ${windSpeed}.` : "Wind details are partially available from current context.";
    const sourceLine = sourceLocation ? `The modeled source location is ${sourceLocation}.` : "Source location is not fully configured in this context.";
    const plumeLine = plumePresent ? `Predicted plume intensity remains limited with peak score near ${formatNumber(maxConcentration)} and spread trending ${formatDirection(dominantSpreadDirection)}.` : "The plume signal remains limited in this forecast window.";
    const provenance = context?.provenance ?? {};
    const activeConvLstm = provenance.forecast_source === "active_model_inference" && provenance.model_family === "ConvLSTM" && provenance.fallback_used !== true;
    const inputSource = String(provenance.input_source ?? ctxForecast.input_source ?? "unknown");
    const inputLine = inputSource === "dataset_window"
      ? "Input is a dataset window seed, not live sensor confirmation."
      : inputSource === "degraded_session_state"
        ? "Input is degraded session state, not live sensor confirmation."
        : "Input source is current session observations or configured session context.";
    const limitationLine = activeConvLstm ? `This briefing is grounded in the active ConvLSTM session forecast context. ${inputLine}` : "Active ConvLSTM forecast context is unavailable or not confirmed by provenance.";
    return cleanAssistantText(`Scenario ${scenarioName} is currently ${riskLevel.toLowerCase()} risk with status ${status}. ${windLine} ${plumeLine} ${sourceLine} ${limitationLine}`);
  };

  const briefingKey = String(activeScenario || ctxForecast.scenario_id || ctxForecast.forecast_id || forecastTime || "default");
  const scenarioLabel = safeText(ctxForecast.scenario_id, activeScenario || "Scenario");

  useEffect(() => {
    if (!hasContext || !isContextReady) {
      setIsContextLoading(true);
      return;
    }
    setIsContextLoading(false);
    if (lastBriefingKeyRef.current === briefingKey && data?.briefing) return;
    void httpGet<DecisionSupportLatest>("/decision-support/latest")
      .then((latest) => {
        if (latest.mode === "context_loading") return;
        setData(latest);
        setLlmWarning(null);
        const summaryText = buildOperatorBriefing(latest.briefing);
        setMessages((prev) => {
          if (prev.length === 0) {
            lastBriefingKeyRef.current = briefingKey;
            return [{ role: "assistant", content: summaryText }];
          }
          if (lastBriefingKeyRef.current === briefingKey) return prev;
          lastBriefingKeyRef.current = briefingKey;
          return [...prev, { role: "assistant", content: `Scenario changed: ${scenarioLabel}. ${summaryText}` }];
        });
      })
      .catch(() => {
        setLlmWarning("LLM unavailable; using forecast context.");
      });
  }, [briefingKey, hasContext, isContextReady, data?.briefing, scenarioLabel]);

  async function activateDatasetScenario(_scenarioId: string) {
    // Dataset playback is not part of the normal Forecast Overview workflow.
    return Promise.resolve();
  }

  async function sendQuestion(question: string) {
    if (!question.trim() || !hasContext) return;
    if (!isContextReady) {
      setMessages((prev) => [...prev, { role: "assistant", content: "Forecast context is still loading. Please wait a moment and try again." }]);
      return;
    }
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setChatQuestion("");
    try {
      const response = await httpPost<{ answer?: string }>("/decision-support/chat", { message: question });
      setLlmWarning(null);
      setMessages((prev) => [...prev, { role: "assistant", content: cleanAssistantText(safeText(response.answer, "No answer available.")) }]);
    } catch {
      setLlmWarning("LLM unavailable; using forecast context.");
      const statusText = plumePresent ? "Plume detected above threshold" : safeText(ctxForecast.status, "No meaningful plume above threshold");
      const riskText = riskLevel;
      const windText = windSpeed !== "Unavailable" || windDirection !== "Unavailable" ? `Wind ${windSpeed} ${windDirection}`.trim() : "Wind details unavailable";
      const datasetNote = "Uses current forecast context only.";
      const fallback = cleanAssistantText(`${statusText}. Risk: ${riskText}. ${windText}. ${datasetNote}`);
      setMessages((prev) => [...prev, { role: "assistant", content: fallback }]);
    }
  }

  return <AppShell title="Forecast Overview" subtitle="Forecast interpretation, current conditions, and plume result.">
    {error ? <section className="panel"><p>{error}</p></section> : null}
    <div className="decision-support-layout">
      <DecisionChatPanel hasContext={hasContext && !isContextLoading} llmWarning={llmWarning} messages={messages} chatQuestion={chatQuestion} setChatQuestion={setChatQuestion} sendQuestion={sendQuestion} threadRef={threadRef} loadingMessage={isContextLoading ? "Loading forecast context..." : undefined} />
      <ConditionsPanel datasetScenarios={[]} activeScenario={activeScenario} activateDatasetScenario={activateDatasetScenario} currentConditionsRows={currentConditionsRows} currentForecastRows={currentForecastRows} plumePresent={plumePresent} plumeDetailRows={plumeDetailRows} detailsRows={detailsRows} filterAvailableRows={filterAvailableRows} rawContext={rawContext} />
    </div>
  </AppShell>;
}
