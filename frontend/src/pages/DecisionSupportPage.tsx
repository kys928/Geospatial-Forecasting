import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { DecisionChatPanel } from "../features/decision-support/components/DecisionChatPanel";
import { ConditionsPanel } from "../features/decision-support/components/ConditionsPanel";
import { CHAT_STORAGE_KEY } from "../features/decision-support/constants";
import type { ActiveDatasetScenarioResponse, ChatMessage, DatasetPlaybackState, DatasetScenarioPreview, DecisionSupportLatest, DisplayRow, ForecastContextResponse } from "../features/decision-support/types";
import { cleanAssistantText, formatArea, formatCoordinate, formatDirection, formatNumber, formatPercent, formatPressure, formatSpeed, formatTemperature, formatTimestamp, formatUnknown, safeText } from "../features/decision-support/formatters";
import { getActiveForecastTechnicalDetails, isModelIdentityQuestion } from "../features/forecast-selection/activeForecastHelpers";
import { httpGet, httpPost } from "../services/api/http";
import { useActiveForecast } from "../features/forecast-selection/context/ActiveForecastContext";

const DEFAULT_ASSISTANT: ChatMessage = { role: "assistant", content: "Forecast context is ready. Ask for risk interpretation, plume spread, or scenario caveats." };

function parseStoredMessages(raw: string | null): ChatMessage[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is ChatMessage => item && typeof item === "object" && (item.role === "assistant" || item.role === "user") && typeof item.content === "string")
      .slice(-50);
  } catch {
    return [];
  }
}


function formatPressureWithPbl(context: ForecastContextResponse | null): string {
  const pressure = formatPressure((context as any)?.conditions?.surface_pressure_hpa);
  const pbl = formatNumber((context as any)?.conditions?.pbl_height_m, 0);
  if (pressure === "Unavailable" && pbl === "Unavailable") return "Unavailable";
  return `${pressure} / ${pbl} m`;
}

function formatSourceCoordinates(context: ForecastContextResponse | null): string {
  const latitude = formatCoordinate((context as any)?.source?.latitude);
  const longitude = formatCoordinate((context as any)?.source?.longitude);
  if (latitude === "Unavailable" && longitude === "Unavailable") return "Unavailable";
  return `${latitude}, ${longitude}`;
}

function formatImpactExtent(context: ForecastContextResponse | null): string {
  const area = formatArea((context as any)?.plume_metrics?.affected_area_m2);
  const hectares = formatNumber((context as any)?.plume_metrics?.affected_area_hectares, 2);
  if (area === "Unavailable" && hectares === "Unavailable") return "Unavailable";
  return `${area} (${hectares} ha)`;
}

export function DecisionSupportPage() {
  const { activeScenarioId, activeModelId, activeModelLabel, activeForecastKind, activeSessionId, activePersistedForecastId, activeForecastBundle, activeFramesMetadata, status, error, lastSuccessfulRunKey, setActiveScenario, setActiveModel, runActiveForecast } = useActiveForecast();
  const [datasetScenarios, setDatasetScenarios] = useState<DatasetScenarioPreview[]>([]);
  const [context, setContext] = useState<ForecastContextResponse | null>(null);
  const [data, setData] = useState<DecisionSupportLatest | null>(null);
  const [llmWarning, setLlmWarning] = useState<string | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const threadRef = useRef<HTMLDivElement>(null);
  const briefingKeyRef = useRef<string>("");

  useEffect(() => {
    setMessages(parseStoredMessages(sessionStorage.getItem(CHAT_STORAGE_KEY)));
  }, []);

  useEffect(() => {
    try { sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages.slice(-50))); } catch { /* noop */ }
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    Promise.all([
      httpGet<{ enabled: boolean; scenarios: DatasetScenarioPreview[] }>("/forecast-context/dataset-scenarios"),
      httpGet<ActiveDatasetScenarioResponse>("/forecast-context/dataset-scenarios/active"),
      httpGet<DatasetPlaybackState>("/forecast-context/dataset-playback/state")
    ]).then(([scn, active, playback]) => {
      const scenarios = scn.scenarios ?? [];
      setDatasetScenarios(scenarios);
      const fallback = playback.active_scenario_id ?? active.selected_scenario_id ?? scenarios[0]?.scenario_id ?? null;
      if (fallback && !activeScenarioId) setActiveScenario(fallback, scenarios.find((x) => x.scenario_id === fallback)?.label);
    }).catch(() => setDatasetScenarios([]));
  }, [activeScenarioId, setActiveScenario]);

  const runKey = `${activeModelId}:${activeScenarioId ?? "none"}`;
  useEffect(() => {
    if (activeModelId === "ridge_baseline" && !activeScenarioId) return;
    if (status === "ready" && lastSuccessfulRunKey === runKey) return;
    void runActiveForecast();
  }, [activeModelId, activeScenarioId, lastSuccessfulRunKey, runActiveForecast, runKey, status]);

  const identityKey = `${activeForecastKind}|${activeSessionId ?? "none"}|${activePersistedForecastId ?? "none"}|${activeScenarioId ?? "none"}|${activeModelId}|${String(activeForecastBundle?.summary?.forecast_id ?? "none")}`;
  useEffect(() => {
    const latestUrl = activeForecastKind === "session_convlstm" && activeSessionId ? `/decision-support/latest?session_id=${encodeURIComponent(activeSessionId)}` : "/decision-support/latest";
    void httpGet<DecisionSupportLatest>(latestUrl).then(setData).catch(() => setLlmWarning("LLM unavailable; using active forecast context."));
    const contextUrl = activeForecastKind === "dataset_ridge" ? "/forecast-context/latest?source=dataset" : (activeForecastKind === "session_convlstm" && activeSessionId ? `/forecast-context/latest?source=session&session_id=${encodeURIComponent(activeSessionId)}` : (activeSessionId ? `/forecast-context/latest?session_id=${encodeURIComponent(activeSessionId)}` : "/forecast-context/latest"));
    void httpGet<ForecastContextResponse>(contextUrl).then(setContext).catch(() => setContext(null));
  }, [identityKey, activeForecastKind, activeSessionId]);

  useEffect(() => {
    if (status !== "ready" || !activeForecastBundle) return;
    if (briefingKeyRef.current === identityKey) return;
    briefingKeyRef.current = identityKey;
    const scenarioLabel = datasetScenarios.find((s) => s.scenario_id === activeScenarioId)?.label ?? activeScenarioId ?? "default";
    const source = formatUnknown((activeForecastBundle.summary.metadata as any)?.input_source ?? context?.source?.label ?? "active forecast");
    const prefix = messages.length === 0 ? DEFAULT_ASSISTANT.content : `Forecast changed: ${activeModelLabel} on ${scenarioLabel}.`;
    setMessages((prev) => [...prev, { role: "assistant", content: `${prefix} Source: ${source}.` }]);
  }, [status, activeForecastBundle, identityKey, activeModelLabel, datasetScenarios, activeScenarioId, context, messages.length]);

  async function activateDatasetScenario(scenarioId: string) {
    const label = datasetScenarios.find((s) => s.scenario_id === scenarioId)?.label ?? scenarioId;
    setActiveScenario(scenarioId, label);
  }

  async function sendQuestion(question: string) {
    if (!question.trim()) return;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setChatQuestion("");
    if (isModelIdentityQuestion(question)) {
      setMessages((prev) => [...prev, { role: "assistant", content: `Active model is ${activeModelLabel} (${activeForecastKind}).` }]);
      return;
    }
    if (activeForecastKind === "batch_gaussian") {
      setMessages((prev) => [...prev, { role: "assistant", content: "Gaussian decision-support chat is not persisted per forecast yet. Use technical details and forecast summary for current context." }]);
      return;
    }
    try {
      const payload: Record<string, unknown> = { message: question };
      if (activeForecastKind === "session_convlstm" && activeSessionId) payload.session_id = activeSessionId;
      const response = await httpPost<{ answer?: string }>("/decision-support/chat", payload);
      setMessages((prev) => [...prev, { role: "assistant", content: cleanAssistantText(safeText(response.answer, "No answer available.")) }]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: `Active model: ${activeModelLabel}. Forecast kind: ${activeForecastKind}.` }]);
    }
  }

  const summary = activeForecastBundle?.summary ?? {};
  const metadata = ((summary.metadata as Record<string, unknown> | undefined) ?? {});
  const currentConditionsRows: DisplayRow[] = [
    ["Wind", `${formatSpeed((context as any)?.conditions?.wind_speed_ms)} ${formatDirection((context as any)?.conditions?.wind_direction_label ?? (context as any)?.conditions?.wind_direction_deg)}`],
    ["Temperature", formatTemperature((context as any)?.conditions?.temperature_c)],
    ["Humidity", formatPercent((context as any)?.conditions?.humidity_pct)],
    ["Pressure / PBL", formatPressureWithPbl(context)],
    ["Source", formatSourceCoordinates(context)]
  ];
  const currentForecastRows: DisplayRow[] = [
    ["Status", formatUnknown((context as any)?.forecast?.status ?? (summary as any).status ?? status)],
    ["Risk", formatUnknown((context as any)?.forecast?.risk_level ?? (summary as any).risk_level ?? "Unavailable")],
    ["Input source", formatUnknown((context as any)?.forecast?.input_source ?? metadata.input_source ?? (context as any)?.source?.label)]
  ];
  const plumeDetailRows: DisplayRow[] = [
    ["Impact extent", formatImpactExtent(context)],
    ["Peak plume score", formatNumber((context as any)?.plume_metrics?.max_concentration, 4)],
    ["Predicted spread", formatDirection((context as any)?.plume_metrics?.dominant_spread_direction)],
    ["Forecast time", formatTimestamp((context as any)?.forecast?.timestamp ?? (context as any)?.forecast?.issued_at ?? (summary as any).created_at)]
  ];

  const detailsRows = getActiveForecastTechnicalDetails({ activeModelId, activeForecastKind, activeSessionId, activePersistedForecastId, summary, rasterMetadata: activeForecastBundle?.rasterMetadata ?? null, framesMetadata: activeFramesMetadata as Record<string, unknown> | null, forecastContextRuntime: (context as any)?.runtime ?? null });
  const enrichedDetails = useMemo(() => ([
    ["Forecast horizon", formatUnknown((summary as any).forecast_horizon ?? (context as any)?.forecast?.horizon)],
    ["Mean plume score", formatNumber((context as any)?.plume_metrics?.mean_concentration, 4)],
    ["Detection threshold", formatNumber((context as any)?.plume_metrics?.threshold_used ?? (summary as any).detection_threshold, 6)],
    ["Grid size", formatUnknown((summary as any).grid_size ?? (activeFramesMetadata as any)?.shape?.join(" × ") ?? ((context as any)?.plume_metrics?.grid_rows != null && (context as any)?.plume_metrics?.grid_columns != null ? `${(context as any).plume_metrics.grid_rows} × ${(context as any).plume_metrics.grid_columns}` : undefined))],
    ...detailsRows
  ] as DisplayRow[]), [summary, context, activeFramesMetadata, detailsRows]);

  return <AppShell title="Forecast Overview" subtitle="Forecast interpretation, current conditions, and plume result.">
    <div className="decision-support-layout">
      <DecisionChatPanel hasContext={true} llmWarning={error ?? llmWarning} messages={messages} chatQuestion={chatQuestion} setChatQuestion={setChatQuestion} sendQuestion={sendQuestion} threadRef={threadRef} />
      <ConditionsPanel datasetScenarios={datasetScenarios} activeScenario={activeScenarioId ?? ""} activateDatasetScenario={activateDatasetScenario} activeModelId={activeModelId} setActiveModel={setActiveModel} currentConditionsRows={currentConditionsRows} currentForecastRows={currentForecastRows} plumePresent={Boolean(activeForecastBundle)} plumeDetailRows={plumeDetailRows} detailsRows={enrichedDetails} filterAvailableRows={(r) => r.filter((x) => String(x[1]).trim().length > 0 && String(x[1]) !== "Unavailable")} rawContext={{ active_model_id: activeModelId, active_forecast_kind: activeForecastKind, summary, decision_latest: data, context }} />
    </div>
  </AppShell>;
}
