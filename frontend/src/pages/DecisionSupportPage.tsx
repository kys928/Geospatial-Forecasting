import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { DecisionChatPanel } from "../features/decision-support/components/DecisionChatPanel";
import { ConditionsPanel } from "../features/decision-support/components/ConditionsPanel";
import { CHAT_STORAGE_KEY } from "../features/decision-support/constants";
import type { ActiveDatasetScenarioResponse, ChatMessage, DatasetPlaybackState, DatasetScenarioPreview, DecisionSupportLatest, DisplayRow, ForecastContextResponse } from "../features/decision-support/types";
import { cleanAssistantText, safeText } from "../features/decision-support/formatters";
import { getActiveForecastTechnicalDetails, isModelIdentityQuestion } from "../features/forecast-selection/activeForecastHelpers";
import { httpGet, httpPost } from "../services/api/http";
import { useActiveForecast } from "../features/forecast-selection/context/ActiveForecastContext";

const DEFAULT_ASSISTANT: ChatMessage = { role: "assistant", content: "Forecast context is ready. Ask for risk interpretation, plume spread, or scenario caveats." };

export function DecisionSupportPage() {
  const { activeScenarioId, activeModelId, activeModelLabel, activeForecastKind, activeSessionId, activePersistedForecastId, activeForecastBundle, activeFramesMetadata, status, error, setActiveScenario, setActiveModel, runActiveForecast } = useActiveForecast();
  const [datasetScenarios, setDatasetScenarios] = useState<DatasetScenarioPreview[]>([]);
  const [context, setContext] = useState<ForecastContextResponse | null>(null);
  const [data, setData] = useState<DecisionSupportLatest | null>(null);
  const [llmWarning, setLlmWarning] = useState<string | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const threadRef = useRef<HTMLDivElement>(null);
  const completedRunKeyRef = useRef<string>("");
  const briefingKeyRef = useRef<string>("");

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(CHAT_STORAGE_KEY);
      if (raw) setMessages(JSON.parse(raw) as ChatMessage[]);
    } catch {
      setMessages([]);
    }
  }, []);

  useEffect(() => {
    try { sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages)); } catch { /* noop */ }
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
    if (status === "running" && completedRunKeyRef.current !== runKey) return;
    if (completedRunKeyRef.current === runKey && status === "ready") return;
    void runActiveForecast();
  }, [activeModelId, activeScenarioId, runActiveForecast, runKey, status]);

  useEffect(() => {
    if (status === "ready") completedRunKeyRef.current = runKey;
    if (status === "error") completedRunKeyRef.current = "";
  }, [runKey, status]);

  const identityKey = `${activeForecastKind}|${activeSessionId ?? "none"}|${activePersistedForecastId ?? "none"}|${activeScenarioId ?? "none"}|${activeModelId}|${String(activeForecastBundle?.summary?.forecast_id ?? "none")}`;
  useEffect(() => {
    const latestUrl = activeForecastKind === "session_convlstm" && activeSessionId ? `/decision-support/latest?session_id=${encodeURIComponent(activeSessionId)}` : "/decision-support/latest";
    void httpGet<DecisionSupportLatest>(latestUrl).then(setData).catch(() => setLlmWarning("LLM unavailable; using active forecast context."));
    const contextUrl = activeForecastKind === "dataset_ridge" ? "/forecast-context/latest?source=dataset" : (activeSessionId ? `/forecast-context/latest?session_id=${encodeURIComponent(activeSessionId)}` : "/forecast-context/latest");
    void httpGet<ForecastContextResponse>(contextUrl).then(setContext).catch(() => setContext(null));
  }, [identityKey, activeForecastKind, activeSessionId]);

  useEffect(() => {
    if (status !== "ready" || !activeForecastBundle) return;
    if (briefingKeyRef.current === identityKey) return;
    briefingKeyRef.current = identityKey;
    const scenarioLabel = datasetScenarios.find((s) => s.scenario_id === activeScenarioId)?.label ?? activeScenarioId ?? "default";
    const source = String((activeForecastBundle.summary.metadata as any)?.input_source ?? context?.source?.label ?? "active forecast");
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
    ["Wind", String((context as any)?.conditions?.wind_speed ?? (summary as any).wind_speed ?? "n/a")],
    ["Temperature", String((context as any)?.conditions?.temperature ?? (summary as any).temperature ?? "n/a")],
    ["Humidity", String((context as any)?.conditions?.humidity ?? (summary as any).humidity ?? "n/a")],
    ["Pressure / PBL", String((context as any)?.conditions?.pressure ?? (context as any)?.conditions?.pbl_height ?? (summary as any).pressure ?? "n/a")],
    ["Source", String((context as any)?.source?.label ?? metadata.input_source ?? "n/a")]
  ];
  const currentForecastRows: DisplayRow[] = [
    ["Status", String((summary as any).status ?? status)],
    ["Risk", String((summary as any).risk_level ?? (context as any)?.forecast?.risk_level ?? "n/a")],
    ["Input source", String(metadata.input_source ?? (context as any)?.source?.label ?? "n/a")]
  ];
  const plumeDetailRows: DisplayRow[] = [
    ["Impact extent", String((context as any)?.plume_metrics?.impact_extent ?? (summary as any).impact_extent ?? "n/a")],
    ["Peak plume score", String((context as any)?.plume_metrics?.peak_plume_score ?? (summary as any).peak_plume_score ?? "n/a")],
    ["Predicted spread", String((context as any)?.plume_metrics?.predicted_spread ?? (summary as any).predicted_spread ?? "n/a")],
    ["Forecast time", String((summary as any).forecast_time ?? (summary as any).created_at ?? "n/a")]
  ];

  const detailsRows = getActiveForecastTechnicalDetails({ activeModelId, activeForecastKind, activeSessionId, activePersistedForecastId, summary, rasterMetadata: activeForecastBundle?.rasterMetadata ?? null, framesMetadata: activeFramesMetadata as Record<string, unknown> | null, forecastContextRuntime: (context as any)?.runtime ?? null });
  const enrichedDetails = useMemo(() => ([
    ["Forecast horizon", String((summary as any).forecast_horizon ?? (context as any)?.forecast?.horizon ?? "n/a")],
    ["Mean plume score", String((context as any)?.plume_metrics?.mean_plume_score ?? (summary as any).mean_plume_score ?? "n/a")],
    ["Detection threshold", String((summary as any).detection_threshold ?? metadata.detection_threshold ?? "n/a")],
    ["Grid size", String((summary as any).grid_size ?? (activeFramesMetadata as any)?.shape?.join("x") ?? "n/a")],
    ...detailsRows
  ] as DisplayRow[]), [summary, context, metadata, activeFramesMetadata, detailsRows]);

  return <AppShell title="Forecast Overview" subtitle="Forecast interpretation, current conditions, and plume result.">
    <div className="decision-support-layout">
      <DecisionChatPanel hasContext={true} llmWarning={error ?? llmWarning} messages={messages} chatQuestion={chatQuestion} setChatQuestion={setChatQuestion} sendQuestion={sendQuestion} threadRef={threadRef} />
      <ConditionsPanel datasetScenarios={datasetScenarios} activeScenario={activeScenarioId ?? ""} activateDatasetScenario={activateDatasetScenario} activeModelId={activeModelId} setActiveModel={setActiveModel} currentConditionsRows={currentConditionsRows} currentForecastRows={currentForecastRows} plumePresent={Boolean(activeForecastBundle)} plumeDetailRows={plumeDetailRows} detailsRows={enrichedDetails} filterAvailableRows={(r) => r.filter((x) => String(x[1]).trim().length > 0)} rawContext={{ active_model_id: activeModelId, active_forecast_kind: activeForecastKind, summary, decision_latest: data, context }} />
    </div>
  </AppShell>;
}
