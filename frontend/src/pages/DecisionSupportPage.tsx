import { useEffect, useRef, useState } from "react";
import { AppShell } from "../app/AppShell";
import { DecisionChatPanel } from "../features/decision-support/components/DecisionChatPanel";
import { ConditionsPanel } from "../features/decision-support/components/ConditionsPanel";
import { CHAT_STORAGE_KEY } from "../features/decision-support/constants";
import type { ActiveDatasetScenarioResponse, ChatMessage, DatasetPlaybackState, DatasetScenarioPreview, DecisionSupportLatest, ForecastContextResponse } from "../features/decision-support/types";
import { cleanAssistantText, safeText } from "../features/decision-support/formatters";
import { getActiveForecastTechnicalDetails, isModelIdentityQuestion } from "../features/forecast-selection/activeForecastHelpers";
import { httpGet, httpPost } from "../services/api/http";
import { useActiveForecast } from "../features/forecast-selection/context/ActiveForecastContext";

export function DecisionSupportPage() {
  const { activeScenarioId, activeModelId, activeModelLabel, activeForecastKind, activeSessionId, activePersistedForecastId, activeForecastBundle, setActiveScenario, setActiveModel, runActiveForecast } = useActiveForecast();
  const [datasetScenarios, setDatasetScenarios] = useState<DatasetScenarioPreview[]>([]);
  const [context, setContext] = useState<ForecastContextResponse | null>(null);
  const [data, setData] = useState<DecisionSupportLatest | null>(null);
  const [llmWarning, setLlmWarning] = useState<string | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const threadRef = useRef<HTMLDivElement>(null);
  const runKeyRef = useRef<string>("");

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

  useEffect(() => {
    if (!activeScenarioId) return;
    const runKey = `${activeModelId}:${activeScenarioId}`;
    if (runKeyRef.current === runKey) return;
    runKeyRef.current = runKey;
    void runActiveForecast();
  }, [activeScenarioId, activeModelId, runActiveForecast]);

  useEffect(() => {
    const latestUrl = activeForecastKind === "session_convlstm" && activeSessionId ? `/decision-support/latest?session_id=${encodeURIComponent(activeSessionId)}` : "/decision-support/latest";
    void httpGet<DecisionSupportLatest>(latestUrl).then(setData).catch(() => setLlmWarning("LLM unavailable; using active forecast context."));
    const contextUrl = activeForecastKind === "dataset_ridge" ? "/forecast-context/latest?source=dataset" : (activeSessionId ? `/forecast-context/latest?session_id=${encodeURIComponent(activeSessionId)}` : "/forecast-context/latest");
    void httpGet<ForecastContextResponse>(contextUrl).then(setContext).catch(() => setContext(null));
  }, [activeForecastKind, activeSessionId]);

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
  const detailsRows = getActiveForecastTechnicalDetails({
    activeModelId,
    activeForecastKind,
    activeSessionId,
    activePersistedForecastId,
    summary
  });

  const rows: Array<[string, string]> = [["Scenario", datasetScenarios.find((s) => s.scenario_id === activeScenarioId)?.label ?? activeScenarioId ?? "Unavailable"], ["Status", String((summary as any)?.status ?? "ready")]];

  return <AppShell title="Forecast Overview" subtitle="Forecast interpretation, current conditions, and plume result.">
    <div className="decision-support-layout">
      <DecisionChatPanel hasContext={true} llmWarning={llmWarning} messages={messages} chatQuestion={chatQuestion} setChatQuestion={setChatQuestion} sendQuestion={sendQuestion} threadRef={threadRef} />
      <ConditionsPanel datasetScenarios={datasetScenarios} activeScenario={activeScenarioId ?? ""} activateDatasetScenario={activateDatasetScenario} activeModelId={activeModelId} setActiveModel={setActiveModel} currentConditionsRows={rows} currentForecastRows={rows} plumePresent={Boolean(activeForecastBundle)} plumeDetailRows={[]} detailsRows={detailsRows} filterAvailableRows={(r) => r} rawContext={{ active_model_id: activeModelId, active_forecast_kind: activeForecastKind, summary, decision_latest: data, context }} />
    </div>
  </AppShell>;
}
