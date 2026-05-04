import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { AppShell } from "../app/AppShell";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";
import { sessionClient } from "../features/sessions/api/sessionClient";
import type { SessionDetail, SessionStateSummary } from "../features/sessions/types/session.types";
import { httpGet, httpPost } from "../services/api/http";

type DecisionSupportLatest = {
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

type ChatMessage = { role: "assistant" | "user"; content: string };

const SUGGESTED_PROMPTS = [
  "Why is the risk low?",
  "What live inputs are missing?",
  "How reliable is this forecast?",
  "What changed since the previous forecast?"
];

function safeText(value: unknown, fallback = "Unavailable"): string {
  return typeof value === "string" && value.trim().length > 0 ? value : fallback;
}
function isPresentValue(value: unknown): boolean {
  if (value == null) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  return true;
}

function formatTimestamp(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatNumber(value: unknown, digits = 2): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  return parsed.toLocaleString(undefined, { maximumFractionDigits: digits });
}
function formatArea(value: unknown): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  if (parsed === 0) return "0 m²";
  if (Math.abs(parsed) >= 10000) return `${(parsed / 10000).toLocaleString(undefined, { maximumFractionDigits: 1 })} ha`;
  return `${parsed.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²`;
}

function formatUnknown(value: unknown): string {
  if (value == null) return "Unavailable";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.length ? value.map((item) => formatUnknown(item)).join(", ") : "Unavailable";
  return "Unavailable";
}

function formatRiskLevel(value: unknown): string {
  const text = safeText(value, "Unknown").toLowerCase();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function DecisionSupportPage() {
  const { activeSessionId, latestForecastBundle } = useSessionForecastView();
  const [data, setData] = useState<DecisionSupportLatest | null>(null);
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [sessionState, setSessionState] = useState<SessionStateSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const threadRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    httpGet<DecisionSupportLatest>("/decision-support/latest").then(setData).catch((e) => setError(e instanceof Error ? e.message : "Unavailable"));
  }, []);

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

  const modeLabel = useMemo(() => {
    const usedLlm = data?.used_llm === true || explanation.used_llm === true;
    if (usedLlm || data?.mode === "llm") return "LLM-generated explanation";
    return "Stub/development explanation";
  }, [data, explanation]);

  const firstBriefing = useMemo(() => {
    if (data?.briefing) return data.briefing;
    const parts = [
      `Situation summary: ${safeText(data?.situation_summary ?? explanation.summary)}`,
      `Risk level: ${formatRiskLevel(data?.risk_level ?? explanation.risk_level)}`,
      `Recommended action: ${safeText(data?.recommended_action ?? explanation.recommendation)}`,
      `Uncertainty/limitations: ${safeText(data?.uncertainty_limitations ?? explanation.uncertainty_note)}`,
      `Forecast evidence: ${safeText(data?.forecast_evidence ?? summary.evidence)}`,
      `System honesty: ${safeText(data?.system_honesty ?? explanation.system_honesty, modeLabel)}`
    ];
    return parts.join("\n\n");
  }, [data, explanation, summary, modeLabel]);

  useEffect(() => {
    if (!firstBriefing) return;
    setMessages([{ role: "assistant", content: firstBriefing }]);
  }, [firstBriefing]);

  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) return;
    thread.scrollTop = thread.scrollHeight;
  }, [messages]);

  const hasContext = Boolean(latestForecastBundle || data);
  const values = summary as Record<string, unknown>;

  const riskLevel = formatRiskLevel(data?.risk_level ?? explanation.risk_level ?? values.risk_level);
  const affectedAreaRaw = values.affected_area;
  const affectedCellsRaw = values.affected_cells_above_threshold;
  const hasNoPlumeSignal = Number(affectedAreaRaw ?? -1) === 0 || Number(affectedCellsRaw ?? -1) === 0 || safeText(explanation.summary, "").toLowerCase().includes("no meaningful plume");
  const plumeStatus = hasNoPlumeSignal ? "No meaningful plume above threshold" : (isPresentValue(affectedAreaRaw) || isPresentValue(affectedCellsRaw) ? "Plume detected above threshold" : "Unavailable");

  const snapshotRows = [
    ["Key finding", safeText(data?.situation_summary ?? explanation.summary, hasNoPlumeSignal ? "No meaningful plume is visible above the selected threshold." : "Forecast signals plume transport above the selected threshold.")],
    ["Recommended action", safeText(data?.recommended_action ?? explanation.recommendation, "Continue monitoring and review local response protocols.")],
    ["Main limitation", safeText(data?.uncertainty_limitations ?? explanation.uncertainty_note, "No major limitation reported.")]
  ] as Array<[string, string]>;

  const keyOutputRows = [
    ["Risk level", riskLevel],
    ["Plume status", plumeStatus],
    ["Affected area", formatArea(affectedAreaRaw)],
    ["Dominant spread direction", formatUnknown(values.dominant_spread_direction ?? values.wind_direction ?? values.direction)],
    ["Peak concentration", formatNumber(values.max_concentration)],
    ["Forecast time", formatTimestamp(data?.last_forecast_time ?? values.timestamp)]
  ] as Array<[string, string]>;

  const meteorologyLine = [
    isPresentValue(values.wind_speed) ? `Wind ${formatNumber(values.wind_speed)} m/s` : null,
    isPresentValue(values.wind_direction) ? `Dir ${formatUnknown(values.wind_direction)}` : null,
    isPresentValue(values.temperature) ? `Temp ${formatNumber(values.temperature)}°` : null
  ].filter(Boolean).join(" • ");
  const meteoStatus = meteorologyLine || "Meteorology unavailable or degraded/default values in use";
  const releaseLine = [
    isPresentValue(values.pollutant) ? `Pollutant ${formatUnknown(values.pollutant)}` : null,
    isPresentValue(values.emission_rate) ? `Emission ${formatNumber(values.emission_rate)}` : null,
    isPresentValue(values.release_height) ? `Height ${formatNumber(values.release_height)} m` : null
  ].filter(Boolean).join(" • ");
  const liveObsCount = sessionState?.observation_count;
  const liveObsLine = typeof liveObsCount === "number" && liveObsCount > 0
    ? `${formatNumber(liveObsCount, 0)} observations • latest ${formatTimestamp(session?.runtime_metadata?.last_observation_time)}`
    : "No live asset observations";
  const qualityItems = [
    liveObsLine === "No live asset observations" ? "No live asset observations" : null,
    meteorologyLine ? null : "Meteorology unavailable",
    data?.mode === "stub" || modeLabel.includes("Stub") ? "Explanation mode: stub/development" : null,
    typeof session?.model_name === "string" && session.model_name.includes("random_init") ? "Model mode indicates random initialization" : null,
    ...(Array.isArray(data?.limitations) ? data.limitations : [])
  ].filter((item, index, arr): item is string => Boolean(item) && arr.indexOf(item as string) === index);

  async function sendQuestion(question: string) {
    if (!question.trim() || !hasContext) return;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setChatQuestion("");
    try {
      const response = await httpPost<{ answer?: string }>("/decision-support/chat", { message: question });
      setMessages((prev) => [...prev, { role: "assistant", content: safeText(response.answer, "No answer available.") }]);
    } catch {
      const fallback = `Deterministic fallback (${modeLabel}): ${safeText(explanation.summary)}. Missing details remain unavailable in current context.`;
      setMessages((prev) => [...prev, { role: "assistant", content: fallback }]);
    }
  }

  return <AppShell title="Decision Support" subtitle="AI briefing and live forecast input visibility." metaItems={[{ label: `Explanation mode: ${modeLabel}` }]}>
    {error ? <section className="panel"><p>{error}</p></section> : null}
    <div className="decision-support-layout">
      <section className="panel decision-support-chat-panel polished-chat-panel">
        <header className="chat-panel-header">
          <h3>AI Decision Support</h3>
          <div className="detail-inline-meta">
            <span className="detail-chip">{modeLabel}</span>
            <span className="detail-chip">Forecast mode: {formatUnknown(data?.runtime_mode ?? session?.model_name)}</span>
            <span className="detail-chip">Last forecast: {formatTimestamp(data?.last_forecast_time)}</span>
          </div>
        </header>

        <div className="chat-thread polished-chat-thread" ref={threadRef}>
          {!hasContext ? <p className="chat-empty-state">No forecast context is available yet. Open the Map page or wait for the automatic forecast to complete.</p> : null}
          {messages.map((message, index) => <article key={`${message.role}-${index}`} className={`chat-message ${message.role}`}><p>{message.content}</p></article>)}
        </div>

        <div className="suggested-prompts">
          {SUGGESTED_PROMPTS.map((prompt) => <button key={prompt} type="button" className="chip-button" onClick={() => void sendQuestion(prompt)} disabled={!hasContext}>{prompt}</button>)}
        </div>

        <div className="chat-composer">
          <textarea value={chatQuestion} onChange={(event) => setChatQuestion(event.target.value)} onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void sendQuestion(chatQuestion);
            }
          }} placeholder="Ask a grounded question about this forecast" disabled={!hasContext} />
          <button className="primary-button" onClick={() => void sendQuestion(chatQuestion)} disabled={!hasContext || !chatQuestion.trim()}>Ask</button>
        </div>
      </section>

      <section className="panel decision-support-live-panel">
        <h3>Forecast Evidence</h3>
        <div className="values-section">
          <h4>Forecast Interpretation Snapshot</h4>
          <div className="values-grid compact-values-grid">
            {snapshotRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}
          </div>
        </div>

        <div className="values-section">
          <h4>Key Forecast Outputs</h4>
          <div className="values-grid compact-values-grid">{keyOutputRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}</div>
        </div>

        <div className="values-section">
          <h4>Input Drivers</h4>
          <div className="values-grid compact-values-grid">
            <div className="status-row"><strong>Meteorology</strong><span>{meteoStatus}</span></div>
            <div className="status-row"><strong>Release / Incident</strong><span>{releaseLine || "Forecast scenario input values are limited"}</span></div>
            <div className="status-row"><strong>Live observations</strong><span>{liveObsLine}</span></div>
          </div>
        </div>

        <div className="values-section">
          <h4>Data Gaps / Quality</h4>
          {qualityItems.length === 0 ? <p>No major input gaps reported.</p> : <div className="chip-list">{qualityItems.map((item) => <span key={item} className="detail-chip">{item}</span>)}</div>}
        </div>

        <details className="technical-details">
          <summary>Technical details</summary>
          <pre>{JSON.stringify({
            backend: data?.forecast_backend ?? session?.backend_name,
            session_id: session?.session_id,
            session_status: session?.status,
            state_version: sessionState?.state_version,
            observation_count: sessionState?.observation_count,
            timestamps: {
              last_update_time: sessionState?.last_update_time,
              last_ingest_time: session?.runtime_metadata?.last_ingest_time,
              last_observation_time: session?.runtime_metadata?.last_observation_time,
              last_prediction_time: data?.last_forecast_time ?? session?.runtime_metadata?.last_prediction_time
            },
            internal_state: sessionState,
            runtime_metadata: session?.runtime_metadata,
            capabilities: session?.capabilities,
            limitations: data?.limitations,
            raw_state_json: { explanation, summary }
          }, null, 2)}</pre>
        </details>
      </section>
    </div>
  </AppShell>;
}
