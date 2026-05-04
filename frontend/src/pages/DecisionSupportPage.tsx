import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
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

function formatTimestamp(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
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

  const hasContext = Boolean(latestForecastBundle || data);

  const liveStatusRows = useMemo(() => {
    const runtimeMeta = session?.runtime_metadata ?? {};
    return [
      ["Backend", formatUnknown(data?.forecast_backend ?? session?.backend_name ?? runtimeMeta.backend_name)],
      ["Session status", formatUnknown(session?.status ?? summary.session_status)],
      ["Observation count", formatUnknown(sessionState?.observation_count)],
      ["State version", formatUnknown(sessionState?.state_version)],
      ["Last update time", formatTimestamp(sessionState?.last_update_time)],
      ["Last ingest time", formatTimestamp(runtimeMeta.last_ingest_time)],
      ["Last observation time", formatTimestamp(runtimeMeta.last_observation_time)],
      ["Last prediction time", formatTimestamp(data?.last_forecast_time ?? runtimeMeta.last_prediction_time)],
      ["Freshness", formatUnknown(runtimeMeta.freshness_status ?? "Unavailable")],
      ["Limitations", formatUnknown(data?.limitations?.[0] ?? runtimeMeta.limitations)]
    ];
  }, [data, session, sessionState, summary]);

  const liveInputRows = useMemo(() => {
    if (data?.live_inputs && typeof data.live_inputs === "object") {
      return Object.entries(data.live_inputs).map(([name, value]) => ({ name, value: formatUnknown(value), unit: "", timestamp: "Unavailable", source: "Live input stream", status: "available" }));
    }
    const scenarioInputs = Object.entries(summary).slice(0, 8).map(([name, value]) => ({
      name,
      value: formatUnknown(value),
      unit: "",
      timestamp: formatTimestamp(data?.last_forecast_time),
      source: "Forecast scenario input",
      status: "scenario"
    }));
    return scenarioInputs;
  }, [data, summary]);

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
      <section className="panel decision-support-chat-panel">
        <h3>AI Decision Support</h3>
        <div className="detail-inline-meta">
          <span className="detail-chip">{modeLabel}</span>
          <span className="detail-chip">Forecast backend: {formatUnknown(data?.forecast_backend ?? session?.backend_name)}</span>
          <span className="detail-chip">Runtime mode: {formatUnknown(data?.runtime_mode ?? session?.model_name)}</span>
          <span className="detail-chip">Last forecast: {formatTimestamp(data?.last_forecast_time)}</span>
          <span className="detail-chip">Live input status: {data?.live_inputs ? "Available" : "Live asset observation unavailable"}</span>
        </div>
        {!hasContext ? <p>No forecast context is available yet. Open the Map page or wait for the automatic forecast to complete.</p> : null}
        <div className="briefing-card chat-thread">
          {messages.map((message, index) => <article key={`${message.role}-${index}`} className={`chat-message ${message.role}`}><p>{message.content}</p></article>)}
        </div>
        <div className="suggested-prompts">
          {SUGGESTED_PROMPTS.map((prompt) => <button key={prompt} type="button" className="chip-button" onClick={() => void sendQuestion(prompt)} disabled={!hasContext}>{prompt}</button>)}
        </div>
        <div className="chat-input-row">
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
        <h3>Live Input Flow</h3>
        <div className="status-grid">
          {liveStatusRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}
        </div>
        <h4>Current values</h4>
        {!data?.live_inputs ? <p>No live asset observations are currently available.</p> : null}
        <div className="live-values-table">
          <table>
            <thead><tr><th>Name</th><th>Value</th><th>Unit</th><th>Timestamp</th><th>Source</th><th>Status</th></tr></thead>
            <tbody>
              {liveInputRows.map((row) => <tr key={row.name}><td>{row.name}</td><td>{row.value}</td><td>{row.unit || "-"}</td><td>{row.timestamp}</td><td>{row.source}</td><td>{row.status}</td></tr>)}
            </tbody>
          </table>
        </div>
        <details className="technical-details">
          <summary>Technical details</summary>
          <pre>{JSON.stringify({ internal_state: sessionState, runtime_metadata: session?.runtime_metadata, capabilities: session?.capabilities, limitations: data?.limitations, raw_state_json: { explanation, summary } }, null, 2)}</pre>
        </details>
      </section>
    </div>
  </AppShell>;
}
