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
function getNestedValue(source: unknown, ...paths: string[]): unknown {
  for (const path of paths) {
    const parts = path.split(".");
    let current: unknown = source;
    let found = true;
    for (const part of parts) {
      if (current && typeof current === "object" && part in (current as Record<string, unknown>)) {
        current = (current as Record<string, unknown>)[part];
      } else {
        found = false;
        break;
      }
    }
    if (found && isPresentValue(current)) return current;
  }
  return undefined;
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
function formatCoordinate(value: unknown): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  return parsed.toFixed(5);
}
function formatSpeed(value: unknown): string {
  const n = formatNumber(value);
  return n === "Unavailable" ? n : `${n} m/s`;
}
function formatDirection(value: unknown): string {
  if (!isPresentValue(value)) return "Unavailable";
  if (typeof value === "number") return `${value}°`;
  return String(value);
}
function formatTemperature(value: unknown): string {
  const n = formatNumber(value, 1);
  return n === "Unavailable" ? n : `${n} °C`;
}
function formatPressure(value: unknown): string {
  const n = formatNumber(value, 1);
  return n === "Unavailable" ? n : `${n} hPa`;
}
function formatPercent(value: unknown): string {
  const n = formatNumber(value, 1);
  return n === "Unavailable" ? n : `${n}%`;
}
function formatDurationMinutes(value: unknown): string {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || Number.isNaN(parsed)) return "Unavailable";
  return `${formatNumber(parsed, 0)} min`;
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

function hasMeaningfulPlume(params: {
  affectedAreaM2: unknown;
  affectedCellsAboveThreshold: unknown;
  maxConcentration: unknown;
  explanationSummary: unknown;
  riskLevel: string;
}): boolean {
  const toNumber = (value: unknown): number | null => {
    const parsed = typeof value === "string" ? Number(value) : value;
    return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : null;
  };
  const affectedArea = toNumber(params.affectedAreaM2);
  const affectedCells = toNumber(params.affectedCellsAboveThreshold);
  const maxConcentration = toNumber(params.maxConcentration);
  const explanationText = safeText(params.explanationSummary, "").toLowerCase();

  if ((affectedArea != null && affectedArea > 0) || (affectedCells != null && affectedCells > 0) || (maxConcentration != null && maxConcentration > 0)) return true;
  if (affectedArea == 0 || affectedCells == 0 || maxConcentration == 0 || explanationText.includes("no meaningful plume")) return false;
  return params.riskLevel.toLowerCase() !== "low";
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
    return parts.join("\n\n").replace(/^Grounded response:\s*/i, "");
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
  const forecastEvidence = getNestedValue(data, "forecast_evidence") as Record<string, unknown> | undefined;
  const affectedAreaM2 = getNestedValue(summary, "affected_area_m2", "affected_area", "summary_statistics.affected_area_m2", "summary_statistics.affected_area") ?? getNestedValue(forecastEvidence, "affected_area_m2", "affected_area");
  const affectedCellsRaw = getNestedValue(summary, "affected_cells_above_threshold", "summary_statistics.affected_cells_above_threshold") ?? getNestedValue(forecastEvidence, "affected_cells_above_threshold");
  const maxConcentration = getNestedValue(summary, "max_concentration", "summary_statistics.max_concentration") ?? getNestedValue(forecastEvidence, "max_concentration");
  const meanConcentration = getNestedValue(summary, "mean_concentration", "summary_statistics.mean_concentration") ?? getNestedValue(forecastEvidence, "mean_concentration");
  const dominantSpreadDirection = getNestedValue(summary, "dominant_spread_direction", "summary_statistics.dominant_spread_direction", "wind_direction", "direction") ?? getNestedValue(forecastEvidence, "dominant_spread_direction");
  const thresholdUsed = getNestedValue(summary, "threshold_used", "threshold", "summary_statistics.threshold_used") ?? getNestedValue(forecastEvidence, "threshold_used");
  const forecastTime = getNestedValue(data, "last_forecast_time") ?? getNestedValue(summary, "timestamp", "issued_at");
  const plumePresent = hasMeaningfulPlume({ affectedAreaM2, affectedCellsAboveThreshold: affectedCellsRaw, maxConcentration, explanationSummary: explanation.summary, riskLevel });
  const hasThreatSignal = safeText(data?.situation_summary ?? explanation.summary, "").toLowerCase().includes("threat");
  const explicitNoPlumeSignal = Number(affectedAreaM2) === 0 || Number(affectedCellsRaw) === 0 || Number(maxConcentration) === 0 || safeText(explanation.summary, "").toLowerCase().includes("no meaningful plume");
  const plumeStatus = plumePresent ? (hasThreatSignal ? "Threat detected" : "Plume detected above threshold") : (explicitNoPlumeSignal ? "No meaningful plume above threshold" : "Unavailable");

  const adapterMeta = getNestedValue(sessionState, "last_input_adapter_metadata", "input_adapter_metadata", "internal_state.last_input_adapter_metadata");
  const windSpeedValue = getNestedValue(summary, "wind_speed", "meteorology.wind_speed", "met.wind_speed", "inputs.wind_speed", "weather.wind_speed", "u10_speed", "wind.speed", "meteo.wind_speed", "source_inputs.wind_speed", "request.wind_speed", "runtime_metadata.wind_speed");
  const windDirectionValue = getNestedValue(summary, "wind_direction", "meteorology.wind_direction", "met.wind_direction", "inputs.wind_direction", "weather.wind_direction", "wind.direction", "meteo.wind_direction");
  const uWindValue = getNestedValue(summary, "u_wind", "u", "meteorology.u_wind", "met.u_wind", "wind.u");
  const vWindValue = getNestedValue(summary, "v_wind", "v", "meteorology.v_wind", "met.v_wind", "wind.v");
  const meteorologyRows = [
    ["Wind speed", formatSpeed(windSpeedValue)],
    ["Wind direction", formatDirection(windDirectionValue)],
    ["U wind", formatSpeed(uWindValue)],
    ["V wind", formatSpeed(vWindValue)],
    ["Temperature", formatTemperature(getNestedValue(summary, "temperature", "meteorology.temperature", "met.temperature", "weather.temperature"))],
    ["Relative humidity", formatPercent(getNestedValue(summary, "relative_humidity", "humidity", "meteorology.relative_humidity", "met.relative_humidity", "weather.humidity"))],
    ["Surface pressure", formatPressure(getNestedValue(summary, "surface_pressure", "pressure", "meteorology.surface_pressure", "met.surface_pressure", "weather.pressure"))],
    ["PBL height", `${formatNumber(getNestedValue(summary, "pbl_height", "meteorology.pbl_height", "met.pbl_height", "weather.pbl_height"), 1)} m`],
    ["Meteorology timestamp", formatTimestamp(getNestedValue(summary, "meteorology.timestamp", "met.timestamp", "weather.timestamp", "timestamp"))],
    ["Meteorology source", formatUnknown(getNestedValue(summary, "meteorology.source", "met.source", "weather.source", "source", "adapter.source", "runtime_metadata.meteorology_source"))]
  ] as Array<[string, string]>;

  const windSpeed = formatSpeed(windSpeedValue);
  const windDirection = formatDirection(windDirectionValue);
  const uWind = formatSpeed(uWindValue);
  const vWind = formatSpeed(vWindValue);
  const windSummary = windSpeed !== "Unavailable" && windDirection !== "Unavailable"
    ? `${windSpeed} ${windDirection}`
    : (uWind !== "Unavailable" || vWind !== "Unavailable"
      ? `U ${uWind}, V ${vWind}`.replace("U Unavailable, ", "").replace(", V Unavailable", "")
      : "Unavailable");

  const weatherCompactRows = [
    ["Wind", windSummary],
    ["Temperature", formatTemperature(getNestedValue(summary, "temperature", "meteorology.temperature", "met.temperature", "weather.temperature"))],
    ["Humidity", formatPercent(getNestedValue(summary, "relative_humidity", "humidity", "meteorology.relative_humidity", "met.relative_humidity", "weather.humidity"))],
    ["Pressure / PBL", (() => {
      const pressure = formatPressure(getNestedValue(summary, "surface_pressure", "pressure", "meteorology.surface_pressure", "met.surface_pressure", "weather.pressure"));
      const pbl = `${formatNumber(getNestedValue(summary, "pbl_height", "meteorology.pbl_height", "met.pbl_height", "weather.pbl_height"), 1)} m`;
      if (pressure === "Unavailable" && pbl === "Unavailable m") return "Unavailable";
      if (pressure !== "Unavailable" && pbl !== "Unavailable m") return `${pressure} / ${pbl}`;
      return pressure !== "Unavailable" ? pressure : pbl;
    })()]
  ] as Array<[string, string]>;

  const keyOutputRows = [
    ...(
    plumePresent
      ? [["Status", plumeStatus], ["Risk", riskLevel], ["Affected area", formatArea(affectedAreaM2)], ["Peak concentration", formatNumber(maxConcentration)], ["Direction", formatDirection(dominantSpreadDirection)], ["Forecast time", formatTimestamp(forecastTime)]]
      : [["Status", plumeStatus], ["Risk", riskLevel], ["Forecast time", formatTimestamp(forecastTime)]]
  )] as Array<[string, string]>;

  const meteorologyConnected = meteorologyRows.some(([_, value]) => value !== "Unavailable");
  const sourceLatitude = formatCoordinate(getNestedValue(summary, "source.latitude", "source_latitude", "source.lat", "release.latitude", "scenario.source_latitude", "request.source_latitude"));
  const sourceLongitude = formatCoordinate(getNestedValue(summary, "source.longitude", "source_longitude", "source.lon", "release.longitude", "scenario.source_longitude", "request.source_longitude"));
  const pollutant = formatUnknown(getNestedValue(summary, "pollutant", "pollutant_type", "release.pollutant", "scenario.pollutant"));
  const emissionRate = formatUnknown(getNestedValue(summary, "emission_rate", "release.emission_rate", "scenario.emission_rate"));
  const releaseRows = [
    ["Source latitude", formatCoordinate(getNestedValue(summary, "source.latitude", "source_latitude", "source.lat", "release.latitude", "scenario.source_latitude", "request.source_latitude"))],
    ["Source longitude", formatCoordinate(getNestedValue(summary, "source.longitude", "source_longitude", "source.lon", "release.longitude", "scenario.source_longitude", "request.source_longitude"))],
    ["Pollutant type", formatUnknown(getNestedValue(summary, "pollutant", "pollutant_type", "release.pollutant", "scenario.pollutant"))],
    ["Emission rate", formatUnknown(getNestedValue(summary, "emission_rate", "release.emission_rate", "scenario.emission_rate"))],
    ["Release height", `${formatNumber(getNestedValue(summary, "release_height", "release.height", "scenario.release_height"))} m`],
    ["Duration", formatUnknown(getNestedValue(summary, "duration", "release.duration", "scenario.duration"))],
    ["Start time", formatTimestamp(getNestedValue(summary, "start_time", "release.start_time", "scenario.start_time"))],
    ["End time", formatTimestamp(getNestedValue(summary, "end_time", "release.end_time", "scenario.end_time"))],
    ["Forecast horizon", formatUnknown(getNestedValue(summary, "forecast_horizon", "horizon", "scenario.forecast_horizon"))]
  ] as Array<[string, string]>;
  const releaseCompactRows = [
    ["Source location", sourceLatitude !== "Unavailable" && sourceLongitude !== "Unavailable" ? `${sourceLatitude}, ${sourceLongitude}` : "Unavailable"],
    ["Pollutant / emission", pollutant !== "Unavailable" && emissionRate !== "Unavailable" ? `${pollutant}, ${emissionRate}` : (pollutant !== "Unavailable" ? pollutant : emissionRate)],
    ["Release height", `${formatNumber(getNestedValue(summary, "release_height", "release.height", "scenario.release_height"))} m`],
    ["Duration / window", (() => {
      const duration = formatDurationMinutes(getNestedValue(summary, "duration", "release.duration", "scenario.duration"));
      const start = formatTimestamp(getNestedValue(summary, "start_time", "release.start_time", "scenario.start_time"));
      const end = formatTimestamp(getNestedValue(summary, "end_time", "release.end_time", "scenario.end_time"));
      if (duration !== "Unavailable") return duration;
      if (start !== "Unavailable" && end !== "Unavailable") return `${start} → ${end}`;
      return start !== "Unavailable" ? `Starts ${start}` : (end !== "Unavailable" ? `Ends ${end}` : "Unavailable");
    })()]
  ] as Array<[string, string]>;
  const liveObsCount = sessionState?.observation_count;
  const liveObsLine = typeof liveObsCount === "number" && liveObsCount > 0
    ? `${formatNumber(liveObsCount, 0)} observations • latest ${formatTimestamp(session?.runtime_metadata?.last_observation_time)}`
    : "No live asset observations";
  async function sendQuestion(question: string) {
    if (!question.trim() || !hasContext) return;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setChatQuestion("");
    try {
      const response = await httpPost<{ answer?: string }>("/decision-support/chat", { message: question });
      setMessages((prev) => [...prev, { role: "assistant", content: safeText(response.answer, "No answer available.") }]);
    } catch {
      const fallback = `${safeText(explanation.summary)} Missing details remain unavailable in current context.`;
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
        <h3>Geospatial Conditions</h3>
        <div className="values-section">
          <h4>Weather / Dispersion</h4>
          <div className="values-grid compact-values-grid">
            {weatherCompactRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}
          </div>
          {!meteorologyConnected ? <p className="subtle-note">Live meteorology is not connected yet.</p> : null}
        </div>

        <div className="values-section">
          <h4>Release Source</h4>
          <p className="subtle-note">Forecast scenario inputs</p>
          <div className="values-grid compact-values-grid">{releaseCompactRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}</div>
        </div>

        <div className="values-section">
          <h4>Plume / Threat</h4>
          <div className="values-grid compact-values-grid">{keyOutputRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}</div>
        </div>

        <div className="values-section">
          <h4>Live Observations</h4>
          <div className="values-grid compact-values-grid"><div className="status-row"><strong>Status</strong><span>{liveObsLine === "No live asset observations" ? "No live asset observations connected." : "Connected"}</span></div>{liveObsLine !== "No live asset observations" ? <div className="status-row"><strong>Latest observation</strong><span>{liveObsLine}</span></div> : null}</div>
        </div>

        <details className="technical-details">
          <summary>More geospatial values</summary>
          <div className="values-grid compact-values-grid">
            {[...meteorologyRows, ...releaseRows, ["Affected area", formatArea(affectedAreaM2)], ["Affected area (ha)", formatNumber(getNestedValue(summary, "affected_area_hectares", "summary_statistics.affected_area_hectares") ?? getNestedValue(forecastEvidence, "affected_area_hectares"), 3)], ["Affected cells", formatNumber(affectedCellsRaw, 0)], ["Peak concentration", formatNumber(maxConcentration)], ["Mean concentration", formatNumber(meanConcentration)], ["Dominant spread direction", formatDirection(dominantSpreadDirection)], ["Threshold used", formatUnknown(thresholdUsed)], ["Grid size", formatUnknown(getNestedValue(summary, "grid_size", "grid_shape", "grid.rows", "grid.columns"))], ["Forecast horizon", formatUnknown(getNestedValue(summary, "forecast_horizon", "horizon", "scenario.forecast_horizon"))], ["Forecast time", formatTimestamp(forecastTime)]].map(([label, value]) => <div key={`more-${label}`} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}
          </div>
        </details>

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
