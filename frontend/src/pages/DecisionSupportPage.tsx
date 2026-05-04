import { useEffect, useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { httpGet, httpPost } from "../services/api/http";

type DecisionSupportLatest = {
  mode: "stub" | "llm";
  briefing?: string;
  situation_summary?: string;
  risk_level?: string;
  recommended_action?: string;
  uncertainty_limitations?: string;
  forecast_evidence?: unknown;
  system_honesty?: string;
  limitations?: string[];
  live_inputs?: Record<string, unknown>;
};

export function DecisionSupportPage() {
  const [data, setData] = useState<DecisionSupportLatest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatAnswer, setChatAnswer] = useState<string>("");

  useEffect(() => {
    httpGet<DecisionSupportLatest>("/decision-support/latest").then(setData).catch((e) => setError(e instanceof Error ? e.message : "Unavailable"));
  }, []);

  const evidenceText = useMemo(() => {
    if (data?.forecast_evidence == null) return "Unavailable";
    if (typeof data.forecast_evidence === "string") return data.forecast_evidence;
    return JSON.stringify(data.forecast_evidence, null, 2);
  }, [data]);

  const liveInputRows = useMemo(() => {
    if (!data?.live_inputs || typeof data.live_inputs !== "object") return [];
    return Object.entries(data.live_inputs).slice(0, 12);
  }, [data]);

  return <AppShell title="Decision Support" subtitle="Operational interpretation of the latest forecast." metaItems={[{ label: `Explanation mode: ${data?.mode ?? "unavailable"}` }]}>
    <section className="panel"><h3>Operational briefing</h3><p>{data?.briefing ?? "No forecast explanation available yet."}</p></section>
    {error ? <section className="panel"><p>{error}</p></section> : null}
    <div className="workspace-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
      <section className="panel"><h3>Situation summary</h3><p>{data?.situation_summary ?? "Unavailable"}</p></section>
      <section className="panel"><h3>Risk level</h3><p>{data?.risk_level ?? "unknown"}</p></section>
      <section className="panel"><h3>Recommended action</h3><p>{data?.recommended_action ?? "Unavailable"}</p></section>
      <section className="panel"><h3>Uncertainty / limitations</h3><p>{data?.uncertainty_limitations ?? "Unavailable"}</p></section>
      <section className="panel"><h3>Forecast evidence</h3><pre style={{ whiteSpace: "pre-wrap", overflow: "auto", maxHeight: 200 }}>{evidenceText}</pre></section>
      <section className="panel"><h3>System honesty</h3><p>{data?.system_honesty ?? "Unavailable"}</p></section>
    </div>
    <section className="panel"><h3>Ask About This Forecast</h3>
      <div style={{ display: "grid", gap: 8 }}>
        <input value={chatQuestion} onChange={(e) => setChatQuestion(e.target.value)} placeholder="Ask a grounded question about the current forecast" />
        <button className="primary-button" onClick={async () => {
          try {
            const response = await httpPost<{ answer?: string }>("/decision-support/chat", { message: chatQuestion });
            setChatAnswer(response.answer ?? "No answer available.");
          } catch (e) { setChatAnswer(e instanceof Error ? e.message : "Chat unavailable"); }
        }}>Ask</button>
        <p>{chatAnswer || "No answer yet."}</p>
      </div>
    </section>
    <section className="panel"><h3>Live Input Values</h3>
      {liveInputRows.length === 0 ? <p>No live input values available.</p> : <div style={{ overflow: "auto" }}><table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>{liveInputRows.map(([k,v]) => <tr key={k}><td>{k}</td><td>{typeof v === "object" ? JSON.stringify(v) : String(v)}</td></tr>)}</tbody></table></div>}
    </section>
  </AppShell>;
}
