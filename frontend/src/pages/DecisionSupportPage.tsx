import { useMemo } from "react";
import { AppShell } from "../app/AppShell";
import { useSessionForecastView } from "../features/sessions/context/SessionForecastViewContext";

export function DecisionSupportPage() {
  const { latestForecastBundle } = useSessionForecastView();
  const explanation = (latestForecastBundle?.explanation ?? {}) as Record<string, unknown>;
  const summary = (latestForecastBundle?.summary ?? {}) as Record<string, unknown>;
  const risk = String(explanation.risk_level ?? "unavailable");
  const runtimeMode = String(explanation.used_llm ? "LLM-generated explanation" : "deterministic summary");
  const situation = String(explanation.summary ?? "No forecast explanation available yet.");
  const recommendation = String(explanation.recommendation ?? "Unavailable");
  const uncertainty = String(explanation.uncertainty_note ?? "Unavailable");
  const evidence = useMemo(() => `Max concentration: ${summary.max_concentration ?? "n/a"}; affected cells: ${summary.affected_cells_above_threshold ?? "n/a"}; direction: ${summary.dominant_spread_direction ?? "n/a"}.`, [summary]);

  return <AppShell title="Decision Support" subtitle="Operational interpretation of the latest forecast." metaItems={[{ label: `Explanation mode: ${runtimeMode}` }]}>
    <section className="panel"><h3>Situation summary</h3><p>{situation}</p></section>
    <section className="panel"><h3>Risk level</h3><p>{risk}</p></section>
    <section className="panel"><h3>Recommended action</h3><p>{recommendation}</p></section>
    <section className="panel"><h3>Uncertainty / limitations</h3><p>{uncertainty}</p></section>
    <section className="panel"><h3>Forecast evidence</h3><p>{evidence}</p></section>
    <section className="panel"><h3>System honesty</h3><p>LLM-ready context available. Explanations are deterministic unless LLM mode is explicitly enabled.</p></section>
  </AppShell>;
}
