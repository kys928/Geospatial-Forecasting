import type {
  DemoScenario,
  ForecastExplanation,
  ForecastSummary
} from "../features/forecast/forecast.types";

interface SummaryCardsProps {
  summary: ForecastSummary | null;
  explanationPayload: ForecastExplanation | null;
  activeScenario: DemoScenario | null;
}

function formatPeak(value: number | null): string {
  if (value == null) {
    return "—";
  }
  return value.toExponential(3);
}

function formatAffectedCells(value: number | null): string {
  if (value == null) {
    return "—";
  }
  return `${value} cells`;
}

function formatRisk(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatDirection(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  return value;
}

export function SummaryCards({
  summary,
  explanationPayload,
  activeScenario
}: SummaryCardsProps) {
  const cards = [
    {
      label: "Scenario",
      value: activeScenario?.label ?? "—"
    },
    {
      label: "Peak concentration",
      value: summary ? formatPeak(summary.summary_statistics.max_concentration) : "—"
    },
    {
      label: "Dispersion footprint",
      value: explanationPayload
        ? formatAffectedCells(explanationPayload.summary.affected_cells_above_threshold)
        : "—"
    },
    {
      label: "Spread direction",
      value: explanationPayload
        ? formatDirection(explanationPayload.summary.dominant_spread_direction)
        : "—"
    },
    {
      label: "Risk",
      value: explanationPayload
        ? formatRisk(explanationPayload.explanation.risk_level)
        : "—"
    }
  ];

  return (
    <section className="summary-cards">
      {cards.map((card) => (
        <article key={card.label} className="summary-card panel">
          <div className="summary-label">{card.label}</div>
          <div className="summary-value">{card.value}</div>
        </article>
      ))}
    </section>
  );
}