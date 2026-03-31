import type { ForecastSummary } from "../features/forecast/forecast.types";

interface SummaryCardsProps {
  summary: ForecastSummary | null;
}

export function SummaryCards({ summary }: SummaryCardsProps) {
  const cards = [
    {
      label: "Max concentration",
      value: summary ? summary.summary_statistics.max_concentration.toExponential(3) : "—"
    },
    {
      label: "Mean concentration",
      value: summary ? summary.summary_statistics.mean_concentration.toExponential(3) : "—"
    },
    {
      label: "Source",
      value: summary ? `${summary.source.latitude}, ${summary.source.longitude}` : "—"
    },
    {
      label: "Grid",
      value: summary ? `${summary.grid.rows} × ${summary.grid.columns}` : "—"
    },
    {
      label: "Timestamp",
      value: summary ? summary.timestamp : "—"
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