import type { DemoScenario } from "../types/forecast.types";

interface ForecastScenarioSummaryProps {
  activeScenario: DemoScenario;
}

export function ForecastScenarioSummary({ activeScenario }: ForecastScenarioSummaryProps) {
  return (
    <section className="control-group">
      <div className="scenario-status">
        <div className="scenario-status-row">
          <h3>Scenario</h3>
          {activeScenario.severity ? (
            <span className={`severity-badge severity-${activeScenario.severity}`}>
              {activeScenario.severity}
            </span>
          ) : null}
        </div>

        <p className="scenario-inline-label">{activeScenario.label}</p>

        {activeScenario.notes ? (
          <p className="muted scenario-inline-note">{activeScenario.notes}</p>
        ) : null}
      </div>

      <p className="control-note muted">
        Run demo scenario samples a new bounded sandbox scenario each time.
      </p>
    </section>
  );
}
