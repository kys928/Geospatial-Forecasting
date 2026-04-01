import type { DemoScenario } from "../forecast/forecast.types";

interface ScenarioControlsProps {
  activeScenario: DemoScenario;
}

export function ScenarioControls({ activeScenario }: ScenarioControlsProps) {
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
        Run Forecast samples a new bounded scenario each time.
      </p>
    </section>
  );
}