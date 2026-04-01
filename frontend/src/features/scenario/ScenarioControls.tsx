import type { DemoScenario, ThresholdPreset } from "../forecast/forecast.types";

interface ScenarioControlsProps {
  activeScenario: DemoScenario;
  threshold: ThresholdPreset;
  onThresholdChange: (value: ThresholdPreset) => void;
}

export function ScenarioControls({
  activeScenario,
  threshold,
  onThresholdChange
}: ScenarioControlsProps) {
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

      <label className="field">
        <span>Threshold</span>
        <select
          value={threshold}
          onChange={(event) => onThresholdChange(event.target.value as ThresholdPreset)}
        >
          <option value="1e-6">1e-6</option>
          <option value="1e-5">1e-5</option>
          <option value="1e-4">1e-4</option>
        </select>
      </label>

      <p className="control-note muted">
        Run Forecast samples a new bounded scenario each time.
      </p>
    </section>
  );
}