import type { ScenarioPreset, ThresholdPreset } from "../forecast/forecast.types";

interface ScenarioControlsProps {
  scenario: ScenarioPreset;
  threshold: ThresholdPreset;
  onScenarioChange: (value: ScenarioPreset) => void;
  onThresholdChange: (value: ThresholdPreset) => void;
}

export function ScenarioControls({
  scenario,
  threshold,
  onScenarioChange,
  onThresholdChange
}: ScenarioControlsProps) {
  return (
    <section className="control-group">
      <h3>Scenario</h3>

      <label className="field">
        <span>Preset</span>
        <select
          value={scenario}
          onChange={(event) => onScenarioChange(event.target.value as ScenarioPreset)}
        >
          <option value="default">Default baseline</option>
          <option value="urban">Urban release</option>
          <option value="industrial">Industrial release</option>
        </select>
      </label>

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
    </section>
  );
}