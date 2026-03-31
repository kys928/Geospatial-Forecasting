export function ScenarioControls() {
  return (
    <section className="control-group">
      <h3>Scenario</h3>

      <label className="field">
        <span>Preset</span>
        <select defaultValue="default">
          <option value="default">Default baseline</option>
          <option value="urban">Urban release</option>
          <option value="industrial">Industrial release</option>
        </select>
      </label>

      <label className="field">
        <span>Threshold</span>
        <select defaultValue="1e-5">
          <option value="1e-6">1e-6</option>
          <option value="1e-5">1e-5</option>
          <option value="1e-4">1e-4</option>
        </select>
      </label>
    </section>
  );
}