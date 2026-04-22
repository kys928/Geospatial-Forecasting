export function RoleSwitcher() {
  return (
    <label className="role-switcher">
      <span className="eyebrow">View role (display only)</span>
      <select defaultValue="Operator" aria-label="Display role selector">
        <option>Operator</option>
        <option>Technical Integrator</option>
        <option>ML Operator</option>
      </select>
    </label>
  );
}
