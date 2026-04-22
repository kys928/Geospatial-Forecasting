export function RoleSwitcher() {
  return (
    <label className="role-switcher">
      <span className="eyebrow">Role</span>
      <select defaultValue="Operator">
        <option>Operator</option>
        <option>Technical Integrator</option>
        <option>ML Operator</option>
      </select>
    </label>
  );
}
