import { NavLink } from "react-router-dom";

const tabs = [
  { label: "Forecast", to: "/forecast" },
  { label: "Sessions", to: "/sessions" },
  { label: "Ops", to: "/ops" },
  { label: "Registry", to: "/registry" },
  { label: "Events", to: "/events" }
];

export function WorkspaceTabs() {
  return (
    <nav className="workspace-tabs panel" aria-label="Workspaces">
      {tabs.map((tab) => (
        <NavLink key={tab.to} to={tab.to} className="badge">
          {tab.label}
        </NavLink>
      ))}
    </nav>
  );
}
