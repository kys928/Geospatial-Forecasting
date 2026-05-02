import { NavLink } from "react-router-dom";

const tabs = [
  { label: "Map", to: "/forecast" },
  { label: "Sessions", to: "/sessions" },
  { label: "Ops", to: "/ops" }
];

export function WorkspaceTabs() {
  return (
    <nav className="workspace-tabs" aria-label="Workspaces">
      {tabs.map((tab) => (
        <NavLink
          key={tab.to}
          to={tab.to}
          className={({ isActive }) => `workspace-tab ${isActive ? "workspace-tab-active" : ""}`}
        >
          {tab.label}
        </NavLink>
      ))}
    </nav>
  );
}
