import type { ReactNode } from "react";

interface SidebarProps {
  children?: ReactNode;
  onRunForecast: () => void;
}

export function Sidebar({ children, onRunForecast }: SidebarProps) {
  return (
    <aside className="sidebar panel">
      <div className="panel-header">
        <h2>Controls</h2>
      </div>

      <div className="panel-body">
        {children}
      </div>

      <div className="sidebar-actions">
        <button className="primary-button" onClick={onRunForecast}>
          Run Forecast
        </button>
      </div>
    </aside>
  );
}