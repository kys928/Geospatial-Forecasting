import type { ReactNode } from "react";

interface ForecastSidebarProps {
  children?: ReactNode;
  onRunForecast: () => void;
}

export function ForecastSidebar({ children, onRunForecast }: ForecastSidebarProps) {
  return (
    <aside className="sidebar panel">
      <div className="panel-header">
        <h2>Controls</h2>
      </div>

      <div className="panel-body sidebar-body">
        {children}
      </div>

      <div className="sidebar-actions">
        <button className="primary-button" onClick={onRunForecast}>
          Run demo scenario
        </button>
      </div>
    </aside>
  );
}
