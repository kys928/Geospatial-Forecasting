import type { ApiMode } from "../features/forecast/forecast.types";

interface TopBarProps {
  apiMode: ApiMode;
  apiHealthy: boolean;
  modelLabel: string;
  scenarioName: string;
}

export function TopBar({ apiMode, apiHealthy, modelLabel, scenarioName }: TopBarProps) {
  return (
    <header className="topbar">
      <div>
        <div className="eyebrow">Geospatial Forecasting</div>
        <h1>Forecast Operations Demo</h1>
      </div>

      <div className="topbar-meta">
        <span className="badge">{scenarioName}</span>
        <span className="badge">{modelLabel}</span>
        <span className={`badge ${apiHealthy ? "badge-ok" : "badge-error"}`}>
          {apiMode.toUpperCase()} · {apiHealthy ? "API OK" : "API DOWN"}
        </span>
      </div>
    </header>
  );
}