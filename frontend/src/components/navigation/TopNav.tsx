import type { ApiMode } from "../../features/forecast/types/forecast.types";

interface TopNavProps {
  apiMode: ApiMode;
  apiHealthy: boolean;
  modelLabel: string;
  scenarioName: string;
}

export function TopNav({ apiMode, apiHealthy, modelLabel, scenarioName }: TopNavProps) {
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