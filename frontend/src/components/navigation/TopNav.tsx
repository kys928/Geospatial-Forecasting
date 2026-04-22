import type { ApiMode } from "../../features/forecast/types/forecast.types";

interface TopNavProps {
  apiMode: ApiMode;
  apiHealthy?: boolean;
  modelLabel: string;
  scenarioName: string;
}

export function TopNav({ apiMode, apiHealthy, modelLabel, scenarioName }: TopNavProps) {
  const healthLabel = apiHealthy === undefined ? "API UNKNOWN" : apiHealthy ? "API OK" : "API DOWN";

  return (
    <header className="topbar">
      <div>
        <div className="eyebrow">Geospatial Forecasting</div>
        <h1>Forecast Operations Demo</h1>
      </div>

      <div className="topbar-meta">
        <span className="badge">{scenarioName}</span>
        <span className="badge">{modelLabel}</span>
        <span className={`badge ${apiHealthy === false ? "badge-error" : apiHealthy === true ? "badge-ok" : ""}`}>
          {apiMode.toUpperCase()} · {healthLabel}
        </span>
      </div>
    </header>
  );
}
