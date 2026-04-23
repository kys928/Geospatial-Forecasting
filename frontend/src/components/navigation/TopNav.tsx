import type { ApiMode } from "../../features/forecast/types/forecast.types";

interface TopNavMetaItem {
  label: string;
  tone?: "default" | "ok" | "error";
}

interface TopNavProps {
  title: string;
  subtitle?: string;
  apiMode?: ApiMode;
  apiHealthy?: boolean;
  metaItems?: TopNavMetaItem[];
}

export function TopNav({ title, subtitle, apiMode, apiHealthy, metaItems = [] }: TopNavProps) {
  const hasApiStatus = apiMode !== undefined;
  const healthLabel = apiHealthy === undefined ? "API unknown" : apiHealthy ? "API ok" : "API down";
  const hasContext = metaItems.length > 0 || hasApiStatus;

  return (
    <header className="topbar panel">
      <div className="topbar-main">
        <div className="eyebrow">Geospatial Forecasting</div>
        <h1>{title}</h1>
        {subtitle ? <p className="topbar-subtitle muted">{subtitle}</p> : null}
      </div>

      {hasContext ? (
        <div className="topbar-meta">
          {metaItems.map((item) => (
            <span
              key={`${item.label}-${item.tone ?? "default"}`}
              className={`badge ${item.tone === "ok" ? "badge-ok" : item.tone === "error" ? "badge-error" : ""}`}
            >
              {item.label}
            </span>
          ))}
          {hasApiStatus ? (
            <span className={`badge ${apiHealthy === false ? "badge-error" : apiHealthy === true ? "badge-ok" : ""}`}>
              {apiMode.toUpperCase()} · {healthLabel}
            </span>
          ) : null}
        </div>
      ) : null}
    </header>
  );
}
