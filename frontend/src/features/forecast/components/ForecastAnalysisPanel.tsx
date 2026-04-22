import type {
  ForecastExplanation,
  SelectedFeatureState
} from "../types/forecast.types";

interface ForecastAnalysisPanelProps {
  selected: SelectedFeatureState | null;
  explanationPayload: ForecastExplanation | null;
  explanationSource?: string;
}

function getFeatureTitle(selected: SelectedFeatureState): string {
  const kind = selected.properties?.kind;

  switch (kind) {
    case "source":
      return "Emission source";
    case "forecast_extent":
      return "Forecast domain";
    case "plume_band_low":
      return "Low plume band";
    case "plume_band_medium":
      return "Medium plume band";
    case "plume_band_high":
      return "High plume band";
    default:
      return selected.title || "Feature details";
  }
}

function getFeatureDescription(selected: SelectedFeatureState): string | null {
  const kind = selected.properties?.kind;

  switch (kind) {
    case "source":
      return "Origin point of the simulated release.";
    case "forecast_extent":
      return "Spatial domain used for the current forecast grid.";
    case "plume_band_low":
      return "Outer edge of the plume where concentration is weakest.";
    case "plume_band_medium":
      return "Middle plume zone where concentration is more noticeable.";
    case "plume_band_high":
      return "Core plume zone where concentration is strongest.";
    default:
      return null;
  }
}

function formatExplanationSource(explanationSource?: string): string {
  if (!explanationSource) {
    return "unknown";
  }

  switch (explanationSource) {
    case "llm":
      return "llm";
    case "fallback":
      return "fallback";
    default:
      return explanationSource;
  }
}

function getThresholdLabel(threshold: unknown): string {
  const value = typeof threshold === "number" ? threshold : Number(threshold);

  if (!Number.isFinite(value)) {
    return "Unknown detection level";
  }

  if (value <= 1e-6) {
    return "Very low concentration band";
  }

  if (value <= 1e-5) {
    return "Moderate concentration band";
  }

  return "High concentration band";
}

function getStrengthLabel(maxValue: unknown): string {
  const value = typeof maxValue === "number" ? maxValue : Number(maxValue);

  if (!Number.isFinite(value)) {
    return "Unknown plume strength";
  }

  if (value >= 1e-3) {
    return "Strong inner plume";
  }

  if (value >= 1e-4) {
    return "Moderately strong plume";
  }

  if (value >= 1e-5) {
    return "Noticeable outer plume";
  }

  return "Faint plume";
}

function formatArea(areaM2: number | null | undefined, areaHectares: number | null | undefined): string {
  if (typeof areaHectares === "number" && Number.isFinite(areaHectares) && areaHectares >= 1) {
    return `About ${areaHectares.toFixed(2)} hectares`;
  }

  if (typeof areaM2 === "number" && Number.isFinite(areaM2)) {
    return `About ${Math.round(areaM2).toLocaleString()} m²`;
  }

  return "Unknown affected area";
}

function formatThresholdRaw(threshold: unknown): string {
  const value = typeof threshold === "number" ? threshold : Number(threshold);
  return Number.isFinite(value) ? value.toExponential(1) : "—";
}

function formatMaxRaw(maxValue: unknown): string {
  const value = typeof maxValue === "number" ? maxValue : Number(maxValue);
  return Number.isFinite(value) ? value.toExponential(3) : "—";
}

function getCoverageText(
  selected: SelectedFeatureState,
  explanationPayload: ForecastExplanation | null
): string {
  const kind = selected.properties?.kind;
  const summary = explanationPayload?.summary;

  if (!summary) {
    const cellCount = selected.properties?.cell_count;
    const count = typeof cellCount === "number" ? cellCount : Number(cellCount);

    if (!Number.isFinite(count)) {
      return "Unknown footprint";
    }

    if (count <= 25) return "Very small local footprint";
    if (count <= 100) return "Small local footprint";
    if (count <= 250) return "Moderate footprint";
    if (count <= 500) return "Large footprint";
    return "Very large footprint";
  }

  if (kind === "plume_band_high") {
    return `${formatArea(summary.affected_area_m2, summary.affected_area_hectares)} in the strongest plume core`;
  }

  if (kind === "plume_band_medium") {
    return `${formatArea(summary.affected_area_m2, summary.affected_area_hectares)} across the main affected zone`;
  }

  if (kind === "plume_band_low") {
    return `${formatArea(summary.affected_area_m2, summary.affected_area_hectares)} including the weaker outer edge`;
  }

  return formatArea(summary.affected_area_m2, summary.affected_area_hectares);
}

function getDetailRows(
  selected: SelectedFeatureState,
  explanationPayload: ForecastExplanation | null
): Array<{ label: string; value: string }> {
  const properties = selected.properties ?? {};
  const kind = properties.kind;

  if (kind === "source") {
    const emissionsRate = properties.emissions_rate;
    return [
      {
        label: "Type",
        value: "Release origin"
      },
      {
        label: "Emission level",
        value:
          typeof emissionsRate === "number" || typeof emissionsRate === "string"
            ? `${emissionsRate} units`
            : "Unknown"
      }
    ];
  }

  if (
    kind === "plume_band_low" ||
    kind === "plume_band_medium" ||
    kind === "plume_band_high"
  ) {
    const threshold = properties.threshold;
    const maxValue = properties.max_value;

    return [
      {
        label: "Detection level",
        value: getThresholdLabel(threshold)
      },
      {
        label: "Coverage",
        value: getCoverageText(selected, explanationPayload)
      },
      {
        label: "Strength",
        value: getStrengthLabel(maxValue)
      }
    ];
  }

  if (kind === "forecast_extent") {
    return [
      {
        label: "Type",
        value: "Forecast reference domain"
      }
    ];
  }

  return Object.entries(properties)
    .slice(0, 4)
    .map(([label, value]) => ({
      label,
      value: typeof value === "string" || typeof value === "number" ? String(value) : "—"
    }));
}

function getRawRows(
  selected: SelectedFeatureState,
  explanationPayload: ForecastExplanation | null
): Array<{ label: string; value: string }> {
  const properties = selected.properties ?? {};
  const kind = properties.kind;

  if (
    kind === "plume_band_low" ||
    kind === "plume_band_medium" ||
    kind === "plume_band_high"
  ) {
    const summary = explanationPayload?.summary;

    return [
      {
        label: "Threshold",
        value: formatThresholdRaw(properties.threshold)
      },
      {
        label: "Active cells",
        value:
          typeof properties.cell_count === "number" || typeof properties.cell_count === "string"
            ? String(properties.cell_count)
            : "—"
      },
      {
        label: "Peak value",
        value: formatMaxRaw(properties.max_value)
      },
      {
        label: "Affected area (m²)",
        value:
          summary && Number.isFinite(summary.affected_area_m2)
            ? Math.round(summary.affected_area_m2).toLocaleString()
            : "—"
      },
      {
        label: "Affected area (ha)",
        value:
          summary && Number.isFinite(summary.affected_area_hectares)
            ? summary.affected_area_hectares.toFixed(2)
            : "—"
      }
    ];
  }

  if (kind === "source") {
    return [
      {
        label: "Emission rate",
        value:
          typeof properties.emissions_rate === "number" || typeof properties.emissions_rate === "string"
            ? String(properties.emissions_rate)
            : "—"
      }
    ];
  }

  return [];
}

export function ForecastAnalysisPanel({
  selected,
  explanationPayload,
  explanationSource
}: ForecastAnalysisPanelProps) {
  const prettySource = formatExplanationSource(explanationSource);
  const detailRows = selected ? getDetailRows(selected, explanationPayload) : [];
  const rawRows = selected ? getRawRows(selected, explanationPayload) : [];
  const explanationText = explanationPayload?.explanation.summary ?? "No explanation loaded.";
  const recommendation = explanationPayload?.explanation.recommendation ?? null;
  const uncertainty = explanationPayload?.explanation.uncertainty_note ?? null;
  const risk = explanationPayload?.explanation.risk_level ?? null;

  return (
    <aside className="detail-drawer panel">
      <div className="panel-header">
        <h2>Explanation</h2>
      </div>

      <div className="panel-body">
        <p className="muted explanation-source">Source: {prettySource}</p>
        <p className="detail-explanation-text">{explanationText}</p>

        {risk ? (
          <div className="detail-inline-meta">
            <span className="detail-chip">Risk: {risk}</span>
          </div>
        ) : null}

        {recommendation ? (
          <>
            <div className="section-spacer" />
            <h3>Recommendation</h3>
            <p className="muted">{recommendation}</p>
          </>
        ) : null}

        {uncertainty ? (
          <>
            <div className="section-spacer" />
            <h3>Uncertainty note</h3>
            <p className="muted">{uncertainty}</p>
          </>
        ) : null}

        <div className="section-spacer" />

        <h3>Inspect map</h3>

        {!selected ? (
          <p className="muted">Click the source marker or any visible plume band on the map.</p>
        ) : (
          <div className="detail-section">
            <h3>{getFeatureTitle(selected)}</h3>

            {getFeatureDescription(selected) ? (
              <p className="muted">{getFeatureDescription(selected)}</p>
            ) : null}

            <dl className="detail-list">
              {detailRows.map((row) => (
                <div key={row.label} className="detail-list-row">
                  <dt>{row.label}</dt>
                  <dd>{row.value}</dd>
                </div>
              ))}
            </dl>

            {rawRows.length > 0 ? (
              <details className="raw-details">
                <summary>Technical values</summary>
                <dl className="detail-list">
                  {rawRows.map((row) => (
                    <div key={row.label} className="detail-list-row">
                      <dt>{row.label}</dt>
                      <dd>{row.value}</dd>
                    </div>
                  ))}
                </dl>
              </details>
            ) : null}
          </div>
        )}
      </div>
    </aside>
  );
}