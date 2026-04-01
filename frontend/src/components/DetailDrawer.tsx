import type { ForecastExplanation, SelectedFeatureState } from "../features/forecast/forecast.types";

interface DetailDrawerProps {
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
      return "Outer hazard band for the lowest configured concentration threshold.";
    case "plume_band_medium":
      return "Intermediate hazard band for the medium configured concentration threshold.";
    case "plume_band_high":
      return "Core hazard band for the highest configured concentration threshold.";
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

function getDetailRows(
  selected: SelectedFeatureState
): Array<{ label: string; value: string }> {
  const properties = selected.properties ?? {};
  const kind = properties.kind;

  if (kind === "source") {
    const emissionsRate = properties.emissions_rate;
    return [
      {
        label: "Kind",
        value: "Point source"
      },
      {
        label: "Emission rate",
        value:
          typeof emissionsRate === "number" || typeof emissionsRate === "string"
            ? String(emissionsRate)
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
    const cellCount = properties.cell_count;
    const maxValue = properties.max_value;

    return [
      {
        label: "Threshold",
        value:
          typeof threshold === "number" || typeof threshold === "string"
            ? String(threshold)
            : "Unknown"
      },
      {
        label: "Active cells",
        value:
          typeof cellCount === "number" || typeof cellCount === "string"
            ? String(cellCount)
            : "Unknown"
      },
      {
        label: "Max value",
        value:
          typeof maxValue === "number" || typeof maxValue === "string"
            ? String(maxValue)
            : "Unknown"
      }
    ];
  }

  if (kind === "forecast_extent") {
    return [
      {
        label: "Kind",
        value: "Reference domain"
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

export function DetailDrawer({
  selected,
  explanationPayload,
  explanationSource
}: DetailDrawerProps) {
  const prettySource = formatExplanationSource(explanationSource);
  const detailRows = selected ? getDetailRows(selected) : [];
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
            <span className="detail-chip">
              Risk: {risk}
            </span>
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
          </div>
        )}
      </div>
    </aside>
  );
}