import type { SelectedFeatureState } from "../../forecast/types/forecast.types";
import type { SessionPredictionResponse } from "../types/session.types";
import { normalizeSessionPrediction } from "../utils/normalizeSessionPrediction";

interface SessionResultSummaryProps {
  lastPrediction: SessionPredictionResponse | null;
  loading: boolean;
  error: string | null;
  selectedFeature: SelectedFeatureState | null;
}

function formatSource(source: string | null): string {
  if (!source) {
    return "session forecast";
  }

  return source;
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

function getDetailRows(selected: SelectedFeatureState): Array<{ label: string; value: string }> {
  const properties = selected.properties ?? {};
  const kind = properties.kind;

  if (kind === "source") {
    return [
      {
        label: "Type",
        value: "Release origin"
      },
      {
        label: "Emission level",
        value:
          typeof properties.emissions_rate === "number" || typeof properties.emissions_rate === "string"
            ? `${properties.emissions_rate} units`
            : "Unknown"
      }
    ];
  }

  if (kind === "plume_band_low" || kind === "plume_band_medium" || kind === "plume_band_high") {
    return [
      {
        label: "Threshold",
        value:
          typeof properties.threshold === "number" || typeof properties.threshold === "string"
            ? String(properties.threshold)
            : "Unknown"
      },
      {
        label: "Active cells",
        value:
          typeof properties.cell_count === "number" || typeof properties.cell_count === "string"
            ? String(properties.cell_count)
            : "Unknown"
      },
      {
        label: "Peak value",
        value:
          typeof properties.max_value === "number" || typeof properties.max_value === "string"
            ? String(properties.max_value)
            : "Unknown"
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

export function SessionResultSummary({ lastPrediction, loading, error, selectedFeature }: SessionResultSummaryProps) {
  const normalized = normalizeSessionPrediction(lastPrediction);
  const detailRows = selectedFeature ? getDetailRows(selectedFeature) : [];

  return (
    <section className="detail-drawer panel">
      <div className="panel-header">
        <h2>Explanation</h2>
      </div>

      <div className="panel-body">
        <p className="muted explanation-source">Source: {formatSource(normalized?.source ?? null)}</p>

        {loading ? <p className="muted">Loading session details...</p> : null}
        {error ? <p className="muted">{error}</p> : null}

        <p className="detail-explanation-text">
          {normalized?.explanation ?? "Run a forecast to generate a readable result summary."}
        </p>

        {normalized?.riskLevel ? (
          <div className="detail-inline-meta">
            <span className="detail-chip">Risk: {normalized.riskLevel}</span>
          </div>
        ) : null}

        {normalized?.recommendation ? (
          <>
            <div className="section-spacer" />
            <h3>Recommendation</h3>
            <p className="muted">{normalized.recommendation}</p>
          </>
        ) : null}

        {normalized?.uncertaintyNote ? (
          <>
            <div className="section-spacer" />
            <h3>Uncertainty note</h3>
            <p className="muted">{normalized.uncertaintyNote}</p>
          </>
        ) : null}

        <div className="section-spacer" />

        <h3>Inspect map</h3>
        {!selectedFeature ? (
          <p className="muted">Go to the Map page and click the source marker or a plume band.</p>
        ) : (
          <div className="detail-section">
            <h3>{getFeatureTitle(selectedFeature)}</h3>
            {getFeatureDescription(selectedFeature) ? (
              <p className="muted">{getFeatureDescription(selectedFeature)}</p>
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
    </section>
  );
}
