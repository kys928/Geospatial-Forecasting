import type { SelectedFeatureState } from "../features/forecast/forecast.types";

interface DetailDrawerProps {
  selected: SelectedFeatureState | null;
  explanation: string;
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

export function DetailDrawer({
  selected,
  explanation,
  explanationSource
}: DetailDrawerProps) {
  const prettySource = formatExplanationSource(explanationSource);

  return (
    <aside className="detail-drawer panel">
      <div className="panel-header">
        <h2>Details</h2>
      </div>

      <div className="panel-body">
        {!selected ? (
          <p className="muted">Select a source or plume region on the map to inspect it.</p>
        ) : (
          <>
            <h3>{getFeatureTitle(selected)}</h3>

            {getFeatureDescription(selected) ? (
              <p className="muted">{getFeatureDescription(selected)}</p>
            ) : null}

            <pre className="json-block">
              {JSON.stringify(selected.properties, null, 2)}
            </pre>
          </>
        )}

        <div className="section-spacer" />

        <h3>Explanation</h3>
        <p className="muted explanation-source">Source: {prettySource}</p>
        <p className="muted">{explanation}</p>

        <div className="section-spacer" />

        <h3>Uncertainty</h3>
        <p className="muted">
          Placeholder only. Uncertainty is not implemented yet.
        </p>
      </div>
    </aside>
  );
}