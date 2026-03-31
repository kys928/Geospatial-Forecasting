import type { SelectedFeatureState } from "../features/forecast/forecast.types";

interface DetailDrawerProps {
  selected: SelectedFeatureState | null;
  explanation: string;
}

export function DetailDrawer({ selected, explanation }: DetailDrawerProps) {
  return (
    <aside className="detail-drawer panel">
      <div className="panel-header">
        <h2>Details</h2>
      </div>

      <div className="panel-body">
        {!selected ? (
          <p className="muted">Select a map feature to inspect details.</p>
        ) : (
          <>
            <h3>{selected.title}</h3>
            <pre className="json-block">
              {JSON.stringify(selected.properties, null, 2)}
            </pre>
          </>
        )}

        <div className="section-spacer" />

        <h3>Explanation</h3>
        <p className="muted">{explanation}</p>

        <div className="section-spacer" />

        <h3>Uncertainty</h3>
        <p className="muted">Placeholder only. Uncertainty is not implemented yet.</p>
      </div>
    </aside>
  );
}