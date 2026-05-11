import { ScenarioSelector } from "./ScenarioSelector";
import type { DatasetScenarioPreview, DisplayRow } from "../types";

type Props = {
  datasetScenarios: DatasetScenarioPreview[];
  activeScenario: string;
  activateDatasetScenario: (scenarioId: string) => Promise<void>;
  currentConditionsRows: DisplayRow[];
  currentForecastRows: DisplayRow[];
  plumePresent: boolean;
  plumeDetailRows: DisplayRow[];
  detailsRows: DisplayRow[];
  filterAvailableRows: (rows: DisplayRow[], options?: { allowZero?: boolean }) => DisplayRow[];
  rawContext: Record<string, unknown>;
};

export function ConditionsPanel(props: Props) {
  const {
    datasetScenarios,
    activeScenario,
    activateDatasetScenario,
    currentConditionsRows,
    currentForecastRows,
    plumePresent,
    plumeDetailRows,
    detailsRows,
    filterAvailableRows,
    rawContext
  } = props;

  return <section className="panel decision-support-live-panel">
    <h3>Geospatial Conditions</h3>
    <ScenarioSelector datasetScenarios={datasetScenarios} activeScenario={activeScenario} activateDatasetScenario={activateDatasetScenario} />
    <div className="values-section">
      <h4>Current Conditions</h4>
      <div className="values-grid compact-values-grid">{currentConditionsRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}</div>
    </div>

    <div className="values-section">
      <h4>Forecast Result</h4>
      <div className="values-grid compact-values-grid">{currentForecastRows.map(([label, value]) => <div key={label} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}</div>
      {plumePresent ? <div className="values-grid compact-values-grid">{plumeDetailRows.map(([label, value]) => <div key={`plume-${label}`} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}</div> : null}
    </div>

    <details className="technical-details">
      <summary>Details</summary>
      <div className="values-grid compact-values-grid">{filterAvailableRows(detailsRows).map(([label, value]) => <div key={`details-${label}`} className="status-row"><strong>{label}</strong><span>{value}</span></div>)}</div>
      <details className="technical-details nested-technical-details">
        <summary>Technical details</summary>
        <pre>{JSON.stringify(rawContext, null, 2)}</pre>
      </details>
    </details>
  </section>;
}
