import type { DatasetScenarioPreview } from "../types";

type Props = {
  datasetScenarios: DatasetScenarioPreview[];
  activeScenario: string;
  activateDatasetScenario: (scenarioId: string) => Promise<void>;
};

export function ScenarioSelector({ datasetScenarios, activeScenario, activateDatasetScenario }: Props) {
  if (datasetScenarios.length === 0) return null;
  return <div className="scenario-control"><label htmlFor="scenario-select"><strong>Scenario</strong></label><div className="scenario-select"><select id="scenario-select" value={activeScenario} onChange={(e) => void activateDatasetScenario(e.target.value)}>{datasetScenarios.map((item) => <option key={item.scenario_id} value={item.scenario_id}>{item.label}</option>)}</select></div></div>;
}
