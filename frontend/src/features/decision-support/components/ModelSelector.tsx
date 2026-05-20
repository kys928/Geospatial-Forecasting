import { FORECAST_MODELS, type ActiveModelId } from "../../forecast-selection/modelRegistry";

export function ModelSelector({ activeModelId, setActiveModel }: { activeModelId: ActiveModelId; setActiveModel: (id: ActiveModelId) => void }) {
  return <div className="scenario-control"><label htmlFor="model-select"><strong>Model</strong></label><div className="scenario-select"><select id="model-select" value={activeModelId} onChange={(e) => setActiveModel(e.target.value as ActiveModelId)}>{FORECAST_MODELS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></div></div>;
}
