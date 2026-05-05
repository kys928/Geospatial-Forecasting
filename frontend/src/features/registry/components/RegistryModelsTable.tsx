import { useState } from "react";
import type { RegistryModelRecord } from "../types/registry.types";

interface RegistryModelsTableProps {
  models: RegistryModelRecord[];
  selectedModelId: string | null;
  onSelectModel: (modelId: string) => void;
  onActivate: (modelId: string) => void;
}

export function RegistryModelsTable({ models, selectedModelId, onSelectModel, onActivate }: RegistryModelsTableProps) {
  const [menuOpen, setMenuOpen] = useState<string | null>(null);
  return <section className="panel"><h3>Model versions</h3><table className="ops-model-table"><thead><tr><th>Actions</th><th>Model ID</th><th>Status</th><th>Approval</th><th>Path</th><th>Updated</th><th>Active</th></tr></thead><tbody>{models.map((model)=>{const id=String(model.model_id??""); const isActive=model.status==="active"; return <tr key={id} className={selectedModelId===id?"selected":""} onClick={()=>id&&onSelectModel(id)}><td><button className="secondary-button" onClick={(e)=>{e.stopPropagation();setMenuOpen(menuOpen===id?null:id);}}>⋮</button>{menuOpen===id?<div className="ops-row-menu"><button onClick={()=>{onActivate(id);setMenuOpen(null);}}>Activate model</button><button onClick={()=>{onSelectModel(id);setMenuOpen(null);}}>Inspect model</button><button disabled title="Delete is not wired yet">Delete model</button></div>:null}</td><td>{id}</td><td>{String(model.status ?? "")}</td><td>{String(model.approval_status ?? "")}</td><td>{String(model.path ?? "")}</td><td>{String(model.created_at ?? model.updated_at ?? "Not reported")}</td><td>{isActive ? "Current" : ""}</td></tr>;})}</tbody></table>{!models.length?<p className="muted">Registry is empty.</p>:null}</section>;
}
