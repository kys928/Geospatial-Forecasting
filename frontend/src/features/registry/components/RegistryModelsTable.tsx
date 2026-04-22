import type { RegistryModelRecord } from "../types/registry.types";

interface RegistryModelsTableProps {
  models: RegistryModelRecord[];
  selectedModelId: string | null;
  onSelectModel: (modelId: string) => void;
}

export function RegistryModelsTable({ models, selectedModelId, onSelectModel }: RegistryModelsTableProps) {
  return (
    <section className="panel">
      <h3>Registry models</h3>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr>
            <th align="left">model_id</th><th align="left">status</th><th align="left">approval_status</th><th align="left">path</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            const id = typeof model.model_id === "string" ? model.model_id : "";
            return (
              <tr key={id} onClick={() => id && onSelectModel(id)} style={{ background: selectedModelId === id ? "#f0f6ff" : "transparent", cursor: "pointer" }}>
                <td>{id}</td><td>{String(model.status ?? "")}</td><td>{String(model.approval_status ?? "")}</td><td>{String(model.path ?? "")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
