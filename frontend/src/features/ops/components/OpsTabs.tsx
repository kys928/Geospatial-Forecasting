export type OpsTabKey = "overview" | "training" | "registry" | "events";

interface OpsTabsProps {
  selected: OpsTabKey;
  onSelect: (tab: OpsTabKey) => void;
}

const tabs: Array<{ key: OpsTabKey; label: string }> = [
  { key: "overview", label: "System Status" },
  { key: "training", label: "Model Training" },
  { key: "registry", label: "Model Versions" },
  { key: "events", label: "Activity Log" }
];

export function OpsTabs({ selected, onSelect }: OpsTabsProps) {
  return (
    <section className="panel" aria-label="Ops views">
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            className={selected === tab.key ? "primary-button" : "secondary-button"}
            onClick={() => onSelect(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </section>
  );
}
