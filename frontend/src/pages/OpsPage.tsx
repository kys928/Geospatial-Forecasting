import { useState } from "react";
import { AppShell } from "../app/AppShell";
import { OpsTabs, type OpsTabKey } from "../features/ops/components/OpsTabs";
import { OpsOverviewTab } from "../features/ops/components/OpsOverviewTab";
import { OpsTrainingTab } from "../features/ops/components/OpsTrainingTab";
import { OpsRegistryTab } from "../features/ops/components/OpsRegistryTab";
import { OpsEventsTab } from "../features/ops/components/OpsEventsTab";

export function OpsPage() {
  const [tab, setTab] = useState<OpsTabKey>("overview");

  return (
    <AppShell
      title="Operations Workspace"
      subtitle="Operational status, retraining controls, registry, and event/audit panels."
      metaItems={[{ label: "Workspace" }]}
    >
      <OpsTabs selected={tab} onSelect={setTab} />
      <section hidden={tab !== "overview"} aria-hidden={tab !== "overview"}>
        <OpsOverviewTab active={tab === "overview"} />
      </section>
      <section hidden={tab !== "training"} aria-hidden={tab !== "training"}>
        <OpsTrainingTab active={tab === "training"} />
      </section>
      <section hidden={tab !== "registry"} aria-hidden={tab !== "registry"}>
        <OpsRegistryTab />
      </section>
      <section hidden={tab !== "events"} aria-hidden={tab !== "events"}>
        <OpsEventsTab />
      </section>
    </AppShell>
  );
}
