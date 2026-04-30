import { useState } from "react";
import { AppShell } from "../app/AppShell";
import { OpsTabs, type OpsTabKey } from "../features/ops/components/OpsTabs";
import { OpsOverviewTab } from "../features/ops/components/OpsOverviewTab";
import { OpsRegistryTab } from "../features/ops/components/OpsRegistryTab";
import { OpsEventsTab } from "../features/ops/components/OpsEventsTab";

export function OpsPage() {
  const [tab, setTab] = useState<OpsTabKey>("overview");

  return (
    <AppShell
      title="Ops workspace"
      subtitle="Operational status, retraining controls, registry, and event/audit panels."
      statusText="Ready"
      metaItems={[{ label: "Ops" }]}
    >
      <OpsTabs selected={tab} onSelect={setTab} />
      {tab === "overview" ? <OpsOverviewTab /> : null}
      {tab === "registry" ? <OpsRegistryTab /> : null}
      {tab === "events" ? <OpsEventsTab /> : null}
    </AppShell>
  );
}
