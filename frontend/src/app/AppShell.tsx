import type { ReactNode } from "react";
import { TopNav } from "../components/navigation/TopNav";
import { WorkspaceTabs } from "../components/navigation/WorkspaceTabs";
import { RoleSwitcher } from "../components/navigation/RoleSwitcher";
import { GlobalStatusStrip } from "../components/navigation/GlobalStatusStrip";
import type { ApiMode } from "../features/forecast/types/forecast.types";

interface AppShellProps {
  children: ReactNode;
  statusText?: string;
  scenarioName?: string;
  modelLabel?: string;
  apiHealthy?: boolean;
}

export function AppShell({
  children,
  statusText,
  scenarioName = "Workspace",
  modelLabel = "Context managed by workspace",
  apiHealthy
}: AppShellProps) {
  const apiMode: ApiMode = "live";

  return (
    <div className="app-shell">
      <TopNav apiMode={apiMode} apiHealthy={apiHealthy} modelLabel={modelLabel} scenarioName={scenarioName} />
      <div className="panel" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <WorkspaceTabs />
        <RoleSwitcher />
      </div>
      <main>{children}</main>
      {statusText ? <GlobalStatusStrip statusText={statusText} /> : null}
    </div>
  );
}
