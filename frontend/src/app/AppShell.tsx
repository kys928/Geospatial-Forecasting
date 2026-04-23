import type { ReactNode } from "react";
import { TopNav } from "../components/navigation/TopNav";
import { WorkspaceTabs } from "../components/navigation/WorkspaceTabs";
import { GlobalStatusStrip } from "../components/navigation/GlobalStatusStrip";
import type { ApiMode } from "../features/forecast/types/forecast.types";

interface AppShellProps {
  children: ReactNode;
  title: string;
  subtitle?: string;
  statusText?: string;
  apiMode?: ApiMode;
  apiHealthy?: boolean;
  metaItems?: Array<{ label: string; tone?: "default" | "ok" | "error" }>;
}

export function AppShell({ children, title, subtitle, statusText, apiMode, apiHealthy, metaItems }: AppShellProps) {
  return (
    <div className="app-shell">
      <TopNav title={title} subtitle={subtitle} apiMode={apiMode} apiHealthy={apiHealthy} metaItems={metaItems} />
      <div className="workspace-nav-row panel">
        <WorkspaceTabs />
      </div>
      <main>{children}</main>
      {statusText ? <GlobalStatusStrip statusText={statusText} /> : null}
    </div>
  );
}
