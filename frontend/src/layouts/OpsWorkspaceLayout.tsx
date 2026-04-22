import type { ReactNode } from "react";
import { ThreePanelLayout } from "./ThreePanelLayout";

interface OpsWorkspaceLayoutProps {
  status: ReactNode;
  controlTower: ReactNode;
  events: ReactNode;
}

export function OpsWorkspaceLayout({ status, controlTower, events }: OpsWorkspaceLayoutProps) {
  return <ThreePanelLayout left={status} center={controlTower} right={events} />;
}
