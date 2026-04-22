import type { ReactNode } from "react";
import { ThreePanelLayout } from "./ThreePanelLayout";

interface MapWorkspaceLayoutProps {
  controls: ReactNode;
  map: ReactNode;
  detail: ReactNode;
}

export function MapWorkspaceLayout({ controls, map, detail }: MapWorkspaceLayoutProps) {
  return <ThreePanelLayout left={controls} center={map} right={detail} />;
}
