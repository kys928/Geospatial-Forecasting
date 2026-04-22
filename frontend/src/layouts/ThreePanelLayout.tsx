import type { ReactNode } from "react";

interface ThreePanelLayoutProps {
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
}

export function ThreePanelLayout({ left, center, right }: ThreePanelLayoutProps) {
  return (
    <div className="layout-grid">
      <div>{left}</div>
      <div>{center}</div>
      <div>{right}</div>
    </div>
  );
}
