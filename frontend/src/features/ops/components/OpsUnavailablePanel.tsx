interface OpsUnavailablePanelProps {
  reason: string | null;
}

export function OpsUnavailablePanel({ reason }: OpsUnavailablePanelProps) {
  return (
    <section className="panel" style={{ maxWidth: 820, margin: "40px auto", textAlign: "center" }}>
      <h3>Ops unavailable</h3>
      <p>{reason ?? "Ops workspace is currently unavailable."}</p>
      <p className="muted">
        Configure VITE_OPS_API_TOKEN on the frontend and PLUME_OPS_API_TOKEN on the backend, or disable ops auth for local
        development.
      </p>
    </section>
  );
}
