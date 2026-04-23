import { AppShell } from "../app/AppShell";

export function OpsPage() {
  return (
    <AppShell
      title="Ops workspace"
      subtitle="Operational tooling is planned but not available in this build yet."
      statusText="Ops is not implemented yet."
      metaItems={[{ label: "Planned" }]}
    >
      <section className="panel">
        <h2>Ops is not implemented yet.</h2>
        <p className="muted">Operational tooling is planned but not available in this build yet.</p>
      </section>
    </AppShell>
  );
}
