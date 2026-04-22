import { AppShell } from "../app/AppShell";

export function NotFoundPage() {
  return (
    <AppShell statusText="Page not found">
      <section className="panel">
        <h2>Page Not Found</h2>
        <p className="muted">Placeholder page for Route does not exist.</p>
      </section>
    </AppShell>
  );
}
