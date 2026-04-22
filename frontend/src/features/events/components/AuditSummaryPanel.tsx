import type { EventRecord } from "../types/event.types";

interface AuditSummaryPanelProps {
  events: EventRecord[];
}

export function AuditSummaryPanel({ events }: AuditSummaryPanelProps) {
  const counts = events.reduce<Record<string, number>>((acc, event) => {
    const key = event.event_type ?? "unknown";
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <section className="panel">
      <h3>Audit summary</h3>
      <p><strong>Total events:</strong> {events.length}</p>
      <pre style={{ margin: 0, maxHeight: 240, overflow: "auto" }}>{JSON.stringify(counts, null, 2)}</pre>
    </section>
  );
}
