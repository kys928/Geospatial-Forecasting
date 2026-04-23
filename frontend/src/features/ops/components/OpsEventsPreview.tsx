import type { OpsEventRecord } from "../types/ops.types";

interface OpsEventsPreviewProps {
  events: OpsEventRecord[];
}

export function OpsEventsPreview({ events }: OpsEventsPreviewProps) {
  return (
    <section className="panel">
      <h3>Recent ops events</h3>
      <div style={{ maxHeight: 260, overflow: "auto", display: "grid", gap: 8 }}>
        {events.slice(-10).reverse().map((event, index) => (
          <div key={`${event.timestamp ?? "na"}-${index}`} className="badge" style={{ borderRadius: 8 }}>
            <div>{event.timestamp ?? "n/a"}</div>
            <strong>{event.event_type ?? "unknown"}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
