import type { EventRecord } from "../types/event.types";

interface IncidentTimelineProps {
  events: EventRecord[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

export function IncidentTimeline({ events, selectedIndex, onSelect }: IncidentTimelineProps) {
  return (
    <section className="panel">
      <h3>Incident timeline</h3>
      <div style={{ maxHeight: 460, overflow: "auto", display: "grid", gap: 8 }}>
        {events.map((event, index) => (
          <button key={`${event.timestamp ?? "na"}-${index}`} className="badge" style={{ textAlign: "left", borderColor: selectedIndex === index ? "#7aa2f7" : undefined }} onClick={() => onSelect(index)}>
            <div>{event.timestamp ?? "n/a"}</div>
            <strong>{event.event_type ?? "unknown"}</strong>
          </button>
        ))}
      </div>
    </section>
  );
}
