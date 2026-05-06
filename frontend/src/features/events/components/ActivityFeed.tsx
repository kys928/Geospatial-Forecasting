import type { PresentedEvent } from "../utils/presentEvent";

interface ActivityFeedProps {
  events: PresentedEvent[];
  selectedEventId: string | null;
  onSelect: (eventId: string) => void;
}

export function ActivityFeed({ events, selectedEventId, onSelect }: ActivityFeedProps) {
  const grouped = events.reduce<Record<string, PresentedEvent[]>>((acc, event) => {
    acc[event.dateGroup] = acc[event.dateGroup] ?? [];
    acc[event.dateGroup].push(event);
    return acc;
  }, {});

  return (
    <section className="panel activity-feed">
      <h3>Activity feed</h3>
      {Object.entries(grouped).map(([dateGroup, dateEvents]) => (
        <div key={dateGroup} className="activity-date-group">
          <h4>{dateGroup}</h4>
          {dateEvents.map((event) => (
            <button
              key={event.id}
              className={`activity-event-row ${selectedEventId === event.id ? "activity-event-row-selected" : ""}`}
              onClick={() => onSelect(event.id)}
              type="button"
            >
              <div className="activity-event-main">
                <div className="activity-event-title">{event.title}</div>
                <span className="activity-event-time muted">{event.timeLabel}</span>
              </div>
              <div className="activity-event-meta">
                <p className="activity-event-summary muted">{event.summary}</p>
                <span className="badge">{`${event.category.charAt(0).toUpperCase()}${event.category.slice(1)} · ${event.severity.charAt(0).toUpperCase()}${event.severity.slice(1)}`}</span>
              </div>
            </button>
          ))}
        </div>
      ))}
    </section>
  );
}
