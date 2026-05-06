import type { PresentedEvent } from "../utils/presentEvent";

interface ActivityFeedProps {
  events: PresentedEvent[];
  selectedEventId: string | null;
  onSelect: (eventId: string) => void;
  isPreview: boolean;
  filteredCount: number;
  visibleCount: number;
  onViewMore: () => void;
}

export function ActivityFeed({ events, selectedEventId, onSelect, isPreview, filteredCount, visibleCount, onViewMore }: ActivityFeedProps) {
  return (
    <section className="panel activity-feed">
      <h3>Activity feed</h3>
      {isPreview ? <p className="muted">Preview examples — real activity will appear here once the backend reports events.</p> : null}
      {events.map((event) => (
        <button
          key={event.id}
          className={`activity-event-row ${selectedEventId === event.id ? "activity-event-row-selected" : ""}`}
          onClick={() => onSelect(event.id)}
          type="button"
        >
          <div className="activity-event-topline">
            <div className="activity-event-title">{event.title}</div>
            <span className="activity-event-meta">{`${event.activityLabel} · ${event.statusLabel} · ${event.timeLabel}`}</span>
          </div>
          <div className="activity-event-summary">{event.summary}</div>
        </button>
      ))}
      <div className="activity-feed-footer">
        <span className="muted">{isPreview ? "Showing 5 preview examples" : `Showing ${Math.min(visibleCount, filteredCount)} of ${filteredCount} events`}</span>
        {!isPreview && filteredCount > visibleCount ? <button className="secondary-button" type="button" onClick={onViewMore}>View more</button> : null}
      </div>
    </section>
  );
}
