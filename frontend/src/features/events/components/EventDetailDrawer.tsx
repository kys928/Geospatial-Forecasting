import type { PresentedEvent } from "../utils/presentEvent";

interface EventDetailDrawerProps {
  event: PresentedEvent;
}

export function EventDetailDrawer({ event }: EventDetailDrawerProps) {

  return (
    <section className="panel activity-event-detail">
      <h3>Event detail</h3>
      <p className="activity-event-title">{event.title}</p>
      <p className="activity-event-summary muted">{event.summary}</p>
      <dl className="activity-detail-grid">
        <div><dt>Time</dt><dd>{event.timestampRaw ?? "Not available"}</dd></div>
        <div><dt>Activity</dt><dd>{event.activityLabel}</dd></div>
        <div><dt>Status</dt><dd>{event.statusLabel}</dd></div>
        {event.objectLabel ? <div><dt>Object</dt><dd>{event.objectLabel}</dd></div> : null}
        <div><dt>Event type</dt><dd>{event.raw.event_type ?? "Not available"}</dd></div>
      </dl>
      <details className="developer-payload">
        <summary>Developer payload</summary>
        <pre>{JSON.stringify(event.raw, null, 2)}</pre>
      </details>
    </section>
  );
}
