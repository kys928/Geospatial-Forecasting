import { useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsEventRecord } from "../types/ops.types";

interface OpsEventsPanelProps {
  enabled: boolean;
}

export function OpsEventsPanel({ enabled }: OpsEventsPanelProps) {
  const [events, setEvents] = useState<OpsEventRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    async function fetchEvents() {
      setLoading(true);
      setError(null);
      try {
        const response = await opsClient.getEvents(50);
        setEvents(response.events);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load events.");
      } finally {
        setLoading(false);
      }
    }

    void fetchEvents();
  }, [enabled]);

  return (
    <section className="panel">
      <h3>Ops events</h3>
      {loading ? <p className="muted">Loading events...</p> : null}
      {error ? <p className="failure-text">Unable to load events: {error}</p> : null}
      {!loading && !error && events.length === 0 ? <p className="muted">No events available.</p> : null}
      <div style={{ display: "grid", gap: 8 }}>
        {events.slice().reverse().map((event, index) => (
          <article key={`${event.timestamp ?? "na"}-${index}`} className="badge" style={{ borderRadius: 8 }}>
            <div>{event.timestamp ?? "n/a"}</div>
            <strong>{event.event_type ?? "unknown"}</strong>
          </article>
        ))}
      </div>
    </section>
  );
}
