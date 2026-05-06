import { useMemo, useState } from "react";
import { ActivityFeed } from "../../events/components/ActivityFeed";
import { EventDetailDrawer } from "../../events/components/EventDetailDrawer";
import { EventFilters } from "../../events/components/EventFilters";
import { useEvents } from "../../events/hooks/useEvents";
import { presentEvent, type EventCategory, type EventSeverity } from "../../events/utils/presentEvent";

export function OpsEventsTab() {
  const eventsState = useEvents();
  const [searchText, setSearchText] = useState("");
  const [category, setCategory] = useState<"all" | EventCategory>("all");
  const [severity, setSeverity] = useState<"all" | EventSeverity>("all");
  const [limit, setLimit] = useState<50 | 100 | 200>(200);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  const presentedEvents = useMemo(() => eventsState.events.map((event, index) => presentEvent(event, index)), [eventsState.events]);

  const filteredEvents = useMemo(() => {
    const needle = searchText.trim().toLowerCase();
    return presentedEvents
      .filter((event) => {
        const matchesCategory = category === "all" || event.category === category;
        const matchesSeverity = severity === "all" || event.severity === severity;
        const blob = `${event.title} ${event.summary} ${event.category} ${event.severity} ${event.raw.event_type ?? ""} ${JSON.stringify(event.raw)}`.toLowerCase();
        const matchesSearch = needle === "" || blob.includes(needle);
        return matchesCategory && matchesSeverity && matchesSearch;
      })
      .slice(0, limit);
  }, [category, limit, presentedEvents, searchText, severity]);

  const selectedEvent = useMemo(
    () => filteredEvents.find((event) => event.id === selectedEventId) ?? null,
    [filteredEvents, selectedEventId]
  );

  return (
    <div className="activity-log-layout">
      <section className="panel activity-log-header">
        <h3>Activity Log</h3>
        <p className="muted">Operational events from training, model registry, workers, and forecast runs.</p>
        <p className="muted">Last updated: {eventsState.lastUpdatedLabel ?? "Not available"}</p>
        <EventFilters
          searchText={searchText}
          onSearchTextChange={setSearchText}
          category={category}
          onCategoryChange={setCategory}
          severity={severity}
          onSeverityChange={setSeverity}
          limit={limit}
          onLimitChange={setLimit}
        />
      </section>

      {eventsState.loading ? <section className="panel muted">Loading activity...</section> : null}
      {eventsState.error ? <section className="panel muted">Unable to load activity: {eventsState.error}</section> : null}
      {!eventsState.loading && !eventsState.error && presentedEvents.length === 0 ? (
        <section className="panel muted">No activity events are available yet.</section>
      ) : null}
      {!eventsState.loading && !eventsState.error && presentedEvents.length > 0 && filteredEvents.length === 0 ? (
        <section className="panel muted">No events match the current filters.</section>
      ) : null}
      {!eventsState.loading && !eventsState.error && filteredEvents.length > 0 ? (
        <ActivityFeed events={filteredEvents} selectedEventId={selectedEventId} onSelect={setSelectedEventId} />
      ) : null}
      {selectedEvent ? <EventDetailDrawer event={selectedEvent} /> : null}
    </div>
  );
}
