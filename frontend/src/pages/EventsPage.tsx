import { useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { useEvents } from "../features/events/hooks/useEvents";
import { EventFilters } from "../features/events/components/EventFilters";
import { IncidentTimeline } from "../features/events/components/IncidentTimeline";
import { EventDetailDrawer } from "../features/events/components/EventDetailDrawer";
import { AuditSummaryPanel } from "../features/events/components/AuditSummaryPanel";

export function EventsPage() {
  const eventsState = useEvents();
  const [selectedIndex, setSelectedIndex] = useState(0);

  const selectedEvent = useMemo(() => eventsState.filteredEvents[selectedIndex] ?? null, [eventsState.filteredEvents, selectedIndex]);

  return (
    <AppShell
      title="Events workspace"
      subtitle="Review the operational event timeline and audit context."
      statusText={eventsState.error ?? (eventsState.loading ? "Loading events..." : "Ready")}
    >
      <div className="workspace-grid">
        <div className="workspace-column">
          <EventFilters
            searchText={eventsState.searchText}
            onSearchTextChange={eventsState.setSearchText}
            eventType={eventsState.eventType}
            eventTypes={eventsState.availableTypes}
            onEventTypeChange={eventsState.setEventType}
          />
          <button className="primary-button" onClick={() => void eventsState.refresh()}>{eventsState.loading ? "Refreshing..." : "Refresh events"}</button>
          <AuditSummaryPanel events={eventsState.filteredEvents} />
        </div>

        <div className="workspace-column">
          <IncidentTimeline events={eventsState.filteredEvents} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
        </div>

        <div className="workspace-column">
          <EventDetailDrawer event={selectedEvent} />
        </div>
      </div>
    </AppShell>
  );
}
