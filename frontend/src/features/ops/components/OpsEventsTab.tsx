import { useMemo, useState } from "react";
import { useEvents } from "../../events/hooks/useEvents";
import { EventFilters } from "../../events/components/EventFilters";
import { IncidentTimeline } from "../../events/components/IncidentTimeline";
import { EventDetailDrawer } from "../../events/components/EventDetailDrawer";

export function OpsEventsTab() {
  const eventsState = useEvents();
  const [selectedEventIndex, setSelectedEventIndex] = useState(0);
  const selectedEvent = useMemo(
    () => eventsState.filteredEvents[selectedEventIndex] ?? null,
    [eventsState.filteredEvents, selectedEventIndex]
  );

  return (
    <div className="workspace-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
      <div className="workspace-column">
        <EventFilters
          searchText={eventsState.searchText}
          onSearchTextChange={eventsState.setSearchText}
          eventType={eventsState.eventType}
          eventTypes={eventsState.availableTypes}
          onEventTypeChange={eventsState.setEventType}
        />
        <button className="primary-button" onClick={() => void eventsState.refresh()}>
          {eventsState.loading ? "Refreshing..." : "Refresh events"}
        </button>
      </div>
      <div className="workspace-column">
        <IncidentTimeline
          events={eventsState.filteredEvents}
          selectedIndex={selectedEventIndex}
          onSelect={setSelectedEventIndex}
        />
      </div>
      <div className="workspace-column">
        <EventDetailDrawer event={selectedEvent} />
      </div>
    </div>
  );
}
