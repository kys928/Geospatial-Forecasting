import { useEffect, useMemo, useState } from "react";
import { ActivityFeed } from "../../events/components/ActivityFeed";
import { EventDetailDrawer } from "../../events/components/EventDetailDrawer";
import { useEvents } from "../../events/hooks/useEvents";
import { presentEvent, type EventSeverity, type PresentedEvent } from "../../events/utils/presentEvent";
import type { EventRecord } from "../../events/types/event.types";

const PREVIEW_EVENTS: PresentedEvent[] = [
  {
    id: "preview-1",
    title: "Model candidate approved",
    summary: "Candidate convlstm_2026_05_06 was approved and is ready for activation.",
    activityLabel: "Model",
    statusLabel: "Success",
    category: "model",
    severity: "success",
    timeLabel: "11:42",
    timestampRaw: null,
    dateGroup: "Preview examples",
    objectLabel: "convlstm_2026_05_06",
    isPreview: true,
    raw: { event_type: "candidate_approved",  payload: { message: "Preview example; no backend payload available." } } as EventRecord
  },
  { id: "preview-2", title: "Retraining job completed", summary: "Job job_184 finished successfully and produced a candidate model.", activityLabel: "Training", statusLabel: "Success", category: "training", severity: "success", timeLabel: "11:18", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "job_184", isPreview: true, raw: { event_type: "retraining_job_completed",  payload: { message: "Preview example; no backend payload available." } } as EventRecord },
  { id: "preview-3", title: "Worker heartbeat received", summary: "Retraining worker is active and reporting status.", activityLabel: "System", statusLabel: "Info", category: "system", severity: "info", timeLabel: "10:45", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "worker", isPreview: true, raw: { event_type: "worker_heartbeat",  payload: { message: "Preview example; no backend payload available." } } as EventRecord },
  { id: "preview-4", title: "Forecast created", summary: "Forecast forecast_2026_05_06_1022 was generated successfully.", activityLabel: "Forecast", statusLabel: "Success", category: "forecast", severity: "success", timeLabel: "10:22", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "forecast_2026_05_06_1022", isPreview: true, raw: { event_type: "forecast_created",  payload: { message: "Preview example; no backend payload available." } } as EventRecord },
  { id: "preview-5", title: "Retraining job failed", summary: "Job job_183 failed because the worker ran out of memory.", activityLabel: "Training", statusLabel: "Error", category: "training", severity: "error", timeLabel: "Yesterday", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "job_183", isPreview: true, raw: { event_type: "retraining_job_failed",  payload: { message: "Preview example; no backend payload available." } } as EventRecord }
];

function rawEventType(event: PresentedEvent): string {
  return typeof event.raw.event_type === "string" ? event.raw.event_type : "";
}

function groupedSummary(event: PresentedEvent, groupCount: number): string {
  if (groupCount <= 1) return event.summary;
  return `${event.summary} Repeated recently.`;
}

function groupDuplicateEvents(events: PresentedEvent[]): PresentedEvent[] {
  const groupedEvents: PresentedEvent[] = [];
  const groupIndexByKey = new Map<string, number>();

  for (const event of events) {
    const key = `${rawEventType(event)}::${event.severity}`;
    const existingIndex = groupIndexByKey.get(key);

    if (existingIndex === undefined) {
      groupIndexByKey.set(key, groupedEvents.length);
      groupedEvents.push({ ...event, groupCount: 1 });
      continue;
    }

    const existingEvent = groupedEvents[existingIndex];
    const nextGroupCount = (existingEvent.groupCount ?? 1) + 1;
    groupedEvents[existingIndex] = {
      ...existingEvent,
      groupCount: nextGroupCount,
      summary: groupedSummary(event, nextGroupCount)
    };
  }

  return groupedEvents;
}

export function OpsEventsTab() {
  const eventsState = useEvents();
  const [searchText, setSearchText] = useState("");
  const [severity, setSeverity] = useState<"all" | EventSeverity>("all");
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [visibleCount, setVisibleCount] = useState(5);

  const presentedEvents = useMemo(() => eventsState.events.map((event, index) => presentEvent(event, index)), [eventsState.events]);
  const hasRealEvents = presentedEvents.length > 0;
  const canShowActivityFeed = !eventsState.loading && (!eventsState.error || hasRealEvents);

  const filteredEvents = useMemo(() => {
    const sourceEvents = hasRealEvents ? presentedEvents : PREVIEW_EVENTS;
    const needle = searchText.trim().toLowerCase();
    return sourceEvents.filter((event) => {
      const matchesSeverity = severity === "all" || event.severity === severity;
      const blob = `${event.title} ${event.summary} ${event.activityLabel} ${event.statusLabel}`.toLowerCase();
      const matchesSearch = needle === "" || blob.includes(needle);
      return matchesSeverity && matchesSearch;
    });
  }, [hasRealEvents, presentedEvents, searchText, severity]);

  useEffect(() => {
    setVisibleCount(5);
  }, [searchText, severity, hasRealEvents]);

  const groupedEvents = useMemo(() => groupDuplicateEvents(filteredEvents), [filteredEvents]);
  const visibleEvents = groupedEvents.slice(0, visibleCount);
  const selectedEvent = useMemo(() => visibleEvents.find((event) => event.id === selectedEventId) ?? null, [visibleEvents, selectedEventId]);

  return (
    <div className="activity-log-layout">
      <section className="panel activity-log-header">
        <div className="activity-log-title-row">
          <div>
            <h3>Activity Log</h3>
            <p className="muted">Recent operational activity from forecasts, training, models, and system health.</p>
            <p className="muted">Last updated: {eventsState.lastUpdatedLabel ?? "Not yet refreshed"}</p>
          </div>
          <button
            className="secondary-button"
            type="button"
            disabled={eventsState.loading || eventsState.refreshing}
            onClick={() => void eventsState.refresh({ force: true })}
          >
            {eventsState.refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
        <div className="activity-toolbar">
          <input className="activity-search" aria-label="Search activity" placeholder="Search activity..." value={searchText} onChange={(e) => setSearchText(e.target.value)} />
          <select className="activity-select" value={severity} onChange={(e) => setSeverity(e.target.value as "all" | EventSeverity)}>
            <option value="all">All status</option><option value="success">Success</option><option value="warning">Warning</option><option value="error">Error</option>
          </select>
        </div>
      </section>

      {eventsState.loading ? <section className="panel muted">Loading activity...</section> : null}
      {eventsState.error ? <section className="panel muted">Unable to load activity: {eventsState.error}</section> : null}
      {!eventsState.loading && !eventsState.error && hasRealEvents && filteredEvents.length === 0 ? <section className="panel muted">No activity matches the current filters.</section> : null}
      {canShowActivityFeed ? (
        <ActivityFeed
          events={visibleEvents}
          selectedEventId={selectedEventId}
          onSelect={setSelectedEventId}
          isPreview={!hasRealEvents}
          filteredCount={groupedEvents.length}
          visibleCount={visibleCount}
          onViewMore={() => setVisibleCount((count) => count + 5)}
        />
      ) : null}
      {selectedEvent ? <EventDetailDrawer event={selectedEvent} /> : null}
    </div>
  );
}
