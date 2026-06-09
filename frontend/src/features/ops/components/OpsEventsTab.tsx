import { useEffect, useMemo, useState } from "react";
import { ActivityFeed } from "../../events/components/ActivityFeed";
import { EventDetailDrawer } from "../../events/components/EventDetailDrawer";
import { useEvents } from "../../events/hooks/useEvents";
import { presentEvent, type EventCategory, type EventSeverity, type PresentedEvent } from "../../events/utils/presentEvent";
import type { EventRecord } from "../../events/types/event.types";

const PREVIEW_EVENTS: PresentedEvent[] = [
  {
    id: "preview-1",
    title: "Model candidate approved",
    summary: "Candidate convlstm_2026_05_06 was approved and is ready for activation.",
    activityLabel: "Registry",
    statusLabel: "Completed",
    category: "registry",
    severity: "success",
    timeLabel: "11:42",
    timestampRaw: null,
    dateGroup: "Preview examples",
    objectLabel: "convlstm_2026_05_06",
    isPreview: true,
    raw: { event_type: "candidate_approved",  payload: { message: "Preview example; no backend payload available." } } as EventRecord
  },
  { id: "preview-2", title: "Retraining job completed", summary: "Job job_184 finished successfully and produced a candidate model.", activityLabel: "Training", statusLabel: "Completed", category: "training", severity: "success", timeLabel: "11:18", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "job_184", isPreview: true, raw: { event_type: "retraining_job_completed",  payload: { message: "Preview example; no backend payload available." } } as EventRecord },
  { id: "preview-3", title: "Worker heartbeat received", summary: "Retraining worker is active and reporting status.", activityLabel: "Worker", statusLabel: "Normal", category: "worker", severity: "info", timeLabel: "10:45", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "worker", isPreview: true, raw: { event_type: "worker_heartbeat",  payload: { message: "Preview example; no backend payload available." } } as EventRecord },
  { id: "preview-4", title: "Forecast created", summary: "Forecast forecast_2026_05_06_1022 was generated successfully.", activityLabel: "Forecast", statusLabel: "Completed", category: "forecast", severity: "success", timeLabel: "10:22", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "forecast_2026_05_06_1022", isPreview: true, raw: { event_type: "forecast_created",  payload: { message: "Preview example; no backend payload available." } } as EventRecord },
  { id: "preview-5", title: "Retraining job failed", summary: "Job job_183 failed because the worker ran out of memory.", activityLabel: "Training", statusLabel: "Failed", category: "training", severity: "error", timeLabel: "Yesterday", timestampRaw: null, dateGroup: "Preview examples", objectLabel: "job_183", isPreview: true, raw: { event_type: "retraining_job_failed",  payload: { message: "Preview example; no backend payload available." } } as EventRecord }
];

export function OpsEventsTab() {
  const eventsState = useEvents();
  const [searchText, setSearchText] = useState("");
  const [category, setCategory] = useState<"all" | EventCategory>("all");
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
      const matchesCategory = category === "all" || event.category === category;
      const matchesSeverity = severity === "all" || event.severity === severity;
      const blob = `${event.title} ${event.summary} ${event.activityLabel} ${event.statusLabel}`.toLowerCase();
      const matchesSearch = needle === "" || blob.includes(needle);
      return matchesCategory && matchesSeverity && matchesSearch;
    });
  }, [category, hasRealEvents, presentedEvents, searchText, severity]);

  useEffect(() => {
    setVisibleCount(5);
  }, [searchText, category, severity, hasRealEvents]);

  const visibleEvents = filteredEvents.slice(0, visibleCount);
  const selectedEvent = useMemo(() => visibleEvents.find((event) => event.id === selectedEventId) ?? null, [visibleEvents, selectedEventId]);

  return (
    <div className="activity-log-layout">
      <section className="panel activity-log-header">
        <div className="activity-log-title-row">
          <div>
            <h3>Activity Log</h3>
            <p className="muted">Recent operational activity from training, model registry, workers, and forecasts.</p>
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
          <select className="activity-select" value={category} onChange={(e) => setCategory(e.target.value as "all" | EventCategory)}>
            <option value="all">All activity</option><option value="training">Training</option><option value="registry">Registry</option><option value="worker">Worker</option><option value="forecast">Forecast</option><option value="system">System</option><option value="unknown">Other</option>
          </select>
          <select className="activity-select" value={severity} onChange={(e) => setSeverity(e.target.value as "all" | EventSeverity)}>
            <option value="all">All status</option><option value="info">Normal</option><option value="success">Completed</option><option value="warning">Warning</option><option value="error">Failed</option>
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
          filteredCount={filteredEvents.length}
          visibleCount={visibleCount}
          onViewMore={() => setVisibleCount((count) => count + 5)}
        />
      ) : null}
      {selectedEvent ? <EventDetailDrawer event={selectedEvent} /> : null}
    </div>
  );
}
