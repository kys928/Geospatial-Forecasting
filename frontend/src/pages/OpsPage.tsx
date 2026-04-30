import { useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { OpsTabs, type OpsTabKey } from "../features/ops/components/OpsTabs";
import { useOpsStatus } from "../features/ops/hooks/useOpsStatus";
import { useOpsAvailability } from "../features/ops/hooks/useOpsAvailability";
import { useOpsJobs } from "../features/ops/hooks/useOpsJobs";
import { useOpsActions } from "../features/ops/hooks/useOpsActions";
import { useRegistry } from "../features/registry/hooks/useRegistry";
import { useEvents } from "../features/events/hooks/useEvents";
import { OpsControlTower } from "../features/ops/components/OpsControlTower";
import { OpsWarningsPanel } from "../features/ops/components/OpsWarningsPanel";
import { OpsEventsPanel } from "../features/ops/components/OpsEventsPanel";
import { RegistryModelsTable } from "../features/registry/components/RegistryModelsTable";
import { ModelVersionInspector } from "../features/registry/components/ModelVersionInspector";
import { ActivationPanel } from "../features/registry/components/ActivationPanel";
import { RollbackPanel } from "../features/registry/components/RollbackPanel";
import { EventFilters } from "../features/events/components/EventFilters";
import { IncidentTimeline } from "../features/events/components/IncidentTimeline";
import { EventDetailDrawer } from "../features/events/components/EventDetailDrawer";

export function OpsPage() {
  const [tab, setTab] = useState<OpsTabKey>("overview");

  const opsStatus = useOpsStatus();
  const availability = useOpsAvailability();
  const jobs = useOpsJobs();
  const actions = useOpsActions(async () => {
    await Promise.all([opsStatus.refresh(), jobs.refresh()]);
  });

  const registryState = useRegistry();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const selectedModel = useMemo(
    () => registryState.registry?.models.find((model) => model.model_id === selectedModelId) ?? null,
    [registryState.registry, selectedModelId]
  );
  const approvedModelIds = useMemo(
    () =>
      (registryState.registry?.models ?? [])
        .filter((model) => model.approval_status === "approved" && typeof model.model_id === "string")
        .map((model) => model.model_id as string),
    [registryState.registry]
  );

  const eventsState = useEvents();
  const [selectedEventIndex, setSelectedEventIndex] = useState(0);
  const selectedEvent = useMemo(
    () => eventsState.filteredEvents[selectedEventIndex] ?? null,
    [eventsState.filteredEvents, selectedEventIndex]
  );

  return (
    <AppShell
      title="Ops workspace"
      subtitle="Operational status, retraining controls, registry, and event/audit panels."
      statusText={opsStatus.error ?? actions.error ?? registryState.error ?? eventsState.error ?? "Ready"}
      metaItems={[{ label: "Ops" }]}
    >
      <OpsTabs selected={tab} onSelect={setTab} />

      {tab === "overview" ? (
        <div className="workspace-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
          <div className="workspace-column">
            <OpsWarningsPanel latestWarningOrError={opsStatus.status?.latest_warning_or_error ?? null} latestFailureReason={opsStatus.status?.last_retraining_job_failure_reason ?? null} />
          </div>
          <div className="workspace-column">
            <OpsControlTower
              status={opsStatus.status}
              loading={opsStatus.loading}
              onRefreshStatus={() => void opsStatus.refresh()}
              onRefreshJobs={() => void jobs.refresh()}
              onRefreshEvents={() => void eventsState.refresh()}
            />
          </div>
          <div className="workspace-column">
            <OpsEventsPanel enabled={Boolean(availability.opsAvailable)} />
          </div>
        </div>
      ) : null}

      {tab === "registry" ? (
        <div className="workspace-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
          <div className="workspace-column">
            <button className="primary-button" onClick={() => void registryState.refresh()}>
              {registryState.loading ? "Refreshing..." : "Refresh registry"}
            </button>
            <ActivationPanel
              approvedModelIds={approvedModelIds}
              disabled={actions.runningAction !== null}
              onActivate={async (modelId) => {
                await actions.activateModel(modelId);
              }}
            />
            <RollbackPanel disabled={actions.runningAction !== null} onRollback={async () => {
              await actions.rollbackModel();
            }} />
          </div>
          <div className="workspace-column">
            <RegistryModelsTable
              models={registryState.registry?.models ?? []}
              selectedModelId={selectedModelId}
              onSelectModel={setSelectedModelId}
            />
          </div>
          <div className="workspace-column">
            <ModelVersionInspector model={selectedModel} />
          </div>
        </div>
      ) : null}

      {tab === "events" ? (
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
      ) : null}
    </AppShell>
  );
}
