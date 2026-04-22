import { useEffect, useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { useOpsStatus } from "../features/ops/hooks/useOpsStatus";
import { useOpsJobs } from "../features/ops/hooks/useOpsJobs";
import { useOpsActions } from "../features/ops/hooks/useOpsActions";
import { opsClient } from "../features/ops/api/opsClient";
import type { OpsEventRecord } from "../features/ops/types/ops.types";
import { OpsControlTower } from "../features/ops/components/OpsControlTower";
import { ActiveModelPanel } from "../features/ops/components/ActiveModelPanel";
import { CandidateModelPanel } from "../features/ops/components/CandidateModelPanel";
import { OpsWarningsPanel } from "../features/ops/components/OpsWarningsPanel";
import { RetrainingPipelinePanel } from "../features/ops/components/RetrainingPipelinePanel";
import { RetrainingTriggerForm } from "../features/ops/components/RetrainingTriggerForm";
import { ApprovalGatePanel } from "../features/ops/components/ApprovalGatePanel";
import { OpsEventsPreview } from "../features/ops/components/OpsEventsPreview";

export function OpsPage() {
  const status = useOpsStatus();
  const jobs = useOpsJobs();
  const [events, setEvents] = useState<OpsEventRecord[]>([]);
  const [eventsError, setEventsError] = useState<string | null>(null);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);

  async function refreshAll() {
    await Promise.all([status.refresh(), jobs.refresh(), refreshEvents()]);
  }

  const actions = useOpsActions(refreshAll);

  async function refreshEvents() {
    try {
      const response = await opsClient.getEvents(30);
      setEvents(response.events);
      setEventsError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to load ops events";
      setEventsError(message);
      console.error(err);
    }
  }

  useEffect(() => {
    void refreshEvents();
  }, []);

  const pendingCandidateId =
    (typeof status.status?.pending_candidate?.model_id === "string" && status.status.pending_candidate.model_id) ||
    (typeof status.status?.candidate_model?.model_id === "string" && status.status.candidate_model.model_id) ||
    null;

  const statusText = useMemo(
    () => eventsError ?? actions.error ?? status.error ?? jobs.error ?? submitMessage ?? "Ready",
    [eventsError, actions.error, status.error, jobs.error, submitMessage]
  );

  return (
    <AppShell
      title="Ops workspace"
      subtitle="Operate retraining, approvals, and promotion workflow with backend-backed actions."
      statusText={statusText}
      metaItems={[{ label: "Local pod testing" }, { label: "Manual approvals" }]}
    >
      <div className="workspace-grid">
        <div style={{ display: "grid", gap: 12 }}>
          <RetrainingTriggerForm
            disabled={actions.runningAction !== null}
            onSubmit={async (payload) => {
              setSubmitMessage(null);
              try {
                await actions.triggerRetraining(payload);
                setSubmitMessage("Retraining job submitted. Refresh jobs/events to track execution.");
              } catch (error) {
                setSubmitMessage(null);
                throw error;
              }
            }}
          />
          {actions.error ? <section className="panel failure-text">Retraining submission failed: {actions.error}</section> : null}
          {submitMessage ? <section className="panel success-text">{submitMessage}</section> : null}
          <section className="panel muted">
            Retraining sequence: submit job → worker executes → jobs/events update → candidate/registry state may change.
          </section>
          <ApprovalGatePanel
            candidateId={pendingCandidateId}
            disabled={actions.runningAction !== null}
            onApprove={async (actor, comment) => {
              if (!pendingCandidateId) return;
              await actions.approveCandidate(pendingCandidateId, actor, comment);
            }}
            onReject={async (actor, comment) => {
              if (!pendingCandidateId) return;
              await actions.rejectCandidate(pendingCandidateId, actor, comment);
            }}
          />
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          <OpsControlTower
            status={status.status}
            loading={status.loading}
            onRefreshStatus={() => void status.refresh()}
            onRefreshJobs={() => void jobs.refresh()}
            onRefreshEvents={() => void refreshEvents()}
          />
          <OpsWarningsPanel
            latestWarningOrError={status.status?.latest_warning_or_error ?? null}
            latestFailureReason={status.status?.last_retraining_job_failure_reason ?? null}
          />
          <RetrainingPipelinePanel jobs={jobs.jobs?.jobs ?? status.status?.current_retraining_jobs ?? []} />
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          <ActiveModelPanel activeModel={status.status?.active_model ?? null} />
          <CandidateModelPanel candidateModel={status.status?.candidate_model ?? null} />
          <OpsEventsPreview events={events} />
          {eventsError ? <section className="panel failure-text">Events refresh failed: {eventsError}</section> : null}
          <section className="panel">
            <h3>Latest action result</h3>
            <pre style={{ margin: 0, maxHeight: 360, overflow: "auto" }}>{JSON.stringify(actions.lastResult, null, 2)}</pre>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
