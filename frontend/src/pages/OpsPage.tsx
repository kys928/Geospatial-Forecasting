import { useMemo, useState } from "react";
import { AppShell } from "../app/AppShell";
import { useOpsStatus } from "../features/ops/hooks/useOpsStatus";
import { useOpsJobs } from "../features/ops/hooks/useOpsJobs";
import { useOpsActions } from "../features/ops/hooks/useOpsActions";
import { useOpsAvailability } from "../features/ops/hooks/useOpsAvailability";
import { OpsControlTower } from "../features/ops/components/OpsControlTower";
import { OpsWarningsPanel } from "../features/ops/components/OpsWarningsPanel";
import { RetrainingPipelinePanel } from "../features/ops/components/RetrainingPipelinePanel";
import { RetrainingTriggerForm } from "../features/ops/components/RetrainingTriggerForm";
import { ApprovalGatePanel } from "../features/ops/components/ApprovalGatePanel";
import { OpsTabs, type OpsTabKey } from "../features/ops/components/OpsTabs";
import { OpsUnavailablePanel } from "../features/ops/components/OpsUnavailablePanel";
import { OpsRegistryPanel } from "../features/ops/components/OpsRegistryPanel";
import { OpsEventsPanel } from "../features/ops/components/OpsEventsPanel";

export function OpsPage() {
  const [selectedTab, setSelectedTab] = useState<OpsTabKey>("overview");
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);
  const [lastActionName, setLastActionName] = useState<string | null>(null);
  const { opsAvailable, reason } = useOpsAvailability();

  const status = useOpsStatus(opsAvailable);
  const jobs = useOpsJobs(opsAvailable);

  async function refreshAll() {
    await Promise.all([status.refresh(), jobs.refresh()]);
  }

  const actions = useOpsActions(opsAvailable ? refreshAll : undefined);

  const pendingCandidateId =
    (typeof status.status?.pending_candidate?.model_id === "string" && status.status.pending_candidate.model_id) ||
    (typeof status.status?.candidate_model?.model_id === "string" && status.status.candidate_model.model_id) ||
    null;

  const statusText = useMemo(
    () => actions.error ?? status.error ?? jobs.error ?? submitMessage ?? "Ready",
    [actions.error, status.error, jobs.error, submitMessage]
  );

  return (
    <AppShell
      title="Ops workspace"
      subtitle="Operate retraining, approvals, and promotion workflow with backend-backed actions."
      statusText={statusText}
      metaItems={[{ label: "Manual approvals" }]}
    >
      <OpsTabs selected={selectedTab} onSelect={setSelectedTab} />

      {!opsAvailable ? (
        <OpsUnavailablePanel reason={reason} />
      ) : null}

      {opsAvailable && selectedTab === "overview" ? (
        <div className="workspace-grid">
          <div className="workspace-column">
            <OpsControlTower
              status={status.status}
              loading={status.loading}
              onRefreshStatus={() => void status.refresh()}
              onRefreshJobs={() => void jobs.refresh()}
              onRefreshEvents={() => undefined}
            />
            <RetrainingTriggerForm
              disabled={actions.runningAction !== null}
              onSubmit={async (payload) => {
                setSubmitMessage(null);
                setLastActionName("trigger");
                try {
                  await actions.triggerRetraining(payload);
                  setSubmitMessage("Retraining job submitted. Refresh jobs to track execution.");
                } catch (error) {
                  setSubmitMessage(null);
                  throw error;
                }
              }}
            />
            {actions.error ? <section className="panel failure-text">Retraining submission failed: {actions.error}</section> : null}
            {submitMessage ? <section className="panel success-text">{submitMessage}</section> : null}
          </div>

          <div className="workspace-column">
            <ApprovalGatePanel
              candidateId={pendingCandidateId}
              disabled={actions.runningAction !== null}
              onApprove={async (actor, comment) => {
                if (!pendingCandidateId) return;
                setLastActionName("approve");
                await actions.approveCandidate(pendingCandidateId, actor, comment);
              }}
              onReject={async (actor, comment) => {
                if (!pendingCandidateId) return;
                setLastActionName("reject");
                await actions.rejectCandidate(pendingCandidateId, actor, comment);
              }}
            />
            <OpsWarningsPanel
              latestWarningOrError={status.status?.latest_warning_or_error ?? null}
              latestFailureReason={status.status?.last_retraining_job_failure_reason ?? null}
            />
            <RetrainingPipelinePanel jobs={jobs.jobs?.jobs ?? status.status?.current_retraining_jobs ?? []} />
          </div>

          <div className="workspace-column">
            <section className="panel">
              <h3>Action status</h3>
              {!lastActionName && !actions.error && !actions.lastResult ? <p className="muted">No recent ops action.</p> : null}
              {lastActionName ? <p><strong>Last action:</strong> {lastActionName}</p> : null}
              <p><strong>Result:</strong> {actions.error ? "Failed" : actions.lastResult ? "Success" : "Idle"}</p>
              <p>{actions.error ?? submitMessage ?? "No new action message."}</p>
              {actions.lastResult ? (
                <details>
                  <summary>Technical details</summary>
                  <pre style={{ margin: 0, maxHeight: 260, overflow: "auto" }}>{JSON.stringify(actions.lastResult, null, 2)}</pre>
                </details>
              ) : null}
            </section>
          </div>
        </div>
      ) : null}

      {opsAvailable && selectedTab === "registry" ? <OpsRegistryPanel enabled={selectedTab === "registry" && opsAvailable} /> : null}
      {opsAvailable && selectedTab === "events" ? <OpsEventsPanel enabled={selectedTab === "events" && opsAvailable} /> : null}
    </AppShell>
  );
}
