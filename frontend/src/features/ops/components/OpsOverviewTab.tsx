import { useOpsStatus } from "../hooks/useOpsStatus";
import { useOpsAvailability } from "../hooks/useOpsAvailability";
import { useOpsJobs } from "../hooks/useOpsJobs";
import { OpsWarningsPanel } from "./OpsWarningsPanel";
import { OpsControlTower } from "./OpsControlTower";
import { OpsEventsPanel } from "./OpsEventsPanel";

export function OpsOverviewTab() {
  const opsStatus = useOpsStatus();
  const availability = useOpsAvailability();
  const jobs = useOpsJobs();

  return (
    <>
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
          />
        </div>
        <div className="workspace-column">
          <OpsEventsPanel enabled={Boolean(availability.opsAvailable)} />
        </div>
      </div>
    </>
  );
}
