import { useOpsJobs } from "../hooks/useOpsJobs";
import { RetrainingRecommendationPanel } from "./RetrainingRecommendationPanel";
import { RetrainingPipelinePanel } from "./RetrainingPipelinePanel";
import { ModelCandidateContextPanel } from "./ModelCandidateContextPanel";

export function OpsTrainingTab() {
  const jobs = useOpsJobs();

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <section className="panel">
        <h3>Training and model decisions</h3>
        <p className="muted">Review retraining recommendations, queued/running jobs, candidate status, and safe next actions.</p>
      </section>

      <div className="workspace-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
        <div className="workspace-column">
          <RetrainingRecommendationPanel />
        </div>
        <div className="workspace-column">
          <RetrainingPipelinePanel jobs={jobs.jobs?.jobs ?? []} />
        </div>
        <div className="workspace-column">
          <ModelCandidateContextPanel />
        </div>
      </div>
    </div>
  );
}
