import type { OpsJobRecord } from "../types/ops.types";

interface RetrainingPipelinePanelProps {
  jobs: OpsJobRecord[];
}

export function RetrainingPipelinePanel({ jobs }: RetrainingPipelinePanelProps) {
  return (
    <section className="panel">
      <h3>Retraining jobs</h3>
      <div style={{ maxHeight: 240, overflow: "auto" }}>
        {jobs.map((job) => (
          <div key={String(job.job_id)} className="detail-list-row">
            <dt>{String(job.job_id)}</dt>
            <dd>{job.status ?? "unknown"}</dd>
          </div>
        ))}
        {jobs.length === 0 ? <p className="muted">No jobs yet.</p> : null}
      </div>
    </section>
  );
}
