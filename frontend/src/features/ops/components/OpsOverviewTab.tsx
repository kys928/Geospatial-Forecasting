import { useOpsSystemStatus } from "../hooks/useOpsSystemStatus";

function pct(value: unknown): string {
  return typeof value === "number" ? `${Math.round(value)}%` : "Not reported";
}

function toTime(value: unknown): string {
  return typeof value === "string" && value ? value : "Not reported";
}

export function OpsOverviewTab() {
  const { status, loading, error } = useOpsSystemStatus(true, 8000);
  const host = status?.host ?? {};
  const gpu = status?.gpu ?? {};
  const jobs = status?.jobs ?? {};
  const retraining = (jobs.retraining as Record<string, unknown>) ?? {};
  const worker = status?.worker_status ?? {};

  return (
    <div className="ops-dashboard">
      <section className="panel ops-health-row">
        <h3>Health summary</h3>
        <div className="ops-chip-row">
          <StatusChip label="System health" value={status?.status_summary?.latest_warning_or_error ? "Warning" : "Healthy"} tone={status?.status_summary?.latest_warning_or_error ? "warn" : "ok"} />
          <StatusChip label="Forecast worker" value={String(worker.forecast_worker_status ?? "Not reported")} tone="neutral" />
          <StatusChip label="Retraining worker" value={String(worker.retraining_worker_status ?? "Not reported")} tone="neutral" />
          <StatusChip label="Last forecast" value={toTime(worker.last_forecast_at)} tone="neutral" />
          <StatusChip label="Active model" value={String((status?.status_summary?.active_model as Record<string, unknown> | undefined)?.model_id ?? "Not reported")} tone="neutral" />
        </div>
      </section>

      <section className="panel">
        <h3>Resource usage</h3>
        {loading && !status ? <p className="muted">Loading system metrics…</p> : null}
        {error ? <p className="muted">{error}</p> : null}
        <div className="ops-metric-grid">
          <MetricCard label="CPU" value={pct(host.cpu_percent)} percent={typeof host.cpu_percent === "number" ? host.cpu_percent : 0} />
          <MetricCard label="Memory" value={pct(host.memory_percent)} percent={typeof host.memory_percent === "number" ? host.memory_percent : 0} />
          <MetricCard label="Disk" value={pct(host.disk_percent)} percent={typeof host.disk_percent === "number" ? host.disk_percent : 0} />
          <MetricCard label="GPU" value={gpu.available ? pct(gpu.utilization_percent) : String(gpu.reason ?? "Not available")} percent={typeof gpu.utilization_percent === "number" ? gpu.utilization_percent : 0} />
          <MetricCard label="GPU VRAM" value={gpu.available && typeof gpu.memory_used_mib === "number" && typeof gpu.memory_total_mib === "number" ? `${Math.round((gpu.memory_used_mib / gpu.memory_total_mib) * 100)}%` : "Not reported"} percent={gpu.available && typeof gpu.memory_used_mib === "number" && typeof gpu.memory_total_mib === "number" ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100 : 0} />
          <article className="ops-stat-card"><strong>Uptime</strong><p>{typeof host.uptime_seconds === "number" ? `${Math.floor(host.uptime_seconds / 3600)}h` : "Not reported"}</p><p className="muted">Processes: {String(host.process_count ?? "Not reported")}</p></article>
          <article className="ops-stat-card"><strong>GPU details</strong><p>Temp: {String(gpu.temperature_c ?? "Not reported")}</p><p className="muted">Power: {String(gpu.power_w ?? "Not reported")} · Driver: {String(gpu.driver_version ?? "Not reported")} · CUDA: {String(gpu.cuda_version ?? "Not reported")}</p></article>
        </div>
      </section>

      <section className="panel ops-service-grid">
        <div>
          <h3>Service and worker status</h3>
          <p className="muted">Worker heartbeat: {toTime(worker.updated_at)}</p>
          <p className="muted">Worker mode: {String(worker.mode ?? "Not reported")}</p>
        </div>
        <div>
          <h3>Job activity summary</h3>
          <p>Retraining queued: {String(retraining.queued ?? 0)} · running: {String(retraining.running ?? 0)} · failed: {String(retraining.failed ?? 0)}</p>
          <p className="muted">Latest retraining: {toTime((jobs.latest_retraining as Record<string, unknown> | undefined)?.completed_at ?? (jobs.latest_retraining as Record<string, unknown> | undefined)?.started_at)}</p>
        </div>
      </section>

      <section className="panel">
        <h3>Recent activity</h3>
        <div className="ops-events-list">
          {(status?.recent_events ?? []).slice(0, 6).map((evt, idx) => (
            <div key={idx} className="ops-event-row">
              <span>{toTime(evt.timestamp)}</span>
              <strong>{String(evt.event_type ?? "event")}</strong>
              <span className="muted">{String(evt.summary ?? evt.message ?? "No summary")}</span>
            </div>
          ))}
          {!status?.recent_events?.length ? <p className="muted">No recent activity reported.</p> : null}
        </div>
      </section>
    </div>
  );
}

function StatusChip({ label, value, tone }: { label: string; value: string; tone: "ok" | "warn" | "neutral" }) {
  return <div className={`ops-chip ops-chip-${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function MetricCard({ label, value, percent }: { label: string; value: string; percent: number }) {
  return <article className="ops-stat-card"><strong>{label}</strong><p>{value}</p><div className="ops-progress"><div style={{ width: `${Math.max(0, Math.min(100, percent))}%` }} /></div></article>;
}
