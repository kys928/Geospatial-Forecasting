import { useOpsSystemStatus } from "../hooks/useOpsSystemStatus";

function percent(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
}

function formatPercent(value: number | null): string {
  return value === null ? "Not reported" : `${Math.round(value)}%`;
}

function formatTime(value: unknown): string {
  return typeof value === "string" && value ? value : "Not reported";
}

function formatBytesGiB(value: unknown): string {
  return typeof value === "number" ? `${(value / 1024 ** 3).toFixed(2)} GiB` : "Not reported";
}

function workerSummary(worker: Record<string, unknown>): string {
  const forecast = worker.forecast_worker_status;
  const retraining = worker.retraining_worker_status;
  const mode = worker.mode;
  if (!forecast && !retraining && !mode) return "No worker heartbeat reported.";
  if (mode === "all") return "1 worker process running, handling forecast and retraining jobs.";
  const running = [forecast, retraining].filter((v) => String(v ?? "").toLowerCase() === "running").length;
  if (running > 0) return `Worker processes: ${running} running.`;
  return "Worker heartbeat received, no running worker reported.";
}

export function OpsOverviewTab() {
  const { status, loading, error } = useOpsSystemStatus(true, 5000);
  const host = status?.host ?? {};
  const gpu = status?.gpu ?? {};
  const jobs = status?.jobs ?? {};
  const retraining = (jobs.retraining as Record<string, unknown>) ?? {};
  const worker = status?.worker_status ?? {};
  const memoryPercent = percent(host.memory_percent);
  const diskPercent = percent(host.disk_percent);
  const cpuPercent = percent(host.cpu_percent);
  const gpuPercent = gpu.available ? percent(gpu.utilization_percent) : null;
  const vramPercent =
    gpu.available && typeof gpu.memory_used_mib === "number" && typeof gpu.memory_total_mib === "number" && gpu.memory_total_mib > 0
      ? percent((gpu.memory_used_mib / gpu.memory_total_mib) * 100)
      : null;

  return (
    <div className="ops-dashboard">
      <section className="panel ops-health-row">
        <h3>System health</h3>
        <div className="ops-chip-row">
          <StatusChip label="System health" value={status?.status_summary?.latest_warning_or_error ? "Warning" : "Healthy"} tone={status?.status_summary?.latest_warning_or_error ? "warn" : "ok"} />
          <StatusChip label="Worker processes" value={workerSummary(worker)} tone="neutral" />
          <StatusChip label="Last forecast" value={formatTime(worker.last_forecast_at)} tone="neutral" />
          <StatusChip label="Last training" value={formatTime((jobs.latest_retraining as Record<string, unknown> | undefined)?.completed_at)} tone="neutral" />
        </div>
      </section>

      <section className="panel">
        <h3>Resource usage</h3>
        {loading && !status ? <p className="muted">Loading system metrics...</p> : null}
        {error ? <p className="muted">{error}</p> : null}
        <div className="ops-gauge-grid">
          <GaugeCard label="CPU usage" value={formatPercent(cpuPercent)} detail={typeof host.cpu_model === "string" ? host.cpu_model : "Not reported"} percent={cpuPercent} />
          <GaugeCard label="Memory usage" value={formatPercent(memoryPercent)} detail={`${formatBytesGiB(host.memory_used_bytes)} / ${formatBytesGiB(host.memory_total_bytes)}`} percent={memoryPercent} />
          <GaugeCard label="GPU usage" value={gpu.available ? formatPercent(gpuPercent) : "Not reported"} detail={typeof gpu.name === "string" ? gpu.name : String(gpu.reason ?? "Not reported")} percent={gpuPercent} />
          <GaugeCard label="GPU VRAM" value={gpu.available ? formatPercent(vramPercent) : "Not reported"} detail={typeof gpu.memory_used_mib === "number" && typeof gpu.memory_total_mib === "number" ? `${Math.round(gpu.memory_used_mib)} MiB / ${(gpu.memory_total_mib / 1024).toFixed(1)} GiB` : "Not reported"} percent={vramPercent} />
        </div>
        <div className="ops-bar-grid">
          <BarCard label="Disk usage" percent={diskPercent} value={formatPercent(diskPercent)} detail={`${formatBytesGiB(host.disk_used_bytes)} / ${formatBytesGiB(host.disk_total_bytes)}`} />
          <BarCard label="Uptime" percent={null} value={typeof host.uptime_seconds === "number" ? `${Math.floor(host.uptime_seconds / 3600)}h` : "Not reported"} detail={`Processes: ${String(host.process_count ?? "Not reported")}`} />
          <BarCard label="Volume usage" percent={null} value="Not reported" detail="No additional volume metrics reported" />
          <BarCard label="GPU details" percent={null} value={`Temp: ${String(gpu.temperature_c ?? "Not reported")}`} detail={`Power: ${String(gpu.power_w ?? "Not reported")} W, Driver: ${String(gpu.driver_version ?? "Not reported")}, CUDA: ${String(gpu.cuda_version ?? "Not reported")}`} />
        </div>
      </section>

      <section className="panel ops-service-grid">
        <div>
          <h3>Worker processes</h3>
          <p>{workerSummary(worker)}</p>
          <p className="muted">Handles: {worker.mode === "all" ? "forecast and retraining" : "Not reported"}</p>
        </div>
        <div>
          <h3>Training jobs</h3>
          <p>Queued: {String(retraining.queued ?? 0)} | Running: {String(retraining.running ?? 0)} | Failed: {String(retraining.failed ?? 0)}</p>
        </div>
      </section>
    </div>
  );
}

function StatusChip({ label, value, tone }: { label: string; value: string; tone: "ok" | "warn" | "neutral" }) { return <div className={`ops-chip ops-chip-${tone}`}><span>{label}</span><strong>{value}</strong></div>; }

function GaugeCard({ label, value, detail, percent }: { label: string; value: string; detail: string; percent: number | null }) {
  const fill = percent === null ? 0 : percent;
  return <article className="ops-gauge-card"><strong>{label}</strong><div className="ops-gauge" style={{ background: `conic-gradient(#4f6fa8 ${fill * 3.6}deg, #e5ebf6 0deg)` }}><div><span>{value}</span></div></div><p className="muted">{detail}</p></article>;
}

function BarCard({ label, percent, value, detail }: { label: string; percent: number | null; value: string; detail: string }) { return <article className="ops-stat-card"><strong>{label}</strong><p>{value}</p><div className="ops-progress"><div style={{ width: `${percent ?? 0}%` }} /></div><p className="muted">{detail}</p></article>; }
