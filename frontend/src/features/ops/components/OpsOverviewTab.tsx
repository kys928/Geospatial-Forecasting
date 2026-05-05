import { useOpsSystemStatus } from "../hooks/useOpsSystemStatus";

function percent(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
}

function formatPercent(value: number | null): string {
  return value === null ? "Not reported" : `${Math.round(value)}%`;
}

function formatBytesGiB(value: unknown): string {
  return typeof value === "number" ? `${(value / 1024 ** 3).toFixed(2)} GiB` : "Not reported";
}

function formatDuration(seconds: unknown): string {
  if (typeof seconds !== "number") return "Not reported";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatTimestamp(value: unknown): string {
  return typeof value === "string" && value ? value : "Not reported";
}

function workerSummary(worker: Record<string, unknown>): string {
  const mode = worker.mode;
  const forecast = String(worker.forecast_worker_status ?? "").toLowerCase();
  const retraining = String(worker.retraining_worker_status ?? "").toLowerCase();
  if (!mode && !forecast && !retraining) return "No worker heartbeat reported.";
  if (mode === "all") return "1 worker process running, handling forecast and retraining jobs.";
  if (forecast === "running" && retraining === "running") return "2 worker processes running for forecast and retraining.";
  if (forecast === "running" && retraining !== "running") return "Forecast worker running. Retraining worker not reported.";
  if (retraining === "running" && forecast !== "running") return "Retraining worker running. Forecast worker not reported.";
  return "Worker heartbeat received, but no active worker process is reported.";
}

export function OpsOverviewTab() {
  const { status, loading, error } = useOpsSystemStatus(true, 5000);
  const host = status?.host ?? {};
  const gpu = status?.gpu ?? {};
  const worker = status?.worker_status ?? {};
  const jobs = status?.jobs ?? {};
  const recentEvents = Array.isArray(status?.recent_events) ? status?.recent_events : [];

  const cpuPercent = percent(host.cpu_percent);
  const memoryPercent = percent(host.memory_percent);
  const diskPercent = percent(host.disk_percent);
  const volumePercent = percent(host.volume_percent);
  const gpuPercent = gpu.available ? percent(gpu.utilization_percent) : null;
  const vramPercent = gpu.available ? percent(gpu.vram_percent) : null;

  const retraining = (jobs.retraining as Record<string, unknown>) ?? {};

  return (
    <div className="ops-dashboard">
      <section className="panel">
        <h3>Resource usage</h3>
        {loading && !status ? <p className="muted">Loading system metrics...</p> : null}
        {error ? <p className="muted">{error}</p> : null}
        <p className="muted">Last updated: {formatTimestamp(status?.generated_at)}</p>

        <div className="ops-gauge-grid">
          <GaugeCard label="CPU" percent={cpuPercent} value={formatPercent(cpuPercent)} detail={typeof host.cpu_model === "string" ? host.cpu_model : "Not reported"} />
          <GaugeCard label="Memory" percent={memoryPercent} value={formatPercent(memoryPercent)} detail={`${formatBytesGiB(host.memory_used_bytes)} / ${formatBytesGiB(host.memory_total_bytes)}`} />
          <GaugeCard label="GPU" percent={gpuPercent} value={gpu.available ? formatPercent(gpuPercent) : "Not reported"} detail={typeof gpu.name === "string" ? gpu.name : String(gpu.reason ?? "Not reported")} />
          <GaugeCard label="GPU VRAM" percent={vramPercent} value={gpu.available ? formatPercent(vramPercent) : "Not reported"} detail={typeof gpu.memory_used_mib === "number" && typeof gpu.memory_total_mib === "number" ? `${Math.round(gpu.memory_used_mib)} MiB / ${(gpu.memory_total_mib / 1024).toFixed(1)} GiB` : "Not reported"} />
        </div>

        <div className="ops-bar-grid">
          <BarCard label="Disk usage" percent={diskPercent} value={formatPercent(diskPercent)} detail={`${formatBytesGiB(host.disk_used_bytes)} / ${formatBytesGiB(host.disk_total_bytes)}`} />
          <BarCard label="Volume usage" percent={volumePercent} value={formatPercent(volumePercent)} detail={`${formatBytesGiB(host.volume_used_bytes)} / ${formatBytesGiB(host.volume_total_bytes)}`} />
          <BarCard label="Uptime" percent={null} value={formatDuration(host.uptime_seconds)} detail={`Processes: ${String(host.process_count ?? "Not reported")}`} />
          <BarCard label="GPU details" percent={null} value={`Temp: ${String(gpu.temperature_c ?? "Not reported")}`} detail={`Power: ${String(gpu.power_w ?? "Not reported")} W, Driver: ${String(gpu.driver_version ?? "Not reported")}, CUDA: ${String(gpu.cuda_version ?? "Not reported")}`} />
        </div>
      </section>

      <section className="panel ops-service-grid">
        <div>
          <h3>Worker status</h3>
          <p>{workerSummary(worker)}</p>
          <p className="muted">Forecast worker: {String(worker.forecast_worker_status ?? "Not reported")}</p>
          <p className="muted">Retraining worker: {String(worker.retraining_worker_status ?? "Not reported")}</p>
        </div>
        <div>
          <h3>Job activity</h3>
          <p>Queued: {String(retraining.queued ?? "Not reported")}</p>
          <p>Running: {String(retraining.running ?? "Not reported")}</p>
          <p>Failed: {String(retraining.failed ?? "Not reported")}</p>
        </div>
      </section>

      <section className="panel">
        <h3>Recent activity</h3>
        {recentEvents.length === 0 ? <p className="muted">No recent activity reported.</p> : null}
        {recentEvents.length > 0 ? (
          <div className="ops-events-list">
            {recentEvents.map((event, index) => (
              <div className="ops-event-row" key={`${String(event.timestamp ?? "event")}-${index}`}>
                <span>{formatTimestamp(event.timestamp)}</span>
                <strong>{String(event.event_type ?? "Not reported")}</strong>
                <span>{typeof event.payload === "object" && event.payload ? JSON.stringify(event.payload) : "Not reported"}</span>
              </div>
            ))}
          </div>
        ) : null}
      </section>
    </div>
  );
}

function GaugeCard({ label, percent, value, detail }: { label: string; percent: number | null; value: string; detail: string }) {
  const fill = percent ?? 0;
  return <article className="ops-gauge-card"><strong>{label}</strong><div className="ops-gauge" style={{ background: `conic-gradient(#4f6fa8 ${fill * 3.6}deg, #e5ebf6 0deg)` }}><div><span>{value}</span></div></div><p className="muted">{detail}</p></article>;
}

function BarCard({ label, percent, value, detail }: { label: string; percent: number | null; value: string; detail: string }) {
  return <article className="ops-stat-card"><strong>{label}</strong><p>{value}</p><div className="ops-progress"><div style={{ width: `${percent ?? 0}%` }} /></div><p className="muted">{detail}</p></article>;
}
