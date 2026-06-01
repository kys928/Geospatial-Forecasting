import { useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import { useOpsSystemStatus } from "../hooks/useOpsSystemStatus";
import type { AdaptationBufferStatus, AdaptationReadiness } from "../types/ops.types";

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

  const cpuPercent = percent(host.cpu_percent);
  const memoryPercent = percent(host.memory_percent);
  const diskPercent = percent(host.disk_percent);
  const volumePercent = percent(host.volume_percent);
  const gpuPercent = gpu.available ? percent(gpu.utilization_percent) : null;
  const vramPercent = gpu.available ? percent(gpu.vram_percent) : null;

  const retraining = (jobs.retraining as Record<string, unknown>) ?? {};
  const [bufferStatus, setBufferStatus] = useState<AdaptationBufferStatus | null>(null);
  const [readiness, setReadiness] = useState<AdaptationReadiness | null>(null);
  const [adaptationLoading, setAdaptationLoading] = useState(true);
  const [checkNowLoading, setCheckNowLoading] = useState(false);
  const [adaptationError, setAdaptationError] = useState<string | null>(null);

  async function refreshAdaptationStatus() {
    setAdaptationLoading(true);
    setAdaptationError(null);
    try {
      const [buffer, readinessPayload] = await Promise.all([opsClient.getAdaptationBufferStatus(), opsClient.getAdaptationReadiness()]);
      setBufferStatus(buffer);
      setReadiness(readinessPayload);
    } catch (err) {
      setAdaptationError(err instanceof Error ? err.message : "Unable to load adaptation status.");
    } finally {
      setAdaptationLoading(false);
    }
  }

  async function handleCheckNow() {
    setCheckNowLoading(true);
    setAdaptationError(null);
    try {
      const readinessPayload = await opsClient.checkAdaptationNow();
      const buffer = await opsClient.getAdaptationBufferStatus();
      setReadiness(readinessPayload);
      setBufferStatus(buffer);
    } catch (err) {
      setAdaptationError(err instanceof Error ? err.message : "Unable to run adaptation readiness check.");
    } finally {
      setCheckNowLoading(false);
    }
  }

  useEffect(() => { void refreshAdaptationStatus(); }, []);

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
          <BarCard label="GPU details" percent={null} value={gpu.available ? (gpuPercent === null ? "Available" : formatPercent(gpuPercent)) : "Not reported"} detail={buildGpuDetail(gpu)} />
        </div>
      </section>

      <section className="panel">
        <h3>Workspace status</h3>
        <p>Forecast workspace is available.</p>
        <p>Training worker is {String(worker.retraining_worker_status ?? "not reported").toLowerCase()}.</p>
        <p>{jobSummary(retraining)}</p>
        <details>
          <summary>Technical worker/job details</summary>
          <div className="ops-service-grid" style={{ marginTop: 10 }}>
            <div>
              <p>{workerSummary(worker)}</p>
              <p className="muted">Forecast worker: {String(worker.forecast_worker_status ?? "Not reported")}</p>
              <p className="muted">Retraining worker: {String(worker.retraining_worker_status ?? "Not reported")}</p>
            </div>
            <div>
              <p>Queued: {String(retraining.queued ?? "Not reported")}</p>
              <p>Running: {String(retraining.running ?? "Not reported")}</p>
              <p>Failed: {String(retraining.failed ?? "Not reported")}</p>
            </div>
          </div>
        </details>
      </section>

      <section className="panel">
        <div className="button-row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ margin: 0 }}>Adaptation readiness</h3>
          <button className="secondary-button" onClick={() => void handleCheckNow()} disabled={checkNowLoading}>
            {checkNowLoading ? "Checking..." : "Check now"}
          </button>
        </div>
        <p className="muted" style={{ marginTop: 6 }}>Runs readiness checks only. It does not start training.</p>
        {adaptationLoading && !readiness ? <p className="muted">Loading adaptation status...</p> : null}
        {adaptationError ? <p className="failure-text">{adaptationError}</p> : null}
        <div className="ops-service-grid" style={{ marginTop: 10 }}>
          <div>
            <div className={`ops-readiness-item ${readinessClass(readiness)}`} style={{ marginBottom: 10 }}>
              <span className="ops-readiness-dot" />
              <div>
                <strong>{readinessLabel(readiness)}</strong>
                <p className="muted" style={{ margin: 0 }}>{readiness?.status ?? "Readiness not reported"}</p>
              </div>
            </div>
            {(readiness?.blocking_reasons ?? []).slice(0, 3).map((reason) => <p key={reason} className="muted" style={{ margin: "4px 0" }}>Blocked: {reason}</p>)}
            {(readiness?.warnings ?? []).slice(0, 2).map((warning) => <p key={warning} className="muted" style={{ margin: "4px 0" }}>Warning: {warning}</p>)}
            {readiness?.next_retry_at ? <p className="muted">Next retry: {readiness.next_retry_at}</p> : null}
            {readiness?.checks?.length ? <details><summary>View readiness checklist</summary><div style={{ marginTop: 8 }}>{readiness.checks.slice(0, 8).map((check, index) => <div key={`${check.name ?? check.label ?? index}`} className={`ops-readiness-item ${checkClass(check)}`}><span className="ops-readiness-dot" /><div><strong>{String(check.label ?? check.name ?? `Check ${index + 1}`)}</strong><p className="muted" style={{ margin: 0 }}>{String(check.message ?? check.reason ?? check.status ?? "Not reported")}</p></div></div>)}</div></details> : null}
          </div>
          <div>
            <div className="ops-metric-grid">
              <MiniMetric label="Pending" value={bufferStatus?.pending} />
              <MiniMetric label="Accepted train" value={bufferStatus?.accepted_train} />
              <MiniMetric label="Accepted val" value={bufferStatus?.accepted_val} />
              <MiniMetric label="Fresh accepted" value={bufferStatus?.fresh_accepted_total} />
              <MiniMetric label="Rejected" value={bufferStatus?.rejected} />
              <MiniMetric label="Reserve used" value={bufferStatus?.reserve_used} />
            </div>
            <p className="muted" style={{ marginBottom: 0 }}>Manifest: {bufferStatus ? (bufferStatus.manifest_readable ? "Readable" : "Not readable") : "Not reported"}</p>
            {bufferStatus?.latest_event_timestamp ? <p className="muted" style={{ marginTop: 4 }}>Latest event: {bufferStatus.latest_event_timestamp}</p> : null}
            {(bufferStatus?.warnings ?? []).slice(0, 2).map((warning) => <p key={warning} className="muted" style={{ margin: "4px 0" }}>Buffer warning: {warning}</p>)}
          </div>
        </div>
      </section>
    </div>
  );
}

function buildGpuDetail(gpu: Record<string, unknown>): string {
  if (!gpu.available) return String(gpu.reason ?? "Not reported");
  const detail: string[] = [];
  if (typeof gpu.driver_version === "string" && gpu.driver_version) detail.push(`Driver: ${gpu.driver_version}`);
  if (typeof gpu.cuda_version === "string" && gpu.cuda_version) detail.push(`CUDA: ${gpu.cuda_version}`);
  return detail.length ? detail.join(" · ") : "Driver/CUDA details unavailable";
}

function GaugeCard({ label, percent, value, detail }: { label: string; percent: number | null; value: string; detail: string }) {
  const fill = percent ?? 0;
  return <article className="ops-gauge-card"><strong>{label}</strong><div className="ops-gauge" style={{ background: `conic-gradient(#4f6fa8 ${fill * 3.6}deg, #e5ebf6 0deg)` }}><div><span>{value}</span></div></div><p className="muted">{detail}</p></article>;
}

function BarCard({ label, percent, value, detail }: { label: string; percent: number | null; value: string; detail: string }) {
  return <article className="ops-stat-card"><strong>{label}</strong><p>{value}</p><div className="ops-progress"><div style={{ width: `${percent ?? 0}%` }} /></div><p className="muted">{detail}</p></article>;
}


function jobSummary(retraining: Record<string, unknown>): string {
  const queued = Number(retraining.queued ?? 0);
  const running = Number(retraining.running ?? 0);
  if (Number.isFinite(queued) && Number.isFinite(running) && (queued > 0 || running > 0)) {
    return `Retraining jobs: ${queued} queued, ${running} running.`;
  }
  if (retraining.queued !== undefined || retraining.running !== undefined) {
    return "No queued or running retraining jobs are reported.";
  }
  return "Retraining job state is not reported.";
}

function readinessClass(readiness: AdaptationReadiness | null): string {
  if (!readiness) return "ops-readiness-unknown";
  const status = String(readiness.status ?? "").toLowerCase();
  if (readiness.ready || status === "green") return "ops-readiness-met";
  if (status === "red") return "ops-readiness-not_met";
  return "ops-readiness-unknown";
}

function readinessLabel(readiness: AdaptationReadiness | null): string {
  if (!readiness) return "Adaptation readiness not reported";
  const status = String(readiness.status ?? "").toLowerCase();
  if (readiness.ready || status === "green") return "Ready";
  if (status === "yellow") return "Waiting";
  if (status === "red") return "Blocked";
  return "Adaptation readiness not reported";
}

function checkClass(check: Record<string, unknown>): string {
  const status = String(check.status ?? "").toLowerCase();
  if (check.ready === true || check.passed === true || status === "green" || status === "ready" || status === "passed") return "ops-readiness-met";
  if (check.blocking === true || check.ready === false || check.passed === false || status === "red" || status === "blocked" || status === "failed") return "ops-readiness-not_met";
  return "ops-readiness-unknown";
}

function MiniMetric({ label, value }: { label: string; value: number | undefined }) {
  return <article className="ops-stat-card"><p className="muted" style={{ margin: 0 }}>{label}</p><strong>{value ?? "Not reported"}</strong></article>;
}
