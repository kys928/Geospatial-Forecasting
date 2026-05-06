import type { EventRecord } from "../types/event.types";

export type EventSeverity = "success" | "info" | "warning" | "error";
export type EventCategory = "training" | "registry" | "worker" | "forecast" | "system" | "unknown";

export interface PresentedEvent {
  id: string;
  timestampRaw: string | null;
  timeLabel: string;
  dateGroup: string;
  title: string;
  summary: string;
  category: EventCategory;
  severity: EventSeverity;
  activityLabel: string;
  statusLabel: string;
  objectLabel: string | null;
  raw: EventRecord;
  isPreview?: boolean;
}

const TITLE_MAP: Record<string, string> = {
  candidate_approved: "Model candidate approved",
  candidate_rejected: "Model candidate rejected",
  model_activated: "Model activated",
  rollback_completed: "Rollback completed",
  retraining_job_submitted: "Retraining job submitted",
  retraining_job_started: "Retraining job started",
  retraining_job_completed: "Retraining job completed",
  retraining_job_failed: "Retraining job failed",
  worker_heartbeat: "Worker heartbeat received",
  forecast_created: "Forecast created",
  forecast_failed: "Forecast failed"
};

const ACTIVITY_LABELS: Record<EventCategory, string> = {
  training: "Training",
  registry: "Registry",
  worker: "Worker",
  forecast: "Forecast",
  system: "System",
  unknown: "Other"
};

const STATUS_LABELS: Record<EventSeverity, string> = {
  error: "Failed",
  warning: "Warning",
  success: "Completed",
  info: "Normal"
};

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function sentenceCase(value: string): string {
  const cleaned = value.replace(/[_-]+/g, " ").trim();
  return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : "Operational event";
}

function inferCategory(blob: string): EventCategory {
  if (/(retraining|training|job)/.test(blob)) return "training";
  if (/(model|candidate|registry|rollback|activate)/.test(blob)) return "registry";
  if (/(worker|heartbeat)/.test(blob)) return "worker";
  if (/(forecast)/.test(blob)) return "forecast";
  if (/(system|ops|health)/.test(blob)) return "system";
  return "unknown";
}

function inferSeverity(blob: string): EventSeverity {
  if (/(fail|failed|error|exception)/.test(blob)) return "error";
  if (/(warn|warning|rejected|high validation loss)/.test(blob)) return "warning";
  if (/(completed|approved|activated|succeeded|success)/.test(blob)) return "success";
  return "info";
}

function buildSummary(eventType: string, payload: Record<string, unknown>): string {
  const candidateId = asString(payload.candidate_model_id);
  const actor = asString(payload.actor);
  const jobId = asString(payload.job_id);
  const workerId = asString(payload.worker_id);
  const errorMessage = asString(payload.error_message) ?? asString(payload.reason);
  const fallbackMessage = asString(payload.message);

  if (eventType === "candidate_approved" && candidateId) return `Candidate ${candidateId} approved by ${actor ?? "operator"}.`;
  if (eventType === "candidate_rejected" && candidateId) return `Candidate ${candidateId} rejected${actor ? ` by ${actor}` : ""}.`;
  if (eventType === "retraining_job_completed" && jobId) return `Job ${jobId} completed successfully.`;
  if (eventType === "retraining_job_failed" && jobId) return `Job ${jobId} failed: ${errorMessage ?? "Unknown reason"}.`;
  if (eventType === "worker_heartbeat") return `Worker ${workerId ?? "worker"} is active.`;
  return fallbackMessage ? `${fallbackMessage.replace(/[.\s]*$/, "")}.` : "No summary available.";
}

export function presentEvent(event: EventRecord, index: number): PresentedEvent {
  const payload = (event.payload ?? {}) as Record<string, unknown>;
  const eventType = asString(event.event_type) ?? "";
  const timestampRaw = asString(event.timestamp);
  const parsedDate = timestampRaw ? new Date(timestampRaw) : null;
  const validDate = parsedDate && !Number.isNaN(parsedDate.getTime()) ? parsedDate : null;

  const dateGroup = validDate
    ? validDate.toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric", year: "numeric" })
    : "Unknown date";
  const timeLabel = validDate
    ? validDate.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false })
    : "Not available";
  const blob = `${eventType} ${JSON.stringify(payload).toLowerCase()}`;
  const category = inferCategory(blob);
  const severity = inferSeverity(blob);

  return {
    id: `${timestampRaw ?? "event"}-${index}`,
    timestampRaw,
    timeLabel,
    dateGroup,
    title: TITLE_MAP[eventType] ?? (eventType ? sentenceCase(eventType) : "Operational event"),
    summary: buildSummary(eventType, payload),
    category,
    severity,
    activityLabel: ACTIVITY_LABELS[category],
    statusLabel: STATUS_LABELS[severity],
    objectLabel:
      asString(payload.candidate_model_id) ??
      asString(payload.model_id) ??
      asString(payload.job_id) ??
      asString(payload.forecast_id) ??
      asString(payload.session_id) ??
      asString(payload.worker_id) ??
      asString(payload.backend_name),
    raw: event
  };
}
