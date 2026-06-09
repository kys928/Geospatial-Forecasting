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
  candidate_approved_by_operator: "Model candidate approved",
  candidate_rejected: "Model candidate rejected",
  candidate_rejected_by_operator: "Model candidate rejected",
  model_activated: "Model activated",
  rollback_completed: "Rollback completed",
  rollback_performed: "Rollback performed",
  adaptation_candidate_auto_activated: "Adaptation candidate auto-activated",
  adaptation_candidate_manual_review_required: "Adaptation candidate needs review",
  adaptation_candidate_rejected: "Adaptation candidate rejected",
  adaptation_checkpoint_file_deleted: "Checkpoint file deleted",
  automatic_retraining_skipped_cooldown: "Automatic training skipped",
  automatic_retraining_job_enqueued: "Automatic training queued",
  retraining_job_submitted: "Retraining job submitted",
  retraining_job_started: "Retraining job started",
  retraining_job_completed: "Retraining job completed",
  retraining_job_failed: "Retraining job failed",
  retraining_job_cancelled: "Retraining job cancelled",
  retraining_stop_requested: "Training stop requested",
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

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sentenceCase(value: string): string {
  const cleaned = value.replace(/[_-]+/g, " ").trim();
  return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : "Operational event";
}

function hasMeaningfulValue(value: unknown): boolean {
  return value !== undefined && value !== null && !(typeof value === "string" && value.trim() === "");
}

function eventField(event: EventRecord, payload: Record<string, unknown>, key: string): unknown {
  const payloadValue = payload[key];
  return hasMeaningfulValue(payloadValue) ? payloadValue : event[key];
}

function eventString(event: EventRecord, payload: Record<string, unknown>, key: string): string | null {
  return asString(eventField(event, payload, key));
}

function eventNumber(event: EventRecord, payload: Record<string, unknown>, key: string): number | null {
  return asNumber(eventField(event, payload, key));
}

function modelLabel(event: EventRecord, payload: Record<string, unknown>): string | null {
  return (
    eventString(event, payload, "candidate_model_id") ??
    eventString(event, payload, "model_id") ??
    eventString(event, payload, "parent_active_model_id")
  );
}

function cooldownReasonLabel(value: string | null): string {
  if (!value) return "cooldown is active";
  const cleaned = value.replace(/[_-]+/g, " ");
  if (cleaned === "cooldown active") return "cooldown is active";
  if (/active/.test(cleaned)) return cleaned;
  if (/cooldown/.test(cleaned)) return `${cleaned} is active`;
  return cleaned;
}

function gateSummary(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const gate = value as Record<string, unknown>;
  if (gate.stage3_rejected_by_gates === true) return "Stage 3 was rejected by promotion gates.";
  if (gate.gates_enabled === true) return "Promotion gates passed.";
  return null;
}

function inferCategory(eventType: string, event: EventRecord, payload: Record<string, unknown>): EventCategory {
  const eventTypeBlob = eventType.toLowerCase();
  if (/(worker|heartbeat)/.test(eventTypeBlob)) return "worker";
  if (/(forecast)/.test(eventTypeBlob)) return "forecast";
  if (/(retraining|training|job|cooldown|stop_requested)/.test(eventTypeBlob)) return "training";
  if (/(model|candidate|registry|rollback|activate|checkpoint|promotion)/.test(eventTypeBlob)) return "registry";

  const blob = `${JSON.stringify(payload)} ${JSON.stringify(event)}`.toLowerCase();
  if (/(worker|heartbeat)/.test(blob)) return "worker";
  if (/(forecast)/.test(blob)) return "forecast";
  if (/(retraining|training|job|cooldown)/.test(blob)) return "training";
  if (/(model|candidate|registry|rollback|activate|checkpoint|promotion)/.test(blob)) return "registry";
  if (/(system|ops|health)/.test(blob)) return "system";
  return "unknown";
}

function inferSeverity(eventType: string, event: EventRecord, payload: Record<string, unknown>): EventSeverity {
  const blob = `${eventType} ${JSON.stringify(payload)} ${JSON.stringify(event)}`.toLowerCase();
  if (/(fail|failed|error|exception)/.test(blob)) return "error";
  if (/(manual_review|required|cooldown|skipped|rejected|cancelled|stop_requested|deleted|warning|high validation loss)/.test(blob)) return "warning";
  if (/(completed|approved|activated|auto_activated|enqueued|queued|succeeded|success)/.test(blob)) return "success";
  return "info";
}

function buildSummary(eventType: string, event: EventRecord, payload: Record<string, unknown>): string {
  const candidateId = modelLabel(event, payload);
  const previousActiveModelId = eventString(event, payload, "previous_active_model_id");
  const parentActiveModelId = eventString(event, payload, "parent_active_model_id");
  const actor = eventString(event, payload, "actor");
  const comment = eventString(event, payload, "comment");
  const jobId = eventString(event, payload, "job_id");
  const workerId = eventString(event, payload, "worker_id");
  const forecastId = eventString(event, payload, "forecast_id");
  const reason = eventString(event, payload, "reason") ?? eventString(event, payload, "cooldown_reason");
  const errorMessage = eventString(event, payload, "error_message") ?? reason;
  const cooldownScope = eventString(event, payload, "cooldown_scope");
  const remainingSeconds =
    eventNumber(event, payload, "remaining_seconds") ?? eventNumber(event, payload, "cooldown_remaining_seconds");
  const fallbackMessage = eventString(event, payload, "message");
  const selectionGateOutcome = eventField(event, payload, "selection_gate_outcome");
  const gateMessage = gateSummary(selectionGateOutcome);

  if (eventType === "model_activated" && candidateId) {
    return `Model ${candidateId} was activated${previousActiveModelId ? `; previous active model was ${previousActiveModelId}` : ""}.`;
  }
  if (eventType === "rollback_performed" || eventType === "rollback_completed") {
    return candidateId
      ? `Rollback restored model ${candidateId}${previousActiveModelId ? ` from ${previousActiveModelId}` : ""}.`
      : "Rollback was performed.";
  }
  if ((eventType === "candidate_approved" || eventType === "candidate_approved_by_operator") && candidateId) {
    return `Candidate ${candidateId} was approved by ${actor ?? "operator"}${comment ? `: ${comment}` : ""}.`;
  }
  if ((eventType === "candidate_rejected" || eventType === "candidate_rejected_by_operator") && candidateId) {
    return `Candidate ${candidateId} was rejected${actor ? ` by ${actor}` : ""}${reason ? `: ${reason}` : ""}.`;
  }
  if (eventType === "adaptation_candidate_auto_activated" && candidateId) {
    return `Adaptation candidate ${candidateId} was auto-activated${previousActiveModelId ? `; previous active model was ${previousActiveModelId}` : ""}.`;
  }
  if (eventType === "adaptation_candidate_manual_review_required" && candidateId) {
    const gateSuffix = gateMessage ? ` ${gateMessage}` : ".";
    return `Candidate ${candidateId} needs manual review after promotion policy evaluation${gateSuffix}`;
  }
  if (eventType === "adaptation_candidate_rejected" && candidateId) {
    return `Adaptation candidate ${candidateId} was rejected${reason ? `: ${reason}` : ""}.`;
  }
  if (eventType === "adaptation_checkpoint_file_deleted" && candidateId) {
    return `Checkpoint file for ${candidateId} was deleted; registry metadata was preserved.`;
  }
  if (eventType === "automatic_retraining_skipped_cooldown") {
    const scope = cooldownScope ? cooldownScope.replace(/[_-]+/g, " ") : null;
    const reasonLabel = cooldownReasonLabel(reason);
    const cooldownLabel = scope && !reasonLabel.includes(scope) ? `${scope} ${reasonLabel}` : reasonLabel;
    const remaining = remainingSeconds !== null ? ` (${remainingSeconds} seconds remaining)` : "";
    return `Automatic training skipped because ${cooldownLabel}${remaining}.`;
  }
  if (eventType === "automatic_retraining_job_enqueued") {
    return `Automatic training queued${jobId ? ` as job ${jobId}` : ""}${parentActiveModelId ? ` from active model ${parentActiveModelId}` : ""}.`;
  }
  if (eventType === "retraining_stop_requested") return `Training stop requested${actor ? ` by ${actor}` : ""}${jobId ? ` for job ${jobId}` : ""}.`;
  if (eventType === "retraining_job_cancelled" && jobId) return `Retraining job ${jobId} was cancelled${reason ? `: ${reason}` : ""}.`;
  if (eventType === "retraining_job_submitted" && jobId) return `Retraining job ${jobId} was submitted.`;
  if (eventType === "retraining_job_started" && jobId) return `Retraining job ${jobId} started.`;
  if (eventType === "retraining_job_completed" && jobId) return `Retraining job ${jobId} completed successfully.`;
  if (eventType === "retraining_job_failed" && jobId) return `Retraining job ${jobId} failed: ${errorMessage ?? "Unknown reason"}.`;
  if (eventType === "worker_heartbeat") return `Worker ${workerId ?? "worker"} is active.`;
  if (eventType === "forecast_created") return `Forecast ${forecastId ?? "forecast"} was created.`;
  if (eventType === "forecast_failed") return `Forecast ${forecastId ?? "forecast"} failed: ${errorMessage ?? "Unknown reason"}.`;
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
  const category = inferCategory(eventType, event, payload);
  const severity = inferSeverity(eventType, event, payload);
  const eventId = asString(event.event_id);

  return {
    id: eventId ?? `${timestampRaw ?? "event"}-${index}`,
    timestampRaw,
    timeLabel,
    dateGroup,
    title: TITLE_MAP[eventType] ?? (eventType ? sentenceCase(eventType) : "Operational event"),
    summary: buildSummary(eventType, event, payload),
    category,
    severity,
    activityLabel: ACTIVITY_LABELS[category],
    statusLabel: STATUS_LABELS[severity],
    objectLabel:
      eventString(event, payload, "candidate_model_id") ??
      eventString(event, payload, "model_id") ??
      eventString(event, payload, "job_id") ??
      eventString(event, payload, "forecast_id") ??
      eventString(event, payload, "session_id") ??
      eventString(event, payload, "worker_id") ??
      eventString(event, payload, "backend_name"),
    raw: event
  };
}
