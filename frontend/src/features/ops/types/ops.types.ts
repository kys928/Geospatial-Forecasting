export interface OpsEventRecord {
  timestamp?: string;
  event_type?: string;
  payload?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface OpsJobRecord {
  job_id?: string;
  status?: string;
  created_sequence?: number;
  created_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  dataset_snapshot_ref?: string | null;
  run_config_ref?: string | null;
  output_dir?: string | null;
  result_run_dir?: string | null;
  error_message?: string | null;
  [key: string]: unknown;
}

export interface OpsStatusResponse {
  phase: string;
  active_model: Record<string, unknown>;
  candidate_model: Record<string, unknown>;
  retraining_readiness: Record<string, unknown>;
  last_promotion_result: Record<string, unknown> | null;
  latest_warning_or_error: string | null;
  latest_run_summary_excerpt: Record<string, unknown> | null;
  has_pending_manual_approval: boolean;
  candidate_approval_status: string | null;
  last_approval_event: Record<string, unknown> | null;
  last_approval_comment: string | null;
  current_retraining_jobs: OpsJobRecord[];
  latest_retraining_job: OpsJobRecord | null;
  retraining_job_statuses: Array<string | null>;
  last_retraining_job_failure_reason: string | null;
  pending_candidate: Record<string, unknown> | null;
}

export interface OpsJobsResponse {
  jobs: OpsJobRecord[];
  latest_job: OpsJobRecord | null;
}

export interface OpsEventsResponse {
  events: OpsEventRecord[];
}

export interface RetrainingTriggerRequest {
  manual_override: boolean;
  dataset_snapshot_ref?: string;
  run_config_ref?: string;
  output_dir?: string;
}

export interface RetrainingTriggerResponse {
  submitted: boolean;
  policy_check: Record<string, unknown>;
  job: OpsJobRecord | null;
}

export interface CandidateDecisionRequest {
  actor: string;
  comment?: string;
}

export interface ApprovalActionResponse {
  candidate_model_id: string;
  approval_status: string;
  resulting_model_status: string;
  actor: string;
  comment: string | null;
  timestamp: string;
  event_index: number;
}

export interface ActivationResponse {
  activated: boolean;
  model_id: string;
  previous_active_model_id: string | null;
}

export interface RollbackResponse {
  rolled_back: boolean;
  active_model_id: string;
}
