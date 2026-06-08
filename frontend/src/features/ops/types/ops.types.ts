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

export interface RetrainingStopResponse {
  stopped: boolean;
  job_id: string | null;
  previous_status: string | null;
  new_status: string | null;
  message: string;
  graceful: boolean;
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

export interface SafeUserAction {
  title?: string;
  description?: string;
  [key: string]: unknown;
}

export interface RetrainingRecommendation {
  should_retrain: boolean;
  reason?: string | null;
  severity?: string | null;
  evidence?: Record<string, unknown>;
  recommended_actions?: string[];
}

export interface RetrainingExplanationContext {
  topic?: string;
  summary_seed?: string;
  recommendation?: RetrainingRecommendation;
  safe_user_actions?: SafeUserAction[];
  system_boundaries?: string[];
  llm_instructions?: string[];
}

export interface ModelCandidateComparison {
  available_metrics?: Record<string, unknown>;
  missing_metrics?: string[];
  can_compare?: boolean;
  comparison_summary?: string;
  [key: string]: unknown;
}

export interface ModelCandidateContext {
  topic?: string;
  active_model?: Record<string, unknown> | null;
  candidate_model?: Record<string, unknown> | null;
  decision_state?: string;
  comparison?: ModelCandidateComparison;
  safe_user_actions?: SafeUserAction[];
  system_boundaries?: string[];
  llm_instructions?: string[];
  recent_events?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}


export interface OpsSystemStatusResponse {
  generated_at: string;
  host: Record<string, unknown>;
  gpu: Record<string, unknown>;
  worker_status: Record<string, unknown>;
  jobs: Record<string, unknown>;
  recent_events: Array<Record<string, unknown>>;
  status_summary: Record<string, unknown>;
}

export interface AdaptationBufferStatus {
  root: string;
  pending: number;
  accepted_train: number;
  accepted_val: number;
  rejected: number;
  reserve_used: number;
  fresh_accepted_total: number;
  used_total: number;
  manifest_readable: boolean;
  latest_event_timestamp?: string | null;
  warnings: string[];
}

export interface AdaptationReadinessCheck {
  name?: string;
  label?: string;
  status?: string;
  ready?: boolean;
  passed?: boolean;
  blocking?: boolean;
  reason?: string | null;
  message?: string | null;
  summary?: string | null;
  [key: string]: unknown;
}

export interface AdaptationReadiness {
  ready: boolean;
  status: string;
  checks: AdaptationReadinessCheck[];
  blocking_reasons: string[];
  warnings: string[];
  next_retry_at?: string | null;
  summary: Record<string, unknown>;
}

export type AdaptationTrainingJob = Record<string, unknown>;

export interface AdaptationTrainingStatus {
  job_counts: Record<string, number>;
  latest_job?: AdaptationTrainingJob | null;
  latest_manual_job?: AdaptationTrainingJob | null;
  latest_readiness_snapshot?: Record<string, unknown> | null;
  candidate_model_id?: string | null;
  output_dir?: string | null;
  result_run_dir?: string | null;
  best_overall_checkpoint?: string | null;
  final_checkpoint?: string | null;
  training_metrics?: Record<string, unknown> | null;
  cooldown_seconds?: number | null;
  cooldown_remaining_seconds?: number | null;
  next_automatic_training_eligible_at?: string | null;
  cooldown_source?: string | null;
  error_message?: string | null;
}

export interface AdaptationCandidate {
  model_id: string;
  status?: string | null;
  approval_status?: string | null;
  path?: string | null;
  timestamp?: string | null;
  created_at?: string | null;
  run_id?: string | null;
  adaptation_run?: Record<string, unknown> | null;
  last_adaptation_promotion_decision?: Record<string, unknown> | null;
  last_promotion_result?: Record<string, unknown> | null;
  best_overall_checkpoint?: string | null;
  final_checkpoint?: string | null;
  checkpoint_file_exists: boolean;
}

export interface AdaptationCandidateList {
  candidates: AdaptationCandidate[];
}

export interface AdaptationPromotionDecision {
  decision?: Record<string, unknown> | null;
  candidate_model_id?: string | null;
  active_model_id?: string | null;
  result?: Record<string, unknown> | null;
}

export interface AdaptationStorageWarning {
  checkpoint_count: number;
  checkpoint_count_warning: boolean;
  checkpoint_count_threshold?: number;
  registered_adaptation_model_count?: number | null;
  disk_usage_percent: number;
  disk_usage_warning: boolean;
  disk_usage_threshold_percent?: number;
  automatic_deletion: boolean;
  message: string;
}

export interface CheckpointFileDeleteResult {
  model_id: string;
  deleted: boolean;
  file_existed_before: boolean;
  checkpoint_path?: string | null;
  metadata_updated: boolean;
  active_model_id?: string | null;
  event_type?: string | null;
  message: string;
}
