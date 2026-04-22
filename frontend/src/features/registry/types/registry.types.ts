export interface RegistryModelRecord {
  model_id?: string;
  status?: string;
  approval_status?: string;
  path?: string;
  checkpoint_metric?: number;
  checkpoint_metric_name?: string;
  target_policy?: string;
  normalization_mode?: string;
  contract_version?: string;
  [key: string]: unknown;
}

export interface RegistryEventRecord {
  timestamp?: string;
  event_type?: string;
  payload?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface OpsRegistryResponse {
  active_model_id: string | null;
  previous_active_model_id: string | null;
  models: RegistryModelRecord[];
  events: RegistryEventRecord[];
  approval_audit: RegistryEventRecord[];
  revision: number;
  next_event_index: number;
}
