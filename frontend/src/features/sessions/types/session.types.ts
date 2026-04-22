export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
  [key: string]: JsonValue;
}

export interface SessionSummary {
  session_id: string;
  backend_name: string;
  model_name: string;
  status: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
  last_error: string | null;
  capabilities: Record<string, unknown>;
  runtime_metadata: Record<string, unknown>;
}

export type SessionDetail = SessionSummary;

export interface SessionStateSummary {
  session_id?: string;
  state_version?: number;
  observation_count?: number;
  last_update_time?: string;
  [key: string]: unknown;
}

export interface CreateSessionRequest {
  backend_name: string;
  model_name?: string;
  metadata?: Record<string, unknown>;
}

export interface IngestObservationsRequest {
  observations: Array<Record<string, unknown>>;
}

export interface IngestObservationsResponse {
  session_id: string;
  observation_count: number;
  state_version: number;
  last_update_time: string;
  auto_update_result: {
    success: boolean;
    updated_at: string;
    state_version: number;
    message: string;
    changed: boolean;
  } | null;
}

export interface SessionUpdateResponse {
  session_id: string;
  success: boolean;
  updated_at: string;
  state_version: number;
  message: string;
  metadata: Record<string, unknown>;
  previous_state_version: number;
  observation_count: number;
  changed: boolean;
}

export interface SessionPredictionRequest {
  [key: string]: unknown;
}

export type SessionPredictionResponse = Record<string, unknown>;
