import { httpGet, httpPost } from "../../../services/api/http";
import type { OpsRegistryResponse } from "../../registry/types/registry.types";
import type {
  ActivationResponse,
  ApprovalActionResponse,
  CandidateDecisionRequest,
  OpsEventsResponse,
  OpsJobsResponse,
  OpsStatusResponse,
  RetrainingExplanationContext,
  RetrainingRecommendation,
  RetrainingTriggerRequest,
  RetrainingTriggerResponse,
  RollbackResponse
} from "../types/ops.types";

const opsToken = import.meta.env.VITE_OPS_API_TOKEN?.trim();

function opsHeaders(): HeadersInit | undefined {
  if (!opsToken) {
    return undefined;
  }

  return {
    Authorization: `Bearer ${opsToken}`
  };
}

export const opsClient = {
  getStatus(): Promise<OpsStatusResponse> {
    return httpGet<OpsStatusResponse>("/ops/status", opsHeaders());
  },

  getRegistry(): Promise<OpsRegistryResponse> {
    return httpGet<OpsRegistryResponse>("/ops/registry", opsHeaders());
  },

  getJobs(): Promise<OpsJobsResponse> {
    return httpGet<OpsJobsResponse>("/ops/jobs", opsHeaders());
  },

  getEvents(limit = 50): Promise<OpsEventsResponse> {
    return httpGet<OpsEventsResponse>(`/ops/events?limit=${limit}`, opsHeaders());
  },

  getRetrainingRecommendation(): Promise<RetrainingRecommendation> {
    return httpGet<RetrainingRecommendation>("/ops/retraining/recommendation", opsHeaders());
  },

  getRetrainingRecommendationContext(): Promise<RetrainingExplanationContext> {
    return httpGet<RetrainingExplanationContext>("/ops/retraining/recommendation/context", opsHeaders());
  },

  triggerRetraining(payload: RetrainingTriggerRequest): Promise<RetrainingTriggerResponse> {
    return httpPost<RetrainingTriggerResponse, RetrainingTriggerRequest>("/ops/retraining/trigger", payload, opsHeaders());
  },

  approveCandidate(candidateId: string, payload: CandidateDecisionRequest): Promise<ApprovalActionResponse> {
    return httpPost<ApprovalActionResponse, CandidateDecisionRequest>(`/ops/candidates/${candidateId}/approve`, payload, opsHeaders());
  },

  rejectCandidate(candidateId: string, payload: CandidateDecisionRequest): Promise<ApprovalActionResponse> {
    return httpPost<ApprovalActionResponse, CandidateDecisionRequest>(`/ops/candidates/${candidateId}/reject`, payload, opsHeaders());
  },

  activateModel(modelId: string): Promise<ActivationResponse> {
    return httpPost<ActivationResponse, { model_id: string }>("/ops/models/activate", { model_id: modelId }, opsHeaders());
  },

  rollbackModel(): Promise<RollbackResponse> {
    return httpPost<RollbackResponse>("/ops/models/rollback", {}, opsHeaders());
  }
};
