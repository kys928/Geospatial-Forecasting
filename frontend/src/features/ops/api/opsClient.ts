import { httpGet, httpPost } from "../../../services/api/http";
import type { OpsRegistryResponse } from "../../registry/types/registry.types";
import type {
  ActivationResponse,
  ApprovalActionResponse,
  CandidateDecisionRequest,
  OpsEventsResponse,
  OpsJobsResponse,
  ModelCandidateContext,
  OpsStatusResponse,
  RetrainingExplanationContext,
  RetrainingRecommendation,
  RetrainingTriggerRequest,
  RetrainingTriggerResponse,
  RetrainingStopResponse,
  RollbackResponse,
  OpsSystemStatusResponse,
  AdaptationBufferStatus,
  AdaptationReadiness,
  AdaptationTrainingStatus,
  AdaptationCandidateList,
  AdaptationPromotionDecision,
  AdaptationStorageWarning,
  CheckpointFileDeleteResult
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

  getSystemStatus(): Promise<OpsSystemStatusResponse> {
    return httpGet<OpsSystemStatusResponse>("/ops/system/status", opsHeaders());
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

  getModelCandidateContext(): Promise<ModelCandidateContext> {
    return httpGet<ModelCandidateContext>("/ops/models/candidate/context", opsHeaders());
  },

  triggerRetraining(payload: RetrainingTriggerRequest): Promise<RetrainingTriggerResponse> {
    return httpPost<RetrainingTriggerResponse, RetrainingTriggerRequest>("/ops/retraining/trigger", payload, opsHeaders());
  },

  stopRetraining(): Promise<RetrainingStopResponse> {
    return httpPost<RetrainingStopResponse>("/ops/retraining/stop", {}, opsHeaders());
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
  },

  getAdaptationBufferStatus(): Promise<AdaptationBufferStatus> {
    return httpGet<AdaptationBufferStatus>("/ops/adaptation/buffer/status", opsHeaders());
  },

  getAdaptationReadiness(): Promise<AdaptationReadiness> {
    return httpGet<AdaptationReadiness>("/ops/adaptation/readiness", opsHeaders());
  },

  checkAdaptationNow(): Promise<AdaptationReadiness> {
    return httpPost<AdaptationReadiness>("/ops/adaptation/check-now", {}, opsHeaders());
  },

  getAdaptationTrainingStatus(): Promise<AdaptationTrainingStatus> {
    return httpGet<AdaptationTrainingStatus>("/ops/adaptation/training/status", opsHeaders());
  },

  getAdaptationCandidates(): Promise<AdaptationCandidateList> {
    return httpGet<AdaptationCandidateList>("/ops/adaptation/candidates", opsHeaders());
  },

  evaluateAdaptationCandidate(modelId: string): Promise<AdaptationPromotionDecision> {
    return httpPost<AdaptationPromotionDecision>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/evaluate`, {}, opsHeaders());
  },

  applyAdaptationPolicy(modelId: string): Promise<AdaptationPromotionDecision> {
    return httpPost<AdaptationPromotionDecision>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/apply-policy`, {}, opsHeaders());
  },

  approveAdaptationCandidate(modelId: string, payload?: CandidateDecisionRequest): Promise<AdaptationPromotionDecision> {
    return httpPost<AdaptationPromotionDecision, CandidateDecisionRequest | undefined>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/approve`, payload, opsHeaders());
  },

  rejectAdaptationCandidate(modelId: string, payload?: CandidateDecisionRequest): Promise<AdaptationPromotionDecision> {
    return httpPost<AdaptationPromotionDecision, CandidateDecisionRequest | undefined>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/reject`, payload, opsHeaders());
  },

  getAdaptationStorageWarnings(): Promise<AdaptationStorageWarning> {
    return httpGet<AdaptationStorageWarning>("/ops/adaptation/storage/warnings", opsHeaders());
  },

  deleteAdaptationCheckpointFile(modelId: string, payload?: CandidateDecisionRequest): Promise<CheckpointFileDeleteResult> {
    return httpPost<CheckpointFileDeleteResult, CandidateDecisionRequest | undefined>(`/ops/adaptation/checkpoints/${encodeURIComponent(modelId)}/delete-file`, payload, opsHeaders());
  }
};
