import { invalidateActiveForecastSession } from "../../sessions/api/sessionClient";
import { httpGet, httpPost } from "../../../services/api/http";
import { cachedOpsRequest, invalidateOpsCache, peekOpsCache } from "./opsCache";
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

export interface OpsRequestOptions {
  force?: boolean;
}

export const OPS_CACHE_KEYS = {
  status: "ops:status",
  systemStatus: "ops:system-status",
  registry: "ops:registry",
  jobs: "ops:jobs",
  events: (limit: number) => `ops:events:${limit}`,
  retrainingRecommendation: "ops:retraining-recommendation",
  retrainingRecommendationContext: "ops:retraining-recommendation-context",
  modelCandidateContext: "ops:model-candidate-context",
  adaptationBufferStatus: "ops:adaptation-buffer-status",
  adaptationReadiness: "ops:adaptation-readiness",
  adaptationTrainingStatus: "ops:adaptation-training-status",
  adaptationCandidates: "ops:adaptation-candidates",
  adaptationStorageWarnings: "ops:adaptation-storage-warnings",
} as const;

const OPS_STALE_MS = {
  status: 8_000,
  systemStatus: 5_000,
  registry: 30_000,
  jobs: 8_000,
  events: 10_000,
  retrainingRecommendation: 20_000,
  retrainingRecommendationContext: 20_000,
  modelCandidateContext: 20_000,
  adaptationBufferStatus: 20_000,
  adaptationReadiness: 5_000,
  adaptationTrainingStatus: 5_000,
  adaptationCandidates: 10_000,
  adaptationStorageWarnings: 20_000,
} as const;

function opsHeaders(): HeadersInit | undefined {
  if (!opsToken) {
    return undefined;
  }

  return {
    Authorization: `Bearer ${opsToken}`
  };
}

async function opsMutation<T>(request: () => Promise<T>): Promise<T> {
  const response = await request();
  invalidateOpsCache();
  return response;
}

async function opsActivationMutation<T>(request: () => Promise<T>): Promise<T> {
  const response = await opsMutation(request);
  invalidateActiveForecastSession("ops_model_activation");
  return response;
}

export const opsClient = {
  getStatus(options: OpsRequestOptions = {}): Promise<OpsStatusResponse> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.status,
      staleMs: OPS_STALE_MS.status,
      force: options.force,
      request: () => httpGet<OpsStatusResponse>("/ops/status", opsHeaders()),
    });
  },

  peekStatus(): OpsStatusResponse | null {
    return peekOpsCache<OpsStatusResponse>(OPS_CACHE_KEYS.status, OPS_STALE_MS.status);
  },

  getSystemStatus(options: OpsRequestOptions = {}): Promise<OpsSystemStatusResponse> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.systemStatus,
      staleMs: OPS_STALE_MS.systemStatus,
      force: options.force,
      request: () => httpGet<OpsSystemStatusResponse>("/ops/system/status", opsHeaders()),
    });
  },

  peekSystemStatus(): OpsSystemStatusResponse | null {
    return peekOpsCache<OpsSystemStatusResponse>(OPS_CACHE_KEYS.systemStatus, OPS_STALE_MS.systemStatus);
  },

  getRegistry(options: OpsRequestOptions = {}): Promise<OpsRegistryResponse> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.registry,
      staleMs: OPS_STALE_MS.registry,
      force: options.force,
      request: () => httpGet<OpsRegistryResponse>("/ops/registry", opsHeaders()),
    });
  },

  getJobs(options: OpsRequestOptions = {}): Promise<OpsJobsResponse> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.jobs,
      staleMs: OPS_STALE_MS.jobs,
      force: options.force,
      request: () => httpGet<OpsJobsResponse>("/ops/jobs", opsHeaders()),
    });
  },

  peekJobs(): OpsJobsResponse | null {
    return peekOpsCache<OpsJobsResponse>(OPS_CACHE_KEYS.jobs, OPS_STALE_MS.jobs);
  },

  getEvents(limit = 50, options: OpsRequestOptions = {}): Promise<OpsEventsResponse> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.events(limit),
      staleMs: OPS_STALE_MS.events,
      force: options.force,
      request: () => httpGet<OpsEventsResponse>(`/ops/events?limit=${limit}`, opsHeaders()),
    });
  },

  getRetrainingRecommendation(options: OpsRequestOptions = {}): Promise<RetrainingRecommendation> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.retrainingRecommendation,
      staleMs: OPS_STALE_MS.retrainingRecommendation,
      force: options.force,
      request: () => httpGet<RetrainingRecommendation>("/ops/retraining/recommendation", opsHeaders()),
    });
  },

  peekRetrainingRecommendation(): RetrainingRecommendation | null {
    return peekOpsCache<RetrainingRecommendation>(OPS_CACHE_KEYS.retrainingRecommendation, OPS_STALE_MS.retrainingRecommendation);
  },

  getRetrainingRecommendationContext(options: OpsRequestOptions = {}): Promise<RetrainingExplanationContext> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.retrainingRecommendationContext,
      staleMs: OPS_STALE_MS.retrainingRecommendationContext,
      force: options.force,
      request: () => httpGet<RetrainingExplanationContext>("/ops/retraining/recommendation/context", opsHeaders()),
    });
  },

  peekRetrainingRecommendationContext(): RetrainingExplanationContext | null {
    return peekOpsCache<RetrainingExplanationContext>(OPS_CACHE_KEYS.retrainingRecommendationContext, OPS_STALE_MS.retrainingRecommendationContext);
  },

  getModelCandidateContext(options: OpsRequestOptions = {}): Promise<ModelCandidateContext> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.modelCandidateContext,
      staleMs: OPS_STALE_MS.modelCandidateContext,
      force: options.force,
      request: () => httpGet<ModelCandidateContext>("/ops/models/candidate/context", opsHeaders()),
    });
  },

  peekModelCandidateContext(): ModelCandidateContext | null {
    return peekOpsCache<ModelCandidateContext>(OPS_CACHE_KEYS.modelCandidateContext, OPS_STALE_MS.modelCandidateContext);
  },

  triggerRetraining(payload: RetrainingTriggerRequest): Promise<RetrainingTriggerResponse> {
    return opsMutation(() =>
      httpPost<RetrainingTriggerResponse, RetrainingTriggerRequest>("/ops/retraining/trigger", payload, opsHeaders()),
    );
  },

  stopRetraining(): Promise<RetrainingStopResponse> {
    return opsMutation(() => httpPost<RetrainingStopResponse>("/ops/retraining/stop", {}, opsHeaders()));
  },

  approveCandidate(candidateId: string, payload: CandidateDecisionRequest): Promise<ApprovalActionResponse> {
    return opsMutation(() =>
      httpPost<ApprovalActionResponse, CandidateDecisionRequest>(`/ops/candidates/${candidateId}/approve`, payload, opsHeaders()),
    );
  },

  rejectCandidate(candidateId: string, payload: CandidateDecisionRequest): Promise<ApprovalActionResponse> {
    return opsMutation(() =>
      httpPost<ApprovalActionResponse, CandidateDecisionRequest>(`/ops/candidates/${candidateId}/reject`, payload, opsHeaders()),
    );
  },

  activateModel(modelId: string): Promise<ActivationResponse> {
    return opsActivationMutation(() =>
      httpPost<ActivationResponse, { model_id: string }>("/ops/models/activate", { model_id: modelId }, opsHeaders()),
    );
  },

  rollbackModel(): Promise<RollbackResponse> {
    return opsActivationMutation(() => httpPost<RollbackResponse>("/ops/models/rollback", {}, opsHeaders()));
  },

  getAdaptationBufferStatus(options: OpsRequestOptions = {}): Promise<AdaptationBufferStatus> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.adaptationBufferStatus,
      staleMs: OPS_STALE_MS.adaptationBufferStatus,
      force: options.force,
      request: () => httpGet<AdaptationBufferStatus>("/ops/adaptation/buffer/status", opsHeaders()),
    });
  },

  peekAdaptationBufferStatus(): AdaptationBufferStatus | null {
    return peekOpsCache<AdaptationBufferStatus>(OPS_CACHE_KEYS.adaptationBufferStatus, OPS_STALE_MS.adaptationBufferStatus);
  },

  getAdaptationReadiness(options: OpsRequestOptions = {}): Promise<AdaptationReadiness> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.adaptationReadiness,
      staleMs: OPS_STALE_MS.adaptationReadiness,
      force: options.force,
      request: () => httpGet<AdaptationReadiness>("/ops/adaptation/readiness", opsHeaders()),
    });
  },

  peekAdaptationReadiness(): AdaptationReadiness | null {
    return peekOpsCache<AdaptationReadiness>(OPS_CACHE_KEYS.adaptationReadiness, OPS_STALE_MS.adaptationReadiness);
  },

  checkAdaptationNow(): Promise<AdaptationReadiness> {
    return opsMutation(() => httpPost<AdaptationReadiness>("/ops/adaptation/check-now", {}, opsHeaders()));
  },

  getAdaptationTrainingStatus(options: OpsRequestOptions = {}): Promise<AdaptationTrainingStatus> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.adaptationTrainingStatus,
      staleMs: OPS_STALE_MS.adaptationTrainingStatus,
      force: options.force,
      request: () => httpGet<AdaptationTrainingStatus>("/ops/adaptation/training/status", opsHeaders()),
    });
  },

  peekAdaptationTrainingStatus(): AdaptationTrainingStatus | null {
    return peekOpsCache<AdaptationTrainingStatus>(OPS_CACHE_KEYS.adaptationTrainingStatus, OPS_STALE_MS.adaptationTrainingStatus);
  },

  getAdaptationCandidates(options: OpsRequestOptions = {}): Promise<AdaptationCandidateList> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.adaptationCandidates,
      staleMs: OPS_STALE_MS.adaptationCandidates,
      force: options.force,
      request: () => httpGet<AdaptationCandidateList>("/ops/adaptation/candidates", opsHeaders()),
    });
  },

  evaluateAdaptationCandidate(modelId: string): Promise<AdaptationPromotionDecision> {
    return opsMutation(() =>
      httpPost<AdaptationPromotionDecision>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/evaluate`, {}, opsHeaders()),
    );
  },

  applyAdaptationPolicy(modelId: string): Promise<AdaptationPromotionDecision> {
    return opsMutation(() =>
      httpPost<AdaptationPromotionDecision>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/apply-policy`, {}, opsHeaders()),
    );
  },

  approveAdaptationCandidate(modelId: string, payload?: CandidateDecisionRequest): Promise<AdaptationPromotionDecision> {
    return opsActivationMutation(() =>
      httpPost<AdaptationPromotionDecision, CandidateDecisionRequest | undefined>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/approve`, payload, opsHeaders()),
    );
  },

  rejectAdaptationCandidate(modelId: string, payload?: CandidateDecisionRequest): Promise<AdaptationPromotionDecision> {
    return opsMutation(() =>
      httpPost<AdaptationPromotionDecision, CandidateDecisionRequest | undefined>(`/ops/adaptation/candidates/${encodeURIComponent(modelId)}/reject`, payload, opsHeaders()),
    );
  },

  getAdaptationStorageWarnings(options: OpsRequestOptions = {}): Promise<AdaptationStorageWarning> {
    return cachedOpsRequest({
      key: OPS_CACHE_KEYS.adaptationStorageWarnings,
      staleMs: OPS_STALE_MS.adaptationStorageWarnings,
      force: options.force,
      request: () => httpGet<AdaptationStorageWarning>("/ops/adaptation/storage/warnings", opsHeaders()),
    });
  },

  deleteAdaptationCheckpointFile(modelId: string, payload?: CandidateDecisionRequest): Promise<CheckpointFileDeleteResult> {
    return opsMutation(() =>
      httpPost<CheckpointFileDeleteResult, CandidateDecisionRequest | undefined>(`/ops/adaptation/checkpoints/${encodeURIComponent(modelId)}/delete-file`, payload, opsHeaders()),
    );
  }
};
