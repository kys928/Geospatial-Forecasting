import { useCallback, useState } from "react";
import { opsClient } from "../api/opsClient";
import type {
  ActivationResponse,
  ApprovalActionResponse,
  RetrainingTriggerRequest,
  RetrainingTriggerResponse,
  RollbackResponse
} from "../types/ops.types";

export function useOpsActions(onSuccess?: () => Promise<void> | void) {
  const [runningAction, setRunningAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<unknown>(null);

  const run = useCallback(async <T,>(name: string, fn: () => Promise<T>) => {
    setRunningAction(name);
    setError(null);
    try {
      const result = await fn();
      setLastResult(result);
      await onSuccess?.();
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed: ${name}`);
      throw err;
    } finally {
      setRunningAction(null);
    }
  }, [onSuccess]);

  return {
    runningAction,
    error,
    lastResult,
    triggerRetraining: (payload: RetrainingTriggerRequest): Promise<RetrainingTriggerResponse> =>
      run("trigger", () => opsClient.triggerRetraining(payload)),
    approveCandidate: (candidateId: string, actor: string, comment?: string): Promise<ApprovalActionResponse> =>
      run("approve", () => opsClient.approveCandidate(candidateId, { actor, comment })),
    rejectCandidate: (candidateId: string, actor: string, comment?: string): Promise<ApprovalActionResponse> =>
      run("reject", () => opsClient.rejectCandidate(candidateId, { actor, comment })),
    activateModel: (modelId: string): Promise<ActivationResponse> => run("activate", () => opsClient.activateModel(modelId)),
    rollbackModel: (): Promise<RollbackResponse> => run("rollback", () => opsClient.rollbackModel())
  };
}
