import { useCallback, useEffect, useRef, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { RetrainingExplanationContext, RetrainingRecommendation } from "../types/ops.types";

interface RetrainingRecommendationState {
  recommendation: RetrainingRecommendation | null;
  context: RetrainingExplanationContext | null;
  loading: boolean;
  error: string | null;
  refresh: (force?: boolean) => Promise<void>;
}

export function useRetrainingRecommendation(enabled = true): RetrainingRecommendationState {
  const cachedRecommendation = enabled ? opsClient.peekRetrainingRecommendation() : null;
  const cachedContext = enabled ? opsClient.peekRetrainingRecommendationContext() : null;
  const [recommendation, setRecommendation] = useState<RetrainingRecommendation | null>(cachedRecommendation);
  const [context, setContext] = useState<RetrainingExplanationContext | null>(cachedContext);
  const hasDataRef = useRef(Boolean(cachedRecommendation));
  const [loading, setLoading] = useState(enabled && !cachedRecommendation);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (force = true) => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    setLoading((current) => current || !hasDataRef.current);
    setError(null);
    try {
      const [recommendationResult, contextResult] = await Promise.allSettled([
        opsClient.getRetrainingRecommendation({ force }),
        opsClient.getRetrainingRecommendationContext({ force })
      ]);

      if (recommendationResult.status === "fulfilled") {
        hasDataRef.current = true;
        setRecommendation(recommendationResult.value);
      }

      if (contextResult.status === "fulfilled") {
        setContext(contextResult.value);
      }

      if (recommendationResult.status === "rejected") {
        throw recommendationResult.reason;
      }

      if (contextResult.status === "rejected") {
        setError("Recommendation loaded, but additional context is temporarily unavailable.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load retraining recommendation");
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    void refresh(false);
  }, [enabled, refresh]);

  return { recommendation, context, loading, error, refresh };
}
