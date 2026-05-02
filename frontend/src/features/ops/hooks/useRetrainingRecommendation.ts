import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { RetrainingExplanationContext, RetrainingRecommendation } from "../types/ops.types";

interface RetrainingRecommendationState {
  recommendation: RetrainingRecommendation | null;
  context: RetrainingExplanationContext | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useRetrainingRecommendation(enabled = true): RetrainingRecommendationState {
  const [recommendation, setRecommendation] = useState<RetrainingRecommendation | null>(null);
  const [context, setContext] = useState<RetrainingExplanationContext | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const [recommendationResult, contextResult] = await Promise.allSettled([
        opsClient.getRetrainingRecommendation(),
        opsClient.getRetrainingRecommendationContext()
      ]);

      if (recommendationResult.status === "fulfilled") {
        setRecommendation(recommendationResult.value);
      } else {
        setRecommendation(null);
      }

      if (contextResult.status === "fulfilled") {
        setContext(contextResult.value);
      } else {
        setContext(null);
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
      setRecommendation(null);
      setContext(null);
      setLoading(false);
      setError(null);
      return;
    }

    void refresh();
  }, [enabled, refresh]);

  return { recommendation, context, loading, error, refresh };
}
