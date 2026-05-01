import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { ModelCandidateContext } from "../types/ops.types";

interface ModelCandidateContextState {
  context: ModelCandidateContext | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useModelCandidateContext(enabled = true): ModelCandidateContextState {
  const [context, setContext] = useState<ModelCandidateContext | null>(null);
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
      setContext(await opsClient.getModelCandidateContext());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load model candidate context");
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      setContext(null);
      setLoading(false);
      setError(null);
      return;
    }

    void refresh();
  }, [enabled, refresh]);

  return { context, loading, error, refresh };
}
