import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { ModelCandidateContext } from "../types/ops.types";

interface ModelCandidateContextState {
  context: ModelCandidateContext | null;
  loading: boolean;
  error: string | null;
  refresh: (force?: boolean) => Promise<void>;
}

export function useModelCandidateContext(enabled = true): ModelCandidateContextState {
  const cachedContext = enabled ? opsClient.peekModelCandidateContext() : null;
  const [context, setContext] = useState<ModelCandidateContext | null>(cachedContext);
  const [loading, setLoading] = useState(enabled && !cachedContext);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (force = true) => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    setLoading((current) => current || !context);
    setError(null);
    try {
      setContext(await opsClient.getModelCandidateContext({ force }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load model candidate context");
    } finally {
      setLoading(false);
    }
  }, [context, enabled]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    void refresh(false);
  }, [enabled, refresh]);

  return { context, loading, error, refresh };
}
