import { useCallback, useEffect, useRef, useState } from "react";
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
  const hasDataRef = useRef(Boolean(cachedContext));
  const [loading, setLoading] = useState(enabled && !cachedContext);
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
      const nextContext = await opsClient.getModelCandidateContext({ force });
      hasDataRef.current = true;
      setContext(nextContext);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load model candidate context");
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

  return { context, loading, error, refresh };
}
