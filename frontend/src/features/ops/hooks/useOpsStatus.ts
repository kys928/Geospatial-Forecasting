import { useCallback, useEffect, useRef, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsStatusResponse } from "../types/ops.types";

export function useOpsStatus(enabled = true) {
  const cachedStatus = enabled ? opsClient.peekStatus() : null;
  const [status, setStatus] = useState<OpsStatusResponse | null>(cachedStatus);
  const hasDataRef = useRef(Boolean(cachedStatus));
  const [loading, setLoading] = useState(enabled && !cachedStatus);
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
      const nextStatus = await opsClient.getStatus({ force });
      hasDataRef.current = true;
      setStatus(nextStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load ops status");
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

  return { status, loading, error, refresh };
}
