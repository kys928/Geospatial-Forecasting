import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsStatusResponse } from "../types/ops.types";

export function useOpsStatus(enabled = true) {
  const [status, setStatus] = useState<OpsStatusResponse | null>(null);
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
      setStatus(await opsClient.getStatus());
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
      setStatus(null);
      return;
    }

    void refresh();
  }, [enabled, refresh]);

  return { status, loading, error, refresh };
}
