import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsSystemStatusResponse } from "../types/ops.types";

export function useOpsSystemStatus(enabled = true, pollMs = 8000) {
  const [status, setStatus] = useState<OpsSystemStatusResponse | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!enabled) return;
    if (!status) setLoading(true);
    try {
      const payload = await opsClient.getSystemStatus();
      setStatus(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load system status");
    } finally {
      setLoading(false);
    }
  }, [enabled, status]);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
    const id = window.setInterval(() => void refresh(), pollMs);
    return () => window.clearInterval(id);
  }, [enabled, pollMs, refresh]);

  return { status, loading, error, refresh };
}
