import { useCallback, useEffect, useRef, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsSystemStatusResponse } from "../types/ops.types";

export function useOpsSystemStatus(enabled = true, pollMs = 8000) {
  const cachedStatus = enabled ? opsClient.peekSystemStatus() : null;
  const [status, setStatus] = useState<OpsSystemStatusResponse | null>(cachedStatus);
  const hasDataRef = useRef(Boolean(cachedStatus));
  const inFlightRef = useRef(false);
  const [loading, setLoading] = useState(enabled && !cachedStatus);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (force = false) => {
    if (!enabled || inFlightRef.current) return;
    inFlightRef.current = true;
    if (!hasDataRef.current) setLoading(true);
    try {
      const payload = await opsClient.getSystemStatus({ force });
      hasDataRef.current = true;
      setStatus(payload);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load system status");
    } finally {
      inFlightRef.current = false;
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    void refresh(false);
    const id = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void refresh(false);
    }, pollMs);
    return () => window.clearInterval(id);
  }, [enabled, pollMs, refresh]);

  return { status, loading, error, refresh };
}
