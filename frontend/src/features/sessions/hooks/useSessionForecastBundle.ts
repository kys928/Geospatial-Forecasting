import { useCallback, useEffect, useRef, useState } from "react";
import { sessionClient } from "../api/sessionClient";
import type { SessionForecastBundle } from "../types/session.types";

export function useSessionForecastBundle(sessionId: string | null) {
  const [bundle, setBundle] = useState<SessionForecastBundle | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  const clear = useCallback(() => {
    requestIdRef.current += 1;
    setBundle(null);
    setLoading(false);
    setError(null);
  }, []);

  const refresh = useCallback(async () => {
    if (!sessionId) {
      clear();
      return;
    }

    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError(null);

    try {
      const latestBundle = await sessionClient.getLatestForecastBundle(sessionId);
      if (requestId !== requestIdRef.current) {
        return;
      }
      setBundle(latestBundle);
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }
      setBundle(null);
      setError(err instanceof Error ? err.message : "Unable to load session forecast artifacts");
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [clear, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      clear();
      return;
    }

    clear();
    void refresh();
  }, [clear, refresh, sessionId]);

  return { bundle, loading, error, refresh, clear };
}
