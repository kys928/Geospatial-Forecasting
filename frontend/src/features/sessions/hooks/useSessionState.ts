import { useCallback, useEffect, useState } from "react";
import { sessionClient } from "../api/sessionClient";
import type { SessionDetail, SessionStateSummary } from "../types/session.types";

export function useSessionState(sessionId: string | null) {
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [state, setState] = useState<SessionStateSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!sessionId) {
      setDetail(null);
      setState(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const [sessionDetail, sessionState] = await Promise.all([
        sessionClient.getSession(sessionId),
        sessionClient.getSessionState(sessionId)
      ]);
      setDetail(sessionDetail);
      setState(sessionState);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load session state");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { detail, state, loading, error, refresh };
}
