import { useCallback, useEffect, useState } from "react";
import { sessionClient } from "../api/sessionClient";
import type { CreateSessionRequest, SessionSummary } from "../types/session.types";

export function useSessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSessions(await sessionClient.listSessions());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load sessions");
    } finally {
      setLoading(false);
    }
  }, []);

  const createSession = useCallback(async (payload: CreateSessionRequest) => {
    const created = await sessionClient.createSession(payload);
    setSessions((previous) => [created, ...previous.filter((item) => item.session_id !== created.session_id)]);
    return created;
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { sessions, loading, error, refresh, createSession };
}
