import { useCallback, useEffect, useState } from "react";
import { sessionClient } from "../api/sessionClient";
import type {
  IngestObservationsResponse,
  SessionPredictionRequest,
  SessionPredictionResponse,
  SessionUpdateResponse
} from "../types/session.types";

export function useSessionActions(sessionId: string | null, onSuccess?: () => Promise<void> | void) {
  const [runningAction, setRunningAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdateResult, setLastUpdateResult] = useState<SessionUpdateResponse | null>(null);
  const [lastIngestResult, setLastIngestResult] = useState<IngestObservationsResponse | null>(null);
  const [lastPrediction, setLastPrediction] = useState<SessionPredictionResponse | null>(null);

  const withAction = useCallback(async <T,>(name: string, fn: () => Promise<T>): Promise<T> => {
    setRunningAction(name);
    setError(null);
    try {
      const result = await fn();
      await onSuccess?.();
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : `Action failed: ${name}`);
      throw err;
    } finally {
      setRunningAction(null);
    }
  }, [onSuccess]);

  const update = useCallback(async () => {
    if (!sessionId) return null;
    const result = await withAction("update", () => sessionClient.updateSession(sessionId));
    setLastUpdateResult(result);
    return result;
  }, [sessionId, withAction]);

  const ingest = useCallback(async (observations: Array<Record<string, unknown>>) => {
    if (!sessionId) return null;
    const result = await withAction("ingest", () => sessionClient.ingestObservations(sessionId, { observations }));
    setLastIngestResult(result);
    return result;
  }, [sessionId, withAction]);

  const predict = useCallback(async (payload: SessionPredictionRequest) => {
    if (!sessionId) return null;
    const result = await withAction("predict", () => sessionClient.predictSession(sessionId, payload));
    setLastPrediction(result);
    return result;
  }, [sessionId, withAction]);

  useEffect(() => {
    setError(null);
    setLastUpdateResult(null);
    setLastIngestResult(null);
    setLastPrediction(null);
  }, [sessionId]);

  return {
    runningAction,
    error,
    lastUpdateResult,
    lastIngestResult,
    lastPrediction,
    update,
    ingest,
    predict
  };
}
