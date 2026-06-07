import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsJobsResponse } from "../types/ops.types";

export function useOpsJobs(enabled = true) {
  const cachedJobs = enabled ? opsClient.peekJobs() : null;
  const [jobs, setJobs] = useState<OpsJobsResponse | null>(cachedJobs);
  const [loading, setLoading] = useState(enabled && !cachedJobs);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (force = true) => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    setLoading((current) => current || !jobs);
    setError(null);
    try {
      setJobs(await opsClient.getJobs({ force }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load ops jobs");
    } finally {
      setLoading(false);
    }
  }, [enabled, jobs]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      return;
    }

    void refresh(false);
  }, [enabled, refresh]);

  return { jobs, loading, error, refresh };
}
