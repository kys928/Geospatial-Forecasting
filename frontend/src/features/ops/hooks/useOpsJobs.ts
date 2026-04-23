import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsJobsResponse } from "../types/ops.types";

export function useOpsJobs(enabled = true) {
  const [jobs, setJobs] = useState<OpsJobsResponse | null>(null);
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
      setJobs(await opsClient.getJobs());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load ops jobs");
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setError(null);
      setJobs(null);
      return;
    }

    void refresh();
  }, [enabled, refresh]);

  return { jobs, loading, error, refresh };
}
