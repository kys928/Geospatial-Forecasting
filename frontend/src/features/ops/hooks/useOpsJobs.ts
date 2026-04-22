import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsJobsResponse } from "../types/ops.types";

export function useOpsJobs() {
  const [jobs, setJobs] = useState<OpsJobsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setJobs(await opsClient.getJobs());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load ops jobs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { jobs, loading, error, refresh };
}
