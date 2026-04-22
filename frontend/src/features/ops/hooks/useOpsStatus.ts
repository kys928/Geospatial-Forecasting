import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../api/opsClient";
import type { OpsStatusResponse } from "../types/ops.types";

export function useOpsStatus() {
  const [status, setStatus] = useState<OpsStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setStatus(await opsClient.getStatus());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load ops status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { status, loading, error, refresh };
}
