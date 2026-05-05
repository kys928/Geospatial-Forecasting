import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../../ops/api/opsClient";
import type { OpsRegistryResponse } from "../types/registry.types";

export function useRegistry(autoRefreshMs = 10000) {
  const [registry, setRegistry] = useState<OpsRegistryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (initial = false) => {
    initial ? setLoading(true) : setRefreshing(true);
    try {
      setRegistry(await opsClient.getRegistry());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load registry");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { void refresh(true); }, [refresh]);
  useEffect(() => {
    const id = window.setInterval(() => { void refresh(false); }, autoRefreshMs);
    return () => window.clearInterval(id);
  }, [autoRefreshMs, refresh]);

  return { registry, loading, refreshing, error, refresh };
}
