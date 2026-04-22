import { useCallback, useEffect, useState } from "react";
import { registryClient } from "../api/registryClient";
import type { OpsRegistryResponse } from "../types/registry.types";

export function useRegistry() {
  const [registry, setRegistry] = useState<OpsRegistryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setRegistry(await registryClient.getRegistry());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load registry");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { registry, loading, error, refresh };
}
