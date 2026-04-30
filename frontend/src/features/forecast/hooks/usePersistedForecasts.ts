import { useCallback, useEffect, useState } from "react";
import { listForecasts } from "../api/forecastArtifactsClient";
import type { ForecastArtifactMetadata } from "../types/forecast.types";

export function usePersistedForecasts(limit = 50) {
  const [forecasts, setForecasts] = useState<ForecastArtifactMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listForecasts(limit);
      setForecasts(Array.isArray(response?.forecasts) ? response.forecasts : []);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load persisted forecasts.";
      setError(message);
      setForecasts([]);
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { forecasts, loading, error, refresh };
}
