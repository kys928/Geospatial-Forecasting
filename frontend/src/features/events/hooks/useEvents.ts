import { useCallback, useEffect, useState } from "react";
import { opsClient } from "../../ops/api/opsClient";
import type { EventRecord } from "../types/event.types";

export function useEvents() {
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedLabel, setLastUpdatedLabel] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await opsClient.getEvents(200);
      setEvents(response.events);
      setLastUpdatedLabel(new Date().toLocaleString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load events");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { events, loading, error, lastUpdatedLabel, refresh };
}
