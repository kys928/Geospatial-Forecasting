import { useCallback, useEffect, useRef, useState } from "react";
import { opsClient } from "../../ops/api/opsClient";
import type { EventRecord } from "../types/event.types";

export interface EventsRefreshOptions {
  force?: boolean;
}

export function useEvents() {
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedLabel, setLastUpdatedLabel] = useState<string | null>(null);
  const hasLoadedRef = useRef(false);

  const refresh = useCallback(async (options: EventsRefreshOptions = {}) => {
    const isInitialLoad = !hasLoadedRef.current;
    if (isInitialLoad) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError(null);
    try {
      const response = await opsClient.getEvents(200, { force: options.force });
      setEvents(response.events);
      setLastUpdatedLabel(new Date().toLocaleString());
      hasLoadedRef.current = true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load events");
    } finally {
      if (isInitialLoad) {
        setLoading(false);
      }
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { events, loading, refreshing, error, lastUpdatedLabel, refresh };
}
