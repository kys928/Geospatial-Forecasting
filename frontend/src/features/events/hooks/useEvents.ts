import { useCallback, useEffect, useMemo, useState } from "react";
import { eventsClient } from "../api/eventsClient";
import type { EventRecord } from "../types/event.types";

export function useEvents() {
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchText, setSearchText] = useState("");
  const [eventType, setEventType] = useState("all");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await eventsClient.getEvents(200);
      setEvents(response.events);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load events");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const filteredEvents = useMemo(() => {
    return events.filter((event) => {
      const matchesType = eventType === "all" || event.event_type === eventType;
      const blob = JSON.stringify(event).toLowerCase();
      const matchesSearch = searchText.trim() === "" || blob.includes(searchText.toLowerCase());
      return matchesType && matchesSearch;
    });
  }, [events, eventType, searchText]);

  const availableTypes = useMemo(() => {
    return Array.from(new Set(events.map((event) => event.event_type).filter((item): item is string => Boolean(item))));
  }, [events]);

  return {
    events,
    filteredEvents,
    availableTypes,
    loading,
    error,
    searchText,
    setSearchText,
    eventType,
    setEventType,
    refresh
  };
}
