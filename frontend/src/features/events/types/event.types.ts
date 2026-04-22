export interface EventRecord {
  timestamp?: string;
  event_type?: string;
  payload?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface OpsEventsResponse {
  events: EventRecord[];
}
