import { opsClient } from "../../ops/api/opsClient";
import type { OpsEventsResponse } from "../types/event.types";

export const eventsClient = {
  getEvents(limit = 100): Promise<OpsEventsResponse> {
    return opsClient.getEvents(limit) as Promise<OpsEventsResponse>;
  }
};
