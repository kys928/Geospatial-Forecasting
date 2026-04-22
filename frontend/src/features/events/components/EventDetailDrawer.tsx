import type { EventRecord } from "../types/event.types";

interface EventDetailDrawerProps {
  event: EventRecord | null;
}

export function EventDetailDrawer({ event }: EventDetailDrawerProps) {
  return (
    <section className="panel">
      <h3>Event detail</h3>
      <pre style={{ margin: 0, maxHeight: 500, overflow: "auto" }}>{JSON.stringify(event, null, 2)}</pre>
    </section>
  );
}
