import type { RegistryEventRecord } from "../types/registry.types";

interface RegistryEventStreamProps {
  events: RegistryEventRecord[];
  approvalAudit: RegistryEventRecord[];
}

export function RegistryEventStream({ events, approvalAudit }: RegistryEventStreamProps) {
  return (
    <section className="panel">
      <h3>Registry events + approval audit</h3>
      <pre style={{ margin: 0, maxHeight: 280, overflow: "auto" }}>
        {JSON.stringify({ events: events.slice(-20), approvalAudit: approvalAudit.slice(-20) }, null, 2)}
      </pre>
    </section>
  );
}
