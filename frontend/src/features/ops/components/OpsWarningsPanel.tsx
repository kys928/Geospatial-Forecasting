interface OpsWarningsPanelProps {
  latestWarningOrError: string | null;
  latestFailureReason: string | null;
}

export function OpsWarningsPanel({ latestWarningOrError, latestFailureReason }: OpsWarningsPanelProps) {
  return (
    <section className="panel">
      <h3>Warnings</h3>
      <p><strong>Latest warning/error:</strong> {latestWarningOrError ?? "none"}</p>
      <p><strong>Latest retraining failure:</strong> {latestFailureReason ?? "none"}</p>
    </section>
  );
}
