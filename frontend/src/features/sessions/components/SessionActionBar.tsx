interface SessionActionBarProps {
  disabled: boolean;
  runningAction: string | null;
  onUpdate: () => Promise<void>;
}

export function SessionActionBar({ disabled, runningAction, onUpdate }: SessionActionBarProps) {
  return (
    <section className="panel">
      <h3>Session maintenance</h3>
      <p className="muted">Use this when you need to force a state refresh between ingest and prediction.</p>
      <button className="secondary-button" disabled={disabled || runningAction !== null} onClick={() => void onUpdate()}>
        {runningAction === "update" ? "Updating..." : "Manual update"}
      </button>
    </section>
  );
}
