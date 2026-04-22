interface SessionActionBarProps {
  disabled: boolean;
  runningAction: string | null;
  onUpdate: () => Promise<void>;
}

export function SessionActionBar({ disabled, runningAction, onUpdate }: SessionActionBarProps) {
  return (
    <section className="panel">
      <h3>Session actions</h3>
      <button className="primary-button" disabled={disabled || runningAction !== null} onClick={() => void onUpdate()}>
        {runningAction === "update" ? "Updating..." : "Manual update"}
      </button>
    </section>
  );
}
