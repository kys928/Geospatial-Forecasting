interface RollbackPanelProps {
  disabled: boolean;
  onRollback: () => Promise<void>;
}

export function RollbackPanel({ disabled, onRollback }: RollbackPanelProps) {
  return (
    <section className="panel">
      <h3>Rollback active model</h3>
      <button className="primary-button" disabled={disabled} onClick={() => void onRollback()}>Rollback</button>
    </section>
  );
}
