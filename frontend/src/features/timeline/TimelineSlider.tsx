export function TimelineSlider() {
  return (
    <section className="timeline panel">
      <div className="panel-header">
        <h2>Timeline</h2>
      </div>
      <div className="panel-body">
        <input type="range" min={0} max={100} defaultValue={0} disabled />
        <p className="muted">Timeline scaffold only. Time progression is not implemented yet.</p>
      </div>
    </section>
  );
}