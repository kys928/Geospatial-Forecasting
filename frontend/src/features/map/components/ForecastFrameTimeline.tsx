interface ForecastFrameTimelineProps {
  frameCount: number;
  frameIndices: number[];
  selectedFrameIndex: number;
  onSelectFrame: (index: number) => void;
  loading: boolean;
  disabled?: boolean;
  metadata?: Record<string, unknown>;
  selectedFrameSummary?: Record<string, unknown> | null;
  predictionTrust?: string | null;
  inputMode?: string | null;
  modelName?: string | null;
  predictionEngine?: string | null;
  errorMessage?: string | null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatFrameLabel(index: number, frameCount: number) {
  if (frameCount <= 1) {
    return "Now";
  }
  if (index === 0) {
    return "Now";
  }
  return `+${index}h`;
}

export function ForecastFrameTimeline({
  frameCount,
  frameIndices,
  selectedFrameIndex,
  onSelectFrame,
  loading,
  disabled = false,
  metadata,
  selectedFrameSummary,
  predictionTrust,
  inputMode,
  modelName,
  predictionEngine,
  errorMessage
}: ForecastFrameTimelineProps) {
  const summaryStats = selectedFrameSummary?.summary_statistics as Record<string, unknown> | undefined;
  const maxConcentration = asNumber(summaryStats?.max_concentration);
  const meanConcentration = asNumber(summaryStats?.mean_concentration);
  const affectedCells = asNumber(summaryStats?.affected_cells_above_threshold);

  const trustState = inputMode === "degraded" || predictionTrust === "low"
    ? "Low-trust forecast: degraded input"
    : inputMode === "complete" || predictionTrust === "high"
      ? "Model input complete"
      : "Trust metadata unavailable";

  const timelineDisabled = disabled || frameCount <= 1;

  return (
    <section className={`forecast-timeline-card ${timelineDisabled ? "is-disabled" : ""}`}>
      <header className="forecast-timeline-header">
        <div className="forecast-frame-badge">Frame {Math.min(selectedFrameIndex + 1, frameCount)} / {Math.max(frameCount, 1)}</div>
        <div className="forecast-timeline-controls">
          <span className="badge">{modelName ?? "ConvLSTM multi-step"}</span>
          <span className="badge">{predictionEngine ?? "unknown engine"}</span>
          <span className={`badge forecast-trust-badge ${trustState.startsWith("Low-trust") ? "badge-error" : "badge-ok"}`}>{trustState}</span>
        </div>
      </header>

      <div className="forecast-timeline-slider">
        <button type="button" className="secondary-button" onClick={() => onSelectFrame(Math.max(0, selectedFrameIndex - 1))} disabled={timelineDisabled || selectedFrameIndex <= 0}>◀</button>
        <input
          type="range"
          min={0}
          max={Math.max(frameCount - 1, 0)}
          step={1}
          value={Math.min(selectedFrameIndex, Math.max(frameCount - 1, 0))}
          onChange={(event) => onSelectFrame(Number(event.currentTarget.value))}
          disabled={timelineDisabled}
        />
        <button type="button" className="secondary-button" onClick={() => onSelectFrame(Math.min(frameCount - 1, selectedFrameIndex + 1))} disabled={timelineDisabled || selectedFrameIndex >= frameCount - 1}>▶</button>
      </div>

      <div className="forecast-frame-ticks">
        {frameIndices.map((frameIndex) => (
          <button key={frameIndex} type="button" className={`frame-tick ${frameIndex === selectedFrameIndex ? "is-active" : ""}`} onClick={() => onSelectFrame(frameIndex)} disabled={timelineDisabled}>
            {formatFrameLabel(frameIndex, frameCount)}
          </button>
        ))}
      </div>

      <div className="forecast-timeline-stats">
        <span>Max: {maxConcentration != null ? maxConcentration.toFixed(3) : "—"}</span>
        <span>Mean: {meanConcentration != null ? meanConcentration.toFixed(3) : "—"}</span>
        <span>Affected cells: {affectedCells != null ? Math.round(affectedCells) : "—"}</span>
        {typeof metadata?.future_steps === "number" ? <span>Future steps: {metadata.future_steps}</span> : null}
      </div>

      {loading ? <p className="timeline-note">Loading frame…</p> : null}
      {errorMessage ? <p className="timeline-note">{errorMessage}. Showing latest available frame.</p> : null}
    </section>
  );
}
