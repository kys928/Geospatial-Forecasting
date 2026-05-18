import { useEffect, useMemo, useState } from "react";

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
  missingChannelsCount?: number | null;
  observedFrameCount?: number | null;
  requiredFrameCount?: number | null;
  meteorologySourceKind?: string | null;
  modelName?: string | null;
  predictionEngine?: string | null;
  frameDurationSeconds?: number | null;
  errorMessage?: string | null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function getForecastHours(frameIndex: number, frameDurationSeconds: number | null): number {
  const seconds = frameDurationSeconds && frameDurationSeconds > 0 ? frameDurationSeconds : 3600;
  return Math.max(Math.round(((frameIndex + 1) * seconds) / 3600), 1);
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
  missingChannelsCount = null,
  observedFrameCount = null,
  requiredFrameCount = null,
  meteorologySourceKind = null,
  modelName,
  predictionEngine,
  frameDurationSeconds,
  errorMessage
}: ForecastFrameTimelineProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [lastSelectionSource, setLastSelectionSource] = useState<"manual" | "playback">("manual");

  useEffect(() => {
    if (lastSelectionSource === "manual" && isPlaying) {
      setIsPlaying(false);
    }
  }, [isPlaying, lastSelectionSource, selectedFrameIndex]);

  useEffect(() => {
    if (!isPlaying || frameCount <= 1 || disabled) {
      return;
    }
    const timer = window.setInterval(() => {
      if (selectedFrameIndex >= frameCount - 1) {
        setIsPlaying(false);
        return;
      }
      setLastSelectionSource("playback");
      onSelectFrame(selectedFrameIndex + 1);
    }, 900);

    return () => window.clearInterval(timer);
  }, [disabled, frameCount, isPlaying, onSelectFrame, selectedFrameIndex]);

  const summaryStats = selectedFrameSummary?.summary_statistics as Record<string, unknown> | undefined;
  const maxConcentration = asNumber(summaryStats?.max_concentration);
  const meanConcentration = asNumber(summaryStats?.mean_concentration);
  const affectedCells = asNumber(summaryStats?.affected_cells_above_threshold);

  const trustState = useMemo(() => {
    if (inputMode === "degraded" || predictionTrust === "low") return "Low-trust: degraded input";
    if (inputMode === "complete") return "Input complete";
    return "Trust metadata unavailable";
  }, [inputMode, predictionTrust]);

  const timelineDisabled = disabled || frameCount <= 0;
  const selectedHours = getForecastHours(selectedFrameIndex, frameDurationSeconds ?? null);

  const handleManualSelect = (index: number) => {
    setLastSelectionSource("manual");
    onSelectFrame(index);
  };

  const handlePlayPause = () => {
    if (timelineDisabled || frameCount <= 1) return;
    if (!isPlaying && selectedFrameIndex >= frameCount - 1) {
      setLastSelectionSource("playback");
      onSelectFrame(frameIndices[0] ?? 0);
    }
    setIsPlaying((current) => !current);
  };

  return (
    <section className={`forecast-timeline-card ${timelineDisabled ? "is-disabled" : ""}`}>
      <header className="forecast-timeline-header">
        <div>
          <strong>ConvLSTM forecast timeline</strong>
          <p className="timeline-note">{frameCount || 0} future frames · hourly steps</p>
        </div>
        <div className="forecast-horizon-badge">+{selectedHours}h</div>
      </header>

      <div className="forecast-timeline-controls">
        <span className="badge">Forecast horizon +{selectedHours}h</span>
        <span className="badge">Step {Math.min(selectedFrameIndex + 1, Math.max(frameCount, 1))} / {Math.max(frameCount, 1)}</span>
        <span className="badge">{modelName ?? "ConvLSTM multi-step"}</span>
        <span className="badge">{predictionEngine ?? "torch_multistep"}</span>
        <span className={`badge forecast-trust-badge ${trustState.startsWith("Low-trust") ? "badge-error" : "badge-ok"}`}>{trustState}</span>
      </div>

      <div className="forecast-timeline-slider">
        <button type="button" className="secondary-button" onClick={handlePlayPause} disabled={timelineDisabled || frameCount <= 1}>{isPlaying ? "Pause" : "Play"}</button>
        <button type="button" className="secondary-button" onClick={() => handleManualSelect(Math.max(0, selectedFrameIndex - 1))} disabled={timelineDisabled || selectedFrameIndex <= 0}>Prev</button>
        <input type="range" min={0} max={Math.max(frameCount - 1, 0)} step={1} value={Math.min(selectedFrameIndex, Math.max(frameCount - 1, 0))} onChange={(event) => handleManualSelect(Number(event.currentTarget.value))} disabled={timelineDisabled} />
        <button type="button" className="secondary-button" onClick={() => handleManualSelect(Math.min(frameCount - 1, selectedFrameIndex + 1))} disabled={timelineDisabled || selectedFrameIndex >= frameCount - 1}>Next</button>
      </div>

      <div className="forecast-frame-ticks">
        {frameIndices.map((frameIndex) => {
          const hours = getForecastHours(frameIndex, frameDurationSeconds ?? null);
          return <button key={frameIndex} type="button" className={`frame-tick ${frameIndex === selectedFrameIndex ? "is-active" : ""}`} onClick={() => handleManualSelect(frameIndex)} disabled={timelineDisabled}>+{hours}h</button>;
        })}
      </div>

      <div className="forecast-timeline-stats">
        <span>Max: {maxConcentration != null ? maxConcentration.toFixed(3) : "—"}</span>
        <span>Mean: {meanConcentration != null ? meanConcentration.toFixed(3) : "—"}</span>
        <span>Affected cells: {affectedCells != null ? Math.round(affectedCells) : "—"}</span>
        {missingChannelsCount != null ? <span>Missing channels: {missingChannelsCount}</span> : null}
        {observedFrameCount != null && requiredFrameCount != null ? <span>Observed/required frames: {observedFrameCount}/{requiredFrameCount}</span> : null}
        {meteorologySourceKind ? <span>Meteorology: {meteorologySourceKind}</span> : null}
        {typeof metadata?.future_steps === "number" ? <span>Future steps: {metadata.future_steps}</span> : null}
      </div>

      {timelineDisabled ? <p className="timeline-note">Run forecast to enable timeline</p> : null}
      {frameCount === 1 ? <p className="timeline-note">Single-frame forecast</p> : null}
      {loading ? <p className="timeline-note timeline-loading">Loading selected frame…</p> : null}
      {errorMessage ? <p className="timeline-note">Could not load selected frame. Showing latest available frame.</p> : null}
    </section>
  );
}
