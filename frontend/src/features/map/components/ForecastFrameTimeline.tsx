import { useEffect, useState } from "react";

interface ForecastFrameTimelineProps {
  frameCount: number;
  frameIndices: number[];
  selectedFrameIndex: number;
  onSelectFrame: (index: number) => void;
  loading: boolean;
  disabled?: boolean;
}

export function ForecastFrameTimeline({
  frameCount,
  frameIndices,
  selectedFrameIndex,
  onSelectFrame,
  loading,
  disabled = false
}: ForecastFrameTimelineProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!isPlaying || frameCount <= 1 || disabled) {
      return;
    }

    const timer = window.setInterval(() => {
      if (selectedFrameIndex >= frameCount - 1) {
        setIsPlaying(false);
        return;
      }
      onSelectFrame(selectedFrameIndex + 1);
    }, 900);

    return () => window.clearInterval(timer);
  }, [disabled, frameCount, isPlaying, onSelectFrame, selectedFrameIndex]);

  const timelineDisabled = disabled || frameCount <= 0;

  const handleManualSelect = (index: number) => {
    setIsPlaying(false);
    onSelectFrame(index);
  };

  const handlePlayPause = () => {
    if (timelineDisabled || frameCount <= 1) return;
    if (!isPlaying && selectedFrameIndex >= frameCount - 1) {
      onSelectFrame(frameIndices[0] ?? 0);
    }
    setIsPlaying((current) => !current);
  };

  return (
    <section className={`forecast-timeline-card ${timelineDisabled ? "is-disabled" : ""}`}>
      <button type="button" className="timeline-icon-button" onClick={() => handleManualSelect(Math.max(0, selectedFrameIndex - 1))} disabled={timelineDisabled || selectedFrameIndex <= 0} aria-label="Previous frame">‹</button>
      <button type="button" className="timeline-icon-button" onClick={handlePlayPause} disabled={timelineDisabled || frameCount <= 1} aria-label={isPlaying ? "Pause" : "Play"}>{isPlaying ? "⏸" : "▶"}</button>
      <button type="button" className="timeline-icon-button" onClick={() => handleManualSelect(Math.min(frameCount - 1, selectedFrameIndex + 1))} disabled={timelineDisabled || selectedFrameIndex >= frameCount - 1} aria-label="Next frame">›</button>
      <input type="range" min={0} max={Math.max(frameCount - 1, 0)} step={1} value={Math.min(selectedFrameIndex, Math.max(frameCount - 1, 0))} onChange={(event) => handleManualSelect(Number(event.currentTarget.value))} disabled={timelineDisabled} aria-label="Forecast frame" />
      <span className="timeline-count">{timelineDisabled ? "—/—" : `${Math.min(selectedFrameIndex + 1, Math.max(frameCount, 1))}/${Math.max(frameCount, 1)}`}</span>
      {loading ? <span className="timeline-inline-status">Loading…</span> : null}
    </section>
  );
}
