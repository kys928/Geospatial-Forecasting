type Props = {
  uncertainty: Record<string, unknown>;
};

type HistogramBin = {
  bin_start: number;
  bin_end: number;
  count: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function toFiniteNumber(value: unknown): number | null {
  const numberValue = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function clampPercent(value: number): number {
  return Math.min(100, Math.max(0, value));
}

function formatHectares(value: unknown): string {
  const numberValue = toFiniteNumber(value);
  return numberValue === null ? "Unavailable" : `${numberValue.toFixed(1)} ha`;
}

function formatTick(value: number): string {
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

function parseHistogram(value: unknown): HistogramBin[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (!isRecord(item)) return null;
      const binStart = toFiniteNumber(item.bin_start);
      const binEnd = toFiniteNumber(item.bin_end);
      const count = toFiniteNumber(item.count);
      if (binStart === null || binEnd === null || count === null || count < 0 || binEnd <= binStart) return null;
      return { bin_start: binStart, bin_end: binEnd, count };
    })
    .filter((item): item is HistogramBin => item !== null);
}

export function MonteCarloUncertaintyPanel({ uncertainty }: Props) {
  const histogram = parseHistogram(uncertainty?.histogram);
  const maxCount = Math.max(...histogram.map((bin) => bin.count), 0);
  const likelyRange = isRecord(uncertainty?.likely_range) ? uncertainty.likely_range : null;
  const centralEstimate = toFiniteNumber(uncertainty?.central_estimate);
  const sampleCount = toFiniteNumber(uncertainty?.sample_count);
  const likelyLow = likelyRange ? toFiniteNumber(likelyRange.low) : null;
  const likelyHigh = likelyRange ? toFiniteNumber(likelyRange.high) : null;
  const chartMin = histogram.length > 0 ? Math.min(...histogram.map((bin) => bin.bin_start)) : 0;
  const chartMax = histogram.length > 0 ? Math.max(...histogram.map((bin) => bin.bin_end)) : 0;
  const chartSpan = chartMax - chartMin;
  const midpoint = chartMin + chartSpan / 2;
  const centralPositionPct = centralEstimate !== null && chartSpan > 0 ? clampPercent(((centralEstimate - chartMin) / chartSpan) * 100) : null;
  const likelyLeftPct = likelyLow !== null && chartSpan > 0 ? clampPercent(((likelyLow - chartMin) / chartSpan) * 100) : null;
  const likelyRightPct = likelyHigh !== null && chartSpan > 0 ? clampPercent(((likelyHigh - chartMin) / chartSpan) * 100) : null;
  const likelyWidthPct = likelyLeftPct !== null && likelyRightPct !== null ? Math.max(0, likelyRightPct - likelyLeftPct) : null;

  return <section className="panel monte-carlo-uncertainty-panel" aria-labelledby="monte-carlo-uncertainty-title">
    <div className="uncertainty-panel-header">
      <div>
        <h3 id="monte-carlo-uncertainty-title">ConvLSTM Prediction Uncertainty</h3>
        <p>Monte Carlo estimate of plume prediction stability</p>
      </div>
    </div>

    {histogram.length === 0 || maxCount <= 0 || chartSpan <= 0 ? <p className="uncertainty-empty-state">Monte Carlo uncertainty data is not available for this forecast.</p> : <>
      <div className="uncertainty-summary-grid" aria-label="Monte Carlo uncertainty summary">
        <div className="uncertainty-stat">
          <span className="uncertainty-stat-icon" aria-hidden="true">◎</span>
          <span className="uncertainty-stat-label">Central estimate</span>
          <strong className="uncertainty-stat-value">{formatHectares(centralEstimate)}</strong>
        </div>
        <div className="uncertainty-stat">
          <span className="uncertainty-stat-icon" aria-hidden="true">↔</span>
          <span className="uncertainty-stat-label">Likely range</span>
          <strong className="uncertainty-stat-value">{likelyLow === null || likelyHigh === null ? "Unavailable" : `${likelyLow.toFixed(1)}–${likelyHigh.toFixed(1)} ha`}</strong>
        </div>
        <div className="uncertainty-stat">
          <span className="uncertainty-stat-icon" aria-hidden="true">▥</span>
          <span className="uncertainty-stat-label">Uncertainty samples</span>
          <strong className="uncertainty-stat-value">{sampleCount === null ? "Unavailable" : sampleCount.toLocaleString()}</strong>
        </div>
      </div>

      <div className="uncertainty-chart-wrap">
        <div className="uncertainty-histogram" role="img" aria-label="Histogram of predicted affected area from Monte Carlo samples">
          {likelyLeftPct !== null && likelyWidthPct !== null && likelyWidthPct > 0 ? <>
            <span className="uncertainty-likely-band" style={{ left: `${likelyLeftPct}%`, width: `${likelyWidthPct}%` }} aria-hidden="true" />
            <span className="uncertainty-chart-label uncertainty-likely-label" style={{ left: `${clampPercent(likelyLeftPct + likelyWidthPct / 2)}%` }} aria-hidden="true">Likely range</span>
          </> : null}
          {centralPositionPct !== null ? <>
            <span className="uncertainty-central-marker" style={{ left: `${centralPositionPct}%` }} aria-hidden="true" />
            <span className="uncertainty-chart-label uncertainty-central-label" style={{ left: `${centralPositionPct}%` }} aria-hidden="true">Central estimate</span>
          </> : null}
          <div className="uncertainty-bars" aria-hidden="true">
            {histogram.map((bin, index) => {
              const height = Math.max(3, (bin.count / maxCount) * 100);
              return <span key={`${bin.bin_start}-${bin.bin_end}-${index}`} className="uncertainty-bar" title={`${bin.bin_start.toFixed(1)}–${bin.bin_end.toFixed(1)} ha: ${bin.count} samples`} style={{ height: `${height}%` }} />;
            })}
          </div>
        </div>
        <div className="uncertainty-axis-ticks" aria-hidden="true">
          <span>{formatTick(chartMin)}</span>
          <span>{formatTick(midpoint)}</span>
          <span>{formatTick(chartMax)}</span>
        </div>
        <div className="uncertainty-axis-label">Predicted affected area (ha)</div>
      </div>
      <p className="uncertainty-note"><span aria-hidden="true">ⓘ</span> Model-output uncertainty estimate, not live sensor confirmation.</p>
    </>}
  </section>;
}
