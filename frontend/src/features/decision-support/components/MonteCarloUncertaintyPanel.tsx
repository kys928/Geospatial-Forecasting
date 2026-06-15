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

function formatHectares(value: unknown): string {
  const numberValue = toFiniteNumber(value);
  return numberValue === null ? "Unavailable" : `${numberValue.toFixed(1)} ha`;
}

function parseHistogram(value: unknown): HistogramBin[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (!isRecord(item)) return null;
      const binStart = toFiniteNumber(item.bin_start);
      const binEnd = toFiniteNumber(item.bin_end);
      const count = toFiniteNumber(item.count);
      if (binStart === null || binEnd === null || count === null || count < 0) return null;
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
  const showCentralMarker = centralEstimate !== null && histogram.some((bin) => centralEstimate >= bin.bin_start && centralEstimate <= bin.bin_end);

  return <section className="panel monte-carlo-uncertainty-panel" aria-labelledby="monte-carlo-uncertainty-title">
    <div className="uncertainty-panel-header">
      <div>
        <h3 id="monte-carlo-uncertainty-title">ConvLSTM Prediction Uncertainty</h3>
        <p>Monte Carlo estimate of plume prediction stability</p>
      </div>
    </div>

    {histogram.length === 0 || maxCount <= 0 ? <p className="uncertainty-empty-state">Monte Carlo uncertainty data is not available for this forecast.</p> : <>
      <div className="uncertainty-summary-grid" aria-label="Monte Carlo uncertainty summary">
        <div className="uncertainty-stat">Uncertainty samples: <strong>{sampleCount === null ? "Unavailable" : sampleCount.toLocaleString()}</strong></div>
        <div className="uncertainty-stat">Central estimate: <strong>{formatHectares(centralEstimate)}</strong></div>
        <div className="uncertainty-stat">Likely range: <strong>{likelyLow === null || likelyHigh === null ? "Unavailable" : `${likelyLow.toFixed(1)}–${likelyHigh.toFixed(1)} ha`}</strong></div>
      </div>

      <div className="uncertainty-histogram" role="img" aria-label="Histogram of predicted affected area from Monte Carlo samples">
        {histogram.map((bin, index) => {
          const height = Math.max(4, (bin.count / maxCount) * 100);
          const containsCentralEstimate = showCentralMarker && centralEstimate !== null && centralEstimate >= bin.bin_start && centralEstimate <= bin.bin_end;
          return <div key={`${bin.bin_start}-${bin.bin_end}-${index}`} title={`${bin.bin_start.toFixed(1)}–${bin.bin_end.toFixed(1)} ha: ${bin.count} samples`} style={{ position: "relative", display: "flex", alignItems: "end", justifyContent: "center", flex: "1 1 8px", minWidth: "4px", height: "100%" }}>
            {containsCentralEstimate ? <span aria-label={`Central estimate ${centralEstimate.toFixed(1)} ha`} style={{ position: "absolute", top: "-6px", width: "2px", height: "calc(100% + 10px)", borderRadius: "999px", background: "#475569", zIndex: 1 }} /> : null}
            <span className="uncertainty-bar" style={{ height: `${height}%` }} aria-hidden="true" />
          </div>;
        })}
      </div>
      <div className="uncertainty-axis-label">Predicted affected area (ha)</div>
      <p className="uncertainty-note">Model-output uncertainty estimate, not live sensor confirmation.</p>
    </>}
  </section>;
}
