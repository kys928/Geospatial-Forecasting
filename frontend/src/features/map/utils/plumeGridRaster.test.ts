import {
  bilinearSample,
  buildPlumeGridRasterOverlay,
  computeVisualWindow,
  colorRamp,
  smoothGrid
} from "./plumeGridRaster";
import type { ForecastFrameRasterPayload } from "../../sessions/types/session.types";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

class FakeCtx {
  createImageData(w: number, h: number) {
    return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h } as ImageData;
  }
  putImageData() {}
}
class FakeCanvas {
  width = 0;
  height = 0;
  ctx = new FakeCtx();
  getContext() {
    return this.ctx as unknown as CanvasRenderingContext2D;
  }
  toDataURL() {
    return "data:image/png;base64,fake";
  }
}
(globalThis as any).document = { createElement: () => new FakeCanvas() };

const centered = Array.from({ length: 5 }, () => Array<number>(5).fill(0));
centered[2][2] = 1;
const smoothed = smoothGrid(centered, 3, 1.8);
assert(smoothed[2][1] > 0, "smoothing should spread intensity into neighbor cells");

const c0 = colorRamp(0);
assert(c0[3] === 0, "colorRamp(0) alpha should be 0");
const c1 = colorRamp(1);
assert(c1[0] <= 160 && c1[2] <= 40 && c1[3] >= 0.9, "colorRamp(1) should be dark-red with high alpha");

assert(bilinearSample([[0, 1], [1, 1]], 0.5, 0.5) > 0.6, "bilinear sampling should interpolate neighboring values");

const invalidBoundsRaster: ForecastFrameRasterPayload = {
  forecast_id: "f1",
  session_id: "s1",
  frame_index: 0,
  shape: [64, 64],
  min: 0,
  max: 1,
  mean: 0,
  threshold: 0.1,
  bounds: { min_lon: Number.NaN, min_lat: 1, max_lon: 2, max_lat: 3 },
  georeferencing_status: "ok",
  grid: Array.from({ length: 64 }, () => Array<number>(64).fill(0))
};
assert(buildPlumeGridRasterOverlay(invalidBoundsRaster) === null, "invalid bounds should return null");

const validGrid = Array.from({ length: 64 }, () => Array<number>(64).fill(0));
validGrid[32][32] = 10;
const validRaster: ForecastFrameRasterPayload = {
  forecast_id: "f1",
  session_id: "s1",
  frame_index: 1,
  shape: [64, 64],
  min: 0,
  max: 10,
  mean: 0.02,
  threshold: 0.2,
  bounds: { min_lon: 4, min_lat: 50, max_lon: 6, max_lat: 52 },
  georeferencing_status: "ok",
  grid: validGrid
};
const overlay = buildPlumeGridRasterOverlay(validRaster, { width: 256, height: 256 });
assert(Boolean(overlay?.imageDataUrl), "valid grid should create raster image");
assert(overlay?.coordinates[0][0] === 4 && overlay?.coordinates[2][1] === 50, "coordinate ordering should be preserved");

const belowModelThresholdRaster: ForecastFrameRasterPayload = {
  ...validRaster,
  max: 1,
  threshold: 0.9,
  grid: Array.from({ length: 64 }, (_, y) => Array.from({ length: 64 }, (_, x) => (x > 8 && x < 18 && y > 8 && y < 18 ? 0.09 : 0.03)))
};
assert(buildPlumeGridRasterOverlay(belowModelThresholdRaster) !== null, "values below model threshold but above visual floor should render");

const highThresholdPositiveRaster: ForecastFrameRasterPayload = {
  ...validRaster,
  threshold: 9.9,
  grid: Array.from({ length: 64 }, () => Array<number>(64).fill(0.2))
};
assert(buildPlumeGridRasterOverlay(highThresholdPositiveRaster) !== null, "high model threshold should not blank positive grid");

const sparsePositiveGrid: ForecastFrameRasterPayload = {
  ...validRaster,
  threshold: 0.5,
  grid: Array.from({ length: 64 }, (_, y) => Array.from({ length: 64 }, (_, x) => ((x === 15 && y === 20) || (x === 44 && y === 50) ? 1 : 0)))
};
assert(buildPlumeGridRasterOverlay(sparsePositiveGrid) !== null, "sparse positive grid should still create visible raster");

const allZeroRaster: ForecastFrameRasterPayload = {
  ...validRaster,
  max: 0,
  threshold: 0.2,
  grid: Array.from({ length: 64 }, () => Array<number>(64).fill(0))
};
assert(buildPlumeGridRasterOverlay(allZeroRaster) === null, "all-zero grid should return null");

const windowStats = computeVisualWindow([0, 0.02, 0.08, 0.2, 0.4, 1.0]);
assert(Boolean(windowStats), "computeVisualWindow should return stats for positive values");
assert((windowStats?.visualFloor ?? 0) >= 0.01, "visualFloor should be at least 1% of max");
assert((windowStats?.visualCeil ?? 0) >= (windowStats?.p95 ?? 0), "visualCeil should be at least p95");
