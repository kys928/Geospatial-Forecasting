import {
  bilinearSample,
  buildPlumeGridRasterOverlay,
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
assert(c1[0] <= 140 && c1[2] <= 40 && c1[3] >= 0.9, "colorRamp(1) should be dark-red with high alpha");

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

const nearZeroRaster: ForecastFrameRasterPayload = {
  ...validRaster,
  max: 1,
  threshold: 0.8,
  grid: Array.from({ length: 64 }, () => Array<number>(64).fill(0.79))
};
assert(buildPlumeGridRasterOverlay(nearZeroRaster) !== null, "under-threshold grids should still return an overlay object");
