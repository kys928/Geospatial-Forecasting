import type { ForecastFrameRasterPayload } from "../../sessions/types/session.types";

export interface PlumeRasterOverlay {
  imageDataUrl: string;
  coordinates: [[number, number], [number, number], [number, number], [number, number]];
  width: number;
  height: number;
  min: number;
  max: number;
  threshold: number | null;
}

const VISUAL_CUTOFF = 0.04;

export function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

export function gaussianKernel(radius: number, sigma: number): number[] {
  const safeRadius = Math.max(1, Math.floor(radius));
  const safeSigma = sigma > 0 ? sigma : 1.8;
  const kernel: number[] = [];
  let sum = 0;
  for (let i = -safeRadius; i <= safeRadius; i++) {
    const weight = Math.exp(-(i * i) / (2 * safeSigma * safeSigma));
    kernel.push(weight);
    sum += weight;
  }
  return kernel.map((weight) => weight / sum);
}

export function smoothGrid(grid: number[][], radius: number, sigma: number): number[][] {
  const h = grid.length;
  const w = h > 0 ? grid[0].length : 0;
  if (!h || !w) return [];
  const kernel = gaussianKernel(radius, sigma);
  const r = Math.floor(kernel.length / 2);

  const horizontal = Array.from({ length: h }, () => Array<number>(w).fill(0));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        const sampleX = Math.max(0, Math.min(w - 1, x + k));
        acc += grid[y][sampleX] * kernel[k + r];
      }
      horizontal[y][x] = acc;
    }
  }

  const vertical = Array.from({ length: h }, () => Array<number>(w).fill(0));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        const sampleY = Math.max(0, Math.min(h - 1, y + k));
        acc += horizontal[sampleY][x] * kernel[k + r];
      }
      vertical[y][x] = clamp01(acc);
    }
  }

  const softened = Array.from({ length: h }, () => Array<number>(w).fill(0));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const center = vertical[y][x];
      let maxNeighbor = center;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          const yy = Math.max(0, Math.min(h - 1, y + oy));
          const xx = Math.max(0, Math.min(w - 1, x + ox));
          maxNeighbor = Math.max(maxNeighbor, vertical[yy][xx]);
        }
      }
      softened[y][x] = clamp01(center * 0.8 + maxNeighbor * 0.2);
    }
  }

  return softened;
}

export function bilinearSample(grid: number[][], x: number, y: number): number {
  const h = grid.length;
  const w = h > 0 ? grid[0].length : 0;
  if (!h || !w) return 0;
  const x0 = Math.max(0, Math.min(w - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(h - 1, Math.floor(y)));
  const x1 = Math.max(0, Math.min(w - 1, x0 + 1));
  const y1 = Math.max(0, Math.min(h - 1, y0 + 1));
  const tx = clamp01(x - x0);
  const ty = clamp01(y - y0);

  const v00 = grid[y0][x0];
  const v10 = grid[y0][x1];
  const v01 = grid[y1][x0];
  const v11 = grid[y1][x1];
  const a = v00 * (1 - tx) + v10 * tx;
  const b = v01 * (1 - tx) + v11 * tx;
  return clamp01(a * (1 - ty) + b * ty);
}

export function colorRamp(value: number): [number, number, number, number] {
  const points: Array<[number, [number, number, number, number]]> = [
    [0.0, [255, 245, 160, 0]],
    [0.1, [255, 245, 160, 0.1]],
    [0.25, [255, 214, 90, 0.28]],
    [0.45, [251, 146, 60, 0.5]],
    [0.65, [239, 68, 68, 0.68]],
    [0.85, [185, 28, 28, 0.82]],
    [1.0, [127, 29, 29, 0.92]]
  ];
  const v = clamp01(value);
  for (let i = 1; i < points.length; i++) {
    const [rightValue, rightColor] = points[i];
    const [leftValue, leftColor] = points[i - 1];
    if (v <= rightValue) {
      const span = rightValue - leftValue || 1;
      const t = (v - leftValue) / span;
      return [
        Math.round(leftColor[0] + (rightColor[0] - leftColor[0]) * t),
        Math.round(leftColor[1] + (rightColor[1] - leftColor[1]) * t),
        Math.round(leftColor[2] + (rightColor[2] - leftColor[2]) * t),
        leftColor[3] + (rightColor[3] - leftColor[3]) * t
      ];
    }
  }
  return points[points.length - 1][1];
}

export function buildPlumeGridRasterOverlay(raster: ForecastFrameRasterPayload | null, options?: { width?: number; height?: number }): PlumeRasterOverlay | null {
  if (!raster || !Array.isArray(raster.grid) || raster.grid.length === 0) return null;
  const srcH = raster.grid.length;
  const srcW = Array.isArray(raster.grid[0]) ? raster.grid[0].length : 0;
  if (!srcW) return null;

  const bounds = raster.bounds;
  if (!bounds || !Number.isFinite(bounds.min_lon) || !Number.isFinite(bounds.min_lat) || !Number.isFinite(bounds.max_lon) || !Number.isFinite(bounds.max_lat)) {
    if (import.meta.env.DEV) console.debug("[forecast-map] invalid raster bounds", { bounds });
    return null;
  }

  const width = options?.width ?? 1024;
  const height = options?.height ?? 1024;

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  const cleanGrid = Array.from({ length: srcH }, (_, y) =>
    Array.from({ length: srcW }, (_, x) => {
      const value = Number(raster.grid[y][x]);
      const safe = Number.isFinite(value) ? value : 0;
      min = Math.min(min, safe);
      max = Math.max(max, safe);
      return safe;
    })
  );

  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= 0) return null;

  const threshold = Number.isFinite(raster.threshold) ? Number(raster.threshold) : max * 0.08;
  const denominator = Math.max(1e-12, max - threshold);

  let normalizedNonZeroCount = 0;
  const normalizedGrid = cleanGrid.map((row) =>
    row.map((value) => {
      if (value <= threshold) return 0;
      const normalized = clamp01((value - threshold) / denominator);
      if (normalized > 0) normalizedNonZeroCount += 1;
      return normalized;
    })
  );

  const smoothedGrid = smoothGrid(normalizedGrid, 4, 1.9);
  let smoothedNonZeroCount = 0;
  for (const row of smoothedGrid) {
    for (const cell of row) {
      if (cell > VISUAL_CUTOFF) smoothedNonZeroCount += 1;
    }
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const image = ctx.createImageData(width, height);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const gx = (px / Math.max(1, width - 1)) * (srcW - 1);
      const gy = (py / Math.max(1, height - 1)) * (srcH - 1);
      const sampled = bilinearSample(smoothedGrid, gx, gy);
      const idx = (py * width + px) * 4;
      if (sampled < 0.05 || sampled < VISUAL_CUTOFF) {
        image.data[idx + 3] = 0;
        continue;
      }
      const [r, g, b, baseA] = colorRamp(sampled);
      const alphaRamp = clamp01((sampled - 0.05) / 0.95);
      const alpha = baseA * alphaRamp;
      image.data[idx] = r;
      image.data[idx + 1] = g;
      image.data[idx + 2] = b;
      image.data[idx + 3] = Math.round(clamp01(alpha) * 255);
    }
  }
  ctx.putImageData(image, 0, 0);

  if (import.meta.env.DEV) {
    console.debug("[forecast-map] grid raster built", {
      shape: [srcH, srcW],
      min,
      max,
      threshold,
      normalizedNonZeroCount,
      smoothedNonZeroCount,
      bounds,
      imageWidth: width,
      imageHeight: height
    });
  }

  return {
    imageDataUrl: canvas.toDataURL("image/png"),
    coordinates: [
      [bounds.min_lon, bounds.max_lat],
      [bounds.max_lon, bounds.max_lat],
      [bounds.max_lon, bounds.min_lat],
      [bounds.min_lon, bounds.min_lat]
    ],
    width,
    height,
    min: raster.min,
    max: raster.max,
    threshold: raster.threshold
  };
}
