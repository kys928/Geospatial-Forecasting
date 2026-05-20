import type { ForecastFrameRasterPayload } from "../../sessions/types/session.types";

export interface PlumeRasterOverlay {
  imageDataUrl: string;
  coordinates: [[number, number],[number, number],[number, number],[number, number]];
  width: number;
  height: number;
  min: number;
  max: number;
  threshold: number | null;
}

export function buildPlumeGridRasterOverlay(raster: ForecastFrameRasterPayload | null, options?: { width?: number; height?: number }): PlumeRasterOverlay | null {
  if (!raster || !Array.isArray(raster.grid) || raster.grid.length === 0) return null;
  const srcH = raster.grid.length;
  const srcW = Array.isArray(raster.grid[0]) ? raster.grid[0].length : 0;
  if (!srcW) return null;
  const width = options?.width ?? 768;
  const height = options?.height ?? 768;
  const threshold = typeof raster.threshold === "number" ? raster.threshold : 0;
  const max = Number.isFinite(raster.max) ? raster.max : 0;
  const low = Math.max(threshold, max * 0.08);
  const med = Math.max(low, max * 0.26);
  const high = Math.max(med, max * 0.55);

  const lowCanvas = document.createElement("canvas");
  lowCanvas.width = srcW;
  lowCanvas.height = srcH;
  const lowCtx = lowCanvas.getContext("2d");
  if (!lowCtx) return null;
  const image = lowCtx.createImageData(srcW, srcH);
  for (let y=0;y<srcH;y++) for (let x=0;x<srcW;x++) {
    const v = Number(raster.grid[y][x]);
    const idx=(y*srcW+x)*4;
    if (!Number.isFinite(v) || v <= threshold) continue;
    const edge = Math.min(1, Math.max(0, (v-threshold)/(Math.max(threshold*0.5, low-threshold)||1)));
    let c:[number,number,number,number] = [255,230,120,0.25*edge];
    if (v >= high) c=[153,27,27,0.92];
    else if (v >= med) c=[239,68,68,0.78];
    else if (v >= low) c=[251,146,60,0.55];
    image.data[idx]=c[0]; image.data[idx+1]=c[1]; image.data[idx+2]=c[2]; image.data[idx+3]=Math.round(c[3]*255);
  }
  lowCtx.putImageData(image,0,0);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.filter = "blur(1.2px)";
  ctx.drawImage(lowCanvas,0,0,width,height);

  return {
    imageDataUrl: canvas.toDataURL("image/png"),
    coordinates: [
      [raster.bounds.min_lon, raster.bounds.max_lat],
      [raster.bounds.max_lon, raster.bounds.max_lat],
      [raster.bounds.max_lon, raster.bounds.min_lat],
      [raster.bounds.min_lon, raster.bounds.min_lat]
    ],
    width,
    height,
    min: raster.min,
    max: raster.max,
    threshold: raster.threshold
  };
}
