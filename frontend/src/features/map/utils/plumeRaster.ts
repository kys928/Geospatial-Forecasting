import type { GeoJsonFeature, GeoJsonFeatureCollection } from "../../forecast/types/forecast.types";

export type PlumeRasterResult = {
  imageDataUrl: string;
  coordinates: [[number, number], [number, number], [number, number], [number, number]];
  width: number;
  height: number;
  featureCount: number;
  bounds: { minLon: number; minLat: number; maxLon: number; maxLat: number };
};

type RasterizeOptions = {
  width?: number;
  height?: number;
  paddingRatio?: number;
  blurPx?: number;
};

const PLUME_KINDS = new Set(["plume_band_high", "plume_band_medium", "plume_band_low", "plume_band", "plume_cell"]);

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function getKind(feature: GeoJsonFeature): string {
  return typeof feature.properties?.kind === "string" ? feature.properties.kind : "";
}

function getWeight(feature: GeoJsonFeature): number {
  const kind = getKind(feature);
  if (kind === "plume_band_high") return 1;
  if (kind === "plume_band_medium") return 0.58;
  if (kind === "plume_band_low") return 0.28;
  if (kind === "plume_cell") {
    const normalized = feature.properties?.normalized_intensity;
    if (typeof normalized === "number" && Number.isFinite(normalized)) return clamp01(normalized);
    return 0.65;
  }
  if (kind === "plume_band") {
    const band = typeof feature.properties?.band === "string" ? feature.properties.band : "";
    if (band === "high") return 1;
    if (band === "medium") return 0.58;
    if (band === "low") return 0.28;
  }
  return 0.5;
}

function eachLonLat(input: unknown, fn: (lon: number, lat: number) => void): void {
  if (!Array.isArray(input)) return;
  if (input.length >= 2 && typeof input[0] === "number" && typeof input[1] === "number") {
    fn(input[0], input[1]);
    return;
  }
  for (const value of input) eachLonLat(value, fn);
}

function drawPolygonRings(
  ctx: CanvasRenderingContext2D,
  rings: number[][][],
  project: (lon: number, lat: number) => [number, number]
): void {
  ctx.beginPath();
  for (const ring of rings) {
    ring.forEach((pair, i) => {
      const [x, y] = project(pair[0], pair[1]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
  }
  ctx.fill("evenodd");
}

export function buildPlumeRasterOverlay(
  geojson: GeoJsonFeatureCollection | null,
  options: RasterizeOptions = {}
): PlumeRasterResult | null {
  if (!geojson || geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) return null;

  const features = geojson.features.filter((feature) => PLUME_KINDS.has(getKind(feature)));
  if (!features.length) return null;

  let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
  for (const feature of features) {
    if (!feature.geometry || !("coordinates" in feature.geometry)) continue;
    eachLonLat(feature.geometry.coordinates, (lon, lat) => {
      minLon = Math.min(minLon, lon);
      minLat = Math.min(minLat, lat);
      maxLon = Math.max(maxLon, lon);
      maxLat = Math.max(maxLat, lat);
    });
  }
  if (!Number.isFinite(minLon) || !Number.isFinite(minLat) || !Number.isFinite(maxLon) || !Number.isFinite(maxLat)) return null;

  const paddingRatio = options.paddingRatio ?? 0.1;
  const lonSpan = Math.max(0.0001, maxLon - minLon);
  const latSpan = Math.max(0.0001, maxLat - minLat);
  minLon -= lonSpan * paddingRatio;
  maxLon += lonSpan * paddingRatio;
  minLat -= latSpan * paddingRatio;
  maxLat += latSpan * paddingRatio;

  const width = options.width ?? 768;
  const height = options.height ?? 768;
  const blurPx = options.blurPx ?? 10;

  const intensityCanvas = document.createElement("canvas");
  intensityCanvas.width = width;
  intensityCanvas.height = height;
  const ictx = intensityCanvas.getContext("2d");
  if (!ictx) return null;

  const project = (lon: number, lat: number): [number, number] => {
    const x = ((lon - minLon) / (maxLon - minLon)) * (width - 1);
    const y = ((maxLat - lat) / (maxLat - minLat)) * (height - 1);
    return [x, y];
  };

  features.sort((a, b) => getWeight(a) - getWeight(b));
  for (const feature of features) {
    if (!feature.geometry) continue;
    const weight = getWeight(feature);
    ictx.fillStyle = `rgba(255,255,255,${clamp01(weight)})`;

    if (feature.geometry.type === "Polygon") {
      drawPolygonRings(ictx, feature.geometry.coordinates as number[][][], project);
    } else if (feature.geometry.type === "MultiPolygon") {
      for (const polygon of feature.geometry.coordinates as number[][][][]) {
        drawPolygonRings(ictx, polygon, project);
      }
    }
  }

  const outCanvas = document.createElement("canvas");
  outCanvas.width = width;
  outCanvas.height = height;
  const octx = outCanvas.getContext("2d");
  if (!octx) return null;

  octx.filter = `blur(${blurPx}px)`;
  octx.drawImage(intensityCanvas, 0, 0, width, height);
  octx.filter = "none";

  const imageData = octx.getImageData(0, 0, width, height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const v = data[i] / 255;
    if (v <= 0.01) {
      data[i + 3] = 0;
      continue;
    }
    const intensity = clamp01(v * 1.12);
    const r = intensity > 0.66 ? 239 : intensity > 0.33 ? 249 : 250;
    const g = intensity > 0.66 ? 68 : intensity > 0.33 ? 115 : 204;
    const b = intensity > 0.66 ? 68 : intensity > 0.33 ? 22 : 21;
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = Math.round(255 * Math.pow(intensity, 0.92) * 0.88);
  }
  octx.putImageData(imageData, 0, 0);

  return {
    imageDataUrl: outCanvas.toDataURL("image/png"),
    coordinates: [
      [minLon, maxLat],
      [maxLon, maxLat],
      [maxLon, minLat],
      [minLon, minLat]
    ],
    width,
    height,
    featureCount: features.length,
    bounds: { minLon, minLat, maxLon, maxLat }
  };
}
