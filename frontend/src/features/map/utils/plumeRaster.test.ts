import type { GeoJsonFeatureCollection } from "../../forecast/types/forecast.types";
import { buildPlumeRasterOverlay } from "./plumeRaster";

function assert(condition: boolean, message: string): void { if (!condition) throw new Error(message); }

class FakeCtx {
  fillStyle = ""; filter = "none";
  beginPath(){} moveTo(){} lineTo(){} closePath(){} fill(){} drawImage(){}
  getImageData(_x:number,_y:number,w:number,h:number){ return { data: new Uint8ClampedArray(w*h*4) }; }
  putImageData(){}
}
class FakeCanvas {
  width = 0; height = 0; ctx = new FakeCtx();
  getContext(){ return this.ctx as unknown as CanvasRenderingContext2D; }
  toDataURL(){ return "data:image/png;base64,fake"; }
}
(globalThis as any).document = { createElement: () => new FakeCanvas() };

const sample: GeoJsonFeatureCollection = { type:"FeatureCollection", features:[
  { type:"Feature", geometry:{ type:"Polygon", coordinates:[[[1,1],[2,1],[2,2],[1,1]]]}, properties:{ kind:"plume_band_high"}},
  { type:"Feature", geometry:{ type:"MultiPolygon", coordinates:[[[[2,2],[3,2],[3,3],[2,2]]]]}, properties:{ kind:"plume_band_medium"}},
  { type:"Feature", geometry:{ type:"Point", coordinates:[4,4]}, properties:{ kind:"source"}},
  { type:"Feature", geometry:{ type:"Polygon", coordinates:[[[0,0],[5,0],[5,5],[0,0]]]}, properties:{ kind:"forecast_extent"}}
]};

const result = buildPlumeRasterOverlay(sample, { width: 256, height: 256, paddingRatio: 0.1 });
assert(Boolean(result), "raster should be built");
assert(result?.featureCount === 2, "only plume features should count");
assert(result?.coordinates[0][1] === result?.bounds.maxLat, "top-left maxLat");
assert(result?.coordinates[2][1] === result?.bounds.minLat, "bottom-right minLat");
assert((result?.bounds.minLon ?? 0) < 1, "padding should expand minLon");
assert((result?.bounds.maxLon ?? 0) > 3, "padding should expand maxLon");
assert(typeof result?.imageDataUrl === "string", "returns image url");

const empty: GeoJsonFeatureCollection = { type:"FeatureCollection", features:[
  { type:"Feature", geometry:{ type:"Point", coordinates:[0,0]}, properties:{ kind:"source"}}
]};
assert(buildPlumeRasterOverlay(empty) === null, "no plume features should return null");
