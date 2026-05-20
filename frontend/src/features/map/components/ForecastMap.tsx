import { useEffect, useRef } from "react";
import maplibregl, {
  GeoJSONSource,
  LngLatBounds,
  Map,
  MapLayerMouseEvent,
  MapMouseEvent
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import type {
  GeoJsonFeature,
  GeoJsonFeatureCollection,
  SelectedFeatureState
} from "../../forecast/types/forecast.types";
import { isValidFeatureCollection } from "../utils/layerBuilders";
import { MapCompassOverlay } from "./MapCompassOverlay";

interface ForecastMapProps {
  geojson: GeoJsonFeatureCollection | null;
  selectedFeature: SelectedFeatureState | null;
  onSelectFeature: (feature: SelectedFeatureState | null) => void;
  center?: [number, number] | null;
  autoFitKey?: string | null;
}

const FORECAST_SOURCE_ID = "forecast-source";
const SELECTED_SOURCE_ID = "selected-feature-source";

const DOMAIN_FILL_LAYER_ID = "forecast-domain-fill";
const DOMAIN_OUTLINE_LAYER_ID = "forecast-domain-outline";

const PLUME_LOW_HIT_LAYER_ID = "forecast-plume-low-hit";
const PLUME_MEDIUM_HIT_LAYER_ID = "forecast-plume-medium-hit";
const PLUME_HIGH_HIT_LAYER_ID = "forecast-plume-high-hit";

const PLUME_HEATMAP_LAYER_ID = "forecast-plume-heatmap";
const PLUME_FALLBACK_FILL_LAYER_ID = "forecast-plume-fallback-fill";

const PLUME_LOW_OUTLINE_LAYER_ID = "forecast-plume-low-outline";
const PLUME_MEDIUM_OUTLINE_LAYER_ID = "forecast-plume-medium-outline";
const PLUME_HIGH_OUTLINE_LAYER_ID = "forecast-plume-high-outline";

const SOURCE_HIT_LAYER_ID = "forecast-source-hit";
const SOURCE_POINT_LAYER_ID = "forecast-source-point";
const SOURCE_GLOW_LAYER_ID = "forecast-source-glow";

const SELECTED_POLYGON_OUTLINE_LAYER_ID = "selected-feature-polygon-outline";
const SELECTED_POLYGON_GLOW_LAYER_ID = "selected-feature-polygon-glow";
const SELECTED_SOURCE_RING_LAYER_ID = "selected-feature-source-ring";

const MAP_STYLE_URL =
  import.meta.env.VITE_MAP_STYLE_URL ||
  "https://demotiles.maplibre.org/style.json";

const INTERACTIVE_LAYER_ORDER = [
  SOURCE_HIT_LAYER_ID,
  SOURCE_POINT_LAYER_ID,
  PLUME_HIGH_HIT_LAYER_ID,
  PLUME_MEDIUM_HIT_LAYER_ID,
  PLUME_LOW_HIT_LAYER_ID
] as const;

function getFallbackTitle(kind: string | null): string {
  switch (kind) {
    case "source":
      return "Emission source";
    case "plume_cell":
      return "Plume cell";
    case "plume_band":
    case "plume_band_low":
    case "plume_band_medium":
    case "plume_band_high":
      return "Plume band";
    case "forecast_extent":
      return "Forecast domain";
    default:
      return "Feature details";
  }
}

function buildStableFeatureId(feature: GeoJsonFeature, index: number): string {
  const kind =
    feature.properties && typeof feature.properties.kind === "string"
      ? feature.properties.kind
      : "feature";

  const threshold =
    feature.properties &&
    (typeof feature.properties.threshold === "number" ||
      typeof feature.properties.threshold === "string")
      ? String(feature.properties.threshold)
      : "none";

  return `${kind}-${threshold}-${index}`;
}

function normalizeGeojson(
  geojson: GeoJsonFeatureCollection | null
): GeoJsonFeatureCollection | null {
  if (!isValidFeatureCollection(geojson)) {
    return geojson;
  }

  return {
    ...geojson,
    features: geojson.features.map((feature, index) => ({
      ...feature,
      id: feature.id ?? buildStableFeatureId(feature, index)
    }))
  };
}

function buildSelectedFeature(
  feature: maplibregl.MapGeoJSONFeature | undefined
): SelectedFeatureState | null {
  if (!feature) {
    return null;
  }

  const properties =
    feature.properties && typeof feature.properties === "object"
      ? (feature.properties as Record<string, unknown>)
      : null;

  const kind = typeof properties?.kind === "string" ? properties.kind : null;

  const normalizedFeature: GeoJsonFeature = {
    type: "Feature",
    geometry: feature.geometry as GeoJSON.Geometry,
    properties: properties ?? undefined,
    id: feature.id != null ? feature.id : undefined
  };

  return {
    id: feature.id != null ? String(feature.id) : `${kind ?? "feature"}-selected`,
    title: getFallbackTitle(kind),
    properties,
    geometry: feature.geometry as GeoJSON.Geometry,
    feature: normalizedFeature
  };
}

function extendBoundsFromCoordinates(bounds: LngLatBounds, coordinates: unknown): void {
  if (!Array.isArray(coordinates)) {
    return;
  }

  const isLonLatPair =
    coordinates.length >= 2 &&
    typeof coordinates[0] === "number" &&
    typeof coordinates[1] === "number";

  if (isLonLatPair) {
    const [lon, lat] = coordinates as [number, number];
    bounds.extend([lon, lat]);
    return;
  }

  for (const item of coordinates) {
    extendBoundsFromCoordinates(bounds, item);
  }
}

function getFeatureCollectionBounds(
  featureCollection: GeoJsonFeatureCollection
): LngLatBounds | null {
  const bounds = new LngLatBounds();

  for (const feature of featureCollection.features) {
    if (!feature.geometry || !("coordinates" in feature.geometry)) continue;
    extendBoundsFromCoordinates(bounds, feature.geometry.coordinates);
  }

  if (bounds.isEmpty()) {
    return null;
  }

  return bounds;
}

function add3DBuildingsIfPossible(map: Map) {
  const style = map.getStyle();
  if (!style?.layers) {
    return;
  }

  const alreadyExists = style.layers.some((layer) => layer.id === "forecast-3d-buildings");
  if (alreadyExists) {
    return;
  }

  const candidateLayer = style.layers.find(
    (layer: any) =>
      layer.type === "fill" &&
      typeof layer["source-layer"] === "string" &&
      layer["source-layer"].toLowerCase().includes("building")
  ) as any;

  if (!candidateLayer || !candidateLayer.source || !candidateLayer["source-layer"]) {
    return;
  }

  const labelLayer = style.layers.find(
    (layer: any) => layer.type === "symbol" && layer.layout?.["text-field"]
  ) as any;

  try {
    map.addLayer(
      {
        id: "forecast-3d-buildings",
        type: "fill-extrusion",
        source: candidateLayer.source,
        "source-layer": candidateLayer["source-layer"],
        minzoom: 14,
        paint: {
          "fill-extrusion-color": "#d6d7db",
          "fill-extrusion-height": [
            "coalesce",
            ["get", "render_height"],
            ["get", "height"],
            8
          ],
          "fill-extrusion-base": [
            "coalesce",
            ["get", "render_min_height"],
            ["get", "min_height"],
            0
          ],
          "fill-extrusion-opacity": 0.82
        }
      },
      labelLayer?.id
    );
  } catch {
    // Ignore styles that do not expose compatible building layers.
  }
}


function moveForecastLayersToTop(map: Map) {
  const orderedLayerIds = [
    DOMAIN_FILL_LAYER_ID,
    DOMAIN_OUTLINE_LAYER_ID,
    PLUME_LOW_HIT_LAYER_ID,
    PLUME_MEDIUM_HIT_LAYER_ID,
    PLUME_HIGH_HIT_LAYER_ID,
    PLUME_HEATMAP_LAYER_ID,
    PLUME_FALLBACK_FILL_LAYER_ID,
    PLUME_LOW_OUTLINE_LAYER_ID,
    PLUME_MEDIUM_OUTLINE_LAYER_ID,
    PLUME_HIGH_OUTLINE_LAYER_ID,
    SOURCE_HIT_LAYER_ID,
    SOURCE_GLOW_LAYER_ID,
    SOURCE_POINT_LAYER_ID,
    SELECTED_POLYGON_GLOW_LAYER_ID,
    SELECTED_POLYGON_OUTLINE_LAYER_ID,
    SELECTED_SOURCE_RING_LAYER_ID
  ];
  for (const layerId of orderedLayerIds) {
    try {
      if (map.getLayer(layerId)) map.moveLayer(layerId);
    } catch {
      // Keep map rendering resilient while style/layers are loading.
    }
  }
}

function applyGeojsonToMap(
  map: Map,
  geojson: GeoJsonFeatureCollection | null,
  shouldFitBounds: boolean
) {
  const normalized = normalizeGeojson(geojson);

  if (!isValidFeatureCollection(normalized)) {
    return;
  }

  const source = map.getSource(FORECAST_SOURCE_ID) as GeoJSONSource | undefined;
  if (!source) {
    if (import.meta.env.DEV) {
      console.debug("[forecast-map] source unavailable; delaying setData");
    }
    return;
  }

  source.setData(normalized as GeoJSON.FeatureCollection);

  const plumeOnly: GeoJsonFeatureCollection = { ...normalized, features: normalized.features.filter((f) => ["plume_band", "plume_band_low", "plume_band_medium", "plume_band_high", "plume_point", "plume_cell", "source"].includes(typeof f.properties?.kind === "string" ? f.properties.kind : "")) };
  const bounds = plumeOnly.features.length ? getFeatureCollectionBounds(plumeOnly) : null;
  const kinds = Array.from(new Set(normalized.features.map((f) => (typeof f.properties?.kind === "string" ? f.properties.kind : "unknown"))));
  if (import.meta.env.DEV) {
    console.debug("[forecast-map] setData", {
      featureCount: normalized.features.length,
      kinds,
      shouldFitBounds,
      bounds: bounds ? bounds.toArray() : null,
      firstFeature: normalized.features[0] ?? null
    });
  }
  if (shouldFitBounds && bounds && !bounds.isEmpty()) {
    map.fitBounds(bounds, {
      padding: { top: 56, right: 56, bottom: 56, left: 56 },
      duration: 850,
      maxZoom: 18
    });
  }
  moveForecastLayersToTop(map);
}

function applySelectedFeatureToMap(map: Map, selectedFeature: SelectedFeatureState | null) {
  const selectedSource = map.getSource(SELECTED_SOURCE_ID) as GeoJSONSource | undefined;
  if (!selectedSource) {
    return;
  }

  selectedSource.setData({
    type: "FeatureCollection",
    features: selectedFeature?.feature ? [selectedFeature.feature as GeoJSON.Feature] : []
  });
}

function getExistingInteractiveLayers(map: Map): string[] {
  return INTERACTIVE_LAYER_ORDER.filter((layerId) => Boolean(map.getLayer(layerId)));
}

export function ForecastMap({
  geojson,
  selectedFeature,
  onSelectFeature,
  center = null,
  autoFitKey = null
}: ForecastMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const hasFittedRef = useRef(false);
  const lastFitKeyRef = useRef<string | null>(null);
  const latestGeojsonRef = useRef<GeoJsonFeatureCollection | null>(geojson);
  const latestSelectedFeatureRef = useRef<SelectedFeatureState | null>(selectedFeature);

  useEffect(() => {
    latestGeojsonRef.current = geojson;
  }, [autoFitKey, geojson]);

  useEffect(() => {
    if (!center || !mapRef.current) return;
    const hasPlumeFeatures = Boolean(geojson?.features?.some((f) => ["Polygon", "Point"].includes(f.geometry?.type ?? "")));
    if (!hasPlumeFeatures) {
      mapRef.current.flyTo({ center, zoom: 11, duration: 800 });
    }
  }, [center, geojson]);

  useEffect(() => {
    latestSelectedFeatureRef.current = selectedFeature;
  }, [selectedFeature]);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) {
      return;
    }

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAP_STYLE_URL,
      center: center ?? [5.1214, 52.0907],
      zoom: 15,
      pitch: 58,
      bearing: -18,
      maxZoom: 20,
      antialias: true,
      clickTolerance: 8,
      maplibreLogo: false,
      attributionControl: false
    });

    map.addControl(
      new maplibregl.NavigationControl({ visualizePitch: true }),
      "top-right"
    );

    map.addControl(
      new maplibregl.AttributionControl({
        compact: true,
        customAttribution: "© OpenMapTiles · Data from OpenStreetMap"
      }),
      "bottom-right"
    );

    map.on("style.load", () => {
      add3DBuildingsIfPossible(map);
      moveForecastLayersToTop(map);
    });

    map.on("load", () => {
      map.addSource(FORECAST_SOURCE_ID, {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: []
        }
      });

      map.addSource(SELECTED_SOURCE_ID, {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: []
        }
      });

      map.addLayer({
        id: DOMAIN_FILL_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#94a3b8",
          "fill-opacity": 0.0
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          ["==", ["get", "kind"], "forecast_extent"]
        ]
      });

      map.addLayer({
        id: DOMAIN_OUTLINE_LAYER_ID,
        type: "line",
        source: FORECAST_SOURCE_ID,
        paint: {
          "line-color": "#94a3b8",
          "line-width": 0.8,
          "line-opacity": 0.18,
          "line-dasharray": [2, 2]
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          ["==", ["get", "kind"], "forecast_extent"]
        ]
      });

      map.addLayer({
        id: PLUME_LOW_HIT_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#facc15",
          "fill-opacity": 0.44
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          [
            "any",
            ["==", ["get", "kind"], "plume_band_low"],
            ["all", ["==", ["get", "kind"], "plume_band"], ["==", ["get", "band"], "low"]]
          ]
        ]
      });

      map.addLayer({
        id: PLUME_MEDIUM_HIT_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#f59e0b",
          "fill-opacity": 0.56
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          [
            "any",
            ["==", ["get", "kind"], "plume_band_medium"],
            ["all", ["==", ["get", "kind"], "plume_band"], ["==", ["get", "band"], "medium"]]
          ]
        ]
      });

      map.addLayer({
        id: PLUME_HIGH_HIT_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#ef4444",
          "fill-opacity": 0.68
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          [
            "any",
            ["==", ["get", "kind"], "plume_band_high"],
            ["all", ["==", ["get", "kind"], "plume_band"], ["==", ["get", "band"], "high"]]
          ]
        ]
      });

      map.addLayer({
        id: PLUME_HEATMAP_LAYER_ID,
        type: "heatmap",
        source: FORECAST_SOURCE_ID,
        paint: {
          "heatmap-weight": ["coalesce", ["to-number", ["get", "normalized_intensity"]], 0],
          "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 9, 0.8, 15, 1.15, 18, 1.35],
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 8, 12, 12, 20, 15, 28, 18, 42],
          "heatmap-opacity": 0.65,
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0, "rgba(255,255,255,0)",
            0.25, "rgba(255,244,179,0.4)",
            0.55, "rgba(245,158,11,0.75)",
            0.8, "rgba(239,93,42,0.9)",
            1, "rgba(220,38,38,0.98)"
          ]
        },
        filter: ["all", ["==", "$type", "Point"], ["==", ["get", "kind"], "plume_point"]]
      });
      map.addLayer({
        id: PLUME_FALLBACK_FILL_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "rgba(245,158,11,0.85)",
          "fill-opacity": [
            "case",
            [
              "in",
              ["get", "kind"],
              ["literal", ["plume_band", "plume_band_low", "plume_band_medium", "plume_band_high", "plume_cell", "source", "plume_point", "forecast_extent"]]
            ],
            0,
            0.35
          ]
        },
        filter: ["all", ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]]]
      });

      map.addLayer({
        id: PLUME_LOW_OUTLINE_LAYER_ID,
        type: "line",
        source: FORECAST_SOURCE_ID,
        paint: {
          "line-color": "#ca8a04",
          "line-width": 0.9,
          "line-opacity": 0.45,
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          [
            "any",
            ["==", ["get", "kind"], "plume_band_low"],
            ["all", ["==", ["get", "kind"], "plume_band"], ["==", ["get", "band"], "low"]]
          ]
        ]
      });

      map.addLayer({
        id: PLUME_MEDIUM_OUTLINE_LAYER_ID,
        type: "line",
        source: FORECAST_SOURCE_ID,
        paint: {
          "line-color": "#d97706",
          "line-width": 0.8,
          "line-opacity": 0.54,
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          [
            "any",
            ["==", ["get", "kind"], "plume_band_medium"],
            ["all", ["==", ["get", "kind"], "plume_band"], ["==", ["get", "band"], "medium"]]
          ]
        ]
      });

      map.addLayer({
        id: PLUME_HIGH_OUTLINE_LAYER_ID,
        type: "line",
        source: FORECAST_SOURCE_ID,
        paint: {
          "line-color": "#b91c1c",
          "line-width": 1,
          "line-opacity": 0.62,
        },
        filter: [
          "all",
          ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
          [
            "any",
            ["==", ["get", "kind"], "plume_band_high"],
            ["all", ["==", ["get", "kind"], "plume_band"], ["==", ["get", "band"], "high"]]
          ]
        ]
      });

      map.setPaintProperty(PLUME_HEATMAP_LAYER_ID, "heatmap-opacity-transition", { duration: 320 } as any);
      map.setPaintProperty(PLUME_HEATMAP_LAYER_ID, "heatmap-radius-transition", { duration: 320 } as any);
      map.setPaintProperty(PLUME_LOW_OUTLINE_LAYER_ID, "line-opacity-transition", { duration: 320 } as any);
      map.setPaintProperty(PLUME_MEDIUM_OUTLINE_LAYER_ID, "line-opacity-transition", { duration: 320 } as any);
      map.setPaintProperty(PLUME_HIGH_OUTLINE_LAYER_ID, "line-opacity-transition", { duration: 320 } as any);

      map.addLayer({
        id: SOURCE_HIT_LAYER_ID,
        type: "circle",
        source: FORECAST_SOURCE_ID,
        paint: {
          "circle-radius": 20,
          "circle-color": "#000000",
          "circle-opacity": 0
        },
        filter: [
          "all",
          ["==", "$type", "Point"],
          ["==", ["get", "kind"], "source"]
        ]
      });

      map.addLayer({
        id: SOURCE_GLOW_LAYER_ID,
        type: "circle",
        source: FORECAST_SOURCE_ID,
        paint: {
          "circle-radius": 18,
          "circle-color": "#ef4444",
          "circle-opacity": 0.16
        },
        filter: [
          "all",
          ["==", "$type", "Point"],
          ["==", ["get", "kind"], "source"]
        ]
      });

      map.addLayer({
        id: SOURCE_POINT_LAYER_ID,
        type: "circle",
        source: FORECAST_SOURCE_ID,
        paint: {
          "circle-radius": 8,
          "circle-color": "#ef4444",
          "circle-stroke-width": 2.5,
          "circle-stroke-color": "#ffffff"
        },
        filter: [
          "all",
          ["==", "$type", "Point"],
          ["==", ["get", "kind"], "source"]
        ]
      });

      map.addLayer({
        id: SELECTED_POLYGON_GLOW_LAYER_ID,
        type: "line",
        source: SELECTED_SOURCE_ID,
        paint: {
          "line-color": "#ffffff",
          "line-width": 7,
          "line-opacity": 0.12,
        },
        filter: ["any", ["==", "$type", "Polygon"], ["==", "$type", "MultiPolygon"]]
      });

      map.addLayer({
        id: SELECTED_POLYGON_OUTLINE_LAYER_ID,
        type: "line",
        source: SELECTED_SOURCE_ID,
        paint: {
          "line-color": "#ffffff",
          "line-width": 3,
          "line-opacity": 0.96
        },
        filter: ["any", ["==", "$type", "Polygon"], ["==", "$type", "MultiPolygon"]]
      });

      map.addLayer({
        id: SELECTED_SOURCE_RING_LAYER_ID,
        type: "circle",
        source: SELECTED_SOURCE_ID,
        paint: {
          "circle-radius": 14,
          "circle-color": "#ffffff",
          "circle-opacity": 0,
          "circle-stroke-width": 3,
          "circle-stroke-color": "#ffffff"
        },
        filter: ["==", "$type", "Point"]
      });

      const handleLayerClick = (event: MapLayerMouseEvent) => {
        onSelectFeature(buildSelectedFeature(event.features?.[0]));
      };

      for (const layerId of getExistingInteractiveLayers(map)) {
        map.on("click", layerId, handleLayerClick);

        map.on("mouseenter", layerId, () => {
          map.getCanvas().style.cursor = "pointer";
        });

        map.on("mouseleave", layerId, () => {
          map.getCanvas().style.cursor = "";
        });
      }

      map.on("click", (event: MapMouseEvent) => {
        const existingLayers = getExistingInteractiveLayers(map);

        if (existingLayers.length === 0) {
          onSelectFeature(null);
          return;
        }

        const features = map.queryRenderedFeatures(event.point, {
          layers: existingLayers
        });

        if (features.length === 0) {
          onSelectFeature(null);
        }
      });

      const shouldFit = autoFitKey ? autoFitKey !== lastFitKeyRef.current : !hasFittedRef.current;
      applyGeojsonToMap(map, latestGeojsonRef.current, shouldFit);
      moveForecastLayersToTop(map);
      if (shouldFit) {
        hasFittedRef.current = true;
        lastFitKeyRef.current = autoFitKey;
      }
      applySelectedFeatureToMap(map, latestSelectedFeatureRef.current);
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [onSelectFeature]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    const shouldFit = autoFitKey ? autoFitKey !== lastFitKeyRef.current : !hasFittedRef.current;
    applyGeojsonToMap(map, geojson, shouldFit);
    if (shouldFit) {
      hasFittedRef.current = true;
      lastFitKeyRef.current = autoFitKey;
    }
  }, [autoFitKey, geojson]);

  useEffect(() => {
    if (center && mapRef.current && (!geojson || !geojson.features?.some((f) => ["Polygon", "Point"].includes(f.geometry?.type ?? "")))) {
      mapRef.current.flyTo({ center, zoom: 11, duration: 800 });
    }
  }, [center]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    applySelectedFeatureToMap(map, selectedFeature);
  }, [selectedFeature]);

  return (
    <div ref={mapContainerRef} className="forecast-map panel forecast-map-canvas">
      <MapCompassOverlay />
    </div>
  );
}
