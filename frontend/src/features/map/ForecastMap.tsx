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
} from "../forecast/forecast.types";
import { isValidFeatureCollection } from "./layers";

interface ForecastMapProps {
  geojson: GeoJsonFeatureCollection | null;
  selectedFeature: SelectedFeatureState | null;
  onSelectFeature: (feature: SelectedFeatureState | null) => void;
}

const FORECAST_SOURCE_ID = "forecast-source";
const SELECTED_SOURCE_ID = "selected-feature-source";

const DOMAIN_FILL_LAYER_ID = "forecast-domain-fill";
const DOMAIN_OUTLINE_LAYER_ID = "forecast-domain-outline";

const PLUME_LOW_HIT_LAYER_ID = "forecast-plume-low-hit";
const PLUME_MEDIUM_HIT_LAYER_ID = "forecast-plume-medium-hit";
const PLUME_HIGH_HIT_LAYER_ID = "forecast-plume-high-hit";

const PLUME_LOW_FILL_LAYER_ID = "forecast-plume-low-fill";
const PLUME_MEDIUM_FILL_LAYER_ID = "forecast-plume-medium-fill";
const PLUME_HIGH_FILL_LAYER_ID = "forecast-plume-high-fill";

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
    case "plume_band_high":
      return "High plume band";
    case "plume_band_medium":
      return "Medium plume band";
    case "plume_band_low":
      return "Low plume band";
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
    if (!feature.geometry) continue;
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

function applyGeojsonToMap(
  map: Map,
  geojson: GeoJsonFeatureCollection | null,
  hasFittedRef: React.MutableRefObject<boolean>
) {
  const normalized = normalizeGeojson(geojson);

  if (!isValidFeatureCollection(normalized)) {
    return;
  }

  const source = map.getSource(FORECAST_SOURCE_ID) as GeoJSONSource | undefined;
  if (!source) {
    return;
  }

  source.setData(normalized as GeoJSON.FeatureCollection);

  const bounds = getFeatureCollectionBounds(normalized);
  if (bounds && !bounds.isEmpty()) {
    map.fitBounds(bounds, {
      padding: {
        top: 56,
        right: 56,
        bottom: 56,
        left: 56
      },
      duration: hasFittedRef.current ? 700 : 1100,
      maxZoom: 18
    });
    hasFittedRef.current = true;
  }
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
  onSelectFeature
}: ForecastMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const hasFittedRef = useRef(false);
  const latestGeojsonRef = useRef<GeoJsonFeatureCollection | null>(geojson);
  const latestSelectedFeatureRef = useRef<SelectedFeatureState | null>(selectedFeature);

  useEffect(() => {
    latestGeojsonRef.current = geojson;
  }, [geojson]);

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
      center: [5.1214, 52.0907],
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
          ["==", "$type", "Polygon"],
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
          ["==", "$type", "Polygon"],
          ["==", ["get", "kind"], "forecast_extent"]
        ]
      });

      map.addLayer({
        id: PLUME_LOW_HIT_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#000000",
          "fill-opacity": 0
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_low"]]
      });

      map.addLayer({
        id: PLUME_MEDIUM_HIT_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#000000",
          "fill-opacity": 0
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_medium"]]
      });

      map.addLayer({
        id: PLUME_HIGH_HIT_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#000000",
          "fill-opacity": 0
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_high"]]
      });

      map.addLayer({
        id: PLUME_LOW_FILL_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#fde68a",
          "fill-opacity": 0.26
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_low"]]
      });

      map.addLayer({
        id: PLUME_LOW_OUTLINE_LAYER_ID,
        type: "line",
        source: FORECAST_SOURCE_ID,
        paint: {
          "line-color": "#facc15",
          "line-width": 1.0,
          "line-opacity": 0.28
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_low"]]
      });

      map.addLayer({
        id: PLUME_MEDIUM_FILL_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#f59e0b",
          "fill-opacity": 0.44
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_medium"]]
      });

      map.addLayer({
        id: PLUME_MEDIUM_OUTLINE_LAYER_ID,
        type: "line",
        source: FORECAST_SOURCE_ID,
        paint: {
          "line-color": "#d97706",
          "line-width": 1.2,
          "line-opacity": 0.34
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_medium"]]
      });

      map.addLayer({
        id: PLUME_HIGH_FILL_LAYER_ID,
        type: "fill",
        source: FORECAST_SOURCE_ID,
        paint: {
          "fill-color": "#ef4444",
          "fill-opacity": 0.62
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_high"]]
      });

      map.addLayer({
        id: PLUME_HIGH_OUTLINE_LAYER_ID,
        type: "line",
        source: FORECAST_SOURCE_ID,
        paint: {
          "line-color": "#b91c1c",
          "line-width": 1.35,
          "line-opacity": 0.42
        },
        filter: ["all", ["==", ["get", "kind"], "plume_band_high"]]
      });

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
          "line-opacity": 0.24
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

      applyGeojsonToMap(map, latestGeojsonRef.current, hasFittedRef);
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

    applyGeojsonToMap(map, geojson, hasFittedRef);
  }, [geojson]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    applySelectedFeatureToMap(map, selectedFeature);
  }, [selectedFeature]);

  return <div ref={mapContainerRef} className="forecast-map panel forecast-map-canvas" />;
}