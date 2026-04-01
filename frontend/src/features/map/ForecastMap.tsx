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
  GeoJsonFeatureCollection,
  SelectedFeatureState
} from "../forecast/forecast.types";
import { isValidFeatureCollection } from "./layers";

interface ForecastMapProps {
  geojson: GeoJsonFeatureCollection | null;
  onSelectFeature: (feature: SelectedFeatureState | null) => void;
}

const SOURCE_ID = "forecast-source";

const DOMAIN_FILL_LAYER_ID = "forecast-domain-fill";
const DOMAIN_OUTLINE_LAYER_ID = "forecast-domain-outline";

const PLUME_LOW_FILL_LAYER_ID = "forecast-plume-low-fill";
const PLUME_MEDIUM_FILL_LAYER_ID = "forecast-plume-medium-fill";
const PLUME_HIGH_FILL_LAYER_ID = "forecast-plume-high-fill";

const PLUME_LOW_OUTLINE_LAYER_ID = "forecast-plume-low-outline";
const PLUME_MEDIUM_OUTLINE_LAYER_ID = "forecast-plume-medium-outline";
const PLUME_HIGH_OUTLINE_LAYER_ID = "forecast-plume-high-outline";

const SOURCE_POINT_LAYER_ID = "forecast-source-point";
const SOURCE_GLOW_LAYER_ID = "forecast-source-glow";

const MAP_STYLE_URL =
  import.meta.env.VITE_MAP_STYLE_URL ||
  "https://demotiles.maplibre.org/style.json";

function buildSelectedFeature(
  feature: maplibregl.MapGeoJSONFeature | undefined,
  fallbackId: string,
  fallbackTitle: string
): SelectedFeatureState | null {
  if (!feature) {
    return null;
  }

  const properties =
    feature.properties && typeof feature.properties === "object"
      ? (feature.properties as Record<string, unknown>)
      : null;

  return {
    id: feature.id != null ? String(feature.id) : fallbackId,
    title:
      properties && typeof properties.kind === "string"
        ? properties.kind
        : fallbackTitle,
    properties
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
  if (!isValidFeatureCollection(geojson)) {
    return;
  }

  const source = map.getSource(SOURCE_ID) as GeoJSONSource | undefined;
  if (!source) {
    return;
  }

  source.setData(geojson as GeoJSON.FeatureCollection);

  const bounds = getFeatureCollectionBounds(geojson);
  if (bounds && !bounds.isEmpty()) {
    map.fitBounds(bounds, {
      padding: {
        top: 72,
        right: 72,
        bottom: 72,
        left: 72
      },
      duration: hasFittedRef.current ? 700 : 1200,
      maxZoom: 18
    });
    hasFittedRef.current = true;
  }
}

export function ForecastMap({ geojson, onSelectFeature }: ForecastMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const hasFittedRef = useRef(false);
  const latestGeojsonRef = useRef<GeoJsonFeatureCollection | null>(geojson);

  useEffect(() => {
    latestGeojsonRef.current = geojson;
  }, [geojson]);

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
      antialias: true
    });

    map.addControl(
      new maplibregl.NavigationControl({ visualizePitch: true }),
      "top-right"
    );

    map.on("style.load", () => {
      add3DBuildingsIfPossible(map);
    });

    map.on("load", () => {
      map.addSource(SOURCE_ID, {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: []
        }
      });

      // Forecast domain: subtle / mostly for reference
      map.addLayer({
        id: DOMAIN_FILL_LAYER_ID,
        type: "fill",
        source: SOURCE_ID,
        paint: {
          "fill-color": "#94a3b8",
          "fill-opacity": 0.03
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
        source: SOURCE_ID,
        paint: {
          "line-color": "#94a3b8",
          "line-width": 1,
          "line-opacity": 0.35,
          "line-dasharray": [2, 2]
        },
        filter: [
          "all",
          ["==", "$type", "Polygon"],
          ["==", ["get", "kind"], "forecast_extent"]
        ]
      });

      // Low plume band
      map.addLayer({
        id: PLUME_LOW_FILL_LAYER_ID,
        type: "fill",
        source: SOURCE_ID,
        paint: {
          "fill-color": "#fde68a",
          "fill-opacity": 0.28
        },
        filter: [
          "all",
          ["==", ["get", "kind"], "plume_band_low"]
        ]
      });

      map.addLayer({
        id: PLUME_LOW_OUTLINE_LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        paint: {
          "line-color": "#facc15",
          "line-width": 1,
          "line-opacity": 0.35
        },
        filter: [
          "all",
          ["==", ["get", "kind"], "plume_band_low"]
        ]
      });

      // Medium plume band
      map.addLayer({
        id: PLUME_MEDIUM_FILL_LAYER_ID,
        type: "fill",
        source: SOURCE_ID,
        paint: {
          "fill-color": "#f59e0b",
          "fill-opacity": 0.42
        },
        filter: [
          "all",
          ["==", ["get", "kind"], "plume_band_medium"]
        ]
      });

      map.addLayer({
        id: PLUME_MEDIUM_OUTLINE_LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        paint: {
          "line-color": "#d97706",
          "line-width": 1.1,
          "line-opacity": 0.5
        },
        filter: [
          "all",
          ["==", ["get", "kind"], "plume_band_medium"]
        ]
      });

      // High plume band
      map.addLayer({
        id: PLUME_HIGH_FILL_LAYER_ID,
        type: "fill",
        source: SOURCE_ID,
        paint: {
          "fill-color": "#ef4444",
          "fill-opacity": 0.62
        },
        filter: [
          "all",
          ["==", ["get", "kind"], "plume_band_high"]
        ]
      });

      map.addLayer({
        id: PLUME_HIGH_OUTLINE_LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        paint: {
          "line-color": "#b91c1c",
          "line-width": 1.2,
          "line-opacity": 0.65
        },
        filter: [
          "all",
          ["==", ["get", "kind"], "plume_band_high"]
        ]
      });

      // Source marker
      map.addLayer({
        id: SOURCE_GLOW_LAYER_ID,
        type: "circle",
        source: SOURCE_ID,
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
        source: SOURCE_ID,
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

      for (const layerId of [
        SOURCE_POINT_LAYER_ID,
        PLUME_HIGH_FILL_LAYER_ID,
        PLUME_MEDIUM_FILL_LAYER_ID,
        PLUME_LOW_FILL_LAYER_ID,
        DOMAIN_FILL_LAYER_ID
      ]) {
        map.on("mouseenter", layerId, () => {
          map.getCanvas().style.cursor = "pointer";
        });

        map.on("mouseleave", layerId, () => {
          map.getCanvas().style.cursor = "";
        });
      }

      map.on("click", SOURCE_POINT_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(event.features?.[0], "source", "Source");
        onSelectFeature(selected);
      });

      map.on("click", PLUME_HIGH_FILL_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "plume-band-high",
          "High plume band"
        );
        onSelectFeature(selected);
      });

      map.on("click", PLUME_MEDIUM_FILL_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "plume-band-medium",
          "Medium plume band"
        );
        onSelectFeature(selected);
      });

      map.on("click", PLUME_LOW_FILL_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "plume-band-low",
          "Low plume band"
        );
        onSelectFeature(selected);
      });

      map.on("click", DOMAIN_FILL_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "forecast-domain",
          "Forecast domain"
        );
        onSelectFeature(selected);
      });

      map.on("click", (event: MapMouseEvent) => {
        const features = map.queryRenderedFeatures(event.point, {
          layers: [
            SOURCE_POINT_LAYER_ID,
            PLUME_HIGH_FILL_LAYER_ID,
            PLUME_MEDIUM_FILL_LAYER_ID,
            PLUME_LOW_FILL_LAYER_ID,
            DOMAIN_FILL_LAYER_ID
          ]
        });

        if (features.length === 0) {
          onSelectFeature(null);
        }
      });

      applyGeojsonToMap(map, latestGeojsonRef.current, hasFittedRef);
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

  return <div ref={mapContainerRef} className="forecast-map panel forecast-map-canvas" />;
}