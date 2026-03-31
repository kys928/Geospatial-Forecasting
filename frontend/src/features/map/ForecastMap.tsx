import { useEffect, useRef } from "react";
import maplibregl, {
  GeoJSONSource,
  LngLatBounds,
  Map,
  MapLayerMouseEvent,
  MapMouseEvent
} from "maplibre-gl";
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
const POLYGON_FILL_LAYER_ID = "forecast-layer-fill";
const POLYGON_OUTLINE_LAYER_ID = "forecast-layer-outline";
const POINT_LAYER_ID = "forecast-layer-circle";

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

function extendBoundsFromCoordinates(
  bounds: LngLatBounds,
  coordinates: unknown
): void {
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

export function ForecastMap({ geojson, onSelectFeature }: ForecastMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const hasFittedRef = useRef(false);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) {
      return;
    }

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: "https://demotiles.maplibre.org/style.json",
      center: [5.1214, 52.0907],
      zoom: 6,
      pitch: 45,
      bearing: -15,
      maxZoom: 18
    });

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");

    map.on("load", () => {
      map.addSource(SOURCE_ID, {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: []
        }
      });

      map.addLayer({
        id: POLYGON_FILL_LAYER_ID,
        type: "fill",
        source: SOURCE_ID,
        paint: {
          "fill-color": "#f59e0b",
          "fill-opacity": 0.22
        },
        filter: ["==", "$type", "Polygon"]
      });

      map.addLayer({
        id: POLYGON_OUTLINE_LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        paint: {
          "line-color": "#d97706",
          "line-width": 2,
          "line-opacity": 0.95
        },
        filter: ["==", "$type", "Polygon"]
      });

      map.addLayer({
        id: POINT_LAYER_ID,
        type: "circle",
        source: SOURCE_ID,
        paint: {
          "circle-radius": 7,
          "circle-color": "#dc2626",
          "circle-stroke-width": 2,
          "circle-stroke-color": "#ffffff"
        },
        filter: ["==", "$type", "Point"]
      });

      map.on("mouseenter", POINT_LAYER_ID, () => {
        map.getCanvas().style.cursor = "pointer";
      });

      map.on("mouseleave", POINT_LAYER_ID, () => {
        map.getCanvas().style.cursor = "";
      });

      map.on("mouseenter", POLYGON_FILL_LAYER_ID, () => {
        map.getCanvas().style.cursor = "pointer";
      });

      map.on("mouseleave", POLYGON_FILL_LAYER_ID, () => {
        map.getCanvas().style.cursor = "";
      });

      map.on("click", POINT_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "point",
          "Source"
        );
        onSelectFeature(selected);
      });

      map.on("click", POLYGON_FILL_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "polygon",
          "Forecast extent"
        );
        onSelectFeature(selected);
      });

      map.on("click", (event: MapMouseEvent) => {
        const features = map.queryRenderedFeatures(event.point, {
          layers: [POINT_LAYER_ID, POLYGON_FILL_LAYER_ID]
        });

        if (features.length === 0) {
          onSelectFeature(null);
        }
      });
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [onSelectFeature]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isValidFeatureCollection(geojson)) {
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
        padding: 48,
        duration: hasFittedRef.current ? 700 : 1100,
        maxZoom: 13,
        pitch: 45,
        bearing: -15
      });
      hasFittedRef.current = true;
    }
  }, [geojson]);

  return <div ref={mapContainerRef} className="forecast-map panel" />;
}