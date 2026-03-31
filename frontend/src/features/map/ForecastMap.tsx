import { useEffect, useRef } from "react";
import maplibregl, {
  GeoJSONSource,
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

export function ForecastMap({ geojson, onSelectFeature }: ForecastMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) {
      return;
    }

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: "https://demotiles.maplibre.org/style.json",
      center: [5.1214, 52.0907],
      zoom: 12
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");

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
          "fill-color": "#3b82f6",
          "fill-opacity": 0.25
        },
        filter: ["==", "$type", "Polygon"]
      });

      map.addLayer({
        id: POLYGON_OUTLINE_LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        paint: {
          "line-color": "#3b82f6",
          "line-width": 1.5
        },
        filter: ["==", "$type", "Polygon"]
      });

      map.addLayer({
        id: POINT_LAYER_ID,
        type: "circle",
        source: SOURCE_ID,
        paint: {
          "circle-radius": 6,
          "circle-color": "#2563eb",
          "circle-stroke-width": 1,
          "circle-stroke-color": "#ffffff"
        },
        filter: ["==", "$type", "Point"]
      });

      map.on("click", POINT_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "point",
          "Point feature"
        );
        onSelectFeature(selected);
      });

      map.on("click", POLYGON_FILL_LAYER_ID, (event: MapLayerMouseEvent) => {
        const selected = buildSelectedFeature(
          event.features?.[0],
          "polygon",
          "Polygon feature"
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
  }, [geojson]);

  return <div ref={mapContainerRef} className="forecast-map panel" />;
}