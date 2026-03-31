export type ApiMode = "mock" | "live";
export type ScenarioPreset = "default" | "urban" | "industrial";
export type ThresholdPreset = "1e-6" | "1e-5" | "1e-4";

export interface MockForecastRequest {
  scenario: ScenarioPreset;
  threshold: ThresholdPreset;
}

export interface CapabilitiesResponse {
  model: string[];
  exports: string[];
}

export interface ForecastExplanationSummary {
  source_latitude: number;
  source_longitude: number;
  grid_rows: number;
  grid_columns: number;
  projection: string | null;
  max_concentration: number;
  mean_concentration: number;
  affected_cells_above_threshold: number;
  dominant_spread_direction: string;
  threshold_used: number;
  note: string | null;
}

export interface ForecastExplanationBody {
  summary: string | null;
  risk_level: string | null;
  recommendation: string | null;
  uncertainty_note: string | null;
}

export interface ForecastExplanation {
  forecast_id: string;
  issued_at: string;
  model: string;
  used_llm: boolean;
  summary: ForecastExplanationSummary;
  explanation: ForecastExplanationBody;
}


export interface ForecastCreateResponse {
  forecast_id: string;
  issued_at: string;
}

export interface ForecastSummary {
  forecast_id: string;
  issued_at: string;
  model: string;
  model_version: string | null;
  summary_statistics: {
    max_concentration: number;
    mean_concentration: number;
  };
  run_name: string;
  grid: {
    rows: number;
    columns: number;
    projection: string;
  };
  source: {
    latitude: number;
    longitude: number;
  };
  timestamp: string;
}

export interface RasterMetadata {
  forecast_id: string;
  rows: number;
  cols: number;
  bounds: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
  };
  projection: string;
  min_value: number;
  max_value: number;
  grid_spacing: number;
}

export interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: GeoJSON.Geometry;
    properties?: Record<string, unknown>;
  }>;
  properties?: Record<string, unknown>;
}

export interface SelectedFeatureState {
  id: string | null;
  title: string;
  properties: Record<string, unknown> | null;
}