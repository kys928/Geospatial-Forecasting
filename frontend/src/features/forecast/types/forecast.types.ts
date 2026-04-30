export type ApiMode = "mock" | "live";
export type ScenarioMockVariant = "default" | "urban" | "industrial";
export type ScenarioPreset = ScenarioMockVariant;

export interface DemoScenario {
  id: string;
  label: string;
  latitude: number;
  longitude: number;
  emissionsRate: number;
  severity?: "low" | "moderate" | "high";
  notes?: string;
  mockVariant?: ScenarioMockVariant;
}

export interface ForecastRunRequest {
  scenario: DemoScenario;
}

export type MockForecastRequest = ForecastRunRequest;

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
  affected_area_m2: number;
  affected_area_hectares: number;
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

export interface ForecastArtifactMetadata {
  forecast_id: string;
  issued_at: string;
  model?: string | null;
  available_artifacts?: string[];
  runtime?: {
    model_family?: string | null;
    model_source?: string | null;
    prediction_trust?: string | null;
    path?: string | null;
    [key: string]: unknown;
  } | null;
  [key: string]: unknown;
}

export interface ForecastListResponse {
  forecasts: ForecastArtifactMetadata[];
}

export interface ForecastArtifactBundle {
  summary: Record<string, unknown>;
  geojson: Record<string, unknown>;
  rasterMetadata: Record<string, unknown>;
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

export interface GeoJsonFeature {
  type: "Feature";
  geometry: GeoJSON.Geometry;
  properties?: Record<string, unknown>;
  id?: string | number;
}

export interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
  properties?: Record<string, unknown>;
}

export interface SelectedFeatureState {
  id: string;
  title: string;
  properties: Record<string, unknown> | null;
  geometry: GeoJSON.Geometry | null;
  feature: GeoJsonFeature | null;
}
