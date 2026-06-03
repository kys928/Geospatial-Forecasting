import { httpGet, httpPost } from "../../../services/api/http";
import type {
  CreateSessionRequest,
  IngestObservationsRequest,
  IngestObservationsResponse,
  SessionDetail,
  SessionForecastBundle,
  ForecastFramesMetadata,
  ForecastFrameRasterPayload,
  SessionPredictionRequest,
  SessionPredictionResponse,
  SessionStateSummary,
  SessionSummary,
  SessionUpdateResponse
} from "../types/session.types";

const ACTIVE_SESSION_STORAGE_KEY = "plume_active_session_id";
const ACTIVE_FORECAST_SESSION_CONTRACT = "active-convlstm-session-v2";

export interface RunSessionForecastResult {
  sessionId: string;
  recreatedSession: boolean;
  resetReason?: string;
  prediction: SessionPredictionResponse;
}

interface EnsureSessionResult {
  sessionId: string;
  recreatedSession: boolean;
  resetReason?: string;
}

function nestedRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function nestedString(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function activeConvLstmSessionCompatibility(session: SessionDetail): { compatible: boolean; reason?: string } {
  if (session.backend_name !== "convlstm_online") {
    return { compatible: false, reason: `stored session backend is ${session.backend_name || "unknown"}` };
  }
  const runtime = nestedRecord(session.runtime_metadata);
  const metadata = nestedRecord(session.metadata);
  const modelLoad = nestedRecord(runtime.model_load);
  const resolvedActive = nestedRecord(modelLoad.resolved_active_model);
  const provenance = nestedRecord(runtime.provenance);
  const configuredContract = nestedString(metadata, "frontend_session_contract") || nestedString(runtime, "frontend_session_contract");
  if (configuredContract && configuredContract !== ACTIVE_FORECAST_SESSION_CONTRACT) {
    return { compatible: false, reason: `stored session contract is ${configuredContract}` };
  }
  const predictionEngine = nestedString(runtime, "prediction_engine") || nestedString(modelLoad, "prediction_engine");
  if (predictionEngine === "ridge_baseline") {
    return { compatible: false, reason: "stored session uses ridge_baseline" };
  }
  if (runtime.temporary_model_substitution === true || modelLoad.temporary_model_substitution === true) {
    return { compatible: false, reason: "stored session used temporary model substitution" };
  }
  const runtimeSource = nestedString(runtime, "forecast_source") || nestedString(provenance, "forecast_source");
  const runtimeInput = nestedString(runtime, "input_source") || nestedString(provenance, "input_source");
  const runtimeOutput = nestedString(runtime, "output_source") || nestedString(provenance, "output_source");
  if (runtimeSource === "dataset_playback" || runtimeInput === "dataset_playback" || runtimeOutput === "dataset_playback") {
    return { compatible: false, reason: "stored session points at dataset_playback output" };
  }
  const activeModelId = nestedString(modelLoad, "active_model_id") || nestedString(runtime, "active_registry_model_id") || nestedString(resolvedActive, "model_id");
  const checkpointPath = nestedString(modelLoad, "checkpoint_path") || nestedString(resolvedActive, "checkpoint_path");
  if (!activeModelId || !checkpointPath) {
    return { compatible: false, reason: "stored session lacks active ConvLSTM model load metadata" };
  }
  return { compatible: true };
}

const activeSessionCreatePayload = (): CreateSessionRequest => ({
  metadata: {
    requested_forecast_mode: "active_model",
    frontend_session_contract: ACTIVE_FORECAST_SESSION_CONTRACT
  }
});

export const sessionClient = {
  listSessions(): Promise<SessionSummary[]> { return httpGet<SessionSummary[]>("/sessions"); },
  createSession(payload: CreateSessionRequest): Promise<SessionSummary> { return httpPost<SessionSummary, CreateSessionRequest>("/sessions", payload); },
  getSession(sessionId: string): Promise<SessionDetail> { return httpGet<SessionDetail>(`/sessions/${sessionId}`); },
  getSessionState(sessionId: string): Promise<SessionStateSummary> { return httpGet<SessionStateSummary>(`/sessions/${sessionId}/state`); },
  ingestObservations(sessionId: string, payload: IngestObservationsRequest): Promise<IngestObservationsResponse> { return httpPost<IngestObservationsResponse, IngestObservationsRequest>(`/sessions/${sessionId}/observations`, payload); },
  updateSession(sessionId: string): Promise<SessionUpdateResponse> { return httpPost<SessionUpdateResponse>(`/sessions/${sessionId}/update`); },
  predictSession(sessionId: string, payload: SessionPredictionRequest): Promise<SessionPredictionResponse> { return httpPost<SessionPredictionResponse, SessionPredictionRequest>(`/sessions/${sessionId}/predict`, payload); },
  async getLatestForecastBundle(sessionId: string, options?: { includeExplanation?: boolean }): Promise<SessionForecastBundle> {
    const includeExplanation = options?.includeExplanation ?? true;
    const [summary, geojson, rasterMetadata, explanation] = await Promise.all([
      httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/summary`),
      httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/geojson`),
      httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/raster-metadata`),
      includeExplanation
        ? httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/explanation`)
        : Promise.resolve({})
    ]);
    return { summary, geojson, rasterMetadata, explanation };
  },
  getLatestForecastFrames(sessionId: string): Promise<ForecastFramesMetadata> {
    return httpGet<ForecastFramesMetadata>(`/sessions/${sessionId}/forecast/latest/frames`);
  },
  getLatestForecastFrameSummary(sessionId: string, frameIndex: number): Promise<Record<string, unknown>> {
    return httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/frames/${frameIndex}/summary`);
  },
  getLatestForecastFrameGeoJson(sessionId: string, frameIndex: number): Promise<Record<string, unknown>> {
    return httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/frames/${frameIndex}/geojson`);
  },
  getLatestForecastFrameRasterMetadata(sessionId: string, frameIndex: number): Promise<Record<string, unknown>> {
    return httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/frames/${frameIndex}/raster-metadata`);
  },
  getLatestForecastFrameRaster(sessionId: string, frameIndex: number): Promise<ForecastFrameRasterPayload> {
    return httpGet<ForecastFrameRasterPayload>(`/sessions/${sessionId}/forecast/latest/frames/${frameIndex}/raster`);
  },
  clearSession() { localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY); },
  async ensureSession(): Promise<EnsureSessionResult> {
    const stored = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
    if (stored) {
      try {
        const session = await this.getSession(stored);
        const compatibility = activeConvLstmSessionCompatibility(session);
        if (compatibility.compatible) {
          return { sessionId: stored, recreatedSession: false };
        }
        this.clearSession();
        const created = await this.createSession(activeSessionCreatePayload());
        localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, created.session_id);
        return { sessionId: created.session_id, recreatedSession: true, resetReason: compatibility.reason };
      } catch (error) {
        this.clearSession();
        const reason = error instanceof Error ? error.message : "stored session unavailable";
        const created = await this.createSession(activeSessionCreatePayload());
        localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, created.session_id);
        return { sessionId: created.session_id, recreatedSession: true, resetReason: reason };
      }
    }
    const created = await this.createSession(activeSessionCreatePayload());
    localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, created.session_id);
    return { sessionId: created.session_id, recreatedSession: true, resetReason: "new active ConvLSTM session created" };
  },
  async runSessionForecast(payload: SessionPredictionRequest = {}): Promise<RunSessionForecastResult> {
    const ensured = await this.ensureSession();
    try {
      const prediction = await this.predictSession(ensured.sessionId, payload);
      return { sessionId: ensured.sessionId, recreatedSession: ensured.recreatedSession, resetReason: ensured.resetReason, prediction };
    } catch (error) {
      const isMissing = typeof error === "object" && error !== null && ("status" in error ? (error as { status?: number }).status === 404 : false)
        || (error instanceof Error && error.message.includes("404"));
      if (!isMissing) { throw error; }
      this.clearSession();
      const recreated = await this.ensureSession();
      const prediction = await this.predictSession(recreated.sessionId, payload);
      return { sessionId: recreated.sessionId, recreatedSession: true, resetReason: "session missing during predict; recreated", prediction };
    }
  }
};
