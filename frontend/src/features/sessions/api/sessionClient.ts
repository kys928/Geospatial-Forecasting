import { httpGet, httpPost } from "../../../services/api/http";
import type {
  CreateSessionRequest,
  IngestObservationsRequest,
  IngestObservationsResponse,
  SessionDetail,
  SessionForecastBundle,
  SessionPredictionRequest,
  SessionPredictionResponse,
  SessionStateSummary,
  SessionSummary,
  SessionUpdateResponse
} from "../types/session.types";

const ACTIVE_SESSION_STORAGE_KEY = "plume_active_session_id";

export interface RunSessionForecastResult {
  sessionId: string;
  recreatedSession: boolean;
  prediction: SessionPredictionResponse;
}

export const sessionClient = {
  listSessions(): Promise<SessionSummary[]> { return httpGet<SessionSummary[]>("/sessions"); },
  createSession(payload: CreateSessionRequest): Promise<SessionSummary> { return httpPost<SessionSummary, CreateSessionRequest>("/sessions", payload); },
  getSession(sessionId: string): Promise<SessionDetail> { return httpGet<SessionDetail>(`/sessions/${sessionId}`); },
  getSessionState(sessionId: string): Promise<SessionStateSummary> { return httpGet<SessionStateSummary>(`/sessions/${sessionId}/state`); },
  ingestObservations(sessionId: string, payload: IngestObservationsRequest): Promise<IngestObservationsResponse> { return httpPost<IngestObservationsResponse, IngestObservationsRequest>(`/sessions/${sessionId}/observations`, payload); },
  updateSession(sessionId: string): Promise<SessionUpdateResponse> { return httpPost<SessionUpdateResponse>(`/sessions/${sessionId}/update`); },
  predictSession(sessionId: string, payload: SessionPredictionRequest): Promise<SessionPredictionResponse> { return httpPost<SessionPredictionResponse, SessionPredictionRequest>(`/sessions/${sessionId}/predict`, payload); },
  async getLatestForecastBundle(sessionId: string): Promise<SessionForecastBundle> {
    const [summary, geojson, rasterMetadata, explanation] = await Promise.all([
      httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/summary`),
      httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/geojson`),
      httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/raster-metadata`),
      httpGet<Record<string, unknown>>(`/sessions/${sessionId}/forecast/latest/explanation`)
    ]);
    return { summary, geojson, rasterMetadata, explanation };
  },
  clearSession() { localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY); },
  async ensureSession(): Promise<{ sessionId: string; recreated: boolean }> {
    const stored = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
    if (stored) {
      try {
        await this.getSession(stored);
        return { sessionId: stored, recreated: false };
      } catch {
        this.clearSession();
      }
    }
    const created = await this.createSession({ backend_name: "convlstm_online" });
    localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, created.session_id);
    return { sessionId: created.session_id, recreated: true };
  },
  async runSessionForecast(payload: SessionPredictionRequest = {}): Promise<RunSessionForecastResult> {
    const ensured = await this.ensureSession();
    try {
      const prediction = await this.predictSession(ensured.sessionId, payload);
      return { sessionId: ensured.sessionId, recreatedSession: ensured.recreated, prediction };
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes("404")) { throw error; }
      this.clearSession();
      const recreated = await this.ensureSession();
      const prediction = await this.predictSession(recreated.sessionId, payload);
      return { sessionId: recreated.sessionId, recreatedSession: true, prediction };
    }
  }
};
