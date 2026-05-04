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
  resetReason?: string;
  prediction: SessionPredictionResponse;
}

interface EnsureSessionResult {
  sessionId: string;
  recreatedSession: boolean;
  resetReason?: string;
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
  async ensureSession(): Promise<EnsureSessionResult> {
    const stored = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
    if (stored) {
      try {
        await this.getSession(stored);
        return { sessionId: stored, recreatedSession: false };
      } catch (error) {
        this.clearSession();
        const reason = error instanceof Error ? error.message : "stored session unavailable";
        const created = await this.createSession({});
        localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, created.session_id);
        return { sessionId: created.session_id, recreatedSession: true, resetReason: reason };
      }
    }
    const created = await this.createSession({});
    localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, created.session_id);
    return { sessionId: created.session_id, recreatedSession: true, resetReason: "new session created" };
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
