import { httpGet, httpPost } from "../../../services/api/http";
import type {
  CreateSessionRequest,
  IngestObservationsRequest,
  IngestObservationsResponse,
  SessionDetail,
  SessionPredictionRequest,
  SessionPredictionResponse,
  SessionStateSummary,
  SessionSummary,
  SessionUpdateResponse
} from "../types/session.types";

export const sessionClient = {
  listSessions(): Promise<SessionSummary[]> {
    return httpGet<SessionSummary[]>("/sessions");
  },

  createSession(payload: CreateSessionRequest): Promise<SessionSummary> {
    return httpPost<SessionSummary, CreateSessionRequest>("/sessions", payload);
  },

  getSession(sessionId: string): Promise<SessionDetail> {
    return httpGet<SessionDetail>(`/sessions/${sessionId}`);
  },

  getSessionState(sessionId: string): Promise<SessionStateSummary> {
    return httpGet<SessionStateSummary>(`/sessions/${sessionId}/state`);
  },

  ingestObservations(sessionId: string, payload: IngestObservationsRequest): Promise<IngestObservationsResponse> {
    return httpPost<IngestObservationsResponse, IngestObservationsRequest>(`/sessions/${sessionId}/observations`, payload);
  },

  updateSession(sessionId: string): Promise<SessionUpdateResponse> {
    return httpPost<SessionUpdateResponse>(`/sessions/${sessionId}/update`);
  },

  predictSession(sessionId: string, payload: SessionPredictionRequest): Promise<SessionPredictionResponse> {
    return httpPost<SessionPredictionResponse, SessionPredictionRequest>(`/sessions/${sessionId}/predict`, payload);
  }
};
