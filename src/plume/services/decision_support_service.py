from __future__ import annotations

from dataclasses import dataclass

from plume.services.explanation_payloads import build_explanation_payload


@dataclass
class DecisionSupportResponse:
    payload: dict[str, object]


class DecisionSupportService:
    def __init__(self, runtime_client, explain_service, forecast_context_service=None):
        self.runtime_client = runtime_client
        self.explain_service = explain_service
        self.forecast_context_service = forecast_context_service

    def latest(self, session_id: str | None = None) -> DecisionSupportResponse:
        if self.forecast_context_service is not None:
            context = self.forecast_context_service.latest(session_id=session_id, source="auto").payload
            forecast = context.get("forecast", {}) if isinstance(context, dict) else {}
            plume_metrics = context.get("plume_metrics", {}) if isinstance(context, dict) else {}
            conditions = context.get("conditions", {}) if isinstance(context, dict) else {}
            risk = str(forecast.get("risk_level") or "unknown")
            status = str(forecast.get("status") or "forecast unavailable")
            wind = conditions.get("wind_direction_label") or conditions.get("wind_direction_deg") or "unknown"
            max_concentration = plume_metrics.get("max_concentration")
            has_plume = ("plume detected" in status.lower()) or (risk.lower() in {"medium", "high"}) or (isinstance(max_concentration, (int, float)) and max_concentration > 0)
            if has_plume:
                briefing = f"Plume is present with {risk} risk. Wind direction: {wind}. Use precautionary controls based on current forecast context."
            else:
                briefing = "No meaningful plume is currently indicated by the active forecast context."
            return DecisionSupportResponse(payload={
                "mode": "context",
                "briefing": briefing,
                "situation_summary": briefing,
                "risk_level": risk,
                "recommended_action": "Continue monitoring current forecast context.",
                "uncertainty_limitations": "Grounded only in current forecast context.",
                "forecast_evidence": context,
                "system_honesty": "Context-derived decision support",
                "follow_up_questions": [],
                "used_context_fields": ["forecast_context.latest"],
                "limitations": ["Grounded only in current forecast/session context"],
                "live_inputs": context.get("runtime", {}) if isinstance(context, dict) else {},
                "runtime_metadata": {"context_session_id": session_id, "used_llm": False},
            })
        if session_id is None:
            sessions = self.runtime_client.list_sessions()
            if not sessions:
                return DecisionSupportResponse(payload={"mode": "stub", "briefing": "No active session.", "limitations": ["No sessions available"], "live_inputs": {"observation_count": 0}, "runtime_metadata": {}})
            session_id = sessions[-1].session_id
        result = self.runtime_client.get_latest_session_forecast_result(session_id)
        explanation_result = self.explain_service.explain(result, use_llm=True)
        explanation_payload = build_explanation_payload(result, explanation_result)
        detail = explanation_payload.get("explanation", {}) if isinstance(explanation_payload, dict) else {}
        return DecisionSupportResponse(payload={
            "mode": "llm" if explanation_payload.get("used_llm") else "stub",
            "briefing": detail.get("summary", "Unavailable"),
            "situation_summary": detail.get("summary", "Unavailable"),
            "risk_level": detail.get("risk_level", "unknown"),
            "recommended_action": detail.get("recommendation", "Unavailable"),
            "uncertainty_limitations": detail.get("uncertainty_note", "Unavailable"),
            "forecast_evidence": explanation_payload.get("summary", {}),
            "system_honesty": "LLM-generated" if explanation_payload.get("used_llm") else "Stub/development explanation",
            "follow_up_questions": [],
            "used_context_fields": ["forecast.summary", "session.state"],
            "limitations": ["Grounded only in current forecast/session context"],
            "live_inputs": self.runtime_client.get_session_state(session_id),
            "runtime_metadata": {"context_session_id": session_id, "used_llm": explanation_payload.get("used_llm")},
        })

    def chat(self, message: str, session_id: str | None = None) -> dict[str, object]:
        latest = self.latest(session_id=session_id).payload
        briefing = str(latest.get("briefing", "")).strip()
        if not briefing or briefing.lower() == "unavailable":
            answer = "I do not have enough forecast context to answer that specific question right now."
        else:
            answer = briefing
        return {
            "mode": latest.get("mode", "stub"),
            "answer": answer,
            "used_context_fields": latest.get("used_context_fields", []),
            "limitations": latest.get("limitations", []),
            "context_forecast_id": None,
            "context_session_id": latest.get("runtime_metadata", {}).get("context_session_id"),
        }
