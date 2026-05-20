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

    def _build_context_llm_prompt(self, context: dict) -> str:
        return (
            "You are an AI decision-support assistant for geospatial plume forecasts. "
            "Use only the provided forecast context. "
            "The first response should summarize current conditions and explain why the risk level is what it is. "
            "Do not mention raw grid cells or cell counts; translate model metrics into plain-language terms such as limited, moderate, broad, weak, stronger, or more widespread. "
            "Do not invent casualties, evacuation orders, exact weather, exact emergency instructions, or certainty. "
            "Do not claim live sensor confirmation unless observations_available=true appears in the context. "
            "Return ONLY strict JSON with exactly these fields: "
            "summary, risk_level, recommendation, uncertainty_note."
        )

    def _interpret_context_with_llm(self, context: dict) -> tuple[dict | None, bool, str | None]:
        llm_service = getattr(self.explain_service, "llm_service", None)
        if llm_service is None:
            print("[decision-support] no llm_service configured")
            return None, False, None

        try:
            llm_result = llm_service.interpret_context(
                system_prompt=self._build_context_llm_prompt(context),
                context=context,
            )
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
            print(f"[decision-support] LLM context interpretation failed with exception: {error}")
            return None, True, error

        if not llm_result.success:
            provider = getattr(llm_result, "provider", None) or getattr(getattr(llm_service, "llm_config", None), "provider", "unknown")
            model = getattr(llm_result, "model", None) or getattr(getattr(llm_service, "llm_config", None), "model", "unknown")
            error = getattr(llm_result, "error", None)
            raw_text = getattr(llm_result, "raw_text", None)
            short_error = str(error).strip() if error else "LLM interpretation returned unsuccessful result"
            print(
                "[decision-support] LLM context interpretation unsuccessful "
                f"provider={provider} model={model} error={short_error} raw_text={raw_text!r}"
            )
            return None, True, short_error

        print("[decision-support] LLM context interpretation succeeded")
        return {
            "summary": llm_result.summary or "Unavailable",
            "risk_level": llm_result.risk_level or "unknown",
            "recommendation": llm_result.recommendation or "Continue monitoring current forecast context.",
            "uncertainty_note": llm_result.uncertainty_note or "Grounded only in current forecast context.",
        }, True, None

    def latest(self, session_id: str | None = None) -> DecisionSupportResponse:
        if self.forecast_context_service is not None:
            context_source = "session" if session_id is not None else "auto"
            context = self.forecast_context_service.latest(session_id=session_id, source=context_source).payload
            forecast = context.get("forecast", {}) if isinstance(context, dict) else {}
            plume_metrics = context.get("plume_metrics", {}) if isinstance(context, dict) else {}
            conditions = context.get("conditions", {}) if isinstance(context, dict) else {}
            risk = str(forecast.get("risk_level") or "unknown")
            status = str(forecast.get("status") or "forecast unavailable")
            wind = conditions.get("wind_direction_label") or conditions.get("wind_direction_deg") or "unknown"
            max_concentration = plume_metrics.get("max_concentration")
            has_plume = ("plume detected" in status.lower()) or (risk.lower() in {"medium", "high"}) or (
                isinstance(max_concentration, (int, float)) and max_concentration > 0
            )
            if has_plume:
                briefing = (
                    f"Plume is present with {risk} risk. Wind direction: {wind}. "
                    "Use precautionary controls based on current forecast context."
                )
            else:
                briefing = "No meaningful plume is currently indicated by the active forecast context."

            llm_explanation, llm_attempted, llm_error = self._interpret_context_with_llm(context if isinstance(context, dict) else {})
            if llm_explanation is not None:
                return DecisionSupportResponse(payload={
                    "mode": "llm",
                    "briefing": llm_explanation["summary"],
                    "situation_summary": llm_explanation["summary"],
                    "risk_level": llm_explanation["risk_level"],
                    "recommended_action": llm_explanation["recommendation"],
                    "uncertainty_limitations": llm_explanation["uncertainty_note"],
                    "forecast_evidence": context,
                    "system_honesty": "LLM-generated from current forecast context",
                    "follow_up_questions": [],
                    "used_context_fields": ["forecast_context.latest"],
                    "limitations": ["Grounded only in current forecast/session context"],
                    "live_inputs": context.get("runtime", {}) if isinstance(context, dict) else {},
                    "runtime_metadata": {"context_session_id": session_id, "used_llm": True, "llm_attempted": llm_attempted, "llm_error": llm_error},
                })

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
                "runtime_metadata": {"context_session_id": session_id, "used_llm": False, "llm_attempted": llm_attempted, "llm_error": llm_error},
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
        context = latest.get("forecast_evidence")
        llm_service = getattr(self.explain_service, "llm_service", None)

        if llm_service is not None and isinstance(context, dict):
            try:
                prompt = (
                    "You are an AI decision-support assistant for geospatial plume forecasts. "
                    "Answer the user's question using only the provided forecast context. "
                    "Do not mention raw grid cell counts; describe plume extent in plain language. "
                    "Be concise and honest about uncertainty."
                )
                result = llm_service.answer_context_question(
                    system_prompt=prompt,
                    context=context,
                    question=message,
                )
                if result.get("success") and result.get("answer"):
                    return {
                        "mode": "llm",
                        "answer": result.get("answer"),
                        "used_context_fields": latest.get("used_context_fields", []),
                        "limitations": latest.get("limitations", []),
                        "context_forecast_id": None,
                        "context_session_id": latest.get("runtime_metadata", {}).get("context_session_id"),
                        "runtime_metadata": {"used_llm": True, "llm_error": None},
                    }
            except Exception as exc:
                print(f"[decision-support] LLM chat failed with exception: {exc}")
                pass

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
            "runtime_metadata": {"used_llm": False},
        }
