from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class DecisionSupportChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def register_decision_support_routes(app: FastAPI, *, decision_support_service) -> None:
    @app.get('/decision-support/latest')
    def latest_decision_support(session_id: str | None = None):
        try:
            return decision_support_service.latest(session_id=session_id).payload
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post('/decision-support/chat')
    def decision_support_chat(payload: DecisionSupportChatRequest):
        return decision_support_service.chat(message=payload.message, session_id=payload.session_id)
