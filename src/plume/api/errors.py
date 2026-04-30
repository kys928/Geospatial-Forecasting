from __future__ import annotations

from fastapi import HTTPException


def api_error(status_code: int, code: str, message: str, details: dict[str, object] | None = None) -> HTTPException:
    payload: dict[str, object] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return HTTPException(status_code=status_code, detail=payload)


def not_found(code: str, message: str, details: dict[str, object] | None = None) -> HTTPException:
    return api_error(404, code, message, details)


def bad_request(code: str, message: str, details: dict[str, object] | None = None) -> HTTPException:
    return api_error(400, code, message, details)


def conflict(code: str, message: str, details: dict[str, object] | None = None) -> HTTPException:
    return api_error(409, code, message, details)
