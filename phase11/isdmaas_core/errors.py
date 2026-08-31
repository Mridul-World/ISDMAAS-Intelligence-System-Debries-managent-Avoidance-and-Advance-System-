"""
errors.py — one error shape for the whole API.

Every failure the client can see comes back as:

    {"error": {"code": "not_found", "message": "...", "request_id": "..."}}

`code` is a stable machine-readable string the console switches on; `message` is
safe to show a human. Unexpected exceptions are logged with their traceback and
reported as a generic `internal_error` — the traceback never reaches the client,
because in this system a stack trace leaks absolute paths, module layout and
sometimes the contents of an operator's TLE.

The `detail` key is mirrored alongside `error` because the existing console
reads `detail`; keeping both means the transition needs no flag day.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging_config import get_logger, request_id_var

log = get_logger("isdmaas.errors")

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "unprocessable",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
    504: "upstream_timeout",
}


class ApiError(Exception):
    """Raise this for any failure the client is meant to understand."""

    def __init__(
        self,
        status_code: int,
        message: str,
        code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code or _STATUS_CODES.get(status_code, "error")
        self.extra = extra or {}
        self.headers = headers or {}


def error_response(
    status_code: int,
    message: str,
    code: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "error": {
            "code": code or _STATUS_CODES.get(status_code, "error"),
            "message": message,
            "request_id": request_id_var.get(),
        },
        # Back-compat: the console (and FastAPI's own convention) reads `detail`.
        "detail": message,
    }
    if extra:
        body["error"].update(extra)
    return JSONResponse(status_code=status_code, content=body, headers=headers or {})


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        if exc.status_code >= 500:
            log.error("api_error", extra={"code": exc.code, "status": exc.status_code})
        return error_response(
            exc.status_code, exc.message, exc.code, exc.extra, exc.headers
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        detail = exc.detail
        message = detail if isinstance(detail, str) else str(detail)
        return error_response(
            exc.status_code, message, headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        problems = []
        for err in exc.errors():
            location = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            problems.append(
                {"field": location or "body", "problem": err.get("msg", "invalid")}
            )
        summary = "; ".join(f"{p['field']}: {p['problem']}" for p in problems[:5])
        return error_response(
            422,
            f"Request validation failed — {summary}" if summary
            else "Request validation failed",
            code="unprocessable",
            extra={"problems": problems},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception(
            "unhandled exception on %s %s", request.method, request.url.path
        )
        return error_response(
            500,
            "An internal error occurred. The request id below identifies this "
            "failure in the server log.",
            code="internal_error",
        )
