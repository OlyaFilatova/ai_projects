"""HTTP middleware for request logging."""

import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger("ai_chat.http")


def register_http_middleware(app: FastAPI, *, max_request_body_bytes: int) -> None:
    """Register request logging middleware."""

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > max_request_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Request body exceeds the configured limit.",
                    "code": "request_too_large",
                },
            )

        started_at = perf_counter()
        response = await call_next(request)
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "request completed",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
