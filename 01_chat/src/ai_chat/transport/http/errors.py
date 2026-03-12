"""HTTP exception handlers for domain-level application errors."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_chat.errors import AppError


def register_exception_handlers(app: FastAPI) -> None:
    """Register HTTP exception handlers for domain errors."""

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """Convert domain errors into a stable JSON API response."""

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
        )
