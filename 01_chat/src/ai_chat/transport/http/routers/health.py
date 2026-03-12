"""Health routes for the AI chat backend."""

from fastapi import APIRouter, Request

from ai_chat.persistence.health import database_is_ready

router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck(request: Request) -> dict[str, str]:
    """Return basic service health information."""

    container = request.app.state.container
    return {
        "status": "ok",
        "environment": container.settings.env,
        "provider": container.settings.llm.provider,
    }


@router.get("/ready")
def readiness(request: Request) -> dict[str, str | bool]:
    """Return readiness information for current dependencies."""

    container = request.app.state.container
    database_ready = database_is_ready(container.session_factory)
    payload = {
        "status": "ready" if database_ready else "degraded",
        "database_ready": database_ready,
        "provider": container.settings.llm.provider,
    }
    return payload
