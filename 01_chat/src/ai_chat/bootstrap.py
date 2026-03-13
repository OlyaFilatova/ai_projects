"""Application bootstrap helpers."""

from fastapi import FastAPI

from ai_chat.config import Settings, get_settings
from ai_chat.dependencies import ServiceContainer, build_container
from ai_chat.observability.logging import configure_logging
from ai_chat.transport.http.errors import register_exception_handlers
from ai_chat.transport.http.middleware import register_http_middleware
from ai_chat.transport.http.routers.auth import router as auth_router
from ai_chat.transport.http.routers.chat import router as chat_router
from ai_chat.transport.http.routers.health import router as health_router


def create_container(settings: Settings | None = None) -> ServiceContainer:
    """Create the application dependency container."""

    return build_container(settings or get_settings())


def create_api_app(container: ServiceContainer) -> FastAPI:
    """Create the FastAPI application from a prepared service container."""

    configure_logging(container.settings)
    app = FastAPI(title=container.settings.app_name, debug=container.settings.debug)
    app.state.container = container
    register_exception_handlers(app)
    register_http_middleware(
        app,
        max_request_body_bytes=container.settings.max_request_body_bytes,
    )
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(health_router)
    return app
