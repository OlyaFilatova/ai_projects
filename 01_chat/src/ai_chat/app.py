"""FastAPI application factory."""

from fastapi import FastAPI

from ai_chat.bootstrap import create_api_app, create_container
from ai_chat.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    container = create_container(settings)
    return create_api_app(container)
