"""ASGI entry point for local development and container execution."""

from ai_chat.app import create_app

app = create_app()
