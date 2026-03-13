"""Shared helpers for end-to-end API tests."""

import tempfile
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_chat.app import create_app
from ai_chat.config import Settings
from ai_chat.dependencies import ServiceContainer


def create_test_database_url() -> str:
    """Create a file-backed SQLite database URL for migration-based app startup."""

    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return f"sqlite+pysqlite:///{Path(handle.name)}"


def create_test_settings(**overrides: Any) -> Settings:
    """Create test settings with sane defaults and optional overrides."""

    base_settings: dict[str, Any] = {
        "env": "test",
        "database": {"url": create_test_database_url()},
        "jwt": {"secret": "test-secret-key-with-32-bytes-min"},
        "llm": {"provider": "mock"},
        "usage_quota": {
            "window_seconds": 3600,
            "chat_requests": 100,
            "chat_stream_requests": 50,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base_settings.get(key), dict):
            merged = dict(base_settings[key])
            merged.update(value)
            base_settings[key] = merged
        else:
            base_settings[key] = value
    return Settings.model_validate(base_settings)


def create_test_client() -> TestClient:
    """Create a test client backed by an isolated SQLite database."""

    settings = create_test_settings()
    return TestClient(create_app(settings))


def get_test_container(client: TestClient) -> ServiceContainer:
    """Return the typed application container behind a test client."""

    app = cast(FastAPI, client.app)
    return cast(ServiceContainer, app.state.container)


def register_and_login(
    client: TestClient,
    *,
    email: str = "alice@example.com",
    password: str = "correct-horse-battery",
) -> str:
    """Register a user and return a bearer token for that user."""

    payload = {"email": email, "password": password}
    client.post("/auth/register", json=payload)
    token_response = client.post("/auth/token", json=payload)
    access_token = token_response.json()["access_token"]
    assert isinstance(access_token, str)
    return access_token
