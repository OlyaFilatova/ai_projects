"""Smoke tests for the application skeleton."""

import pytest
from fastapi.testclient import TestClient

from ai_chat.app import create_app
from ai_chat.config import get_settings
from tests.support import create_test_settings


def test_app_starts_with_custom_settings() -> None:
    """The app factory should preserve injected settings."""

    settings = create_test_settings()
    app = create_app(settings=settings)

    assert app.title == "AI Chat Backend"
    assert app.state.container.settings.env == "test"
    assert app.debug is False


def test_health_endpoint_reports_service_state() -> None:
    """The health endpoint should return a basic success payload."""

    app = create_app(
        create_test_settings()
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "test",
        "provider": "mock",
    }


def test_readiness_endpoint_reports_database_state() -> None:
    """The readiness endpoint should reflect database availability."""

    app = create_app(
        create_test_settings()
    )
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database_ready": True,
        "provider": "mock",
    }


def test_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should populate nested settings."""

    get_settings.cache_clear()
    monkeypatch.setenv("AI_CHAT_ENV", "test")
    monkeypatch.setenv("AI_CHAT_DEBUG", "true")
    monkeypatch.setenv("AI_CHAT_DATABASE__URL", "postgresql://db/test")
    monkeypatch.setenv("AI_CHAT_JWT__SECRET", "test-secret-key-with-32-bytes-min")
    monkeypatch.setenv("AI_CHAT_LLM__PROVIDER", "OPENAI")
    monkeypatch.setenv("AI_CHAT_RATE_LIMIT__STORAGE_URI", "memory://")
    monkeypatch.setenv("AI_CHAT_USAGE_QUOTA__STORAGE_URI", "memory://")

    settings = get_settings()

    assert settings.env == "test"
    assert settings.debug is True
    assert settings.database.url == "postgresql://db/test"
    assert settings.jwt.secret is not None
    assert settings.jwt.secret.get_secret_value() == "test-secret-key-with-32-bytes-min"
    assert settings.llm.provider == "openai"
    assert settings.rate_limit.storage_uri == "memory://"
    assert settings.usage_quota.storage_uri == "memory://"
    get_settings.cache_clear()


def test_invalid_environment_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid configuration should surface as a runtime configuration error."""

    get_settings.cache_clear()
    monkeypatch.setenv("AI_CHAT_ENV", "invalid")
    with pytest.raises(RuntimeError, match="Invalid application configuration"):
        get_settings()
    get_settings.cache_clear()


def test_short_jwt_secret_is_rejected() -> None:
    """Configured JWT secrets should meet the minimum length requirement."""

    with pytest.raises(ValueError, match="at least 32 characters"):
        create_test_settings(
            env="test",
            jwt={"secret": "too-short"},
        )


def test_large_request_is_rejected_early() -> None:
    """Oversized requests should be blocked before route processing."""

    app = create_app(
        create_test_settings(max_request_body_bytes=32)
    )
    client = TestClient(app)
    response = client.post(
        "/auth/register",
        content='{"email":"alice@example.com","password":"correct-horse-battery"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"
