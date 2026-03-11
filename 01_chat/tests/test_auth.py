"""Authentication API tests."""

import jwt
from fastapi.testclient import TestClient

from ai_chat.app import create_app
from tests.support import (
    create_test_client as shared_create_test_client,
)
from tests.support import (
    create_test_settings,
)


def create_test_client() -> TestClient:
    """Create a test client backed by an isolated SQLite database."""

    return shared_create_test_client()


def test_register_and_authenticate_user() -> None:
    """A user should be able to register and exchange credentials for a token."""

    client = create_test_client()

    register_response = client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": "correct-horse-battery"},
    )
    token_response = client.post(
        "/auth/token",
        json={"email": "alice@example.com", "password": "correct-horse-battery"},
    )

    assert register_response.status_code == 202
    assert (
        register_response.json()["detail"]
        == "If registration is available for this email, you can now sign in."
    )
    assert token_response.status_code == 200
    assert token_response.json()["token_type"] == "bearer"
    assert token_response.json()["access_token"]
    assert token_response.json()["refresh_token"]


def test_duplicate_registration_returns_same_public_response() -> None:
    """Registration should not reveal whether the email already exists."""

    client = create_test_client()
    payload = {"email": "alice@example.com", "password": "correct-horse-battery"}

    first_response = client.post("/auth/register", json=payload)
    second_response = client.post("/auth/register", json=payload)

    assert first_response.status_code == 202
    assert second_response.status_code == 202
    assert first_response.json() == second_response.json()


def test_login_with_bad_password_is_rejected() -> None:
    """Invalid credentials should fail with a 401 response."""

    client = create_test_client()
    client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": "correct-horse-battery"},
    )

    response = client.post(
        "/auth/token",
        json={"email": "alice@example.com", "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."
    assert response.json()["code"] == "auth_credentials_invalid"


def test_login_with_unknown_email_matches_bad_password_response() -> None:
    """Unknown users should receive the same public failure as bad passwords."""

    client = create_test_client()
    known_payload = {
        "email": "alice@example.com",
        "password": "correct-horse-battery",
    }
    client.post("/auth/register", json=known_payload)

    unknown_user_response = client.post(
        "/auth/token",
        json={"email": "missing@example.com", "password": "correct-horse-battery"},
    )
    bad_password_response = client.post(
        "/auth/token",
        json={"email": "alice@example.com", "password": "wrong-password"},
    )

    assert unknown_user_response.status_code == 401
    assert unknown_user_response.json() == bad_password_response.json()


def test_authenticated_route_requires_valid_bearer_token() -> None:
    """Bearer token verification should protect authenticated routes."""

    client = create_test_client()
    client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": "correct-horse-battery"},
    )
    token_response = client.post(
        "/auth/token",
        json={"email": "alice@example.com", "password": "correct-horse-battery"},
    )
    access_token = token_response.json()["access_token"]

    authorized_response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    unauthorized_response = client.get("/auth/me")

    assert authorized_response.status_code == 200
    assert authorized_response.json()["email"] == "alice@example.com"
    assert unauthorized_response.status_code == 401
    assert unauthorized_response.json()["code"] == "auth_credentials_missing"


def test_blank_registration_password_is_rejected() -> None:
    """Whitespace-only passwords should fail validation before service logic."""

    client = create_test_client()
    response = client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": "        "},
    )

    assert response.status_code == 422


def test_non_access_token_is_rejected() -> None:
    """Only access tokens issued by the app should authorize protected routes."""

    client = create_test_client()
    forged_token = jwt.encode(
        {
            "sub": "user-id",
            "email": "alice@example.com",
            "exp": 9999999999,
            "iss": "ai-chat",
            "token_type": "refresh",
        },
        "test-secret-key-with-32-bytes-min",
        algorithm="HS256",
    )

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {forged_token}"},
    )

    assert response.status_code == 401
    assert response.json()["code"] == "auth_token_invalid"


def test_refresh_flow_rotates_and_revokes_refresh_tokens() -> None:
    """Refreshing should return a new token pair and revoke the old refresh token."""

    client = create_test_client()
    credentials = {
        "email": "alice@example.com",
        "password": "correct-horse-battery",
    }
    client.post("/auth/register", json=credentials)
    token_response = client.post("/auth/token", json=credentials)
    original_pair = token_response.json()

    refresh_response = client.post(
        "/auth/refresh",
        json={"refresh_token": original_pair["refresh_token"]},
    )
    rotated_pair = refresh_response.json()
    reused_refresh_response = client.post(
        "/auth/refresh",
        json={"refresh_token": original_pair["refresh_token"]},
    )

    assert refresh_response.status_code == 200
    assert rotated_pair["token_type"] == "bearer"
    assert rotated_pair["access_token"]
    assert rotated_pair["refresh_token"]
    assert rotated_pair["refresh_token"] != original_pair["refresh_token"]
    assert reused_refresh_response.status_code == 401
    assert reused_refresh_response.json()["code"] == "auth_refresh_token_invalid"


def test_refresh_token_can_be_revoked() -> None:
    """Revoked refresh tokens should no longer be accepted for rotation."""

    client = create_test_client()
    credentials = {
        "email": "alice@example.com",
        "password": "correct-horse-battery",
    }
    client.post("/auth/register", json=credentials)
    token_response = client.post("/auth/token", json=credentials)
    refresh_token = token_response.json()["refresh_token"]

    revoke_response = client.post("/auth/revoke", json={"refresh_token": refresh_token})
    refresh_response = client.post("/auth/refresh", json={"refresh_token": refresh_token})

    assert revoke_response.status_code == 204
    assert refresh_response.status_code == 401
    assert refresh_response.json()["code"] == "auth_refresh_token_invalid"


def test_invalid_refresh_token_is_rejected() -> None:
    """Unknown refresh tokens should fail with the same 401 response."""

    client = create_test_client()

    response = client.post(
        "/auth/refresh",
        json={"refresh_token": "not-a-real-refresh-token-value-that-is-long-enough"},
    )

    assert response.status_code == 401
    assert response.json()["code"] == "auth_refresh_token_invalid"


def test_register_rate_limit_is_enforced() -> None:
    """Registration should return 429 once the configured limit is exceeded."""

    client = TestClient(
        create_app(
            create_test_settings(rate_limit={"window_seconds": 60, "register_requests": 2})
        )
    )

    response_one = client.post(
        "/auth/register",
        json={"email": "one@example.com", "password": "correct-horse-battery"},
    )
    response_two = client.post(
        "/auth/register",
        json={"email": "two@example.com", "password": "correct-horse-battery"},
    )
    response_three = client.post(
        "/auth/register",
        json={"email": "three@example.com", "password": "correct-horse-battery"},
    )

    assert response_one.status_code == 202
    assert response_two.status_code == 202
    assert response_three.status_code == 429
    assert response_three.json()["code"] == "rate_limit_exceeded"
