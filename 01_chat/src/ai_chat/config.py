"""Configuration models for the AI chat application."""

from functools import lru_cache

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    """Connection settings for the application database."""

    url: str = "sqlite+pysqlite:///./ai_chat.db"


class JwtSettings(BaseModel):
    """JWT configuration placeholders for future authentication flows."""

    secret: SecretStr | None = None
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    issuer: str = "ai-chat"

    @field_validator("access_token_expire_minutes")
    @classmethod
    def validate_expiration(cls, value: int) -> int:
        """Ensure the JWT expiration is a positive duration."""

        if value <= 0:
            raise ValueError("JWT access token expiration must be greater than zero.")
        return value

    @field_validator("refresh_token_expire_days")
    @classmethod
    def validate_refresh_expiration(cls, value: int) -> int:
        """Ensure the refresh-token expiration is a positive duration."""

        if value <= 0:
            raise ValueError("JWT refresh token expiration must be greater than zero.")
        return value

    @field_validator("secret")
    @classmethod
    def validate_secret_length(cls, value: SecretStr | None) -> SecretStr | None:
        """Require strong enough JWT secrets when a secret is configured."""

        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("JWT secrets must be at least 32 characters long.")
        return value


class LlmSettings(BaseModel):
    """Settings for selecting and configuring the LLM provider layer."""

    provider: str = "mock"
    model: str = "gpt-4o-mini"
    api_base_url: str | None = None
    api_key: SecretStr | None = None
    timeout_seconds: float = 10.0
    max_retries: int = 1

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        """Ensure the provider name is present and normalized."""

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("LLM provider must not be empty.")
        return normalized

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        """Ensure the provider timeout is positive."""

        if value <= 0:
            raise ValueError("LLM timeout must be greater than zero.")
        return value

    @field_validator("max_retries")
    @classmethod
    def validate_retries(cls, value: int) -> int:
        """Ensure retry count is not negative."""

        if value < 0:
            raise ValueError("LLM max retries must not be negative.")
        return value


class RateLimitSettings(BaseModel):
    """Configuration for endpoint rate limiting."""

    window_seconds: int = 60
    storage_uri: str = "memory://"
    register_requests: int = 5
    token_requests: int = 10
    chat_requests: int = 30
    chat_stream_requests: int = 15

    @field_validator(
        "window_seconds",
        "register_requests",
        "token_requests",
        "chat_requests",
        "chat_stream_requests",
    )
    @classmethod
    def validate_positive_values(cls, value: int) -> int:
        """Ensure all rate-limiting values are positive."""

        if value <= 0:
            raise ValueError("Rate-limiting values must be greater than zero.")
        return value

    @field_validator("storage_uri")
    @classmethod
    def validate_storage_uri(cls, value: str) -> str:
        """Require a non-empty storage URI for the limiter backend."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("Rate-limit storage URI must not be empty.")
        return normalized


class UsageQuotaSettings(BaseModel):
    """Configuration for per-user chat quotas."""

    window_seconds: int = 3600
    storage_uri: str = "memory://"
    chat_requests: int = 100
    chat_stream_requests: int = 50

    @field_validator("window_seconds", "chat_requests", "chat_stream_requests")
    @classmethod
    def validate_positive_values(cls, value: int) -> int:
        """Ensure all quota values are positive."""

        if value <= 0:
            raise ValueError("Quota values must be greater than zero.")
        return value

    @field_validator("storage_uri")
    @classmethod
    def validate_storage_uri(cls, value: str) -> str:
        """Require a non-empty storage URI for the quota backend."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("Quota storage URI must not be empty.")
        return normalized


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AI_CHAT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "AI Chat Backend"
    env: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    max_request_body_bytes: int = 16_384
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    usage_quota: UsageQuotaSettings = Field(default_factory=UsageQuotaSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    jwt: JwtSettings = Field(default_factory=JwtSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)

    @field_validator("env")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Restrict the environment value to known modes."""

        normalized = value.strip().lower()
        if normalized not in {"development", "test", "production"}:
            raise ValueError(
                "AI_CHAT_ENV must be one of: development, test, production."
            )
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure the configured log level is usable."""

        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("AI_CHAT_LOG_LEVEL must be a standard logging level.")
        return normalized

    @field_validator("max_request_body_bytes")
    @classmethod
    def validate_max_request_body_bytes(cls, value: int) -> int:
        """Ensure the request size limit is positive."""

        if value <= 0:
            raise ValueError("AI_CHAT_MAX_REQUEST_BODY_BYTES must be greater than zero.")
        return value

    @model_validator(mode="after")
    def validate_secret_requirements(self) -> "Settings":
        """Require explicit JWT secrets in production deployments."""

        if self.env == "production" and self.jwt.secret is None:
            raise ValueError(
                "AI_CHAT_JWT__SECRET is required when AI_CHAT_ENV=production."
            )
        if (
            self.env == "production"
            and self.llm.provider == "openai-compatible"
            and self.llm.api_base_url is not None
            and not self.llm.api_base_url.startswith("https://")
        ):
            raise ValueError(
                "Production OpenAI-compatible providers must use an https:// base URL."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings for process-wide reuse."""

    try:
        return Settings()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc
