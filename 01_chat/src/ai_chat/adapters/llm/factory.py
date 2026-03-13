"""Factory for building LLM provider adapters."""

from fastapi import status

from ai_chat.adapters.llm.base import LlmProvider
from ai_chat.adapters.llm.mock import (
    FailingMockLlmProvider,
    FlakyMockLlmProvider,
    MockLlmProvider,
    PartialFailingMockLlmProvider,
    ProviderError,
    TimeoutMockLlmProvider,
)
from ai_chat.adapters.llm.openai_compatible import OpenAiCompatibleProvider
from ai_chat.adapters.llm.reliable import ReliabilityPolicy, ReliableLlmProvider
from ai_chat.config import Settings


def _build_base_provider(settings: Settings) -> LlmProvider:
    """Build the configured provider implementation."""

    if settings.llm.provider == "mock":
        return MockLlmProvider()
    if settings.llm.provider == "openai-compatible":
        if settings.llm.api_base_url is None or settings.llm.api_key is None:
            raise ProviderError(
                message=(
                    "OpenAI-compatible provider requires AI_CHAT_LLM__API_BASE_URL "
                    "and AI_CHAT_LLM__API_KEY."
                ),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="provider_configuration_invalid",
            )
        return OpenAiCompatibleProvider(
            base_url=settings.llm.api_base_url,
            api_key=settings.llm.api_key.get_secret_value(),
            model=settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
        )
    if settings.llm.provider == "failing-mock":
        return FailingMockLlmProvider()
    if settings.llm.provider == "partial-failing-mock":
        return PartialFailingMockLlmProvider()
    if settings.llm.provider == "flaky-mock":
        return FlakyMockLlmProvider()
    if settings.llm.provider == "timeout-mock":
        return TimeoutMockLlmProvider()
    raise ProviderError(
        message=f"Unsupported LLM provider: {settings.llm.provider}",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        code="provider_unsupported",
    )


def build_provider(settings: Settings) -> LlmProvider:
    """Build the configured provider implementation with reliability policy."""

    return ReliableLlmProvider(
        _build_base_provider(settings),
        ReliabilityPolicy(
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        ),
    )
