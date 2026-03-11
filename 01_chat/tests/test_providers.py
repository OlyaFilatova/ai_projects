"""Provider adapter tests."""

import httpx
import pytest
from openai import APIStatusError

from ai_chat.adapters.llm.base import ChatTurn
from ai_chat.adapters.llm.factory import _build_base_provider
from ai_chat.adapters.llm.mock import ProviderError
from ai_chat.adapters.llm.openai_compatible import OpenAiCompatibleProvider
from ai_chat.adapters.llm.reliable import ReliabilityPolicy, ReliableLlmProvider
from tests.support import create_test_settings


def test_factory_supports_openai_compatible_provider() -> None:
    """Provider selection should support the real provider path."""

    settings = create_test_settings(
        llm={
            "provider": "openai-compatible",
            "api_base_url": "https://example.test/v1",
            "api_key": "test-api-key",
        },
    )

    provider = _build_base_provider(settings)

    assert isinstance(provider, OpenAiCompatibleProvider)


def test_factory_rejects_missing_openai_configuration() -> None:
    """The real provider path should fail clearly when config is incomplete."""

    settings = create_test_settings(llm={"provider": "openai-compatible"})

    with pytest.raises(ProviderError) as exc_info:
        _build_base_provider(settings)

    assert "requires AI_CHAT_LLM__API_BASE_URL" in str(exc_info.value)


def test_reliable_openai_provider_maps_http_errors() -> None:
    """Provider-facing HTTP failures should now be mapped in the reliability layer."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "upstream unavailable"})

    provider = ReliableLlmProvider(
        OpenAiCompatibleProvider(
            base_url="https://example.test/v1",
            api_key="test-api-key",
            model="gpt-4o-mini",
            timeout_seconds=5,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        ReliabilityPolicy(timeout_seconds=5, max_retries=0),
    )

    with pytest.raises(ProviderError, match="status 503"):
        provider.generate_reply(messages=[ChatTurn(role="user", content="Hello")])


def test_openai_compatible_provider_surfaces_raw_sdk_errors() -> None:
    """The base adapter should leave SDK transport errors unmapped."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "upstream unavailable"}, request=request)

    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1",
        api_key="test-api-key",
        model="gpt-4o-mini",
        timeout_seconds=5,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(APIStatusError):
        provider.generate_reply(messages=[ChatTurn(role="user", content="Hello")])


def test_openai_compatible_provider_streams_deltas() -> None:
    """The real provider adapter should parse streaming deltas."""

    body = (
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    provider = OpenAiCompatibleProvider(
        base_url="https://example.test/v1",
        api_key="test-api-key",
        model="gpt-4o-mini",
        timeout_seconds=5,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    chunks = list(provider.stream_reply(messages=[ChatTurn(role="user", content="Hello")]))

    assert chunks == ["Hello ", "world"]
