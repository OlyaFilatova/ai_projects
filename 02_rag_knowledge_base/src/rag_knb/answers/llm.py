"""Built-in LLM providers for grounded answer generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

from rag_knb.answers.context_building import render_evidence_set
from rag_knb.answers.prompt_injection import build_prompt_injection_policy_text
from rag_knb.config import DEFAULT_LLM_BASE_URL, RuntimeConfig
from rag_knb.errors import DependencyUnavailableError
from rag_knb.models import RetrievalResult

DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


@dataclass(frozen=True, slots=True)
class OpenAIChatTextGenerator:
    """OpenAI-compatible chat-completions generator."""

    api_key: str
    model: str
    base_url: str
    timeout_seconds: int

    def generate(self, question: str, context: list[RetrievalResult]) -> str:
        """Generate a grounded answer from retrieved context."""
        request_payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided context. "
                        f"{build_prompt_injection_policy_text()} "
                        "Return at most two short sentences before citations. "
                        "Do not restate all context. "
                        "If the context is insufficient, say so briefly. "
                        "Include supporting chunk citations in the form [chunk_id]."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(question, context),
                },
            ],
        }
        response_payload = self._post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            request_payload,
        )
        choices = response_payload.get("choices", [])
        if not choices:
            raise DependencyUnavailableError("The LLM provider returned no completion choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DependencyUnavailableError("The LLM provider returned an empty completion.")
        return content.strip()

    def _build_user_prompt(self, question: str, context: list[RetrievalResult]) -> str:
        """Render the grounded prompt body for the LLM request."""
        rendered_context = render_evidence_set(question, context)
        return (
            f"Question: {question}\n\n"
            "Write a direct grounded answer in no more than two short sentences.\n\n"
            f"Safety policy: {build_prompt_injection_policy_text()}\n\n"
            f"Context:\n{rendered_context}"
        )

    def _post_json(self, url: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON request and return the decoded JSON payload."""
        request_body = json.dumps(request_payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise DependencyUnavailableError("The LLM provider returned a non-object JSON payload.")
        return response_payload


def build_builtin_text_generator(config: RuntimeConfig) -> OpenAIChatTextGenerator:
    """Build the default built-in LLM generator for generative answers."""
    api_key = os.getenv(DEFAULT_OPENAI_API_KEY_ENV)
    if not api_key:
        raise DependencyUnavailableError(
            "Generative answer mode requires the OPENAI_API_KEY environment variable."
        )
    if not config.allow_custom_llm_base_url and config.llm_base_url != DEFAULT_LLM_BASE_URL:
        raise DependencyUnavailableError(
            "Custom llm_base_url values are disabled by the current library policy."
        )
    return OpenAIChatTextGenerator(
        api_key=api_key,
        model=config.llm_model,
        base_url=config.llm_base_url,
        timeout_seconds=config.llm_request_timeout_seconds,
    )
