"""Shared service-construction helpers for interface layers."""

from __future__ import annotations

from rag_knb.runtime_options import RuntimeOptionValues
from rag_knb.service import KnowledgeBaseService


def build_service_from_options(
    source: object,
    existing_service: KnowledgeBaseService | None = None,
) -> KnowledgeBaseService:
    """Build a service from runtime options, reusing an existing default service when possible."""
    runtime_options = RuntimeOptionValues.from_object(source)
    if not runtime_options.has_overrides():
        return existing_service or KnowledgeBaseService()
    return KnowledgeBaseService(config=runtime_options.to_runtime_config())
