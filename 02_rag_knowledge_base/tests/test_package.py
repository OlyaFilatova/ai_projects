"""Package smoke tests."""

from rag_knb import DeterministicEmbedder, KnowledgeBaseService, RuntimeConfig


def test_package_exports_basic_types() -> None:
    """The package should expose the service and config entry points."""
    service = KnowledgeBaseService()
    assert isinstance(service.config, RuntimeConfig)
    assert DeterministicEmbedder().embed("alpha beta")
