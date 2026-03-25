"""Package smoke tests."""

from rag_knb_app.service import KnowledgeBaseService
from rag_knb_core.config import RuntimeConfig
from rag_knb_retrieval.embeddings import DeterministicEmbedder


def test_package_exports_basic_types() -> None:
    """The package should expose the service and config entry points."""
    service = KnowledgeBaseService()
    assert isinstance(service.config, RuntimeConfig)
    assert DeterministicEmbedder().embed("alpha beta")
