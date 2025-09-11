import os
import types

import pytest


from llama_index.core.base.embeddings.base import BaseEmbedding


class _StubEmbedModel(BaseEmbedding):
    def _get_text_embedding(self, text: str):
        # Simple deterministic embedding stub (length 8)
        vec = [0.0] * 8
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % 8] += (ch % 97) / 97.0
        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    # Also satisfy query path
    def _get_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str):
        return self._get_text_embedding(query)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    # Disable Redis usage implicitly; rely on in-memory fallback
    monkeypatch.delenv("REDIS_URL", raising=False)
    # Enable semantic cache (default true, keep explicit)
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")

    # Patch LlamaIndex global Settings.embed_model with BaseEmbedding subclass
    from llama_index.core import Settings

    original_embed = Settings.embed_model
    Settings.embed_model = _StubEmbedModel()
    try:
        yield
    finally:
        Settings.embed_model = original_embed


def test_semantic_cache_put_and_get(monkeypatch):
    from src.cache import SemanticCache

    cache = SemanticCache()
    assert cache.enabled is True

    # Initially empty
    assert cache.get("what is rag?") is None

    # Build a minimal response-like object with expected attributes
    class _Resp:
        def __init__(self):
            self.response = "RAG is retrieval-augmented generation."
            self.source_nodes = []
            self.metadata = {}

    resp = _Resp()
    ok = cache.put("what is rag?", resp, estimated_cost=0.001)
    assert ok is True

    # Exact query should be a hit
    hit = cache.get("what is rag?")
    assert hit is not None
    response_data, nodes, similarity = hit
    assert "response" in response_data
    assert similarity >= 0.85


def test_cache_stats_update():
    from src.cache import SemanticCache

    cache = SemanticCache()
    # Miss increments totals
    assert cache.get("unknown query") is None
    stats = cache.get_stats()
    assert stats.total_queries >= 1

