"""Integration tests that load the real local models (bge-small + Qwen3-Reranker).

Skipped by default because they download/load models (~1.3GB) and are slow.
Run with: PKB_RUN_INTEGRATION=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration -q
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PKB_RUN_INTEGRATION") != "1",
    reason="set PKB_RUN_INTEGRATION=1 to run (loads real models)",
)


def test_local_embedding_real_model_loads_and_normalizes() -> None:
    from pkb_ingestion.embeddings import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider()
    vectors = provider.embed(["ReentrantLock is a reentrant lock", "Redis is a cache"])

    assert len(vectors) == 2
    assert provider.dimensions == 512
    assert len(vectors[0]) == 512
    # L2-normalized.
    assert abs(sum(x * x for x in vectors[0]) - 1.0) < 1e-4


def test_qwen_reranker_ranks_relevant_doc_first() -> None:
    from pkb_api.reranker import QwenReranker
    from pkb_api.retrieval import RetrievalHit

    reranker = QwenReranker()
    hits = [
        RetrievalHit(
            "d", "c1", "t", None, None, "ReentrantLock is a reentrant lock in Java.", 0.5, "rrf"
        ),
        RetrievalHit(
            "d", "c2", "t", None, None, "Redis is an in-memory key-value store.", 0.4, "rrf"
        ),
    ]
    ranked = reranker.rerank("What is ReentrantLock?", hits, limit=2)

    assert len(ranked) == 2
    assert ranked[0].chunk_id == "c1"  # the relevant doc ranks first
    assert ranked[0].source == "reranked"
