from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from opensearchpy import OpenSearch
from pkb_ingestion.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from pkb_ingestion.pipeline import IngestionPipeline
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from pkb_api.backends import (
    OpenSearchIndexer,
    OpenSearchKeywordBackend,
    QdrantIndexer,
    QdrantVectorBackend,
)
from pkb_api.db import SessionLocal
from pkb_api.retrieval import (
    LLMAnswerGenerator,
    Reranker,
    SimpleAnswerGenerator,
    make_answer_generator,
)
from pkb_api.settings import Settings, settings
from pkb_api.storage import DocumentStorage


@dataclass
class Services:
    """The wired collaborators the API endpoints depend on.

    Real instances are built lazily from settings; tests inject a fake ``Services``
    via ``app.dependency_overrides`` so they never touch Postgres/Qdrant/OpenSearch
    or load the local embedding/reranker models.
    """

    pipeline: IngestionPipeline
    storage: DocumentStorage
    qdrant_indexer: QdrantIndexer
    opensearch_indexer: OpenSearchIndexer
    vector_backend: QdrantVectorBackend
    keyword_backend: OpenSearchKeywordBackend
    answer_generator: SimpleAnswerGenerator | LLMAnswerGenerator
    reranker: Reranker | None
    # Factory the background ingestion task uses to open its own session (decoupled from
    # the request's session, which closes when the response is sent). Tests inject a
    # SQLite factory so the background task writes to the same in-memory DB.
    session_factory: Callable[[], Session] = SessionLocal


def make_embedding_provider(config: Settings) -> EmbeddingProvider:
    """Select the embedding provider from settings.

    ``provider=openai`` requires ``EMBEDDING_API_BASE_URL``, ``EMBEDDING_API_KEY``,
    and ``EMBEDDING_MODEL`` to all be set; it raises ``ValueError`` if any are
    missing rather than silently falling back to a local model (a silent switch to a
    different embedding model would produce invisibly-different vectors and break
    the Qdrant collection's dimension). Any other value uses the local bge model.
    """
    if config.embedding_provider == "openai":
        missing = [
            name
            for name, value in (
                ("EMBEDDING_API_BASE_URL", config.embedding_api_base_url),
                ("EMBEDDING_API_KEY", config.embedding_api_key),
                ("EMBEDDING_MODEL", config.embedding_model),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai but "
                + ", ".join(missing)
                + " not set; set it/them or use EMBEDDING_PROVIDER=local"
            )
        return OpenAICompatibleEmbeddingProvider(
            base_url=config.embedding_api_base_url,
            api_key=config.embedding_api_key,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
    return LocalEmbeddingProvider(
        model_name=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )


def make_reranker(config: Settings) -> Reranker | None:
    """Build the reranker, or None when disabled.

    ``provider == "api"`` calls an Aliyun Model Studio ``/reranks`` endpoint
    (the ``compatible-api`` path, distinct from embeddings' ``compatible-mode``);
    ``RERANKER_API_BASE_URL`` must be set explicitly, while the key reuses
    ``EMBEDDING_API_KEY`` when ``RERANKER_API_KEY`` is empty. Otherwise the local
    Qwen3-Reranker cross-encoder runs in-process.
    """
    if not config.reranker_enabled:
        return None
    if config.reranker_provider == "api":
        base_url = config.reranker_api_base_url
        api_key = config.reranker_api_key or config.embedding_api_key
        if base_url and api_key and config.reranker_model:
            from pkb_api.reranker import ApiReranker

            return ApiReranker(
                base_url=base_url,
                api_key=api_key,
                model=config.reranker_model,
            )
        return None
    from pkb_api.reranker import QwenReranker

    return QwenReranker(model_name=config.reranker_model)


def build_services(config: Settings) -> Services:
    """Build a ``Services`` instance from settings. Clients are lazy - construction
    does not connect or load models, so importing the app (and the test suite)
    needs no live stores or model downloads."""
    embedding_provider = make_embedding_provider(config)
    pipeline = IngestionPipeline(embedding_provider=embedding_provider)

    storage = DocumentStorage(config.storage_docs_path)

    qdrant_client = QdrantClient(url=config.qdrant_url)
    opensearch_client = OpenSearch(hosts=[config.opensearch_url])

    qdrant_indexer = QdrantIndexer(
        qdrant_client, config.qdrant_collection, config.embedding_dimensions
    )
    opensearch_indexer = OpenSearchIndexer(opensearch_client, config.opensearch_index)
    vector_backend = QdrantVectorBackend(qdrant_client, config.qdrant_collection)
    keyword_backend = OpenSearchKeywordBackend(opensearch_client, config.opensearch_index)

    answer_generator = make_answer_generator(config)
    reranker = make_reranker(config)

    return Services(
        pipeline=pipeline,
        storage=storage,
        qdrant_indexer=qdrant_indexer,
        opensearch_indexer=opensearch_indexer,
        vector_backend=vector_backend,
        keyword_backend=keyword_backend,
        answer_generator=answer_generator,
        reranker=reranker,
        session_factory=SessionLocal,
    )


_services_singleton: Services | None = None
_services_lock = threading.Lock()


def get_services() -> Services:
    """FastAPI dependency returning the lazily-built real services singleton.

    Thread-safe (sync endpoints run in a Starlette threadpool, so the first concurrent
    requests could otherwise build services twice). Tests override this via
    ``app.dependency_overrides[get_services] = ...``."""
    global _services_singleton
    if _services_singleton is None:
        with _services_lock:
            if _services_singleton is None:
                _services_singleton = build_services(settings)
    return _services_singleton
