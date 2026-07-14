from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import OpenSearch
from pkb_ingestion.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from pkb_ingestion.pipeline import IngestionPipeline
from qdrant_client import QdrantClient

from pkb_api.backends import (
    OpenSearchIndexer,
    OpenSearchKeywordBackend,
    QdrantIndexer,
    QdrantVectorBackend,
)
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


def make_embedding_provider(config: Settings) -> EmbeddingProvider:
    """Select the embedding provider from settings; defaults to the local bge model."""
    if (
        config.embedding_provider == "openai"
        and config.embedding_api_base_url
        and config.embedding_api_key
        and config.embedding_model
    ):
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
    """Build the cross-encoder reranker, or None when disabled."""
    if not config.reranker_enabled:
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
    )


_services_singleton: Services | None = None


def get_services() -> Services:
    """FastAPI dependency returning the lazily-built real services singleton.

    Tests override this via ``app.dependency_overrides[get_services] = ...``."""
    global _services_singleton
    if _services_singleton is None:
        _services_singleton = build_services(settings)
    return _services_singleton
