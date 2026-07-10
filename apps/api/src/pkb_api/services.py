from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import OpenSearch
from pkb_ingestion.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
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
from pkb_api.retrieval import LLMAnswerGenerator, SimpleAnswerGenerator, make_answer_generator
from pkb_api.settings import Settings, settings
from pkb_api.storage import DocumentStorage


@dataclass
class Services:
    """The wired collaborators the API endpoints depend on.

    Real instances are built lazily from settings; tests inject a fake ``Services``
    via ``app.dependency_overrides`` so they never touch Postgres/Qdrant/OpenSearch.
    """

    pipeline: IngestionPipeline
    storage: DocumentStorage
    qdrant_indexer: QdrantIndexer
    opensearch_indexer: OpenSearchIndexer
    vector_backend: QdrantVectorBackend
    keyword_backend: OpenSearchKeywordBackend
    answer_generator: SimpleAnswerGenerator | LLMAnswerGenerator


def make_embedding_provider(config: Settings) -> EmbeddingProvider:
    """Select the embedding provider from settings; defaults to the local hash provider."""
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
    return HashEmbeddingProvider(dimensions=config.embedding_dimensions)


def build_services(config: Settings) -> Services:
    """Build a ``Services`` instance from settings. Clients are lazy - construction
    does not connect, so importing the app (and the test suite) needs no live stores."""
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

    return Services(
        pipeline=pipeline,
        storage=storage,
        qdrant_indexer=qdrant_indexer,
        opensearch_indexer=opensearch_indexer,
        vector_backend=vector_backend,
        keyword_backend=keyword_backend,
        answer_generator=answer_generator,
    )


_services_singleton: Services | None = None


def get_services() -> Services:
    """FastAPI dependency returning the lazily-built real services singleton.

    Tests override this via ``app.dependency_overrides[get_services] = ...``."""
    global _services_singleton
    if _services_singleton is None:
        _services_singleton = build_services(settings)
    return _services_singleton
