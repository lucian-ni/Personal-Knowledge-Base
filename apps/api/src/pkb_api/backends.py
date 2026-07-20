from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch, helpers
from pkb_ingestion.models import QdrantPoint
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from pkb_api.retrieval import RetrievalHit

logger = logging.getLogger(__name__)

# OpenSearch index mapping mirrors the ``document`` payload produced by
# ``build_opensearch_document`` (see packages/ingestion/src/pkb_ingestion/index_contracts.py).
# Fields are stored at the top level (bulk_index uses the inner ``document`` dict as the
# ``_source``), so search must query ``text`` (not ``document.text``). The ``text`` and
# ``title`` fields use the built-in Lucene ``cjk`` analyzer so Chinese BM25 tokenizes by
# bigram instead of the standard analyzer's poor per-ideogram handling. No plugin is
# required. An existing local index must be dropped + recreated to pick up a mapping change
# (``DELETE /<index>`` then re-ingest).
OPENSEARCH_INDEX_MAPPING: dict[str, Any] = {
    "settings": {
        "analysis": {
            "analyzer": {
                "cjk": {"type": "cjk"},
            }
        }
    },
    "mappings": {
        "properties": {
            "docId": {"type": "keyword"},
            "chunkId": {"type": "keyword"},
            "chunkIndex": {"type": "integer"},
            "text": {"type": "text", "analyzer": "cjk"},
            "title": {"type": "text", "analyzer": "cjk"},
            "section": {"type": "keyword"},
            "page": {"type": "integer"},
            "checksum": {"type": "keyword"},
        }
    },
}


class QdrantIndexer:
    """Owns the Qdrant collection lifecycle and point upserts for ingestion."""

    def __init__(self, client: QdrantClient, collection: str, dimensions: int) -> None:
        self.client = client
        self.collection = collection
        self.dimensions = dimensions

    def ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection)
            return
        except Exception:
            # Collection does not exist (or store is unreachable); fall through and try
            # to create it so the real error - if any - surfaces from create_collection.
            pass
        self.client.create_collection(
            self.collection,
            vectors_config=qmodels.VectorParams(
                size=self.dimensions,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def upsert(self, points: list[QdrantPoint]) -> None:
        if not points:
            return
        self.client.upsert(
            self.collection,
            points=[
                qmodels.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
        )

    def delete_by_document(self, document_id: str) -> None:
        """Best-effort delete of all points whose payload ``docId`` matches.

        Swallows transport errors so a failed ingestion or a document delete never
        raises just because Qdrant is unreachable; the Postgres row is still removed.
        """
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="docId",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.warning("Qdrant delete_by_document failed for %s: %s", document_id, exc)

    def health(self) -> bool:
        """True if the Qdrant server is reachable (lists collections)."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


class QdrantVectorBackend:
    """Vector search backend over Qdrant; implements ``VectorSearchBackend``.

    Degrades to an empty result list when the store is unreachable so the search
    endpoint stays up when the local Docker stack is down. Connectivity problems
    during ingestion (upsert) are intentionally not swallowed.
    """

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def search(self, vector: list[float], limit: int) -> list[RetrievalHit]:
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Qdrant vector search failed, returning no vector hits: %s", exc)
            return []
        return [_hit_from_qdrant(point) for point in results]


class OpenSearchIndexer:
    """Owns the OpenSearch index lifecycle and bulk indexing for ingestion."""

    def __init__(self, client: OpenSearch, index: str) -> None:
        self.client = client
        self.index = index

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(index=self.index, body=OPENSEARCH_INDEX_MAPPING)

    def bulk_index(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        actions = [
            {
                "_op_type": "index",
                "_index": self.index,
                "_id": doc["_id"],
                "_source": doc["document"],
            }
            for doc in documents
        ]
        helpers.bulk(self.client, actions)

    def delete_by_document(self, document_id: str) -> None:
        """Best-effort delete of all docs whose ``docId`` matches (keyword term query).

        Swallows transport errors so cleanup never raises when OpenSearch is unreachable.
        """
        try:
            self.client.delete_by_query(
                index=self.index,
                body={"query": {"term": {"docId": document_id}}},
            )
        except Exception as exc:
            logger.warning("OpenSearch delete_by_document failed for %s: %s", document_id, exc)

    def health(self) -> bool:
        """True if the OpenSearch cluster is reachable (cluster info)."""
        try:
            self.client.info()
            return True
        except Exception:
            return False


class OpenSearchKeywordBackend:
    """Keyword (BM25) search backend over OpenSearch; implements ``KeywordSearchBackend``.

    Like the vector backend, search degrades to an empty list on transport errors.
    """

    def __init__(self, client: OpenSearch, index: str) -> None:
        self.client = client
        self.index = index

    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        try:
            response = self.client.search(
                index=self.index,
                body={
                    "query": {"match": {"text": query}},
                    "size": limit,
                },
            )
        except Exception as exc:
            logger.warning(
                "OpenSearch keyword search failed, returning no keyword hits: %s", exc
            )
            return []
        hits = response.get("hits", {}).get("hits", [])
        return [_hit_from_opensearch(hit) for hit in hits]


def _hit_from_qdrant(point: Any) -> RetrievalHit:
    payload = point.payload or {}
    return RetrievalHit(
        document_id=payload.get("docId", ""),
        chunk_id=payload.get("chunkId", str(point.id)),
        title=payload.get("title", ""),
        section=payload.get("section"),
        page=payload.get("page"),
        text=payload.get("text", ""),
        score=float(point.score or 0.0),
        source="vector",
    )


def _hit_from_opensearch(hit: dict[str, Any]) -> RetrievalHit:
    source = hit.get("_source", {})
    return RetrievalHit(
        document_id=source.get("docId", ""),
        chunk_id=source.get("chunkId", hit.get("_id", "")),
        title=source.get("title", ""),
        section=source.get("section"),
        page=source.get("page"),
        text=source.get("text", ""),
        score=float(hit.get("_score") or 0.0),
        source="keyword",
    )
