from pkb_api.models import Document, DocumentChunk, IngestionJob


def test_document_table_separates_file_path_from_business_metadata() -> None:
    columns = Document.__table__.columns

    assert "id" in columns
    assert "title" in columns
    assert "original_filename" in columns
    assert "file_path" in columns
    assert "status" in columns
    assert "version" in columns
    assert "created_at" in columns
    assert "updated_at" in columns
    assert "binary" not in columns


def test_document_chunk_table_links_postgres_to_search_indexes() -> None:
    columns = DocumentChunk.__table__.columns

    assert "document_id" in columns
    assert "chunk_id" in columns
    assert "chunk_index" in columns
    assert "qdrant_point_id" in columns
    assert "opensearch_document_id" in columns
    assert "text" not in columns


def test_ingestion_job_tracks_status_and_failures() -> None:
    columns = IngestionJob.__table__.columns

    assert "document_id" in columns
    assert "status" in columns
    assert "failure_reason" in columns
    assert "retry_count" in columns
