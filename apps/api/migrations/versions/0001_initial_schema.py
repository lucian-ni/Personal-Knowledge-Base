"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    document_status = sa.Enum("uploaded", "processing", "ready", "failed", name="document_status")
    ingestion_job_status = sa.Enum(
        "queued",
        "running",
        "completed",
        "failed",
        name="ingestion_job_status",
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("status", document_status, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=255), nullable=True),
        sa.Column("owner", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(length=64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
        ),
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("section", sa.String(length=512), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("qdrant_point_id", sa.String(length=128), nullable=False),
        sa.Column("opensearch_document_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_document_chunks_chunk_id", "document_chunks", ["chunk_id"], unique=True)
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(length=64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
        ),
        sa.Column("status", ingestion_job_status, nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_document_chunks_chunk_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    sa.Enum(name="ingestion_job_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="document_status").drop(op.get_bind(), checkfirst=True)
