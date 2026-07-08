from __future__ import annotations

from pathlib import Path

from pkb_ingestion.models import DocumentArtifact


class DoclingMarkdownConverter:
    def convert(
        self,
        document_id: str,
        source_path: Path,
        title: str | None = None,
    ) -> DocumentArtifact:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError(
                "Docling is required for PDF conversion. Install project dependencies."
            ) from exc

        converter = DocumentConverter()
        result = converter.convert(source_path)
        markdown = result.document.export_to_markdown()

        return DocumentArtifact(
            document_id=document_id,
            title=title or source_path.stem,
            source_path=source_path,
            markdown=markdown,
        )
