import type { DocumentRead } from "@/lib/api";
import { DeleteButton } from "@/components/DeleteButton";

export function DocumentList({ documents }: { documents: DocumentRead[] }) {
    if (documents.length === 0) {
        return <p className="muted">No documents indexed yet.</p>;
    }

    return (
        <div className="stack">
            {documents.map((document) => (
                <article className="citation" key={document.id}>
                    <strong>{document.title}</strong>
                    <p className="muted">
                        {document.original_filename} · {document.status} · v{document.version}
                    </p>
                    <DeleteButton documentId={document.id} />
                </article>
            ))}
        </div>
    );
}
