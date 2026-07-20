import { DocumentList } from "@/components/DocumentList";
import { DocumentUploader } from "@/components/DocumentUploader";
import { listDocuments } from "@/lib/api";

export default async function DocumentsPage() {
    const documents = await listDocuments().catch(() => []);

    return (
        <main className="shell">
            <section className="panel stack">
                <h1>Documents</h1>
                <p className="muted">Upload a PDF to ingest, then search across it from Chat.</p>
                <DocumentUploader />
                <DocumentList documents={documents} />
            </section>
        </main>
    );
}
