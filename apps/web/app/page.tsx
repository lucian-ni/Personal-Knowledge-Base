import { ChatPanel } from "@/components/ChatPanel";
import { DocumentList } from "@/components/DocumentList";
import { DocumentUploader } from "@/components/DocumentUploader";
import { listDocuments } from "@/lib/api";

export default async function Home() {
    const documents = await listDocuments().catch(() => []);

    return (
        <main className="shell">
            <section className="hero">
                <p className="muted">Local-first RAG workspace</p>
                <h1>Personal Knowledge Base</h1>
                <p>
                    Store raw documents locally, keep business metadata in PostgreSQL, and retrieve
                    chunk text directly from Qdrant and OpenSearch.
                </p>
            </section>
            <section className="grid">
                <aside className="panel stack">
                    <h2>Documents</h2>
                    <DocumentUploader />
                    <DocumentList documents={documents} />
                </aside>
                <section className="panel stack">
                    <h2>Ask your knowledge base</h2>
                    <ChatPanel />
                </section>
            </section>
        </main>
    );
}
