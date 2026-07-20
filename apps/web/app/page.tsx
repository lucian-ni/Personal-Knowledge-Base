import Link from "next/link";
import { listDocuments } from "@/lib/api";

export default async function Home() {
    const documents = await listDocuments().catch(() => []);
    const ready = documents.filter((document) => document.status === "ready").length;

    return (
        <main className="shell">
            <section className="hero">
                <p className="muted">Local-first RAG workspace</p>
                <h1>Personal Knowledge Base</h1>
                <p>
                    Store raw documents locally, keep business metadata in PostgreSQL, and retrieve
                    chunk text directly from Qdrant and OpenSearch.
                </p>
                {documents.length > 0 ? (
                    <p className="muted">
                        {ready} of {documents.length} documents indexed.
                    </p>
                ) : null}
            </section>
            <section className="cards">
                <Link className="card" href="/documents">
                    <h2>Documents</h2>
                    <p className="muted">Upload PDFs and manage your indexed knowledge base.</p>
                    <span className="card-cta">Open documents →</span>
                </Link>
                <Link className="card" href="/chat">
                    <h2>Chat</h2>
                    <p className="muted">Ask questions and get cited answers from your documents.</p>
                    <span className="card-cta">Start chatting →</span>
                </Link>
            </section>
        </main>
    );
}
