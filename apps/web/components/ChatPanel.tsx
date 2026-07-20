"use client";

import { useState } from "react";
import type { ChunkCitation } from "@/lib/api";
import { streamSearch } from "@/lib/api";

export function ChatPanel() {
    const [query, setQuery] = useState("");
    const [answer, setAnswer] = useState("");
    const [citations, setCitations] = useState<ChunkCitation[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault();
        if (!query.trim()) {
            return;
        }
        setIsLoading(true);
        setAnswer("");
        setCitations([]);
        setError(null);
        try {
            // Citations stream first, then the answer token-by-token.
            await streamSearch(query, {
                onCitations: setCitations,
                onDelta: (delta) => setAnswer((prev) => prev + delta)
            });
        } catch {
            setError("Search failed. Is the API running?");
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <form className="stack" onSubmit={onSubmit}>
            <textarea
                className="textarea"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask a question about your local documents..."
            />
            <button className="button" type="submit" disabled={isLoading}>
                {isLoading ? "Searching..." : "Ask"}
            </button>
            {error ? <p className="muted">{error}</p> : null}
            {answer ? (
                <section className="stack">
                    <h3>Answer</h3>
                    <p>{answer}</p>
                </section>
            ) : null}
            {citations.length > 0 ? (
                <section className="stack">
                    <h3>Citations</h3>
                    {citations.map((citation) => (
                        <article className="citation" key={citation.chunk_id}>
                            <strong>{citation.title}</strong>
                            <p>{citation.text}</p>
                            <p className="muted">
                                {citation.section ?? "Untitled section"}
                                {citation.page ? `, page ${citation.page}` : ""}
                            </p>
                        </article>
                    ))}
                </section>
            ) : null}
        </form>
    );
}
