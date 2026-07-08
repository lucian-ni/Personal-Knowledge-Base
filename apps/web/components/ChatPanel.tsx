"use client";

import { useState } from "react";
import type { SearchResult } from "@/lib/api";
import { searchKnowledgeBase } from "@/lib/api";

export function ChatPanel() {
    const [query, setQuery] = useState("");
    const [result, setResult] = useState<SearchResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault();
        if (!query.trim()) {
            return;
        }
        setIsLoading(true);
        try {
            setResult(await searchKnowledgeBase(query));
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
            {result ? (
                <section className="stack">
                    <h3>Answer</h3>
                    <p>{result.answer}</p>
                    <h3>Citations</h3>
                    {result.citations.length === 0 ? (
                        <p className="muted">No citations yet. Upload and ingest documents first.</p>
                    ) : (
                        result.citations.map((citation) => (
                            <article className="citation" key={citation.chunk_id}>
                                <strong>{citation.title}</strong>
                                <p>{citation.text}</p>
                                <p className="muted">
                                    {citation.section ?? "Untitled section"}
                                    {citation.page ? `, page ${citation.page}` : ""}
                                </p>
                            </article>
                        ))
                    )}
                </section>
            ) : null}
        </form>
    );
}
