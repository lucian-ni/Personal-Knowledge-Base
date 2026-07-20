export type DocumentRead = {
    id: string;
    title: string;
    original_filename: string;
    mime_type: string;
    status: string;
    version: number;
};

export type ChunkCitation = {
    document_id: string;
    chunk_id: string;
    title: string;
    section: string | null;
    page: number | null;
    text: string;
    score: number;
};

export type SearchResult = {
    answer: string;
    citations: ChunkCitation[];
};

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function listDocuments(): Promise<DocumentRead[]> {
    const response = await fetch(`${apiBaseUrl}/documents`, { cache: "no-store" });
    if (!response.ok) {
        throw new Error("Failed to load documents");
    }
    return response.json();
}

export async function uploadDocument(file: File): Promise<{ document_id: string; status: string }> {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        body: form
    });
    if (!response.ok) {
        throw new Error("Upload failed");
    }
    return response.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
    const response = await fetch(`${apiBaseUrl}/documents/${documentId}`, { method: "DELETE" });
    if (!response.ok && response.status !== 404) {
        throw new Error("Delete failed");
    }
}

export async function searchKnowledgeBase(query: string): Promise<SearchResult> {
    const response = await fetch(`${apiBaseUrl}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 8 })
    });
    if (!response.ok) {
        throw new Error("Search failed");
    }
    return response.json();
}

/** Stream a cited answer over SSE: citations arrive first, then answer deltas. */
export async function streamSearch(
    query: string,
    handlers: {
        onCitations: (citations: ChunkCitation[]) => void;
        onDelta: (delta: string) => void;
        signal?: AbortSignal;
    }
): Promise<void> {
    const response = await fetch(`${apiBaseUrl}/search/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 8 }),
        signal: handlers.signal
    });
    if (!response.ok || !response.body) {
        throw new Error("Search failed");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by a blank line.
        let separator: number;
        while ((separator = buffer.indexOf("\n\n")) !== -1) {
            const raw = buffer.slice(0, separator);
            buffer = buffer.slice(separator + 2);
            const event = parseSseEvent(raw);
            if (event.type === "citations") {
                handlers.onCitations(event.data as ChunkCitation[]);
            } else if (event.type === "delta") {
                handlers.onDelta((event.data as { content: string }).content);
            }
        }
    }
}

function parseSseEvent(raw: string): { type: string; data: unknown } {
    let type = "message";
    let data = "";
    for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) {
            type = line.slice("event: ".length);
        } else if (line.startsWith("data: ")) {
            data += line.slice("data: ".length);
        }
    }
    return { type, data: data ? (JSON.parse(data) as unknown) : {} };
}
