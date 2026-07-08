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

export async function uploadDocument(file: File): Promise<void> {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        body: form
    });
    if (!response.ok) {
        throw new Error("Upload failed");
    }
}
