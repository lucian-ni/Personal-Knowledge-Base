"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { listDocuments, uploadDocument } from "@/lib/api";

export function DocumentUploader() {
    const router = useRouter();
    const [status, setStatus] = useState<string>("Select a PDF to start ingestion.");

    async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault();
        const input = event.currentTarget.elements.namedItem("file") as HTMLInputElement;
        const file = input.files?.[0];
        if (!file) {
            setStatus("Choose a file first.");
            return;
        }

        setStatus("Uploading...");
        try {
            const { document_id } = await uploadDocument(file);
            setStatus("Ingesting… (chunking + embedding + indexing in the background)");
            const finalStatus = await pollUntilDone(document_id);
            setStatus(
                finalStatus === "ready"
                    ? "Ingestion complete."
                    : `Ingestion ${finalStatus}.`
            );
            router.refresh();
        } catch {
            setStatus("Upload failed.");
        }
        event.currentTarget.reset();
    }

    return (
        <form className="stack" onSubmit={onSubmit}>
            <input className="input" name="file" type="file" accept="application/pdf" />
            <button className="button" type="submit">
                Upload document
            </button>
            <p className="muted">{status}</p>
        </form>
    );
}

/** Poll GET /documents until the just-uploaded doc reaches a terminal state. */
async function pollUntilDone(documentId: string): Promise<string> {
    for (let i = 0; i < 120; i++) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        try {
            const docs = await listDocuments();
            const doc = docs.find((d) => d.id === documentId);
            if (doc && (doc.status === "ready" || doc.status === "failed")) {
                return doc.status;
            }
        } catch {
            // transient fetch error - keep polling
        }
    }
    return "timed out";
}
