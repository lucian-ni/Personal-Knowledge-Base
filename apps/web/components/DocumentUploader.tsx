"use client";

import { useState } from "react";
import { uploadDocument } from "@/lib/api";

export function DocumentUploader() {
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
        await uploadDocument(file);
        setStatus("Upload accepted. Ingestion job queued.");
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
