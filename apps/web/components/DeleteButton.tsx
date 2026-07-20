"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { deleteDocument } from "@/lib/api";

export function DeleteButton({ documentId }: { documentId: string }) {
    const router = useRouter();
    const [deleting, setDeleting] = useState(false);

    async function onClick() {
        setDeleting(true);
        try {
            await deleteDocument(documentId);
            router.refresh();
        } catch {
            setDeleting(false);
        }
    }

    return (
        <button
            className="button"
            type="button"
            onClick={onClick}
            disabled={deleting}
        >
            {deleting ? "Deleting…" : "Delete"}
        </button>
    );
}
