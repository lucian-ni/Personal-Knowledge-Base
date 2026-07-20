import { ChatPanel } from "@/components/ChatPanel";

export default function ChatPage() {
    return (
        <main className="shell">
            <section className="panel stack">
                <h1>Ask your knowledge base</h1>
                <p className="muted">
                    Answers are grounded in your indexed documents, with citations.
                </p>
                <ChatPanel />
            </section>
        </main>
    );
}
