"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
    { href: "/", label: "Home" },
    { href: "/documents", label: "Documents" },
    { href: "/chat", label: "Chat" }
] as const;

export function Nav() {
    const pathname = usePathname();
    return (
        <nav className="nav">
            <Link className="nav-brand" href="/">
                Personal Knowledge Base
            </Link>
            <div className="nav-links">
                {LINKS.map((link) => {
                    const active = pathname === link.href;
                    return (
                        <Link
                            key={link.href}
                            href={link.href}
                            className={active ? "nav-link active" : "nav-link"}
                        >
                            {link.label}
                        </Link>
                    );
                })}
            </div>
        </nav>
    );
}
