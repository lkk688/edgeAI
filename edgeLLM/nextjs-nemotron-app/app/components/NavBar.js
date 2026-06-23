"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LABS = [
  { href: "/",          label: "Chat",          sub: "streaming Nemotron" },
  { href: "/retrieval", label: "Retrieval Lab", sub: "embed → rerank"     },
  { href: "/omni",      label: "Omni Lab",      sub: "image + audio"      },
  { href: "/asr",       label: "ASR Lab",       sub: "speech-to-text"     },
  { href: "/tts",       label: "TTS Lab",       sub: "zero-shot voice"    },
  { href: "/agent",     label: "Agent Lab",     sub: "files + web"        },
];

export default function NavBar() {
  const pathname = usePathname() || "/";

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link href="/" className="navbar-brand">
          <span className="brand-dot" />
          <span>Jetson × NVIDIA Build</span>
        </Link>
        <div className="navbar-links">
          {LABS.map((lab) => {
            const active =
              lab.href === "/"
                ? pathname === "/"
                : pathname.startsWith(lab.href);
            return (
              <Link
                key={lab.href}
                href={lab.href}
                className={`navbar-link ${active ? "is-active" : ""}`}
              >
                <span className="navbar-link-label">{lab.label}</span>
                <span className="navbar-link-sub">{lab.sub}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
