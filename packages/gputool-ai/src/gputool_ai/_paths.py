"""Where things live, without hardcoding anyone's checkout.

The Jetson images happened to keep the repo at /Developer/edgeAI, and that path
leaked into the chat client and the agent sidecar. On a desktop, a lab node, or
a pip-installed copy there is no /Developer at all, so those lookups silently
did nothing and features quietly went missing.

Resolution order everywhere in this package:

  1. an explicit environment variable  (always wins)
  2. the current working directory     (the sane default: act on what you cd'd into)
  3. the user's home directory         (for the shared ~/.env.local key file)

Nothing here reaches for an absolute path outside the user's control.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that overrides the agent's working directory.
WORKSPACE_ENV = "GPUTOOL_WORKSPACE"
#: Legacy name still honoured so existing sidecar deployments keep working.
LEGACY_WORKSPACE_ENV = "AGENT_WORKSPACE"
#: Environment variable pointing at a specific .env file.
ENV_FILE_ENV = "GPUTOOL_ENV_FILE"


def workspace_root() -> Path:
    """Directory the agent is allowed to read and write.

    Defaults to the current working directory, so `cd myproject && gputool-agent`
    does the obvious thing.
    """
    for var in (WORKSPACE_ENV, LEGACY_WORKSPACE_ENV):
        val = os.environ.get(var)
        if val:
            return Path(val).expanduser().resolve()
    return Path.cwd().resolve()


def state_dir() -> Path:
    """Per-user directory for logs, PID files and the cached chat client."""
    d = Path(os.environ.get("GPUTOOL_DIR", Path.home() / ".gputool")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_file_candidates() -> list[Path]:
    """Files to read API keys from, most specific first.

    ~/.env.local is the shared one the chat client writes to; a project-local
    .env.local lets a checkout carry its own keys without touching home.
    """
    out: list[Path] = []
    explicit = os.environ.get(ENV_FILE_ENV)
    if explicit:
        out.append(Path(explicit).expanduser())
    out.append(Path.home() / ".env.local")
    out.append(Path.cwd() / ".env.local")
    # A repo checkout keeps the Next.js app's keys here; include it only if the
    # directory actually exists relative to where we are, never as an absolute path.
    nested = Path.cwd() / "edgeLLM" / "nextjs-nemotron-app" / ".env.local"
    out.append(nested)
    seen, uniq = set(), []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq
