"""Tool: tiny grep-style code search the LLM agents can use to inspect
the target codebase.

We intentionally do not call out to `ripgrep` so the tutorial has zero
non-Python dependencies.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# Files larger than this are skipped — keeps the agent from being shown
# 10 MB of vendored code.
MAX_FILE_BYTES = 200_000

# Filename suffixes we treat as "source we care about" for triage. Tweak
# for non-Python projects.
SCANNED_SUFFIXES = (".py", ".pyi", ".toml", ".cfg", ".ini", ".txt", ".md")


def _safe_join(project_dir: str, rel: str) -> Path:
    """Resolve `rel` under `project_dir`, refusing path traversal."""
    base = Path(project_dir).resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise PermissionError(
            f"refusing to read {rel!r} — outside project_dir {project_dir}"
        ) from exc
    return target


def _iter_source_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SCANNED_SUFFIXES:
            continue
        # Skip virtual envs and dot-dirs by convention.
        parts = set(path.parts)
        if {".venv", "venv", "__pycache__", ".git", ".cve_cache"} & parts:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def search_usage(
    pattern: str,
    project_dir: str,
    *,
    is_regex: bool = False,
    max_hits: int = 30,
) -> dict[str, Any]:
    """Return up to `max_hits` matches of `pattern` across `project_dir`.

    `pattern` is a literal substring by default; set `is_regex=True` to
    treat it as a Python regular expression.
    """
    root = Path(project_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{project_dir} is not a directory")

    if is_regex:
        rx = re.compile(pattern)
        match_fn = rx.search
    else:
        needle = pattern
        match_fn = lambda line: needle in line  # noqa: E731

    hits: list[dict[str, Any]] = []
    for path in _iter_source_files(root):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if match_fn(line):
                hits.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": lineno,
                        "snippet": line.strip()[:200],
                    }
                )
                if len(hits) >= max_hits:
                    return {
                        "pattern": pattern,
                        "hits": hits,
                        "truncated": True,
                    }
    return {"pattern": pattern, "hits": hits, "truncated": False}


def read_file(
    path: str,
    project_dir: str,
    *,
    start: int = 1,
    end: int | None = None,
) -> dict[str, Any]:
    """Read a slice of a file inside `project_dir`."""
    target = _safe_join(project_dir, path)
    if not target.is_file():
        raise FileNotFoundError(f"{path} is not a file under {project_dir}")
    text = target.read_text(errors="replace")
    lines = text.splitlines()
    start = max(1, start)
    end = len(lines) if end is None else min(end, len(lines))
    excerpt = "\n".join(
        f"{i:5}: {ln}" for i, ln in enumerate(lines[start - 1 : end], start=start)
    )
    return {
        "path": str(target.relative_to(Path(project_dir).resolve())),
        "lines": (start, end),
        "total_lines": len(lines),
        "content": excerpt,
    }


if __name__ == "__main__":
    import json
    import sys

    proj = sys.argv[1] if len(sys.argv) > 1 else "./sample_project"
    pat = sys.argv[2] if len(sys.argv) > 2 else "requests"
    print(json.dumps(search_usage(pat, proj), indent=2))
