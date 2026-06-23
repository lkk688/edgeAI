"""agent_tools.py — a tiny, safe file-tool kit for LLM agents.

Five file tools an agent can call to explore and edit a codebase — the same verbs
a human coder uses:

    read_file(path, start, end)      read a slice of a file (with line numbers)
    grep(pattern, path, is_regex)    search file/dir contents for a string/regex
    search_files(glob, dir)          find files by name pattern
    write_file(path, content)        create or overwrite a file
    edit_file(path, old, new)        replace an exact, unique snippet (find/replace)

…plus one *optional* online tool that is only enabled when a key is configured:

    web_search(query, num)           SerpAPI Google search (needs SERPAPI_API_KEY)

Every path is confined to a `root` directory, so an agent cannot wander outside
the project it was pointed at. The file tools are pure standard library; the
web_search tool uses urllib so there is no third-party dependency either.

Used by:
  - react_loop.py     the ReAct text-protocol loop (the agent foundation)
  - tool_calling.py   the OpenAI native tool-calling loop
  - chat.py           `sjsujetsontool chat --agent` / the in-chat `/agent on`
  - agent_sidecar/    the Next.js Agent Lab streams the trace from this loop
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import urllib.parse
import urllib.request

FILE_TOOL_NAMES = ["read_file", "grep", "search_files", "write_file", "edit_file"]
WEB_TOOL_NAMES = ["web_search"]


def web_search_available() -> bool:
    """True if the SerpAPI key is set in the environment."""
    return bool(os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY"))


def _serpapi_key() -> str:
    return os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY") or ""


def tool_names() -> list[str]:
    """List of currently-available tool names. Adds `web_search` when configured."""
    names = list(FILE_TOOL_NAMES)
    if web_search_available():
        names.extend(WEB_TOOL_NAMES)
    return names


# Snapshot at import time, for back-compat with code that imports the constant.
# Callers that need to react to runtime env changes should use `tool_names()`.
TOOL_NAMES = tool_names()


class Tools:
    """File tools bound to (and confined to) a single project root."""

    _SKIP = {".git", "node_modules", "__pycache__", ".next", ".venv", "dist", "build"}

    def __init__(self, root="."):
        self.root = os.path.abspath(root)

    # ----------------------------------------------------------------- safety
    def _resolve(self, path):
        p = os.path.abspath(os.path.join(self.root, path))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError("path escapes the project root: %s" % path)
        return p

    def _walk(self, path):
        base = self._resolve(path)
        if os.path.isfile(base):
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in self._SKIP]
            for fn in filenames:
                yield os.path.join(dirpath, fn)

    # ------------------------------------------------------------------ tools
    def read_file(self, path, start=1, end=None):
        """Return lines [start, end] of a file, each prefixed with its number."""
        with open(self._resolve(path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(1, int(start))
        end = min(int(end) if end else len(lines), len(lines))
        body = ["%5d  %s" % (i, lines[i - 1].rstrip("\n")) for i in range(start, end + 1)]
        return "\n".join(body) or "(empty range)"

    def grep(self, pattern, path=".", is_regex=False, max_hits=40):
        """Search file (or every file under a dir) for `pattern`. Returns file:line: text."""
        rx = re.compile(pattern if is_regex else re.escape(pattern))
        hits = []
        for fp in self._walk(path):
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for n, line in enumerate(f, 1):
                        if rx.search(line):
                            rel = os.path.relpath(fp, self.root)
                            hits.append("%s:%d: %s" % (rel, n, line.strip()[:200]))
                            if len(hits) >= max_hits:
                                return "\n".join(hits)
            except (OSError, UnicodeError):
                continue
        return "\n".join(hits) or "(no matches)"

    def search_files(self, glob="*", dir="."):
        """Find files whose name matches a glob (e.g. '*.py', 'route.js')."""
        out = [os.path.relpath(fp, self.root) for fp in self._walk(dir)
               if fnmatch.fnmatch(os.path.basename(fp), glob)]
        return "\n".join(sorted(out)[:200]) or "(no files)"

    def write_file(self, path, content):
        """Create or overwrite a file with `content`."""
        p = self._resolve(path)
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return "wrote %d bytes to %s" % (len(content), path)

    def edit_file(self, path, old, new):
        """Replace one exact, unique occurrence of `old` with `new` (find/replace)."""
        p = self._resolve(path)
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        count = text.count(old)
        if count == 0:
            return "ERROR: `old` text not found — read_file first and copy an exact snippet."
        if count > 1:
            return "ERROR: `old` matches %d places — add surrounding context to make it unique." % count
        with open(p, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
        return "edited %s (1 replacement)" % path

    # ------------------------------------------------------------- web_search
    def web_search(self, query, num=5):
        """Google web search via SerpAPI. Returns a compact list of hits.

        Disabled (returns an error string) when SERPAPI_API_KEY is not set —
        the agent should fall back to file tools.
        """
        key = _serpapi_key()
        if not key:
            return ("ERROR: web_search is disabled (no SERPAPI_API_KEY in env). "
                    "Use file tools instead, or ask the user to configure a key.")
        num = max(1, min(int(num or 5), 10))
        params = {
            "engine": "google",
            "q": str(query),
            "num": str(num),
            "api_key": key,
        }
        url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.load(resp)
        except Exception as exc:
            return "ERROR: web_search failed: %s" % exc
        if data.get("error"):
            return "ERROR: SerpAPI: %s" % data["error"]
        results = data.get("organic_results") or []
        out = []
        for r in results[:num]:
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or "").strip().replace("\n", " ")
            out.append("- %s\n  %s\n  %s" % (title, link, snippet))
        if not out:
            ab = (data.get("answer_box") or {}).get("snippet") or ""
            if ab:
                return "answer_box: " + ab.strip()
            return "(no results)"
        return "\n".join(out)

    # --------------------------------------------------------------- dispatch
    def dispatch(self, name, args):
        """Run a tool by name with a dict of args; always returns a string."""
        names = tool_names()
        if name not in names:
            return "ERROR: unknown tool %r. Available: %s" % (name, ", ".join(names))
        if not isinstance(args, dict):
            return "ERROR: arguments must be a JSON object."
        try:
            return str(getattr(self, name)(**args))[:6000]
        except TypeError as e:
            return "ERROR: bad arguments for %s: %s" % (name, e)
        except FileNotFoundError as e:
            return "ERROR: no such file: %s" % e
        except Exception as e:  # surfaced to the model so it can recover
            return "ERROR: %s: %s" % (type(e).__name__, e)


_FILE_TOOL_DOCS = """\
- read_file(path, start=1, end=None) — read a slice of a file with line numbers.
- grep(pattern, path=".", is_regex=false) — search contents; returns file:line: text.
- search_files(glob="*", dir=".") — list files whose name matches a glob.
- write_file(path, content) — create or overwrite a file.
- edit_file(path, old, new) — replace ONE exact, unique snippet (read_file first)."""

_WEB_TOOL_DOCS = """\
- web_search(query, num=5) — Google search via SerpAPI; returns title/link/snippet bullets."""


def tool_docs() -> str:
    """Tool reference to inject into the ReAct system prompt — adapts to env."""
    parts = [_FILE_TOOL_DOCS]
    if web_search_available():
        parts.append(_WEB_TOOL_DOCS)
    return "\n".join(parts)


# Snapshot — see `tool_docs()` for the env-reactive form.
TOOL_DOCS = tool_docs()


_FILE_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a slice of a text file from the project, with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "path relative to the project root"},
                    "start": {"type": "integer", "default": 1},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search a file or directory for a substring (or regex). Returns up to 40 file:line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "is_regex": {"type": "boolean", "default": False},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "List files in the project whose name matches a glob like '*.py'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "default": "*"},
                    "dir": {"type": "string", "default": "."},
                },
                "required": ["glob"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact, unique snippet in a file (find & replace). Read the file first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "exact text to find (must be unique)"},
                    "new": {"type": "string", "description": "replacement text"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]

_WEB_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Google web search via SerpAPI. Returns up to `num` "
                "title / link / snippet bullets. Use ONLY when the answer "
                "cannot come from the project's own files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


def openai_schemas() -> list[dict]:
    """Tool schemas adapted to env — adds web_search when SERPAPI is configured."""
    return list(_FILE_SCHEMAS) + (list(_WEB_SCHEMAS) if web_search_available() else [])


# Snapshot — see `openai_schemas()` for the env-reactive form.
OPENAI_SCHEMAS = openai_schemas()
