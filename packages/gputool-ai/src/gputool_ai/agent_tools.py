"""The tool set the agent actually gets: edge_agent's file tools, plus a terminal.

`edge_agent.Tools` stays untouched — a deployment that only wants file access
keeps exactly what it had. `build_tools()` is the one place that decides whether
a shell is attached, so there is a single switch to audit.
"""
from __future__ import annotations

import os

import edge_agent

from .terminal import (
    TERMINAL_TOOL_DOCS,
    TerminalMixin,
    make_backend,
    terminal_schema,
)


class ShellTools(TerminalMixin, edge_agent.Tools):
    """File tools + run_terminal, sharing one workspace root."""

    def __init__(self, root=".", backend_mode="host", image="", gpus=True):
        edge_agent.Tools.__init__(self, root=root)
        self._term_init(make_backend(backend_mode, self.root, image=image, gpus=gpus))

    def dispatch(self, name, args):
        # run_terminal is not in edge_agent's registry, so it has to be handled
        # before delegating or the base class rejects it as unknown.
        if name == "run_terminal":
            if not isinstance(args, dict):
                return "ERROR: arguments must be a JSON object."
            try:
                return str(self.run_terminal(**args))[:MAX_TOOL_CHARS]
            except TypeError as e:
                return "ERROR: bad arguments for run_terminal: %s" % e
            except Exception as e:
                return "ERROR: %s: %s" % (type(e).__name__, e)
        return edge_agent.Tools.dispatch(self, name, args)

    def cleanup(self):
        be = getattr(self, "_backend", None)
        if be is not None and hasattr(be, "cleanup"):
            be.cleanup()


MAX_TOOL_CHARS = 14000


def terminal_enabled(explicit: bool | None = None) -> bool:
    """Shell access is opt-in.

    Off unless asked for, because attaching a shell changes what the agent can
    do to the machine, and that should be a decision someone made rather than a
    default they inherited.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("GPUTOOL_AGENT_TERMINAL", "").lower() in ("1", "true", "yes", "on")


def backend_mode(explicit: str = "") -> str:
    return (explicit or os.environ.get("GPUTOOL_AGENT_BACKEND", "host")).lower()


def build_tools(root=".", terminal=None, mode="", image="", gpus=True):
    """Return (tools, docs, schemas) for the requested configuration."""
    if terminal_enabled(terminal):
        tools = ShellTools(root=root, backend_mode=backend_mode(mode), image=image, gpus=gpus)
        docs = edge_agent.tool_docs() + "\n" + TERMINAL_TOOL_DOCS
        schemas = list(edge_agent.openai_schemas()) + [terminal_schema()]
    else:
        tools = edge_agent.Tools(root=root)
        docs = edge_agent.tool_docs()
        schemas = list(edge_agent.openai_schemas())
    return tools, docs, schemas


def tool_names(terminal=None) -> list:
    names = list(edge_agent.tool_names())
    if terminal_enabled(terminal):
        names.append("run_terminal")
    return names
