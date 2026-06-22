"""edge_agent — a tiny ReAct agent loop + safe file tools.

The core of an AI agent in pure standard library:

    from edge_agent import ReActAgent, Tools

    agent = ReActAgent(complete, Tools(root="./project"))
    answer = agent.run("Read app.py and summarize it.")

See the tutorial: https://lkk688.github.io/edgeAI/curriculum/13_react_agent/
"""
from .react_loop import ReActAgent, REACT_SYSTEM
from .tool_calling import run_tool_calling
from .tools import OPENAI_SCHEMAS, TOOL_DOCS, TOOL_NAMES, Tools

__version__ = "0.1.0"
__all__ = [
    "Tools",
    "ReActAgent",
    "REACT_SYSTEM",
    "run_tool_calling",
    "TOOL_NAMES",
    "TOOL_DOCS",
    "OPENAI_SCHEMAS",
]
