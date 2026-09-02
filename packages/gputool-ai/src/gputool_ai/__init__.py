"""gputool-ai — the chat client and agent sidecar, packaged for pip.

Installable on a bare node without cloning edgeAI:

    pip install "gputool-ai[all] @ git+https://github.com/lkk688/edgeAI@main#subdirectory=packages/gputool-ai"

Then:

    gputool-chat --url http://localhost:8080/v1     # terminal chat
    gputool-agent                                   # ReAct sidecar on :8002

`gputool chat` and `gputool agent` use these when present and fall back to
downloading the single-file client, so nothing breaks on nodes that have not
installed the package.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
