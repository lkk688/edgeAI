# gputool-ai

Terminal chat client and ReAct agent sidecar for `gputool` and `sjsujetsontool`,
packaged so a node can install them **without cloning the edgeAI repo**.

## Install

```bash
pip install "gputool-ai[all] @ git+https://github.com/lkk688/edgeAI@main#subdirectory=packages/gputool-ai"
```

Extras: `rich` (nicer chat rendering), `agent` (FastAPI sidecar), `all` (both).
The base install is dependency-light on purpose — the chat client falls back to
a stdlib ANSI renderer, so it works on locked-down machines.

## Use

```bash
gputool-chat --url http://localhost:8080/v1   # talk to a local llama.cpp / vLLM server
gputool-agent                                 # ReAct sidecar on :8002
```

`gputool chat` and `gputool agent` prefer these entry points when installed.

## Paths

No absolute paths are baked in. Resolution order is always:

1. an environment variable — `GPUTOOL_WORKSPACE`, `GPUTOOL_ENV_FILE`, `GPUTOOL_DIR`
2. the current working directory — so `cd myproject && gputool-agent` acts on it
3. `~` — for the shared `~/.env.local` key file

The old `/Developer/edgeAI/...` lookups are gone; `AGENT_WORKSPACE` is still
honoured so existing sidecar deployments keep working.
