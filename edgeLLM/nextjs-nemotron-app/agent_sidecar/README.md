# Edge-Agent sidecar (FastAPI + edge_agent)

A small FastAPI service that streams the
[`edge_agent`](../../edge_agent/) ReAct loop over Server-Sent Events so the
Next.js Agent Lab (`/agent` page) can display each Thought / Action /
Observation as it happens.

## Endpoints

| Method | Path        | Notes                                                |
|--------|-------------|------------------------------------------------------|
| GET    | `/health`   | Returns `{ok, tools, web_search, default_root}`      |
| POST   | `/run`      | Body JSON; streams `text/event-stream` of agent steps |
| GET    | `/docs`     | Auto-generated Swagger UI                            |

## Event shapes (per SSE line)

```jsonc
data: {"type":"start",       "root":"…","model":"…","tools":[…],"max_steps":8}
data: {"type":"step",        "n":1,"thought":"…","action":"grep","input":{…},"raw":"…"}
data: {"type":"observation", "n":1,"text":"…"}
data: {"type":"final",       "n":4,"answer":"…","elapsed_ms":12345}
data: {"type":"nudge",       "n":3,"raw":"…"}            // model produced no Action
data: {"type":"error",       "message":"…"}
data: [DONE]
```

## Why a sidecar?

The Next.js `/api/agent` route could implement the same loop in JS — but
the agent's tool surface (read/grep/search/write/edit files, plus
SerpAPI search) is already a small, well-tested Python package
(`edge_agent`). Reusing it server-side keeps the JS surface trivial
(forward an SSE stream) and lets the same agent code power both
`sjsujetsontool chat --agent` and the web Lab.

## Run on the Jetson

```bash
ssh sjsujetson@headscale.forgengi.org -p 20065
# (one-time) install dependencies into a venv:
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -r ~/nextjs-nemotron-app/agent_sidecar/requirements.txt
pip install -e ~/edge_agent

export NVIDIA_API_KEY=nvapi-...
export SERPAPI_API_KEY=...           # optional — enables web_search

cd ~/nextjs-nemotron-app/agent_sidecar
python agent_sidecar.py
# → INFO  starting edge-agent sidecar on 0.0.0.0:8002 — docs at /docs
```

Open <http://localhost:8002/docs> for the live Swagger UI. The
*Try it out* button on `/run` lets you fire one ReAct loop without going
through Next.js.

## Configuration (env vars)

| Variable              | Default                              | Meaning |
|-----------------------|--------------------------------------|---------|
| `AGENT_SIDECAR_PORT`  | `8002`                               | Port Uvicorn binds to |
| `AGENT_WORKSPACE`     | `./workspace`                        | Project dir the agent reads/edits |
| `AGENT_MAX_STEPS`     | `12`                                 | Hard cap; clients may request fewer |
| `NVIDIA_API_KEY`      | *(none)*                             | Default key when the request omits one |
| `SERPAPI_API_KEY`     | *(none)*                             | Enables the optional `web_search` tool |
