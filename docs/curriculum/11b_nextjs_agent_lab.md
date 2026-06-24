# 🛠️ Putting an Agent Inside the Next.js App — the Agent Lab

**Author:** Dr. Kaikai Liu, Ph.D.
**Position:** Associate Professor, Computer Engineering
**Institution:** San Jose State University
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

> **Class goal.** Extend the five-lab Next.js app from
> [Lesson 11](./11_nextjs_nemotron_app.md) with a **sixth lab** — a
> **multi-round, file-and-web agent** that reads, greps, searches,
> writes, and edits files in a Jetson workspace, optionally backed by a
> SerpAPI web-search tool. The agent itself is the standalone
> [`edge_agent`](../../edgeLLM/edge_agent/) package from
> [Lesson 13](./13_react_agent.md); the new work here is everything
> *around* it that makes it a browser-visible lab.
>
> **Companion code:**
> [`edgeLLM/nextjs-nemotron-app/lib/providers.js`](../../edgeLLM/nextjs-nemotron-app/lib/providers.js) ·
> [`agent_sidecar/agent_sidecar.py`](../../edgeLLM/nextjs-nemotron-app/agent_sidecar/agent_sidecar.py) ·
> [`app/api/agent/route.js`](../../edgeLLM/nextjs-nemotron-app/app/api/agent/route.js) ·
> [`app/components/AgentLab.js`](../../edgeLLM/nextjs-nemotron-app/app/components/AgentLab.js) ·
> [`edgeLLM/edge_agent/`](../../edgeLLM/edge_agent/) (now v0.2.0).

---

## 1. 🗺️ What we are adding

A glance at the multi-lab nav after this lesson:

```
[Chat]  [Retrieval Lab]  [Omni Lab]  [ASR Lab]  [TTS Lab]  [Agent Lab ← new]
```

The Agent Lab page is a chat-style task box plus a **live trace pane**:
every Thought / Action / Observation that the model emits arrives over
a Server-Sent Event stream and is rendered as a separate card. When the
loop finishes you see the model's `Final Answer` and a metrics line.

The agent has six tools (when configured):

| Tool          | Stdlib? | When used                                          |
|---------------|---------|----------------------------------------------------|
| `read_file`   | ✅      | Read a slice of a file with line numbers           |
| `grep`        | ✅      | Substring or regex search across a tree            |
| `search_files`| ✅      | List files by glob pattern                         |
| `write_file`  | ✅      | Create or overwrite                                |
| `edit_file`   | ✅      | Find-and-replace one *unique* snippet              |
| `web_search`  | ⚙️       | Google via SerpAPI — auto-enabled if a key is set  |

The first five are pure standard library (the `edge_agent` package has
**zero runtime deps**). `web_search` is an *optional sixth tool* that
appears only when `SERPAPI_API_KEY` is in the environment.

---

## 2. 🤔 Why an agent in the Next.js app at all?

Lessons 11 and 12 demonstrated two important patterns:

- **Lesson 11** — a *chat lab*: a single model call per user turn, no
  side effects.
- **Lesson 12** — a *CLI triage agent*: a Python `for` loop that uses
  tools, run from a terminal.

The Agent Lab is the **browser version of the CLI agent**. The same
ReAct loop now drives a web UI where students can:

1. Type any task (*"Read calculator.py and summarize it"*,
   *"Find every TODO"*, *"Fix the typo"*).
2. **Watch each Thought / Action / Observation** scroll past in real
   time — the model's reasoning becomes visible, not hidden behind a
   `tool_calls` JSON.
3. Switch models from a dropdown — the route auto-resolves NVIDIA Build
   vs. OpenAI vs. Anthropic from the model id (see §4).
4. Optionally let the agent search the web (gated on a SerpAPI key —
   §6).

The pedagogical payoff: students get to *see* what an LLM "decides" to
do when it has six tools, a goal, and a budget of eight steps.

---

## 3. 🏗️ Architecture

Six moving parts; only the bottom two are new in this lesson, plus
`lib/providers.js`:

```
                       ┌─────────────────────────────────┐
                       │  NVIDIA Build / OpenAI / Anthropic │
                       │  (chat/completions endpoint)     │
                       └────────────────▲────────────────┘
                                        │  OpenAI-compatible JSON
            ┌───────────────────────────┴───────────────────────────┐
            │  Jetson Orin Nano                                     │
            │  ┌───────────────────────────────────────────────────┐│
            │  │  agent_sidecar.py (FastAPI)                       ││
            │  │     POST /run    → SSE  {start, step, observation,││
            │  │                          nudge, final, error}     ││
            │  │     GET  /health → {tools, web_search, root, …}   ││
            │  │     GET  /docs   → Swagger UI                     ││
            │  │  uses  edge_agent.ReActAgent                      ││
            │  │         + Tools(root=workspace)                   ││
            │  └─────────▲──────────────────────────────▲──────────┘│
            │            │ stream-forward                │ resolve   │
            │            │ (text/event-stream)           │ base_url  │
            │            │                               │ + apiKey  │
            │  ┌─────────┴───────────┐    ┌──────────────┴────────┐ │
            │  │  Next.js route       │    │ lib/providers.js     │ │
            │  │  app/api/agent       │◀───┤  (NVIDIA / OpenAI /  │ │
            │  │                       │    │   Anthropic resolver)│ │
            │  └─────────▲───────────┘    └─────────────▲────────┘ │
            │            │                                │         │
            │  ┌─────────┴───────────┐    reads ~/.env.local│       │
            │  │  AgentLab.js (UI)   │                       │       │
            │  │  • SSE parser       │                                │
            │  │  • event renderer   │                                │
            │  │  • model picker     │                                │
            │  └─────────────────────┘                                │
            └────────────────────────────────────────────────────────┘
```

Two architectural choices worth pausing on:

1. **Sidecar, not pure JS.** The agent's loop, regex parser, and file
   tools already exist as a tested Python package
   ([`edge_agent`](../../edgeLLM/edge_agent/)) used by
   `sjsujetsontool chat --agent`. Wrapping it in 250 lines of FastAPI
   reuses everything; a JS reimplementation would mean re-deriving the
   ReAct parser and the safe-file-path enforcement.
2. **Provider resolution on the Next.js side.** The route resolves the
   model id to a `(base_url, api_key)` pair *before* calling the
   sidecar, so the sidecar never has to know which provider it is
   talking to — it just makes one OpenAI-style call per step.

---

## 4. 🧩 The provider resolver — [`lib/providers.js`](../../edgeLLM/nextjs-nemotron-app/lib/providers.js)

The chat lab in Lesson 11 hard-coded `https://integrate.api.nvidia.com/v1`
and `NVIDIA_API_KEY`. With three labs that need to switch providers, we
factor that into a single helper:

```js
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// Lazy: merge ~/.env.local into process.env on first call.
let _homeEnvLoaded = false;
function loadEnv() {
  if (_homeEnvLoaded) return;
  _homeEnvLoaded = true;
  const file = path.join(os.homedir(), ".env.local");
  let text; try { text = fs.readFileSync(file, "utf8"); } catch { return; }
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")))
      value = value.slice(1, -1);
    if (key && !(key in process.env)) process.env[key] = value;
  }
}

const PROVIDERS = [
  { name: "NVIDIA Build", keyEnv: "NVIDIA_API_KEY",
    baseUrlDefault: "https://integrate.api.nvidia.com/v1", thinking: true,
    test: (id) => id.startsWith("nvidia/") || id.startsWith("qwen/")
                || id.startsWith("minimaxai/") || id.startsWith("z-ai/")
                || id.startsWith("meta/")     || id.startsWith("mistralai/")
                || id.startsWith("deepseek-ai/") },
  { name: "OpenAI",     keyEnv: "OPENAI_API_KEY",
    baseUrlDefault: "https://api.openai.com/v1", thinking: false,
    test: (id) => /^gpt-/i.test(id) || id.startsWith("o1") || id.startsWith("o3") },
  { name: "Anthropic",  keyEnv: "ANTHROPIC_API_KEY",
    baseUrlDefault: "https://api.anthropic.com/v1", thinking: false,
    test: (id) => id.startsWith("claude-") },
];

export function resolveProvider(modelId) {
  loadEnv();
  const p = PROVIDERS.find((p) => p.test(modelId)) || PROVIDERS[0];
  return {
    name: p.name,
    keyEnv: p.keyEnv,
    apiKey: process.env[p.keyEnv] || "",
    baseUrl: process.env[`${p.keyEnv.split("_")[0]}_BASE_URL`] || p.baseUrlDefault,
    thinking: p.thinking,
  };
}
```

Three things to internalise:

| Pattern | Why |
|---|---|
| **`~/.env.local` is merged on first call.** | Lets you set `NVIDIA_API_KEY` once (with `sjsujetsontool chat` or `setup-nvapi`) and have *every* lab pick it up — no per-app `.env`. |
| **The resolver returns `thinking`** | Only Nemotron / llama.cpp accept `chat_template_kwargs.enable_thinking`. Sending it to OpenAI returns 400. The chat route uses this flag to gate the field. |
| **First prefix wins** | New NVIDIA-Build model families (`qwen/`, `minimaxai/`, `z-ai/`, `deepseek-ai/`) are added to the NVIDIA prefix list, not as new providers. |

The same `resolveProvider()` is used by `/api/chat` (lesson 11) and
`/api/agent` (this lesson). Lessons 11d–11f could trivially add new
labs and reuse it.

---

## 5. 🔁 The `edge_agent` upgrade — now v0.2.0

[Lesson 13](./13_react_agent.md) introduced the standalone package
(`tools.py`, `react_loop.py`, `tool_calling.py`). For the Agent Lab we
made three small changes — they are all in
[`edge_agent/src/edge_agent/tools.py`](../../edgeLLM/edge_agent/src/edge_agent/tools.py):

### 5.1 An optional 6th tool: `web_search`

```python
def web_search(self, query, num=5):
    """Google web search via SerpAPI. Returns title / link / snippet bullets."""
    key = _serpapi_key()
    if not key:
        return ("ERROR: web_search is disabled (no SERPAPI_API_KEY in env). "
                "Use file tools instead, or ask the user to configure a key.")
    num = max(1, min(int(num or 5), 10))
    params = {"engine": "google", "q": str(query),
              "num": str(num), "api_key": key}
    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.load(resp)
    results = data.get("organic_results") or []
    out = []
    for r in results[:num]:
        out.append("- %s\n  %s\n  %s" % (
            r.get("title") or "",
            r.get("link") or "",
            (r.get("snippet") or "").replace("\n", " "),
        ))
    return "\n".join(out) or "(no results)"
```

Notes:

- **Pure `urllib`** — no `requests` dependency. The whole `edge_agent`
  package still has zero runtime requirements.
- **Graceful absence.** When the key is missing, the tool returns an
  explanatory error string. The model sees it as an `Observation` and
  *falls back to file tools* — no exceptions, no crashes.
- **`num` is clamped to 1–10** so a misbehaving model can't ask for
  10 000 results.

### 5.2 Env-reactive tool lists

The package now exposes both a *callable* and a *snapshot* for each
metadata constant, so a `SERPAPI_API_KEY` exported *after* import still
unlocks `web_search`:

```python
def tool_names() -> list[str]:
    names = list(FILE_TOOL_NAMES)              # always 5
    if web_search_available():                 # checks env at call time
        names.extend(WEB_TOOL_NAMES)
    return names

# Constant snapshot at import time for back-compat:
TOOL_NAMES = tool_names()
```

…with the same shape for `tool_docs()` and `openai_schemas()`. The
ReAct loop in `react_loop.py` was updated to re-resolve at run() time,
so a fresh key takes effect on the next agent call, not the next
restart.

### 5.3 `__init__` exports the helpers

```python
from .tools import (
    OPENAI_SCHEMAS, TOOL_DOCS, TOOL_NAMES, Tools,
    openai_schemas, tool_docs, tool_names, web_search_available,
)
__version__ = "0.2.0"
```

That's the whole upgrade. Existing code that imported the constants
keeps working; new code can call the functions when it needs an
env-reactive answer.

---

## 6. 🐍 The FastAPI sidecar

[`agent_sidecar/agent_sidecar.py`](../../edgeLLM/nextjs-nemotron-app/agent_sidecar/agent_sidecar.py)
is the longest new file at ~250 lines — but most of it is the SSE
emitter. The core endpoint is small:

```python
@app.post("/run")
async def run(request: Request) -> StreamingResponse:
    body = await request.json()
    task     = body["task"]
    root     = os.path.abspath(body.get("root") or DEFAULT_ROOT)
    api_key  = body["api_key"]
    base_url = body["base_url"]
    model    = body.get("model", "minimaxai/minimax-m2.7")
    max_steps = min(int(body.get("max_steps") or 8), MAX_STEPS_HARD)

    complete = _make_complete(base_url, api_key, model, body.get("temperature", 0.1))
    tools    = edge_agent.Tools(root=root)
    system   = edge_agent.REACT_SYSTEM.format(
        tools=edge_agent.tool_docs(), names=", ".join(edge_agent.tool_names()))
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": task}]

    def event_stream():
        yield _sse({"type": "start", "root": root, "model": model,
                    "tools": edge_agent.tool_names(), "max_steps": max_steps})
        for step in range(1, max_steps + 1):
            reply = complete(messages)
            messages.append({"role": "assistant", "content": reply})
            parsed = edge_agent.react_loop.parse_step(reply)

            if parsed and parsed[0] == "final":
                yield _sse({"type": "final", "n": step, "answer": parsed[1]})
                yield _sse_done(); return

            if not parsed:
                yield _sse({"type": "nudge", "n": step, "raw": reply})
                messages.append({"role": "user",
                    "content": "Observation: ERROR: no Action found. Reply with "
                               "either an Action + Action Input (JSON), or a "
                               "Final Answer."})
                continue

            _, name, args = parsed
            thought = next((ln.split(":", 1)[1].strip()
                            for ln in reply.splitlines()
                            if ln.strip().lower().startswith("thought:")), "")
            yield _sse({"type": "step", "n": step, "thought": thought,
                        "action": name, "input": args, "raw": reply})

            obs = tools.dispatch(name, args)
            yield _sse({"type": "observation", "n": step, "text": obs})
            messages.append({"role": "user", "content": "Observation: " + obs})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 6.1 The five SSE event shapes

```jsonc
data: {"type":"start",       "root":"…","model":"…","tools":[…],"max_steps":8}
data: {"type":"step",        "n":1,"thought":"…","action":"grep","input":{…},"raw":"…"}
data: {"type":"observation", "n":1,"text":"…"}
data: {"type":"final",       "n":4,"answer":"…","elapsed_ms":12345}
data: {"type":"nudge",       "n":3,"raw":"…"}       // model produced no Action
data: {"type":"error",       "message":"…"}
data: [DONE]
```

The `nudge` event is a real teaching moment: real models sometimes
reply with just a `Thought:` and no `Action:`. Our loop adds a fake
`Observation:` reminding the model of the protocol and continues. The
UI shows it as an amber card so students notice when the model goes
off-protocol.

### 6.2 Why a sync generator (not async)?

Riva-style streaming would have used `async def`. We don't: the OpenAI
client used inside `complete()` is *blocking*, and FastAPI's
`StreamingResponse` iterates a sync generator in a thread pool, which
is exactly the bridge we need. The event loop stays free while the
worker thread waits on the model.

### 6.3 Defaults you can override with env

| Var                  | Default | What it does                                 |
|----------------------|---------|----------------------------------------------|
| `AGENT_SIDECAR_PORT` | `8002`  | Port Uvicorn binds to                        |
| `AGENT_WORKSPACE`    | `./workspace` | Root the agent reads/edits             |
| `AGENT_MAX_STEPS`    | `12`    | Hard ceiling — clients may request fewer    |
| `NVIDIA_API_KEY`     | *(none)* | Default key when the request omits one      |
| `SERPAPI_API_KEY`    | *(none)* | Enables the optional `web_search` tool      |

---

## 7. 🌐 The Next.js route — [`app/api/agent/route.js`](../../edgeLLM/nextjs-nemotron-app/app/api/agent/route.js)

The route is intentionally **thin** — it does *not* implement any
agent logic. Two responsibilities:

```js
import { resolveProvider, envFromHome } from "@/lib/providers";

const SIDECAR_URL = process.env.AGENT_SIDECAR_URL || "http://localhost:8002";

export async function POST(req) {
  const body = await req.json();
  const provider = resolveProvider(body.model);          // ① pick provider
  if (!provider.apiKey) return Response.json(
    { error: `${provider.keyEnv} is not set.` }, { status: 500 });
  envFromHome();                                          // ② merge ~/.env.local

  const upstream = await fetch(`${SIDECAR_URL}/run`, {    // ③ forward
    method: "POST",
    headers: { "Content-Type": "application/json",
               Accept: "text/event-stream" },
    body: JSON.stringify({
      task:        body.task,
      root:        body.root,
      model:       body.model,
      base_url:    provider.baseUrl,
      api_key:     provider.apiKey,
      temperature: body.temperature ?? 0.1,
      max_steps:   body.max_steps || 8,
    }),
  });

  return new Response(upstream.body, {                    // ④ pipe SSE
    status: 200,
    headers: { "Content-Type": "text/event-stream; charset=utf-8",
               "Cache-Control": "no-cache, no-transform",
               Connection: "keep-alive" },
  });
}

export async function GET() {                              // for the UI's
  const r = await fetch(`${SIDECAR_URL}/health`);          // health badge
  return Response.json(await r.json());
}
```

`upstream.body` is a `ReadableStream`. Returning it as the response
body forwards every SSE chunk byte-for-byte to the browser — no
re-serialisation, no buffering.

---

## 8. 🎛️ The client — [`app/components/AgentLab.js`](../../edgeLLM/nextjs-nemotron-app/app/components/AgentLab.js)

The UI has three responsibilities, all React-standard:

```jsx
const CODING_MODELS = [
  { id: "minimaxai/minimax-m2.7",                       label: "MiniMax M2.7 (default)" },
  { id: "minimaxai/minimax-m3",                         label: "MiniMax M3" },
  { id: "z-ai/glm-5.1",                                 label: "Z-AI GLM 5.1" },
  { id: "nvidia/llama-3.3-nemotron-super-49b-v1.5",     label: "Nemotron Super 49B v1.5" },
  { id: "nvidia/llama-3.1-nemotron-70b-instruct",       label: "Nemotron Llama 3.1 70B" },
  { id: "deepseek-ai/deepseek-v4-pro",                  label: "DeepSeek v4 Pro" },
  { id: "mistralai/mistral-large-3-675b-instruct-2512", label: "Mistral Large 3 675B" },
  { id: "claude-sonnet-4-6",                            label: "Anthropic Claude Sonnet 4.6" },
  { id: "gpt-4o-mini",                                  label: "OpenAI GPT-4o mini" },
];

async function runAgent() {
  const res = await fetch("/api/agent", { method: "POST",
    body: JSON.stringify({ task, model, max_steps: maxSteps, root: root || undefined })});
  await readSSE(res, (evt) => {
    if (evt.type === "final") setFinal(evt);
    else                       setEvents((prev) => [...prev, evt]);
  });
}
```

…plus a one-time `GET /api/agent` on mount that fills the *"tools
available"* badge in the header — students can see at a glance whether
`web_search` is enabled, without having to read the docs.

Each event is rendered as a card with an icon (`📄 read_file`,
`🔎 grep`, `🌐 web_search`, …) so the trace is easy to skim during a
live demo.

---

## 9. ▶️ Run it on the Jetson

The Agent Lab needs two processes side-by-side:

- The **FastAPI agent backend** (Python) — this is the *server* that owns the
  ReAct loop and the file tools. The directory is called `agent_sidecar/`
  because the program is technically a "sidecar" in microservice
  terminology, but for *students* we call it the **agent backend** — that
  is the term the UI, the error messages, and the tutorials all use.
- The **Next.js dev server** (Node) — this is the browser-facing UI we
  built in Lesson 11; it proxies to the backend.

### 9.1 The one-step path: `sjsujetsontool agent` + `sjsujetsontool node`

Recent versions of `sjsujetsontool` include an **`agent`** subcommand that
mirrors the `node` subcommand from Lesson 11 §3.1 — it installs
dependencies into `~/.venv`, reads keys from `~/.env.local`, and starts the
FastAPI backend in the foreground or background.

```bash
ssh jetsonorin                            # any of the SSH paths from §5.5
sjsujetsontool agent bg                   # ← starts the FastAPI agent backend on :8002
sjsujetsontool node bg                    # ← starts the Next.js dev server on :3000
```

That is the whole startup. Two background processes, no `cd`, no `pip
install` to remember, no `python agent_sidecar.py` to copy-paste. Behind
the scenes:

| Step | What `sjsujetsontool agent` does |
|------|----------------------------------|
| 1 | Creates `~/.venv` (one-time) and `pip install`s `fastapi`, `uvicorn`, and editable `edge_agent` |
| 2 | `source`s `~/.env.local` and the project's `.env.local` so `NVIDIA_API_KEY`, `SERPAPI_API_KEY`, etc. are visible |
| 3 | Kills any previous backend (so port `:8002` is always free) |
| 4 | Runs `python agent_sidecar.py` — fg writes to your terminal, bg writes to `/tmp/sjsujetsontool-agent.log` |

To check on the backend and tear it down:

```bash
sjsujetsontool agent status   # → 🟢 up on :8002, with tools + workspace listed
sjsujetsontool agent stop     # → 🛑 Stopped the agent FastAPI backend.
```

After both are up, you have four useful URLs:

| URL                                   | What it is                              |
|---------------------------------------|-----------------------------------------|
| `http://<jetson>:3000/agent`          | The Agent Lab UI                        |
| `http://<jetson>:8002/docs`           | FastAPI Swagger UI (try-it-out for the route) |
| `http://<jetson>:8002/health`         | JSON status (also proxied at `/api/agent` GET) |
| `http://<jetson>:8002/openapi.json`   | Machine-readable schema, for typed clients |

If you are not on the same LAN as the Jetson, use the SSH tunnel pattern
from [Lesson 11 §5.5](./11_nextjs_nemotron_app.md#55-open-it-from-your-laptop--over-ssh-off-lan)
— forward **both** ports (`-L 3000:localhost:3000 -L 8002:localhost:8002`).

### 9.2 Manual install (what `sjsujetsontool agent` does for you)

If you want to know what is actually happening — or you need to install
the backend on a fresh box that does not have `sjsujetsontool` yet — here
is the same install done by hand:

```bash
ssh jetsonorin
python3 -m venv ~/.venv && source ~/.venv/bin/activate
pip install -r /Developer/edgeAI/edgeLLM/nextjs-nemotron-app/agent_sidecar/requirements.txt
pip install -e /Developer/edgeAI/edgeLLM/edge_agent

# Pull the keys the chat lab already saved (same convention as Lesson 11 §5).
set -a
[ -f ~/.env.local ] && source ~/.env.local
[ -f /Developer/edgeAI/edgeLLM/nextjs-nemotron-app/.env.local ] && \
   source /Developer/edgeAI/edgeLLM/nextjs-nemotron-app/.env.local
set +a

cd /Developer/edgeAI/edgeLLM/nextjs-nemotron-app/agent_sidecar
python agent_sidecar.py
# → INFO  starting edge-agent sidecar on 0.0.0.0:8002 — docs at /docs
#         (workspace=/Developer/.../agent_sidecar/workspace, web_search=False)
```

For the Next.js side, see [Lesson 11 §3.1 / §5](./11_nextjs_nemotron_app.md#31-tooling--sjsujetsontool-node-the-one-step-path).

### 9.3 The "sidecar" terminology 

The directory is named `agent_sidecar/` because in microservice
architecture a *sidecar* is a helper process that runs alongside a main
application, sharing its lifecycle and network namespace — which is
exactly what this Python program does relative to the Next.js server. 

---

## 10. 🧪 Verified run on `sjsujetson` (Ubuntu 24.04 aarch64)

Here is the *actual* trace from the test we ran while writing this
lesson. Model: `minimaxai/minimax-m2.7`. Workspace:
`~/nextjs-nemotron-app/agent_sidecar/workspace/` (ships with a
deliberately typo'd `calculator.py`). Task:

> *"Read calculator.py, find the typo where `doubel` should be
> `double`, fix it using `edit_file`, then read the file again to
> confirm."*

Result (summarised from the SSE stream):

```
event types observed:
       1 "type": "final"
       1 "type": "nudge"
       5 "type": "observation"
       1 "type": "start"
       5 "type": "step"

step actions in order:
  1.  read_file
  2.  edit_file        ← `def doubel` → `def double`
  3.  read_file
  4.  edit_file        ← `doubel(7)`  → `double(7)` (call site!)
  5.  read_file

final answer:
  "Fixed the typo in calculator.py. Changed `def doubel(x: float)` to
   `def double(x: float)` on line 27, and also updated the function
   call `doubel(7)` to `double(7)` on line 34. The file now correctly
   uses `double`."
```

And the on-disk verification, *after* the agent run:

```bash
$ grep -n 'doubel\|def double' calculator.py
4:`def doubel(x): ...` should be `double`. Ask the agent to fix it.
27:def double(x: float) -> float:
```

(Line 4 is the docstring that *describes* the typo and was left alone
— exactly what we wanted. Lines 27 and 34 are the real code, both
correct now.)

The `nudge` event in step 4 is real: the model paused mid-loop to write
just a Thought without an Action. The sidecar's fallback prompt
recovered it, and step 5 ran cleanly. **This is the moment to point
out in class** — students learn that production agents need a
"protocol-violation recovery" path, and they can see it work.

End-to-end wall-clock: **~85 s** for the whole 5-step run on the
Jetson over the public NVIDIA Build endpoint.

### 10.2 Same lab, **`node05` backend** (no cloud key required)

We re-ran the lab on `jetsonorin` with the **`Shared SJSU llama.cpp (node05)`**
backend selected from the dropdown — same workspace, simpler task,
**zero cloud quota used**:

> *"List the files in the workspace using `search_files`, then tell me
> what types of files you found."*

```
event types observed:
       1 "type": "start"
       1 "type": "step"        ← Qwen3.5-9B emitted a clean Action line
       1 "type": "observation"
       1 "type": "final"

step actions in order:
  1.  search_files

final answer (verbatim):
  "I found 3 files in the workspace:
   1. **README.md** - Markdown file (documentation)
   2. **calculator.py** - Python source code file
   3. **notes.md** - Markdown file (documentation/notes)
   Summary by type:
   - **Markdown files (.md)**: 2 files (README.md, notes.md)
   - **Python files (.py)**: 1 file (calculator.py)"
```

One step, one observation, one final answer — no `nudge`. **`Qwen3.5-9B`
at Q6_K_XL is large enough to follow the ReAct text protocol reliably**;
the 0.8 B variant we tried earlier was not. Both run on the same shared
server, so changing the model is a one-line edit in
[`AgentLab.js`'s `node05` entry](../../edgeLLM/nextjs-nemotron-app/app/components/AgentLab.js).

The `node05` backend is what we recommend for **classroom demos where you
don't want students to burn cloud quota**: it's free, on-prem, and the
model is good enough.

---

## 11. ⚖️ Model selection — and what you'll hit in practice

The single biggest gotcha while writing this lesson:

> **`qwen/qwen3-coder-480b-a35b-instruct` reached end-of-life on
> 2026-06-11.** The agent fails fast with `HTTP 410 Gone` and a JSON
> `detail` field naming the EOL date.

The earlier vuln-triage tutorials in [Lesson 12](./12_vulnerability_triage_intro.md)
used that model as their default. **Use `minimaxai/minimax-m2.7` or
`z-ai/glm-5.1` instead.** Both are currently available on NVIDIA
Build's free tier and both reliably emit OpenAI-format tool calls.

### 11.1 Coding-capable models we tested or saw

The Agent Lab now offers **five named backends + one "Custom"** in the dropdown
(same menu the `sjsujetsontool chat` CLI uses):

| Backend (`backend:` field)  | URL                                              | Default model                       | Key needed?    |
|-----------------------------|--------------------------------------------------|-------------------------------------|----------------|
| **NVIDIA Build** (`nvidia`) | `https://integrate.api.nvidia.com/v1`            | `minimaxai/minimax-m2.7`            | `NVIDIA_API_KEY` |
| **Local llama.cpp** (`llama`) | `http://localhost:8080/v1`                     | whatever `sjsujetsontool llama bg` started | no           |
| **Shared SJSU `node05`** (`node05`) | `https://llm.forgengi.org/node05/v1`     | `Qwen3.5-9B-UD-Q6_K_XL.gguf`       | no            |
| **OpenAI** (`openai`)       | `https://api.openai.com/v1`                      | `gpt-4o-mini`                       | `OPENAI_API_KEY` |
| **Anthropic** (`anthropic`) | `https://api.anthropic.com/v1`                   | `claude-sonnet-4-6`                 | `ANTHROPIC_API_KEY` |
| **Custom** (`custom`)       | user enters in the UI                            | user enters                         | optional       |

The **`node05` row is the same shared llama.cpp server** the
`sjsujetsontool chat` terminal client uses for its *"Our shared LLM server"*
option. No API key needed — it is reachable from any Jetson on the
Headscale network at `https://llm.forgengi.org/node05/v1`. Currently
serves `Qwen3.5-9B-UD-Q6_K_XL.gguf` (9 B parameters, Q6_K_XL quant), which
is capable enough to follow the ReAct text protocol — verified in §10.2
below.

Within the NVIDIA Build backend, the coding-capable models we tested or
saw:

| Model id                                         | Status (2026-06)       | Notes |
|--------------------------------------------------|------------------------|-------|
| `minimaxai/minimax-m2.7`                         | ✅ default for §10 demo | 5 steps, ~85 s wall, recovered from one off-protocol reply |
| `minimaxai/minimax-m3`                           | ✅ available             | Newer release; haven't benchmarked agent-side |
| `z-ai/glm-5.1`                                   | ✅ available             | Strong on tool-calling; good alternative if MiniMax is slow |
| `qwen/qwen3-next-80b-a3b-instruct`               | ✅ available             | Newer Qwen line — the *next* generation after the EOL'd coder |
| `qwen/qwen3.5-122b-a10b`                         | ✅ available             | Slow but capable |
| `qwen/qwen3.5-397b-a17b`                         | ✅ available, **slow**    | Plan for 30 s+ per step |
| `qwen/qwen3-coder-480b-a35b-instruct`            | ❌ **EOL 2026-06-11**     | Returns HTTP 410 Gone |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5`       | ✅ available             | Honest fallback; less aggressive at calling tools |
| `nvidia/llama-3.1-nemotron-70b-instruct`         | ✅ available             | Solid generalist |
| `deepseek-ai/deepseek-v4-pro`                    | ✅ available             | Strong code reasoning |
| `mistralai/mistral-large-3-675b-instruct-2512`   | ✅ available, **slow**    | Very large MoE — wall time dominates |
| `claude-sonnet-4-6` (via `ANTHROPIC_API_KEY`)    | ✅ if key set            | Tool-calling is rock-solid; pricier per token |
| `gpt-4o-mini` (via `OPENAI_API_KEY`)             | ✅ if key set            | Cheapest and fastest of the OpenAI line |

### 11.2 Things you *will* hit in class

- **EOL during a semester.** NVIDIA periodically retires models. The
  symptom is **HTTP 410** with a `detail` mentioning the EOL date.
  Update [`AgentLab.js`'s `CODING_MODELS`](../../edgeLLM/nextjs-nemotron-app/app/components/AgentLab.js)
  and the `AGENT_MODEL` default in your `.env.local`. *No backend code
  change is needed.*
- **HTTP 429 rate limits.** Free-tier quotas are modest. Agents that
  call the model 5–8 times per task hit the limit fastest. Pause the
  lab for a minute, or move to a different free-tier model.
- **HTTP 503 / cold-start.** First call to a less-popular model on a
  fresh day can take 60 s+ while the backend warms up. Subsequent
  calls are normal.
- **Token-per-second variance is huge.** A 9-billion-parameter model
  vs. a 675-billion MoE is a 5–10× wall-time difference at the same
  step count. For class demos, prefer **MiniMax M2.7** or
  **GLM-5.1** — both finish a 5-step agent run in well under two
  minutes.
- **Anthropic + OpenAI work via OpenAI-compatible endpoints.** As long
  as the relevant key is in `~/.env.local`, the provider resolver
  picks them up — no extra route work required. They both reliably
  follow the ReAct text protocol, but cost more per step.

### 11.3 The default we ship with

`minimaxai/minimax-m2.7` — it's free, currently available, fast enough
for class (a typical 5-step run finishes in under 90 s end-to-end),
and reliably emits well-formed `Action:` lines. Override per-request
from the UI dropdown, or globally via:

```bash
# .env.local (in either ~/.env.local or this app's .env.local)
AGENT_MODEL=z-ai/glm-5.1
```

---

## 12. 🧪 Things to try in class

1. **Fix the typo, then read the file.** This is the demo above. The
   "did it actually change the file?" moment is the punchline.
2. **Add a function.** *"Add a `power(base, exp)` function to
   calculator.py and a one-line docstring."* Watch the agent
   `read_file` first to learn the file style, then `write_file` (or
   `edit_file`) to insert.
3. **Find every TODO.** *"Find every TODO in the project."* This is
   pure `grep` — one tool call. Counterexample for how cheap simple
   tasks can be.
4. **Set `SERPAPI_API_KEY` and ask for a web fact.** *"Find the latest
   release version of LangChain and write a one-paragraph summary into
   webnote.md."* The agent now uses **two** tools end-to-end:
   `web_search` to fetch the fact, `write_file` to materialise it.
   Verify by reading `webnote.md` after the run.
5. **Try a slow model.** Switch to
   `mistralai/mistral-large-3-675b-instruct-2512`. The trace becomes a
   *much* better visualisation aid because each step takes 30+ seconds
   — students can see Thought / Action / Observation appearing one by
   one rather than as a fast blur.
6. **Break the protocol on purpose.** Edit `REACT_SYSTEM` in
   `edge_agent/react_loop.py` to remove the explicit examples. Re-run
   the demo. You will see *many more* `nudge` events. Restore the
   examples and the rate drops back to near zero.

---

## 13. 🔒 Safety notes for the Agent Lab

The Agent Lab is **more dangerous** than the other labs in this app
because it writes to disk. Three guardrails are already in place; one
more is worth doing in production:

| Guardrail                                         | Where it lives |
|---------------------------------------------------|----------------|
| Every path is resolved under a *root* directory   | [`Tools._resolve`](../../edgeLLM/edge_agent/src/edge_agent/tools.py) — refuses to leave the root |
| Hard cap on agent steps                           | `AGENT_MAX_STEPS` env var on the sidecar |
| `edit_file` refuses non-unique `old` snippets     | [`Tools.edit_file`](../../edgeLLM/edge_agent/src/edge_agent/tools.py) — prevents clobbering |
| **TODO if exposing publicly**: require auth on `/api/agent` | …because anyone who reaches it can call `write_file` inside your workspace. |

The default `AGENT_WORKSPACE` is a sample directory shipped *with the
app*. Don't point it at `~/` or `/etc` without thinking. The
file-path enforcement is robust against `../../` traversal, but it
will still let the agent delete `foo.py` if you set the root to a
folder you care about.

---

## 14. ➡️ Where to go next

- 🤝 **Add agent ↔ chat.** Add a button on the Chat page (lesson 11)
  that pipes the last assistant turn into `/api/agent` as a task. The
  agent then has the conversation as *context*, not just a single
  task. Now the chat itself is "agentic."
- 🧠 **Save traces.** Append every `{task, events, final}` to a JSONL
  file. Three hundred runs later you have an *agent dataset* — useful
  for fine-tuning a small local Nemotron Nano on the way agents are
  *supposed* to call your tools.
- 🛠️ **More tools.** Add `run_python_snippet(code)`, `git_status()`,
  `npm_test()`, etc. Each is ~20 lines in `tools.py`, one OpenAI
  schema entry, and one icon in `AgentLab.js`. The whole `edge_agent`
  package is designed to absorb tools you write.
- 🧪 **Vuln triage as an agent.** Re-run the
  [Lesson 12](./12_vulnerability_triage_intro.md) triage prompts inside
  this lab instead of from the CLI. The file tools + `web_search`
  cover almost everything `triage_basic.py` does — students can see
  the difference in *one* UI.

---

**Source folders:** [`edgeLLM/edge_agent/`](../../edgeLLM/edge_agent/),
[`edgeLLM/nextjs-nemotron-app/agent_sidecar/`](../../edgeLLM/nextjs-nemotron-app/agent_sidecar/),
[`edgeLLM/nextjs-nemotron-app/lib/`](../../edgeLLM/nextjs-nemotron-app/lib/),
[`edgeLLM/nextjs-nemotron-app/app/agent/`](../../edgeLLM/nextjs-nemotron-app/app/agent/),
[`edgeLLM/nextjs-nemotron-app/app/api/agent/`](../../edgeLLM/nextjs-nemotron-app/app/api/agent/),
[`edgeLLM/nextjs-nemotron-app/app/components/AgentLab.js`](../../edgeLLM/nextjs-nemotron-app/app/components/AgentLab.js).

**Tested on:** Ubuntu 24.04 LTS aarch64 (Jetson reached via
`ssh sjsujetson@headscale.forgengi.org -p 20065`), Python 3.12,
Node v20.20.2, Next.js 15.5.18, FastAPI 0.138, edge_agent 0.2.0,
and the NVIDIA Build chat endpoint with
**`minimaxai/minimax-m2.7`**.
