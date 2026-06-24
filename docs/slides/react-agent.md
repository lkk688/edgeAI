---
marp: true
theme: sjsu
paginate: true
size: 16:9
title: The ReAct Loop — Core of AI Agents
---

<!-- _class: lead -->
# 🤖 The ReAct Loop
### The core of every AI agent

`SJSU · Edge AI`

<span class="tiny">Reason + Act: an LLM in a loop with tools — run it as <code>sjsujetsontool chat --agent</code>.</span>

---

## <span class="step">1</span> Chatbot vs. agent

<div class="cols">
<div>

**Chatbot** — one prompt → one answer.
Knows only its weights + your prompt.

**Agent** — a chatbot put in a **loop** with **tools**.
Given a goal, it repeatedly decides *what to do next*, acts, observes, and continues.

</div>
<div>

Two ingredients turn a model into an agent:

1. **Tools** — functions it may call (read / grep / search / write / edit).
2. **A control loop** — code that runs the tool and feeds the result back.

> That loop is the whole idea. RAG, coding assistants, multi-agent systems — all variations on it.

</div>
</div>

---

## <span class="step">2</span> ReAct = Reason + Act

The model interleaves reasoning and actions in a strict text protocol — **one step per turn**:

```text
Thought: I should look at app.py first
Action: read_file
Action Input: {"path": "app.py", "end": 40}
```

Our code runs the tool and appends the result; the model reads it and continues:

```text
Observation: 1  import requests  ...
Thought: I now know the answer
Final Answer: app.py calls requests.get(url) with a caller-supplied URL.
```

> Plain text → works on **any** chat model (even local base models) and the reasoning is visible to debug.

---

## <span class="step">3</span> Architecture — **one core, multiple interfaces**

We don't ship "an agent" — we ship a tiny **`edge_agent`** Python package, and *wrap it* in
different transports. **Same loop, same tools, different surface.**

```text
                  ┌──────────────────────────────────────────────────┐
                  │                edgeLLM/edge_agent/               │
                  │   ┌──────────┐  ┌──────────────┐  ┌────────────┐ │
                  │   │ tools.py │  │ react_loop.py│  │tool_calling│ │
                  │   │ Tools()  │  │ ReActAgent   │  │ run_tool_… │ │
                  │   └──────────┘  └──────────────┘  └────────────┘ │
                  └────────────▲────────────────▲────────────▲───────┘
                               │                │            │
       ┌───────────────────────┘                │            └────────────────────┐
       │                                        │                                 │
┌──────┴────────────┐         ┌─────────────────┴────────────────┐    ┌───────────┴──────────┐
│ sjsujetsontool    │         │ nextjs-nemotron-app/             │    │ vuln-triage scripts  │
│ chat --agent      │         │ agent_sidecar/agent_sidecar.py   │    │ triage_react.py …    │
│ (CLI, terminal)   │         │ FastAPI :8002 + SSE → browser    │    │ (lesson 12c / 12d)   │
└───────────────────┘         └──────────────────────────────────┘    └──────────────────────┘
```

<span class="tiny">**Zero runtime deps** in the core. Whoever calls the loop just passes a
<code>complete(messages)→str</code> closure — that's the only seam between the agent and the model.</span>

---

## <span class="step">4</span> The tools — `edge_agent/tools.py`

Five built-in verbs (plus one optional online tool) — every path **confined to a project root**:

| Tool | Purpose |
|---|---|
| `read_file(path, start, end)` | read a slice, with line numbers |
| `grep(pattern, path, is_regex)` | search contents → `file:line: text` |
| `search_files(glob, dir)` | find files by name |
| `write_file(path, content)` | create / overwrite |
| `edit_file(path, old, new)` | replace one **exact, unique** snippet |
| `web_search(query, num=5)` *(opt-in)* | Google via SerpAPI, **auto-on** when `SERPAPI_API_KEY` is set |

> `edit_file` refuses unless `old` matches **exactly once** → forces a `read_file` first.
> `dispatch()` always returns a string, so a bad call becomes an `Observation`, not a crash.
> `web_search` reads its key from `~/.env.local` — see Step 9 for the one-line setup.

---

## <span class="step">5</span> The loop — `react_loop.py`

Decoupled from HTTP: you pass a `complete(messages) -> str` (llama.cpp / NVIDIA / OpenAI / Anthropic).

```python
def run(self, task):
    messages = [{"role": "system", "content": REACT_SYSTEM},
                {"role": "user",   "content": task}]
    for step in range(self.max_steps):
        text = self.complete(messages)                  # 1) reason
        messages.append({"role": "assistant", "content": text})
        if (m := _FINAL.search(text)):                  # 2) done?
            return m.group(1).strip()
        act = _ACTION.search(text)                      # 3) parse Action + JSON
        obs = self.tools.dispatch(act.group(1).strip(), json.loads(act.group(2)))  # 4) act
        messages.append({"role": "user", "content": "Observation: " + obs})        # 5) observe
```

<span class="tiny">reason → done? → parse → act → observe, with a <code>max_steps</code> cap so it can't loop forever.</span>

---

## <span class="step">6</span> Interface #1 — CLI: `sjsujetsontool chat --agent`

The terminal client imports `edge_agent` directly and gives the loop a `complete()` that
points at whatever backend is currently selected (local llama.cpp · NVIDIA Build · OpenAI · Anthropic).

```bash
sjsujetsontool chat --agent --agent-dir ./sample_project
# …or inside a normal chat session:
/agent on
/agent dir ./sample_project
What does app.py do, and is the requests CVE reachable?
```

Watch the trace stream by:

```text
[step 1] Thought: read the app   Action: read_file  {"path":"app.py","end":40}
   Observation: 1  import requests ...
[step 2] Thought: confirm URL    Action: grep        {"pattern":"requests.get"}
   Observation: app.py:37: response = requests.get(url, timeout=5)
🤖 requests.get(url) takes a caller-supplied URL → the HTTP CVE path is reachable.
```

---

## 🎥 Video Demo: Enabling Agent Mode in Chat

Observe how the ReAct agent loops through thoughts, actions (calling tools), and observations to solve tasks.

<div class="fig-center">
  <img src="Screencast1.webp" width="760" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
  <span class="caption">Enabling agent mode using <code>/agent on</code> and pointing to a workspace directory</span>
</div>

---

## <span class="step">7</span> Interface #2 — FastAPI: the **Next.js Agent Lab** backend

Same `edge_agent` core — wrapped in a **~250-line FastAPI** service so a browser can see it
([`agent_sidecar/agent_sidecar.py`](https://github.com/lkk688/edgeAI/tree/main/edgeLLM/nextjs-nemotron-app/agent_sidecar)).
One endpoint, **streamed SSE per step**:

```python
import edge_agent
from fastapi.responses import StreamingResponse

@app.post("/run")
async def run(request):
    body  = await request.json()
    tools = edge_agent.Tools(root=body["root"])
    msgs  = [{"role":"system","content":edge_agent.REACT_SYSTEM.format(...)},
             {"role":"user",  "content":body["task"]}]
    def stream():
        yield _sse({"type":"start","tools":edge_agent.tool_names()})
        for step in range(body["max_steps"]):
            reply = openai_call(msgs)                       # any OpenAI-compatible URL
            parsed = edge_agent.react_loop.parse_step(reply)
            yield _sse({"type":"step","action":parsed[1],"input":parsed[2]})
            obs = tools.dispatch(parsed[1], parsed[2])      # ← SAME tools.py as the CLI
            yield _sse({"type":"observation","text":obs})
            ...
    return StreamingResponse(stream(), media_type="text/event-stream")
```

Started on the Jetson with **`sjsujetsontool agent bg`**. The browser sees every Thought / Action
/ Observation card in real time at `/agent`.

---

## <span class="step">8</span> Two ways to call tools

| | ReAct text loop | Native tool-calling |
|---|---|---|
| File | `react_loop.py` | `tool_calling.py` |
| Works on any chat model | ✅ even base/local | ❌ needs a tool model |
| Reasoning visible | ✅ | partly |
| Parsing | regex | provider-enforced JSON |
| Used by | `chat --agent`, CVE lab **12c**, Agent Lab | CVE lab **12b**, Next.js Agent Lab (cloud) |

> Same `tools.py` — only the transport differs. This loop is the connective tissue across the labs.

---

## <span class="step">9</span> Enabling `web_search` — one line in `~/.env.local`

`web_search` is the **opt-in 6th tool** — pure `urllib` (no `requests` dep) calling Google via
[SerpAPI](https://serpapi.com) (free 100 searches/month).

```bash
# 1) Get a free key:  https://serpapi.com   → copy the API key from the dashboard
# 2) Stash it in the same file every lab reads:
echo "SERPAPI_API_KEY=…" >> ~/.env.local && chmod 600 ~/.env.local
```

That's it. The agent now **automatically** has a 6th tool:

- `sjsujetsontool chat --agent` — re-launches the CLI; `tool_names()` returns 6 entries.
- `sjsujetsontool agent bg` — restarts the FastAPI backend, which re-reads the env; the Agent Lab UI's *web_search disabled* banner disappears on next refresh.

<span class="tiny">**Graceful absence** — without a key, calling <code>web_search</code> returns
<code>"ERROR: web_search is disabled (no SERPAPI_API_KEY in env)."</code> The model sees that as an
Observation and falls back to file tools, never crashes.</span>

---

## <span class="step">10</span> Extend it — adding your own tool

Add a `run_python` tool in **three places**, all in [`tools.py`](https://github.com/lkk688/edgeAI/blob/main/edgeLLM/edge_agent/src/edge_agent/tools.py):

```python
class Tools:
    # 1) the method itself (confined to root by self._resolve)
    def run_python(self, code):
        import subprocess, sys
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=15, cwd=self.root)
        return (out.stdout + out.stderr)[:6000] or "(no output)"

# 2) advertise it to the ReAct prompt
TOOL_NAMES = ["read_file","grep","search_files","write_file","edit_file","run_python"]

# 3) (optional) advertise it to native tool-calling too
OPENAI_SCHEMAS.append({"type":"function","function":{
    "name":"run_python",
    "description":"Run a short Python snippet inside the workspace and return stdout+stderr.",
    "parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}})
```

**That's it.** Both the CLI agent and the FastAPI Agent Lab pick it up automatically — no other
file touched. Also try:

- 🎚️ **Swap the brain** — `/server` inside chat, or change the backend dropdown in the Agent Lab.
- 🛡️ **Change the policy** — edit `REACT_SYSTEM`: *require a plan first*, or *forbid `write_file`*
  (a read-only code auditor).

---

## <span class="step">11</span> Bring your own project — `sjsujetsontool` is **path-agnostic**

Cloned your own Next.js / Vite / agent repo into `/Developer/edgeAI/edgeLLM/my-app`? Or anywhere
else under `/Developer`? Same two commands, **three ways** to point them at it:

<div class="cols">
<div>

**(1) Pass the path as an arg**

```bash
sjsujetsontool node  bg /Developer/my-app
sjsujetsontool agent bg /Developer/my-app/agent_sidecar
```

**(2) Export an env var (per-shell or in `~/.bashrc`)**

```bash
export SJSUJETSONTOOL_NODE_DIR=/Developer/my-app
export SJSUJETSONTOOL_AGENT_DIR=/Developer/my-app/agent_sidecar
export SJSUJETSONTOOL_EDGE_AGENT_DIR=/Developer/my-edge-agent  # custom Python core
# now `sjsujetsontool node` / `agent` defaults follow YOUR project
```

</div>
<div>

**(3) Change it inside the Agent Lab UI** (no restart)

- **Workspace** input — sets the agent's project root for `read_file`/`grep`/`write_file`/`edit_file`
- **Backend** dropdown — switch live: NVIDIA · Local llama.cpp · `node05` · OpenAI · Anthropic
- **Custom** option — paste **any OpenAI-compatible URL + optional key**:
  - your own vLLM, Ollama, Together.ai, a corporate gateway, …
  - **base_url** + **api_key** fields appear, get forwarded to the sidecar as-is.

<span class="tiny">Path must live **under `/Developer/`** (that's the dir the container mounts 1:1
from the host). The framework doesn't matter — `node` runs whatever `package.json` says (`dev` /
`start`), `agent` runs whatever `agent_sidecar.py` exposes.</span>

</div>
</div>

---

<!-- _class: lead -->

Full lesson → [lkk688.github.io/edgeAI/curriculum/13_react_agent](https://lkk688.github.io/edgeAI/curriculum/13_react_agent/)
