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

## <span class="step">3</span> How does an LLM *"call"* a tool? *(spoiler: it writes text)*

A common misconception: *"the model has an Execute button."*
**Reality:** an LLM is a **next-token predictor**. It can only write text. **Period.**

When you see this in a transcript, the model is **literally typing those characters**:

```text
Action: read_file
Action Input: {"path": "calculator.py", "end": 40}
```

It does **not** open any file. **Our runtime** reads that text and dispatches the call:

```python
m    = _ACTION.search(text)                          # find "Action: read_file"
args = json.loads(_INPUT_JSON.search(text).group(1)) # find {"path": …}
obs  = tools.dispatch(m.group(1), args)              # actually run it in Python
```

The string the tool returns is **glued back** into the conversation as the next turn:

```python
messages.append({"role": "user", "content": "Observation: " + obs})
```

> **Why does this even work?** The model was trained on millions of similar transcripts —
> code, ChatGPT logs, agent demos. It *learned* that emitting these characters reliably produces
> useful follow-ups. ReAct therefore works on **any** chat model — even a 0.8 B local base model.

---

## <span class="step">4</span> The ReAct loop, visualized

<div style="font-family: ui-monospace, monospace; font-size: 13px; max-width: 920px; margin: 12px auto;">

<div style="text-align: center;">
<span style="display: inline-block; padding: 6px 14px; background: #eef; border-radius: 6px;">
🧑 <strong>USER GOAL</strong> &nbsp;<span class="tiny">"Fix the typo in calculator.py"</span>
</span>
</div>

<div style="text-align: center; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px dashed #0073e6; border-radius: 12px; padding: 10px 14px;">
  <div style="text-align: center; color: #0073e6; margin-bottom: 8px;">
    <strong>ReAct LOOP</strong> &nbsp;<span class="tiny">(repeat up to <code>max_steps</code> times)</span>
  </div>

  <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <div style="flex: 1; border: 1.5px solid #444; padding: 8px 10px; border-radius: 8px; background: #fff;">
      🧠 <strong>Brain</strong> — LLM API<br/>
      <span class="tiny">reads <code>messages[]</code> →<br/>writes <code>Thought:</code> + <code>Action:</code></span>
    </div>
    <div style="font-size: 22px; color: #0073e6;">→</div>
    <div style="flex: 1; border: 1.5px solid #444; padding: 8px 10px; border-radius: 8px; background: #fff;">
      🛠️ <strong>Parser + Tool</strong><br/>
      <span class="tiny">regex finds <code>Action:</code> →<br/><code>tools.dispatch()</code> runs in Python</span>
    </div>
  </div>

  <div style="text-align: center; margin-top: 8px; color: #c00;">
    ↑ Observation &nbsp;<span class="tiny">(tool result appended as a "user" message for the next turn)</span>
  </div>
</div>

<div style="text-align: center; color: #888; margin: 4px 0;">↓ &nbsp;<span class="tiny">on <code>Final Answer:</code></span></div>

<div style="text-align: center;">
<span style="display: inline-block; padding: 6px 14px; background: #efe; border-radius: 6px;">
🤖 <strong>FINAL ANSWER</strong> back to the user
</span>
</div>

</div>

<span class="tiny">Five movable parts. Edit any one to change the agent: the <strong>brain</strong>
(any <code>complete(messages)→str</code>), the <strong>parser</strong> (regex), the
<strong>tools</strong>, the <strong>policy</strong> (<code>REACT_SYSTEM</code> prompt), and the
<strong>stop condition</strong> (<code>max_steps</code> / <code>Final Answer</code>).</span>

---

## <span class="step">5</span> You're standing on giants — the agent landscape

You've probably seen or used some of these. **All implement the same fundamental pattern.**

<div class="cols">
<div>

**Closed / commercial:**
- 🟠 **Claude Code** (Anthropic) — terminal coding agent
- 🟢 **OpenAI Codex CLI** — terminal coding agent
- 🐙 **GitHub Copilot — Agent mode** — VSCode
- ⚡ **Cursor** — IDE with agent

</div>
<div>

**Open-source:**
- 🤖 **OpenHands** (was OpenDevin) — autonomous agent
- ✂️ **Aider** — terminal coding agent
- 🔁 **Continue.dev** — VSCode + JetBrains
- 🛡️ **OpenCodex** — community Codex clone

</div>
</div>

All variations on **LLM + tools + loop** (sometimes + RAG / multi-agent / browser / sandbox).
The differences are the **tool kit** (browser? terminal exec? Git? screenshots?) and the
**policy** (prompt + guardrails) — *not* the core algorithm.

> **Our `edge_agent`** is **~120 lines** that distills the essence. Read it in one sitting,
> hack it on day 1. The same loop powers `sjsujetsontool chat --agent`, the Next.js Agent Lab,
> *and* the Lesson 12 vuln-triage scripts.

---

## <span class="step">6</span> Architecture — **one core, multiple interfaces**

<div style="font-family: ui-monospace, monospace; font-size: 13px;">

<div style="border: 2px solid #0073e6; border-radius: 10px; padding: 10px 14px; margin: 0 70px; background: rgba(0,115,230,0.05);">
  <div style="text-align: center;">
    <strong>edgeLLM/edge_agent/</strong> &nbsp;<span class="tiny">— pure-stdlib Python (zero runtime deps)</span>
  </div>
  <div style="display: flex; gap: 6px; margin-top: 8px;">
    <div style="flex:1; border: 1px solid #888; padding: 6px; border-radius: 6px; background: #fff;">
      <code>tools.py</code><br/><span class="tiny">read · grep · search · write · edit · web_search</span>
    </div>
    <div style="flex:1; border: 1px solid #888; padding: 6px; border-radius: 6px; background: #fff;">
      <code>react_loop.py</code><br/><span class="tiny">ReActAgent + tolerant parser</span>
    </div>
    <div style="flex:1; border: 1px solid #888; padding: 6px; border-radius: 6px; background: #fff;">
      <code>tool_calling.py</code><br/><span class="tiny">native OpenAI <code>tools=[…]</code> loop</span>
    </div>
  </div>
  <div style="margin-top: 8px; text-align: center; padding: 4px; background: #ffe; border: 1px dashed #c80; border-radius: 6px;">
    ⚡ <strong>LLM API seam</strong>: <code>complete(messages) → str</code>
    &nbsp;<span class="tiny">any chat backend plugs here — local llama.cpp · NVIDIA · OpenAI · Anthropic · …</span>
  </div>
</div>

<div style="display: flex; justify-content: space-around; color: #555; font-size: 18px; margin-top: 4px;">
  <span>↓</span><span>↓</span><span>↓</span>
</div>

<div style="display: flex; gap: 8px; margin-top: 2px;">
  <div style="flex:1; border: 1px dashed #888; padding: 8px; border-radius: 6px;">
    🐚 <strong>CLI</strong><br/>
    <code>sjsujetsontool chat --agent</code><br/>
    <span class="tiny">terminal, in-chat <code>/agent on</code></span>
  </div>
  <div style="flex:1; border: 1px dashed #888; padding: 8px; border-radius: 6px;">
    🌐 <strong>FastAPI</strong><br/>
    <code>agent_sidecar.py</code> :8002<br/>
    <span class="tiny">POST /run → SSE → browser</span>
  </div>
  <div style="flex:1; border: 1px dashed #888; padding: 8px; border-radius: 6px;">
    🛡️ <strong>Vuln triage</strong><br/>
    <code>triage_react.py</code><br/>
    <span class="tiny">Lesson 12c / 12d scripts</span>
  </div>
</div>

</div>

<span class="tiny">Three consumers, **same import**: <code>from edge_agent import Tools, ReActAgent</code>.
The only seam between the agent and "intelligence" is the highlighted <code>complete()</code>
callable — see Step 8 for what that signature buys us.</span>

---

## <span class="step">7</span> The tools — `edge_agent/tools.py`

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

## <span class="step">8</span> The loop — `react_loop.py`

The loop is **decoupled from HTTP**. It doesn't know whether the model lives on a cloud
endpoint, in-process llama.cpp, or a unit-test stub. It just gets a callable:

```python
complete(messages: list[dict]) -> str   # ← the ONLY seam between agent and "intelligence"
```

…takes an OpenAI-style message list, returns the assistant reply text. **Three real
implementations** — all drop-in interchangeable behind the same `ReActAgent`:

```python
# (a) NVIDIA Build via the OpenAI client (or OpenAI / Anthropic / any compat URL)
def complete(msgs):
    return client.chat.completions.create(model="…", messages=msgs).choices[0].message.content

# (b) local llama-cpp-python in-process — no network at all
def complete(msgs):
    return llm.create_chat_completion(messages=msgs)["choices"][0]["message"]["content"]

# (c) mocked, for unit tests — deterministic, zero quota burned
def complete(msgs):
    return 'Thought: stub\nAction: read_file\nAction Input: {"path":"x.py"}'
```

The loop itself stays tiny — **5 steps, one for-loop**:

```python
def run(self, task):
    messages = [{"role": "system", "content": REACT_SYSTEM},
                {"role": "user",   "content": task}]
    for step in range(self.max_steps):
        text = self.complete(messages)                                # 1) reason
        messages.append({"role": "assistant", "content": text})
        if (m := _FINAL.search(text)): return m.group(1).strip()      # 2) done?
        act = _ACTION.search(text)                                    # 3) parse Action
        obs = self.tools.dispatch(act.group(1).strip(), json.loads(act.group(2)))  # 4) act
        messages.append({"role": "user", "content": "Observation: " + obs})        # 5) observe
```

<span class="tiny">Swap providers by passing a different <code>complete</code> — zero loop changes,
zero provider lock-in. Same code runs against a 0.5 B local model or GPT-5.</span>

---

## <span class="step">9</span> Interface #1 — CLI: `sjsujetsontool chat --agent`

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

## <span class="step">10</span> Interface #2 — FastAPI: the **Next.js Agent Lab** backend

Same `edge_agent` core — wrapped in a **~250-line FastAPI** service so a browser can see it
([`agent_sidecar/agent_sidecar.py`](https://github.com/lkk688/edgeAI/tree/main/edgeLLM/nextjs-nemotron-app/agent_sidecar)).
One streaming endpoint, **SSE per step**:

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
    return StreamingResponse(stream(), media_type="text/event-stream")
```

**The defined HTTP surface — small on purpose:**

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health`        | Liveness + lists tools + workspace root + max_steps cap |
| `POST` | `/run`           | Body JSON → **SSE stream** of `{start, step, observation, nudge, final, error}` |
| `GET`  | `/docs`          | Auto Swagger UI — try `/run` from the browser, no curl needed |
| `GET`  | `/openapi.json`  | Machine-readable schema (drives typed JS clients) |

<div class="cols">
<div>

**Why modular?**
- Each endpoint is independent. Add `/sessions/{id}/resume` without touching `/run`.
- Swap the agent algorithm (LangChain, AutoGen) → only `/run` changes; UI stays the same.
- Browser, Postman, curl, Python — anyone with HTTP gets in.

</div>
<div>

**Why scalable?**
- **Stateless** by default → `uvicorn --workers N` for N× concurrency.
- SSE doesn't hold sessions → each `/run` finishes and lets go.
- Async-friendly: FastAPI threadpools blocking work (Riva, edge_agent) automatically.
- Decoupled from UI: the same backend serves the Next.js Lab AND a CLI client.

</div>
</div>

Started on the Jetson with **`sjsujetsontool agent bg`**. Browser sees every Thought / Action /
Observation card in real time at `/agent`.

---

## <span class="step">11</span> Context window — the **hidden ceiling**

Every model has a **max context**. Your whole conversation must fit inside:

```
system_prompt + task + (Thought + Action + Observation) × N steps
```

| Backend | Context window |
|---|---|
| Local Qwen 3.5 (Q4) on Jetson | **4 – 8 K** tokens |
| Shared `node05` Qwen 3.5-9B    | typically **8 – 32 K** |
| NVIDIA Nemotron 49 B / 70 B    | **128 K** |
| OpenAI GPT-4o                  | 128 K |
| Claude Sonnet 4.6              | **200 K** |

**Where it bites in practice:** an `Observation` from `read_file` on a 1000-line file is ~6 KB
→ ~1.5 K tokens. Three such observations + system prompt + task and a local 8 K window is half-full.

**Built-in defense** — in `tools.py`'s dispatcher:

```python
return str(getattr(self, name)(**args))[:6000]   # ← every tool result is truncated
```

…plus `max_steps = 8` by default → naturally bounded.

**Mitigations** when you need more headroom:

- ✂️ Use **`read_file(path, start=10, end=40)`** — *never* whole files
- 🔎 **`grep` first**, then `read_file` a narrow range around the hit
- 📉 Lower **`max_steps`** if your task is small
- 📚 Pick a model with a **bigger window** (Nemotron 128 K beats local 8 K)
- 🪄 Have the agent **summarize old observations** into a short note for long sessions

> **Symptom of exhaustion:** the agent forgets the original task, hallucinates files, or re-runs
> the same Action twice. Fix is **almost always smaller observations**, not a smarter prompt.

---

## <span class="step">12</span> Two ways to call tools — same outcome, different fence

**Under the hood both paths are just the model emitting tokens.** The difference is **who parses
those tokens** — you (regex) or the provider (built-in JSON unpacker).

| | ReAct text loop | Native tool-calling |
|---|---|---|
| File | `react_loop.py` | `tool_calling.py` |
| What the model emits | free text following the protocol | tokens forming a JSON-schema'd object |
| Who parses | **you** — regex in `react_loop.py` | **the provider** — built into its API |
| Works on any chat model | ✅ even base / local | ❌ needs a tool-fine-tuned model |
| Reasoning visible | ✅ <code>Thought:</code> in plain text | partial — text + opaque <code>tool_calls</code> |
| Used by | `chat --agent`, lesson 12c, Agent Lab | lesson 12b, OpenAI / Anthropic SDKs |

What native tool-calling looks like (the provider does the parsing):

```python
resp = client.chat.completions.create(
    model="qwen/qwen3.5-9b", messages=msgs,
    tools=[{"type": "function", "function": {"name": "read_file",
            "parameters": {...JSON schema...}}}])
for tc in resp.choices[0].message.tool_calls:                # ← already parsed for you
    args   = json.loads(tc.function.arguments)
    result = tools.dispatch(tc.function.name, args)          # SAME tools.py
```

> Same `tools.py` powers both paths — **only the transport differs**. This is the connective
> tissue that ties every lab in the curriculum together.

---

## <span class="step">13</span> Enabling `web_search` — one line in `~/.env.local`

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

## <span class="step">14</span> Extend it — adding tools *and* **revising the prompt**

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
file touched.

**Where to revise the prompt** — three knobs, biggest to smallest:

| Knob | File | What it controls |
|---|---|---|
| `REACT_SYSTEM` | `edge_agent/react_loop.py` | The ReAct rules + examples shown to **every** agent run |
| Per-backend tweaks | `agent_sidecar.py` `event_stream()` | Append extras *only* for a specific backend |
| User task | Agent Lab UI input | What this run is asked to do |

Three popular `REACT_SYSTEM` tweaks:

- 🛡️ **Read-only auditor** — drop `write_file`/`edit_file` from `TOOL_NAMES` *and* add
  *"You must NOT modify files."* to `REACT_SYSTEM`.
- 🗺️ **Plan-first** — *"Output a numbered plan in your first Thought; emit Actions only after."*
- 📋 **Tone control** — *"Final Answer must be a markdown bullet list, ≤ 5 bullets."*

> The system prompt is **half of what the model effectively is** at runtime. Every refinement
> propagates instantly to all three interfaces (CLI · FastAPI · CVE labs) — they import the same
> `REACT_SYSTEM` constant.

---

## <span class="step">15</span> Bring your own project — `sjsujetsontool` is **path-agnostic**

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
