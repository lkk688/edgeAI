---
marp: true
paginate: true
size: 16:9
title: The ReAct Loop — Core of AI Agents
---

<style>
:root { --blue:#0055A2; --gold:#E5A823; --ink:#202a3c; }
section { background:#fff; color:var(--ink); font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:21px; line-height:1.45; padding:46px 62px 54px; border-top:7px solid var(--blue); }
section::before { content:""; position:absolute; left:0; right:0; top:7px; height:3px; background:var(--gold); }
h1 { color:var(--blue); font-size:1.85em; margin:0 0 .3em; }
h2 { color:var(--blue); font-size:1.3em; border-bottom:2px solid var(--gold); padding-bottom:6px; margin:0 0 .5em; }
h3 { color:#0a3d7a; }
strong { color:var(--blue); }
a { color:var(--blue); text-decoration:none; border-bottom:1px solid var(--gold); }
code { background:#eef2f8; color:#0a3d7a; border-radius:5px; padding:.05em .35em; font-size:.92em; }
pre { background:#0f1830; border-radius:10px; box-shadow:0 6px 18px rgba(8,20,50,.12); }
pre code { background:transparent; color:#e8eefc; font-size:.72em; line-height:1.5; }
blockquote { border-left:4px solid var(--gold); background:#fbf6e9; color:#5b4a22; padding:.4em .9em; border-radius:6px; }
table { font-size:.8em; border-collapse:collapse; } th { background:var(--blue); color:#fff; } td,th { border:1px solid #d4dce8; padding:5px 10px; }
.step { background:var(--blue); color:#fff; border-radius:999px; padding:.03em .6em; font-weight:700; font-size:.85em; }
.tiny { font-size:.78em; color:#5d6b82; }
.cols { display:flex; gap:26px; align-items:flex-start; } .cols > * { flex:1; }
section.lead { text-align:center; border-top-width:10px; }
section.lead h1 { font-size:2.2em; }
</style>

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

## <span class="step">3</span> The tools — `edge_agent/tools.py`

Five verbs a human coder uses — every path **confined to a project root**:

| Tool | Purpose |
|---|---|
| `read_file(path, start, end)` | read a slice, with line numbers |
| `grep(pattern, path, is_regex)` | search contents → `file:line: text` |
| `search_files(glob, dir)` | find files by name |
| `write_file(path, content)` | create / overwrite |
| `edit_file(path, old, new)` | replace one **exact, unique** snippet |

> `edit_file` refuses unless `old` matches **exactly once** → forces a `read_file` first.
> `dispatch()` always returns a string, so a bad call becomes an `Observation`, not a crash.

---

## <span class="step">4</span> The loop — `react_loop.py`

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

## <span class="step">5</span> Run it

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

## <span class="step">6</span> Two ways to call tools

| | ReAct text loop | Native tool-calling |
|---|---|---|
| File | `react_loop.py` | `tool_calling.py` |
| Works on any chat model | ✅ even base/local | ❌ needs a tool model |
| Reasoning visible | ✅ | partly |
| Parsing | regex | provider-enforced JSON |
| Used by | `chat --agent`, CVE lab **12c** | CVE lab **12b**, Next.js API |

> Same `tools.py` — only the transport differs. This loop is the connective tissue across the labs.

---

## <span class="step">7</span> Extend it — a one-function change

1. **Add a tool** → a method on `Tools` + its name in `TOOL_NAMES` (+ one `OPENAI_SCHEMAS` entry). Usable instantly.
2. **Swap the brain** → pass any `complete()`; switch models with `/server`.
3. **Change the policy** → edit `REACT_SYSTEM`: require a plan first, or forbid `write_file` (a read-only auditor).

<!-- _class: lead -->

Full lesson → [lkk688.github.io/edgeAI/curriculum/13_react_agent](https://lkk688.github.io/edgeAI/curriculum/13_react_agent/)
