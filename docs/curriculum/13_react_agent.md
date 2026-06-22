# 🤖 The ReAct Loop — the Core of Every AI Agent

**Author:** Dr. Kaikai Liu, Ph.D.
**Position:** Associate Professor, Computer Engineering
**Institution:** San Jose State University
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

> **Class goal.** Understand what actually makes a chatbot into an *agent*: a
> **reason-and-act (ReAct) loop** wrapped around a small set of **tools**. You
> will read a ~120-line reusable implementation, run it as
> `sjsujetsontool chat --agent`, and see how the *same* loop powers the Next.js
> app follow-ups and the CVE-triage labs (12b/12c).
>
> 🎞️ **Overview slides:** [**ReAct Agents ▶**](https://lkk688.github.io/edgeAI/slides/react-agent.html)
>
> **Companion code** — the installable [`edge_agent`](../../edgeLLM/edge_agent/) package:
> [`tools.py`](../../edgeLLM/edge_agent/src/edge_agent/tools.py) ·
> [`react_loop.py`](../../edgeLLM/edge_agent/src/edge_agent/react_loop.py) ·
> [`tool_calling.py`](../../edgeLLM/edge_agent/src/edge_agent/tool_calling.py)
>
> ```bash
> pip install -e edgeLLM/edge_agent      # then:  from edge_agent import ReActAgent, Tools
> ```

---

## 1. 🎯 Chatbot vs. agent

A plain chatbot maps **one prompt → one answer**. It cannot look anything up; it
only knows what is in its weights and your prompt.

An **agent** is a chatbot put in a **loop** with **tools**. Given a goal, it
repeatedly decides *"what should I do next?"*, takes an action in the world
(reads a file, runs a search, calls an API), observes the result, and continues
until the goal is met. Two ingredients turn a model into an agent:

1. **Tools** — functions the model may call (here: read/grep/search/write/edit files).
2. **A control loop** — code that runs the chosen tool and feeds the result back.

That loop is the whole idea. Everything else (RAG, multi-agent systems, coding
assistants like Claude Code) is a variation on it.

---

## 2. 🔁 ReAct = Reason + Act

[**ReAct**](https://arxiv.org/abs/2210.03629) is the most common loop shape. The
model is asked to interleave **reasoning** and **actions** in a strict text
protocol, **one step per turn**:

```text
Thought: I should look at app.py first
Action: read_file
Action Input: {"path": "app.py", "end": 40}
```

Our code parses that, runs the tool, and appends the result:

```text
Observation: 1  import requests
2  def fetch_status(url): ...
```

The model sees the observation and produces the next `Thought / Action`, looping
until it is confident enough to finish:

```text
Thought: I now know the answer
Final Answer: app.py calls requests.get(url) with a caller-supplied URL.
```

**Why a text protocol?** Because it works against *any* chat endpoint — even a
local base model with no "function-calling" feature. The model's reasoning is
also right there in the terminal, which makes the agent easy to debug. (When the
backend *does* support structured tool-calling, you can use that instead — see
§6.)

---

## 3. 🧰 The tools — [`edge_agent/tools.py`](../../edgeLLM/edge_agent/src/edge_agent/tools.py)

We give the agent the same five verbs a human coder uses. Every path is
**confined to a project root**, so the agent cannot wander outside the folder
you point it at:

| Tool | Signature | Purpose |
|---|---|---|
| `read_file` | `(path, start=1, end=None)` | read a slice, with line numbers |
| `grep` | `(pattern, path=".", is_regex=False)` | search contents → `file:line: text` |
| `search_files` | `(glob="*", dir=".")` | find files by name |
| `write_file` | `(path, content)` | create / overwrite a file |
| `edit_file` | `(path, old, new)` | replace one **exact, unique** snippet |

`edit_file` is the interesting one — it is a **find-and-replace** that *refuses*
to run unless the `old` text matches exactly once. That forces the agent to
`read_file` first and quote a unique snippet, which is exactly how real coding
agents avoid clobbering the wrong line:

```python
def edit_file(self, path, old, new):
    text = open(self._resolve(path)).read()
    count = text.count(old)
    if count == 0:
        return "ERROR: `old` text not found — read_file first and copy an exact snippet."
    if count > 1:
        return "ERROR: `old` matches %d places — add surrounding context." % count
    open(self._resolve(path), "w").write(text.replace(old, new, 1))
    return "edited %s (1 replacement)" % path
```

The `dispatch(name, args)` method runs a tool by name and **always returns a
string** (errors included), so a bad tool call becomes an `Observation` the
model can recover from rather than a crash.

---

## 4. 🧠 The loop — [`react_loop.py`](../../edgeLLM/edge_agent/src/edge_agent/react_loop.py)

The whole engine is one class. It is deliberately **decoupled from any HTTP
client**: you hand it a `complete(messages) -> str` callable, so the same loop
runs on llama.cpp, NVIDIA Build, OpenAI, or Anthropic.

```python
class ReActAgent:
    def __init__(self, complete, tools, *, max_steps=8, log=print):
        self.complete, self.tools, self.max_steps, self.log = complete, tools, max_steps, log

    def run(self, task):
        messages = [{"role": "system", "content": REACT_SYSTEM},
                    {"role": "user",   "content": task}]
        for step in range(self.max_steps):
            text = self.complete(messages)                 # 1) the model reasons
            messages.append({"role": "assistant", "content": text})
            if (m := _FINAL.search(text)):                 # 2) done?
                return m.group(1).strip()
            act = _ACTION.search(text)                     # 3) parse Action + JSON input
            name, args = act.group(1).strip(), json.loads(act.group(2))
            obs = self.tools.dispatch(name, args)          # 4) run the tool
            messages.append({"role": "user", "content": "Observation: " + obs})  # 5) feed back
        return "(stopped: reached max_steps without a Final Answer)"
```

Five lines of logic: **reason → check-for-done → parse → act → observe**, with a
`max_steps` cap so a confused model can't loop forever (or burn your API quota).
The `REACT_SYSTEM` prompt tells the model the exact format and lists the tools.

---

## 5. ▶️ Run it: `sjsujetsontool chat --agent`

Agent mode is built into the chat client. It uses the `edge_agent` package shipped
in the repo (`/Developer/edgeAI/edgeLLM/edge_agent`); `chat.py` finds it on
`sys.path` automatically — no install required on the Jetson.

```bash
sjsujetsontool shell                 # (keys from ~/.env.local come with you)
sjsujetsontool chat --agent --agent-dir /Developer/edgeAI/edgeLLM/vuln-triage/sample_project
```

…or toggle it inside an ordinary chat session:

```text
/agent on                            # turn the ReAct loop on
/agent dir ./sample_project          # point it at a folder
What does app.py do, and is the requests CVE reachable?
```

You will watch the trace stream by — each `Thought / Action / Observation` —
before the final answer:

```text
[step 1]
Thought: read the app to see how requests is used
Action: read_file
Action Input: {"path": "app.py", "end": 40}
   Observation: 1  import requests ...
[step 2]
Thought: confirm the URL is caller-supplied
Action: grep
Action Input: {"pattern": "requests.get"}
   Observation: app.py:37: response = requests.get(url, timeout=5)
🤖 app.py calls requests.get(url) with a caller-supplied URL → the HTTP CVE path is reachable.
```

> **Tip.** Any backend works, but a tool-following model (NVIDIA Nemotron, a
> Qwen3.5-Coder, GPT-4o, Claude) follows the ReAct format far more reliably than
> a tiny base model. Pick one with `/server`.

---

## 6. 🔌 Two ways to call tools — and where each fits

`react_loop.py` uses the **text protocol** (works everywhere).
[`tool_calling.py`](../../edgeLLM/edge_agent/src/edge_agent/tool_calling.py) does the same job with the
provider's **native** `tools=` field (structured JSON function-calling):

```bash
pip install -e "edgeLLM/edge_agent[toolcalling]"      # installs the `edge-agent` CLI + openai
edge-agent "Summarize app.py and list its risky calls" \
    --dir edgeLLM/vuln-triage/sample_project
```

| | ReAct text loop (`react_loop.py`) | Native tool-calling (`tool_calling.py`) |
|---|---|---|
| Works on any chat model | ✅ even base / local | ❌ needs a tool-calling model |
| Reasoning visible in terminal | ✅ | partly |
| Parsing robustness | regex (can mis-parse) | provider-enforced JSON |
| Used by | `chat --agent`, CVE lab **12c** | CVE lab **12b**, the Next.js API routes |

They share the **same `tools.py`** — only the transport differs. This is
the connective tissue across the curriculum: the
[CVE-triage labs](./12_vulnerability_triage_intro.md) are this exact pattern with
*security* tools, and the [Next.js app](./11_nextjs_nemotron_app.md) can add an
agent endpoint by calling the same loop server-side.

---

## 7. 🛠️ Extend it

The design makes extension a one-function change:

1. **Add a tool.** Write a method on `Tools` (e.g. `run_tests()`,
   `http_get(url)`), add its name to `TOOL_NAMES`, and — for native
   tool-calling — one entry in `OPENAI_SCHEMAS`. The model can use it immediately.
2. **Swap the brain.** Pass any `complete()` — point it at a different model with
   `/server`, or wrap a different SDK.
3. **Change the policy.** Edit `REACT_SYSTEM` to require a plan first, or to
   forbid `write_file` (a read-only "auditor" agent).

---

## 8. ✅ Where you are now

You should be able to:

- [ ] Explain why an agent = **LLM + tools + a loop**.
- [ ] Trace one ReAct turn: `Thought → Action → Action Input → Observation`.
- [ ] Run `sjsujetsontool chat --agent` and read the live trace.
- [ ] Name the trade-off between the ReAct text loop and native tool-calling.
- [ ] Add a sixth tool to `edge_agent/tools.py`.

Next: see the loop applied to security in
**[Lesson 12b — Basic tool-calling triage](./12b_basic_tool_calling_triage.md)**.
