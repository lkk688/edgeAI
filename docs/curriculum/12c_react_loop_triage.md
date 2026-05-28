# 🔁 AI-Powered CVE Triage on Jetson — Part 3: ReAct Without a Framework

**Author:** Dr. Kaikai Liu, Ph.D.
**Institution:** San Jose State University

> **Prerequisite:** [Lesson 12b](./12b_basic_tool_calling_triage.md).
> You should have `triage_basic.py` working end-to-end.
>
> **Companion code:** [`edgeLLM/vuln-triage/triage_react.py`](../../edgeLLM/vuln-triage/triage_react.py).

---

## 1. 🎯 What you'll build

The same triage you wrote in 12b, but the OpenAI `tools=[...]`
parameter is **gone**. Instead, the model is asked to follow a
text-only protocol:

```text
Thought: <one short sentence reasoning about the next step>
Action: <tool_name>({"arg": "value", ...})

      ↓  (you, the runtime, execute the tool)

Observation: <JSON result of the tool, possibly truncated>

      ↓  (model produces next Thought + Action, or stops)

Thought: <final reasoning>
Final Answer: <single-line JSON verdict>
```

This is the **ReAct** ("Reason + Act") pattern from
[Yao et al. 2022](https://arxiv.org/abs/2210.03629), in its purest form.
The "framework" is **one regex parser** and **one `while` loop**.

Two reasons it matters:

1. **Portability.** It runs against *any* chat endpoint — including
   self-hosted vLLM serving base or fine-tuned models that do not
   expose the OpenAI `tools` schema.
2. **Visibility.** Every Thought lands in your terminal verbatim, so
   the model's reasoning is debuggable in a way that opaque
   `tool_calls` JSON never is.

---

## 2. 🧩 Step 1 — The protocol prompt

The whole thing hinges on a system prompt that defines the format *and*
the available tools. We give the model concrete examples so it cannot
fall back on Python-style `tool("arg")` syntax (it will try, often):

```python
REACT_SYSTEM = textwrap.dedent("""
    You are a security analyst triaging one CVE against a small Python
    project on an NVIDIA Jetson edge device. You work in a strict
    ReAct loop — every reply MUST follow this template exactly:

        Thought: <one short sentence reasoning about the next step>
        Action: <tool_name>({"arg": "value", ...})

    After your Action you will receive:

        Observation: <JSON result of the tool, possibly truncated>

    You then produce the next Thought + Action. When you have enough
    evidence to decide, reply with NO action block, only:

        Thought: <final reasoning>
        Final Answer: <single-line JSON object>

    Available tools (and their argument names):
      lookup_cve(cve_id)
      pip_audit_findings(requirements_path)
      search_usage(pattern, project_dir, is_regex=false)
      read_file(path, project_dir, start=1, end=null)

    Examples of valid Action lines (any of these forms is accepted):
      Action: lookup_cve({"cve_id": "CVE-2024-1234"})
      Action: search_usage({"pattern": "yaml.load", "project_dir": "/tmp/proj"})
      Action: read_file({"path": "app.py", "project_dir": "/tmp/proj", "start": 10, "end": 40})

    Rules:
      - One Action per turn. Multiple actions are not allowed.
      - Prefer the JSON-object form shown above.
      - Do not invent tool names. Do not fabricate observations.
""").strip()
```

The `Examples` block is doing real work — without it, qwen3-coder
defaults to writing `Action: lookup_cve("CVE-2024-1234")` (Python style)
and our strict JSON parser would reject it. We could fail loudly *or*
make the parser tolerant of both forms. We do both, because real models
in real classrooms misbehave.

---

## 3. 🧩 Step 2 — The line-oriented parser

The model's reply per turn is plain text. We need to pull out
`Thought:`, `Action:`, and `Final Answer:` lines reliably, despite the
model occasionally wrapping them in `**bold**` or producing extra
blank lines.

```python
RE_ACTION  = re.compile(r"^\s*\**\s*Action\s*:\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$")
RE_FINAL   = re.compile(r"^\s*\**\s*Final Answer\s*:\s*(.+)$", re.IGNORECASE)
RE_THOUGHT = re.compile(r"^\s*\**\s*Thought\s*:\s*(.+)$",      re.IGNORECASE)

def _parse_step(text: str):
    thought = action = args_str = final = None
    for line in text.splitlines():
        if m := RE_THOUGHT.match(line):
            thought = m.group(1).strip()
        elif m := RE_ACTION.match(line):
            action   = m.group(1).strip()
            args_str = m.group(2).strip()
        elif m := RE_FINAL.match(line):
            final = m.group(1).strip()
    return thought, action, args_str, final
```

Three deliberate choices:

- **`\**` (optional `**`)** — models love to bold their headers.
- **The action regex anchors on `^...$`** — so a stray `Action:` inside
  a thought string is *not* mistaken for a real action.
- **No state machine.** We don't care about the *order* of lines — only
  that one of each appears. Simpler == fewer bugs.

---

## 4. 🧩 Step 3 — The forgiving argument parser

OpenAI's structured tool calling guarantees you a JSON string. With our
plain-text protocol, we get *whatever the model typed*. A robust
runtime accepts all three real-world dialects:

```python
def parse_action_args(args_str: str, fn) -> dict:
    """Parse `Action:` args. Accepts:
       {"cve_id": "CVE-2024-1234"}     # canonical JSON
       cve_id="CVE-2024-1234"           # Python kwargs
       "CVE-2024-1234"                  # positional
       "yaml load", k=3                 # mixed
    """
    args_str = (args_str or "").strip()
    if not args_str:
        return {}

    # Fast path: a JSON object.
    if args_str.startswith("{"):
        try:
            obj = json.loads(args_str)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass  # fall through

    # General path: parse as the argument list of a Python call.
    node = ast.parse(f"_({args_str})", mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError("Action arguments are not a call form")

    param_names = list(inspect.signature(fn).parameters)
    out = {}
    for i, arg in enumerate(node.args):
        out[param_names[i]] = ast.literal_eval(arg)
    for kw in node.keywords:
        out[kw.arg] = ast.literal_eval(kw.value)
    return out
```

`ast.literal_eval` rejects anything that isn't a Python literal
(`"strings"`, numbers, `True/False/None`, lists, dicts) — so the model
cannot smuggle code execution through the parser. That property is
important: if you ever expose this runtime to user-controlled text, the
parser does not become a sandbox-escape oracle.

---

## 5. 🧩 Step 4 — The loop

Now the actual ReAct loop. ~25 lines:

```python
def react_triage(client, *, finding, project_dir, requirements_path,
                 model, verbose=True) -> Verdict:
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": initial_user_message(finding,
                                                           project_dir,
                                                           requirements_path)},
    ]

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stop=["\nObservation:", "Observation:"],   # ← see below
            temperature=0.1,
            max_tokens=1024,
        )
        reply = (resp.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": reply})

        _, action, args_str, final = _parse_step(reply)

        if final is not None:
            return parse_verdict(final, ...)        # done

        if action is None:                          # model produced
            messages.append({"role": "user",        # only a Thought —
                "content": "I did not see an Action line. "
                "Continue with `Action: tool(...)` or finish "
                "with `Final Answer: <json>`."})
            continue

        obs = _run_tool(action, args_str or "{}")   # execute the tool
        messages.append({"role": "user",
            "content": f"Observation: {obs}"})       # feed back the result
```

The one non-obvious knob is `stop=["\nObservation:", "Observation:"]`.

The model is *supposed* to stop after writing its `Action:` line and
wait for our Observation. But once a model has been trained on enough
ReAct transcripts, it will happily **role-play** the Observation too —
hallucinating tool results. The `stop` parameter tells NVIDIA Build to
truncate the assistant message before the literal string "Observation:"
ever appears in its output. The model physically cannot fake an
observation past that boundary.

This single line is the difference between a fragile prototype and a
reliable agent.

---

## 6. ▶️ Run it

```bash
cd ~/vuln-triage
python3 triage_react.py --project sample_project --cve CVE-2019-10906
```

Verified output on Jetson Orin Nano (jinja2 sandbox escape, the
canonical "vulnerable but not exploitable here" case):

```text
[1/1] jinja2 2.10 — CVE-2019-10906
  ─ step 1: calling model … 6.5s  (p=452 c=56)
      │ Thought: I need to first look up the details of CVE-2019-10906 to understand its impact and affected versions.
      │ Action: lookup_cve({"cve_id": "CVE-2019-10906"})
      │ Observation: {"cve_id": "CVE-2019-10906", "description": "In Pallets Jinja before 2.10.1, str.format_map allows …
  ─ step 2: calling model … 38.8s  (p=846 c=56)
      │ Thought: I need to check if the project uses the Jinja2 template engine in a way that could be exploited by this CVE.
      │ Action: search_usage({"pattern": "jinja2", "project_dir": "/home/cmpe/vuln-triage/sample_project"})
      │ Observation: {"pattern": "jinja2", "hits": [{"file": "app.py", "line": 11, ...
  ─ step 3: calling model … 18.1s  (p=1192 c=65)
      │ Thought: I need to examine the actual usage of jinja2 in the code to determine if it's exploitable.
      │ Action: read_file({"path": "app.py", "project_dir": "/home/cmpe/vuln-triage/sample_project"})
      │ Observation: {"path": "app.py", "lines": [1, 49], "total_lines": 49, "content": ...
  ─ step 4: calling model … 4.3s  (p=2061 c=61)
      │ Thought: I need to check the requirements.txt file to confirm the version of jinja2.
      │ Action: read_file({"path": "requirements.txt", "project_dir": "/home/cmpe/vuln-triage/sample_project"})
      │ Observation: {"path": "requirements.txt", "lines": [1, 18], ...
  ─ step 5: calling model … 6.6s  (p=2495 c=172)
      │ Thought: Based on the evidence from the code analysis, I can now make a final determination.
      │ Final Answer: {"exploitable_here": false, "confidence": "high", ...}

┌─ CVE-2019-10906  [jinja2 2.10]
│  verdict   : NOT EXPLOITABLE HERE   (confidence: high)
│  reason    : CVE-2019-10906 affects Jinja2 versions prior to 2.10.1, and while the
│              project uses jinja2==2.10, the vulnerability is not
│              exploitable because the template is hard-coded and not
│              user-controllable. …
│  action    : Upgrade jinja2 to version 2.10.1 or later …
└──
```

Five steps, ~75 s wall time, every word of the model's reasoning
visible. **Compare this with the 12b output** — same triage decision,
but here you can see *how* it got there.

---

## 7. 🆚 ReAct vs. provider tool-calling

| Dimension              | 12b — provider `tools=[...]`            | 12c — text ReAct (this) |
|------------------------|------------------------------------------|--------------------------|
| Provider requirements  | Endpoint must support OpenAI tool schema | Any chat endpoint        |
| Reasoning visibility   | Hidden inside `tool_calls`               | Every Thought in the log |
| Tokens / round         | Lower — schema is implicit               | Higher — protocol is in the prompt |
| Robustness             | Provider does parsing for you            | You write the parser     |
| Debuggability          | "Why did it call X?"                     | The Thought line is the answer |
| Fits inside `n8n`, `Airflow`, `cron` | Yes                       | Yes — no different |

Rule of thumb: **prototype with 12b**, **demo and debug with 12c**.
Production systems often run both — 12b for speed, 12c spawned only
when 12b's output is `confidence: low`.

---

## 8. 🧪 Try in class

1. **Drop the `stop` parameter.** Re-run and watch the model produce
   fake `Observation:` lines, then try to act on them. This is the
   single most common ReAct failure mode — students remember it forever
   after seeing it once.
2. **Add a noisy tool.** Wire up `read_file` to return 50 KB of HTML.
   The model usually starts calling `search_usage` *first* once
   `read_file` becomes expensive — that's the right behaviour for
   real RAG-like agents.
3. **Compare per-CVE token cost** between 12b and 12c. The text-ReAct
   protocol consistently uses 20–40 % more tokens. Is the visibility
   worth it for your use case?
4. **Make it loop forever.** Lower `MAX_STEPS = 3`. Watch what the
   forced "out of steps" prompt produces. Most models gracefully emit
   a confidence:"low" verdict — a nice property to verify.

---

## 9. ⚖️ What's still missing

This is a strong, framework-free agent. But for many CVE classes the
model has to *rediscover* the exploit pattern from scratch every run:

> *"What does `yaml.load` look like in user code? Is `yaml.safe_load`
> safe? Does importing pyyaml without calling it count?"*

A human triage engineer answers those in a second from memory. An LLM
without prior knowledge wastes 2–3 steps relearning them. The fix in
**12d** is to give the agent a one-tool RAG capability — `similar_cves`
— backed by a small corpus of hand-written triage notes and NVIDIA's
embedding model.

Continue to **[Lesson 12d — RAG-enhanced triage](./12d_rag_cve_triage.md)**.
