# 🛠️ AI-Powered CVE Triage on Jetson — Part 2: Basic Tool-Calling

**Author:** Dr. Kaikai Liu, Ph.D.
**Institution:** San Jose State University

> **Prerequisite:** [Lesson 12 — Intro](./12_vulnerability_triage_intro.md)
> and a working `python -m pip_audit` against `sample_project/`.
>
> **Companion code:** [`edgeLLM/vuln-triage/triage_basic.py`](../../edgeLLM/vuln-triage/triage_basic.py)
> — every snippet below is an excerpt from that file.

---

## 1. 🎯 What you'll build

> ⚠️ **2026-06 update.** This lesson originally targeted
> `qwen/qwen3-coder-480b-a35b-instruct`, which **reached EOL on
> 2026-06-11**. Run the script with
> `--model minimaxai/minimax-m2.7` (or `z-ai/glm-5.1`) instead — both
> are free-tier-available and OpenAI-tools-compatible. The verified
> output block in §7 is from the original qwen run and is kept for the
> reasoning-pattern walkthrough.

A single Python script that:

1. Runs `pip-audit` against `sample_project/requirements.txt`.
2. For each CVE finding, opens an **OpenAI tool-calling loop** against
   `minimaxai/minimax-m2.7` on NVIDIA Build (or any other model
   listed in [Lesson 11b §11.1](./11b_nextjs_agent_lab.md#111-coding-capable-models-we-tested-or-saw)).
3. Lets the model call four read-only tools to gather evidence:
   - `lookup_cve(cve_id)` — official NVD record.
   - `pip_audit_findings(requirements_path)` — re-run the scanner.
   - `search_usage(pattern, project_dir)` — grep the source tree.
   - `read_file(path, project_dir)` — slice a file.
4. Stops the loop when the model emits a final JSON verdict, and
   pretty-prints it.

No agent framework. ~130 lines of Python plus four ~50-line tool files.

---

## 2. 🔧 Setup

```bash
ssh jetsonorin
source ~/.venv/bin/activate
/usr/bin/python3 -m pip install --target ~/.venv/lib/python3.10/site-packages \
    -r ~/vuln-triage/requirements.txt
export NVIDIA_API_KEY=nvapi-...
cd ~/vuln-triage
```

The toolbox lives in `tools/`:

```
tools/
├── __init__.py
├── cve_lookup.py        # lookup_cve(cve_id)
├── code_search.py       # search_usage + read_file
└── pip_audit_runner.py  # pip_audit_findings
```

Each tool is **plain Python** with a JSON-serialisable return. None of
them know about LLMs.

---

## 3. 🧩 Step 1 — Define tools as OpenAI JSON schemas

NVIDIA Build accepts the same `tools=[...]` schema OpenAI's API expects.
Each tool is a JSON description of the function the model is allowed to
call. The `name` here **must match** the key in our `TOOL_IMPL` dict —
that is how the agent loop dispatches.

```python
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_cve",
            "description": "Fetch the official NVD record for a CVE id and "
                           "return its description, CVSS score, CWE ids, and "
                           "affected packages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cve_id": {
                        "type": "string",
                        "description": "e.g. CVE-2018-18074",
                    }
                },
                "required": ["cve_id"],
            },
        },
    },
    # ... pip_audit_findings, search_usage, read_file ...
]

TOOL_IMPL = {
    "lookup_cve":         lookup_cve,
    "pip_audit_findings": pip_audit_findings,
    "search_usage":       search_usage,
    "read_file":          read_file,
}
```

Key design choices:

- **`description`** is the *only* documentation the model has. Make it
  explicit about when to call this tool ("Use this *after* search_usage
  finds an interesting line"). The model uses these descriptions as a
  decision criterion.
- **`parameters.required`** is small. We let the model omit optional
  args like `is_regex` and `start`.
- **No write tools.** Everything is read-only by construction —
  important for the safety properties of *any* agent that runs against
  source code.

---

## 4. 🧩 Step 2 — Dispatch a tool call safely

The agent loop will eventually receive a `tool_calls` array from the
model. Each entry has a `function.name` and a `function.arguments`
string (JSON). Our job is to:

```python
def _dispatch_tool(name: str, arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": f"bad JSON arguments: {arguments_json!r}"})

    fn = TOOL_IMPL.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        result = fn(**args)
    except TypeError as exc:
        return json.dumps({"error": f"bad args for {name}: {exc}"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return json.dumps(result, default=str)[:6000]
```

Three robustness rules that bit us during development:

1. **Never trust the JSON.** Models sometimes emit trailing commas or
   unquoted keys. Catch `json.JSONDecodeError`, return it as a tool
   observation — the model will retry.
2. **Catch `TypeError` separately.** It is the signature mismatch case
   ("got an unexpected keyword `pat`"). Reporting the actual TypeError
   to the model lets it self-correct on the next turn.
3. **Truncate.** Long observations balloon the prompt for the next
   round. 6 KB is plenty for our triage tools; bigger projects might
   summarise instead.

---

## 5. 🧩 Step 3 — The actual agent loop

This is the heart of the whole lesson. ~50 lines of Python; no agent
classes, no decorators, no callbacks.

```python
def triage_one(client, *, finding, project_dir, requirements_path,
               model, verbose=True) -> Verdict:
    cve_id = finding["primary_cve"]
    package, version = finding["package"], finding["version"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASIC},
        {"role": "user",   "content": f"""
            Triage this finding:
                CVE         : {cve_id}
                Package     : {package}
                Version     : {version}
                Requirements: {requirements_path}
                Project dir : {project_dir}

            Decide whether this codebase actually exposes the vulnerable
            code path. Call tools as needed; reply with the JSON verdict
            when done.

            {VERDICT_SCHEMA_HINT}
        """.strip()},
    ]

    for round_idx in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            tools=TOOL_SCHEMAS, tool_choice="auto",
            temperature=0.1, max_tokens=4096,
        )
        msg = resp.choices[0].message
        messages.append({                                       # ← keep the
            "role": "assistant", "content": msg.content,        #   assistant
            "tool_calls": [                                     #   turn in
                {"id": tc.id, "type": "function",               #   history
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None,
        })

        if not msg.tool_calls:                                  # ← model is
            return parse_verdict(msg.content,                   #   done; we
                                 cve_id=cve_id,                 #   parse its
                                 package=package,               #   final JSON
                                 version=version)

        for tc in msg.tool_calls:                               # ← execute
            result_str = _dispatch_tool(tc.function.name,       #   every tool
                                        tc.function.arguments)  #   it asked
            messages.append({                                   #   for and
                "role": "tool",                                 #   feed each
                "tool_call_id": tc.id,                          #   result
                "name": tc.function.name,                       #   back
                "content": result_str,
            })

    # ... fall through to a forced no-tool final round on budget exhaustion ...
```

**Five things to internalize from those 50 lines:**

| Concept | Where |
|---|---|
| `tool_choice="auto"` lets the model pick *whether* to call a tool, or just return text. | Line with `tool_choice` |
| The whole conversation is one growing `messages` list. The model never has memory across `.create()` calls — *we* provide it. | Appends |
| Tool replies use the special role `"tool"` and must carry the matching `tool_call_id`. | Inner loop |
| The exit condition is `not msg.tool_calls` — the model stops calling tools when it has its answer. | `if not msg.tool_calls` |
| `MAX_TOOL_ROUNDS` is a hard ceiling so a confused model can't burn quota forever. | `for round_idx in range(...)` |

That last point is non-negotiable. A two-line guard prevents a runaway
loop from costing you $50 in NVIDIA Build credits.

---

## 6. 🧩 Step 4 — Parse the verdict

The system prompt instructs the model to return a JSON object as its
final assistant message. Real chat models sometimes wrap it in
triple-back-tick fences anyway. We extract defensively:

```python
def extract_json_block(text: str) -> dict:
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"no JSON found in:\n{text[:400]}")
```

This is forgiving on purpose — the alternative is to error out when the
model adds one stray line of commentary, which would be a frustrating
classroom experience.

---

## 7. ▶️ Run it

```bash
cd ~/vuln-triage
python3 triage_basic.py --project sample_project --limit 2
```

Verified output on Jetson Orin Nano while writing this lesson
(`qwen/qwen3-coder-480b-a35b-instruct`, two CVEs against `requests`):

```text
⚙  project        : /home/cmpe/vuln-triage/sample_project
⚙  requirements   : /home/cmpe/vuln-triage/sample_project/requirements.txt
⚙  model          : qwen/qwen3-coder-480b-a35b-instruct

→ running pip-audit …
  pip-audit reported 33 finding(s).

[1/2] requests 2.19.1 — CVE-2018-18074
  · round 1: calling model … 2.8s  (tokens: prompt=1021, completion=36)
      → lookup_cve({"cve_id":"CVE-2018-18074"})
  · round 2: calling model … 2.5s  (tokens: prompt=1417, completion=38)
      → pip_audit_findings({"requirements_path":".../sample_project/requirements.txt"})
  · round 3: calling model … 20.3s  (tokens: prompt=3012, completion=42)
      → search_usage({"project_dir":".../sample_project","pattern":"requests"})
  · round 4: calling model … 50.6s  (tokens: prompt=3471, completion=62)
      → read_file({"end":38,"project_dir":".../sample_project","start":23,"path":"app.py"})
  · round 5: calling model … 14.1s  (tokens: prompt=3824, completion=114)

┌─ CVE-2018-18074  [requests 2.19.1]
│  verdict   : EXPLOITABLE HERE   (confidence: high)
│  reason    : The codebase directly uses the `requests` library in `app.py` (line
│              37) to make HTTP requests with caller-supplied URLs,
│              which matches the vulnerable code path described in
│              CVE-2018-18074 …
│  action    : Upgrade the `requests` package to version 2.20.0 or later
└──
```

Five rounds, 11–13 k tokens total, ~90 seconds wall time. The agent
took *exactly* the path a human analyst would: read the CVE → look at
what other findings live in this project → grep for usage → read the
exact lines → conclude.

---

## 8. 🧪 Try in class

1. **Catch the agent being lazy.** Re-run with
   `--cve CVE-2019-10906` (the jinja2 sandbox issue). Does the model
   bother to *read* the actual template definition in `app.py`, or does
   it just trust the package name? The 12c lesson will show the same
   case under a more visible loop.
2. **Swap the coding model.** Try `--model "z-ai/glm-5.1"`,
   `--model "deepseek-ai/deepseek-v4-pro"`, or — if you have the keys
   — `--model "claude-sonnet-4-6"` / `--model "gpt-4o-mini"`. Compare
   token usage, latency, and the quality of the justification text.
   The [Lesson 11b §11 status table](./11b_nextjs_agent_lab.md#111-coding-capable-models-we-tested-or-saw)
   notes which are currently slow or EOL'd.
3. **Inject a wrong package.** Edit `app.py` to add
   `import yaml; yaml.load(open("x.yaml"))`. Re-run with
   `--cve CVE-2020-1747`. The agent should flip to **Exploitable here**.
4. **Read the messages array.** Add `print(json.dumps(messages,
   indent=2))` just before each `.create()` call. You'll see exactly
   how the conversation grows turn by turn — this is *the* mental
   model you need for agent debugging.

---

## 9. ⚖️ Limitations of this design

This 130-line agent is great, but it leaves three sharp edges:

| Limitation | What it means | Fixed in |
|---|---|---|
| Only works on providers that support OpenAI `tools` schema. | A self-hosted vLLM serving a base model won't accept it. | **12c** uses the text-protocol ReAct alternative. |
| Hides the model's reasoning behind opaque `tool_calls`. | When the agent picks a weird tool, you can only see *what* it called, not *why*. | **12c** shows the chain of thought line by line. |
| No mechanism for prior knowledge. | The model re-discovers "yaml.load is the danger pattern" every single run. | **12d** adds an embedding-search tool over a small corpus of triage notes. |

Continue to **[Lesson 12c — ReAct without a framework](./12c_react_loop_triage.md)**.
