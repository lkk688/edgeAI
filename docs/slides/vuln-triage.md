---
marp: true
paginate: true
size: 16:9
title: AI CVE Triage on Jetson — Part 1
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
table { font-size:.82em; border-collapse:collapse; } th { background:var(--blue); color:#fff; } td,th { border:1px solid #d4dce8; padding:5px 10px; }
.step { background:var(--blue); color:#fff; border-radius:999px; padding:.03em .6em; font-weight:700; font-size:.85em; }
.tiny { font-size:.78em; color:#5d6b82; }
.cols { display:flex; gap:26px; align-items:flex-start; } .cols > * { flex:1; }
section.lead { text-align:center; border-top-width:10px; }
section.lead h1 { font-size:2.2em; }
</style>

<!-- _class: lead -->
# 🛡️ AI-Powered CVE Triage
### on the Jetson Orin Nano — Part 1: The Idea

`SJSU · Edge AI · Cyber-AI`

<span class="tiny">An LLM + a few tools that decides which scanner findings are *actually* exploitable.</span>

---

## <span class="step">1</span> Finding a CVE ≠ triaging it

A scanner (`pip-audit`) cross-checks your `requirements.txt` against the CVE database:

```text
$ pip-audit -r requirements.txt
Found 33 known vulnerabilities in 4 packages.
  requests 2.19.1  CVE-2018-18074  leak Proxy-Authorization on redirect
  jinja2   2.10    CVE-2019-10906  str.format_map sandbox escape
  pyyaml   5.3     CVE-2020-1747   yaml.load arbitrary code execution  ...
```

Every line is *technically* true. The real question for the engineer is:

> **"Is this CVE actually reachable from *our* code — or a false alarm?"**

---

## <span class="step">2</span> Three buckets, one hard decision

| Bucket | Meaning | Cost of error |
|---|---|---|
| **Exploitable here** | Patch now — real bug. | High if missed |
| **Not exploitable** | Suppress / defer to next upgrade. | High if misclassified |
| **Inconclusive** | Needs a human. | Bounded |

Humans triage ~30 findings/hour. We compress NVIDIA's production
[vulnerability-analysis blueprint](https://github.com/NVIDIA-AI-Blueprints/vulnerability-analysis)
into **~600 lines of single-file Python** on one Jetson.

---

## <span class="step">3</span> Why an LLM (not a regex)

The scanner knows the **facts** (version X has CVE Y). It can't answer the **semantic** question:
*does our code even call the vulnerable function, with attacker-controlled input?*

A coding LLM (`qwen/qwen3-coder-480b`) on NVIDIA Build can:

1. **Read** the CVE → identify the vulnerable pattern (`yaml.load(untrusted)`).
2. **Search** our codebase for that pattern (a tool we hand it).
3. **Reason** about context — is the input really attacker-controlled?
4. **Emit** a JSON verdict downstream CI can ingest.

> The model **never sees the whole codebase** — it pulls only the bytes it needs via tools.

---

## <span class="step">4</span> The architecture

<div class="cols">
<div>

```text
   NVIDIA Build (cloud LLM)
   qwen3-coder-480b + nv-embedqa
            ▲  OpenAI-compatible
            │  /chat/completions
 ┌──────────┴───────────┐
 │  Jetson Orin Nano    │
 │  triage_basic .py     │ 12b
 │  triage_react .py     │ 12c
 │  triage_rag   .py     │ 12d
 └──────────────────────┘
   tools: lookup_cve · search_usage
          read_file · similar_cves
   pip_audit_findings → pip-audit
```

</div>
<div>

**3 entrypoints · 4 tools · 1 sample project.**

We cut from the blueprint:
- Morpheus pipeline → one `for` loop
- LangGraph → OpenAI tool-calling
- Triton → NVIDIA Build endpoints
- Milvus → in-memory cosine over ~12 rows
- Docker/Helm → `python triage_basic.py`

**Kept:** an LLM with tools, classifying each finding.

</div>
</div>

---

## <span class="step">5</span> The sample project — a triage puzzle

`app.py` deliberately exercises three distinct shapes:

```python
import jinja2, requests
_STATUS_TEMPLATE = jinja2.Template("Status for {{ url }}: {{ status }}")  # constant!

def fetch_status(url: str) -> dict:           # requests: caller-supplied URL
    response = requests.get(url, timeout=5)    # → vulnerable path REACHABLE
    return {"status": response.status_code, "length": len(response.content)}

def render_status(url, status):               # jinja2: only a fixed template
    return _STATUS_TEMPLATE.render(url=url, status=status)   # → NOT reachable
# pyyaml: in requirements.txt, never imported  → dead weight
```

| Package | Used? | Expected verdict |
|---|---|---|
| `requests` | ✅ caller-supplied URL | **Exploitable** |
| `jinja2` | ✅ hard-coded template | **Not exploitable** |
| `pyyaml` | ❌ never imported | **Not exploitable** (dead weight) |

---

## <span class="step">6</span> Run it — inside the container

```bash
sjsujetsontool shell                       # Jetson AI container; brings in ~/.env.local (NVIDIA_API_KEY)
cd /Developer/edgeAI/edgeLLM/vuln-triage
pip install -r requirements.txt            # openai · httpx · pip-audit
```

Prove the **scanner half** works before adding any LLM:

```bash
python3 -m pip_audit -r sample_project/requirements.txt --format json --no-deps \
  | jq '.dependencies[] | {name, version, n: (.vulns|length)}'
```

```json
{"name": "requests", "version": "2.19.1", "n": 6}
{"name": "jinja2",   "version": "2.10",   "n": 6}
{"name": "pyyaml",   "version": "5.3",    "n": 4}
```

---

## <span class="step">7</span> The agent loop (lesson 12b)

Hand the model **four tools** as OpenAI JSON schemas, then loop:

```python
TOOL_SCHEMAS = [ lookup_cve, pip_audit_findings, search_usage, read_file ]  # JSON schemas

for round in range(MAX_TOOL_ROUNDS):                 # ~6
    resp = client.chat.completions.create(
        model=model, messages=messages,
        tools=TOOL_SCHEMAS, tool_choice="auto")       # model picks a tool
    msg = resp.choices[0].message
    if not msg.tool_calls:                            # no tool → it's the verdict
        return parse_verdict(msg.content)
    for tc in msg.tool_calls:                         # run each tool, feed result back
        result = TOOL_IMPL[tc.function.name](**json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

<span class="tiny">One OpenAI tool-calling loop, zero frameworks. 12c rewrites it as a manual ReAct loop; 12d adds retrieval.</span>

---

## <span class="step">8</span> The four-part series

| Part | What you learn | Code |
|---|---|---|
| **12** (here) | Problem · sample data · architecture | — |
| **12b** | Single-turn OpenAI tool-calling | `triage_basic.py` |
| **12c** | Manual ReAct loop (any chat model) | `triage_react.py` |
| **12d** | ReAct + embedding retrieval (agentic RAG) | `triage_rag.py` |

<!-- _class: lead -->

Full lesson → [lkk688.github.io/edgeAI/curriculum/12_vulnerability_triage_intro](https://lkk688.github.io/edgeAI/curriculum/12_vulnerability_triage_intro/)
