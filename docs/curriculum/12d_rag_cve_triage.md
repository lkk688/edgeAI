# 🔎 AI-Powered CVE Triage on Jetson — Part 4: Embedding-Based RAG

**Author:** Dr. Kaikai Liu, Ph.D.
**Institution:** San Jose State University

> **Prerequisite:** [Lesson 12c](./12c_react_loop_triage.md). You should
> have the ReAct loop working against `triage_react.py`.
>
> **Companion code:**
> [`edgeLLM/vuln-triage/triage_rag.py`](../../edgeLLM/vuln-triage/triage_rag.py)
> · [`tools/embedding_search.py`](../../edgeLLM/vuln-triage/tools/embedding_search.py)
> · [`examples/cve_corpus.jsonl`](../../edgeLLM/vuln-triage/examples/cve_corpus.jsonl)

---

## 1. 🎯 What you'll build

The ReAct triage from lesson 12c, plus **one extra tool**:

```python
similar_cves(query: str, k: int = 3) -> dict
```

It performs cosine-similarity search over a tiny in-process corpus of
hand-written **triage notes** — one row per CVE class, with the *code
patterns* and the *one-paragraph rule of thumb* a human analyst would
have memorised. The embedding model is
[`nvidia/nv-embedqa-e5-v5`](https://build.nvidia.com) — the same one we
used in the Next.js Retrieval Lab in [Lesson 11 §7](./11_nextjs_nemotron_app.md).

The point: **retrieval is just one more tool in the loop, not a
separate pipeline**. The model decides when (and whether) to use it.
This is the simplest possible *agentic RAG*.

---

## 2. 🧠 Why a triage corpus (and not the full CVE database)?

A "normal" RAG system embeds the entire knowledge base — Wikipedia,
your docs, the NVD JSON — and retrieves the top-k chunks. For
vulnerability triage that is *wasteful*:

- The CVE description is already provided to the model from
  `lookup_cve`. We don't need to retrieve it.
- What the model **does** need is the *operational* knowledge a senior
  analyst would carry — the kind of advice you'd find in an internal
  wiki:

> *"`pyyaml` CVEs are exploitable only when the program actually calls
> `yaml.load` or `yaml.full_load` on untrusted input. Code that imports
> pyyaml transitively but never invokes its loaders is **not exposed**."*

We hand-write ~12 of these notes in
[`examples/cve_corpus.jsonl`](../../edgeLLM/vuln-triage/examples/cve_corpus.jsonl):

```json
{"cve_id": "CVE-2020-1747", "package": "pyyaml",
 "summary":  "PyYAML yaml.load(..., Loader=FullLoader) allows arbitrary "
             "Python object construction in versions <5.3.1",
 "patterns": ["yaml.load", "yaml.FullLoader", "yaml.unsafe_load"],
 "guidance": "Exploitable only if the program actually calls yaml.load "
             "on untrusted input. Projects that import pyyaml "
             "transitively but never call its loaders are not exposed. "
             "Move to yaml.safe_load regardless."}
```

`patterns` is the *what to grep for*; `guidance` is the *what to
conclude*. Both will end up inside the model's prompt after retrieval.

Twelve rows might sound tiny — but the cosine search is **per query**,
not per row. The agent issues *one* `similar_cves` call per CVE, and
the top-3 matches are typically all the prior knowledge it needs.

---

## 3. 🧩 Step 1 — Embed the corpus once, retrieve forever

[`tools/embedding_search.py`](../../edgeLLM/vuln-triage/tools/embedding_search.py)
is ~90 lines. The interesting parts:

```python
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"        # 1024-dim, asymmetric
_INDEX = None                                  # module-level cache

def _embed_batch(texts, *, input_type):
    """One POST to /v1/embeddings — both `query` and `passage` modes."""
    with httpx.Client(timeout=60.0, trust_env=False) as client:
        resp = client.post(
            f"{NVIDIA_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type":  "application/json"},
            json={"model": EMBED_MODEL,
                  "input": texts,
                  "input_type": input_type},
        )
        resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]

def _build_index(corpus_path):
    rows  = _load_corpus(corpus_path)
    texts = [f"{r['package']}: {r['summary']}. Patterns: "
             f"{', '.join(r.get('patterns', []))}" for r in rows]
    # Asymmetric: corpus = "passage", queries will use "query".
    vecs  = _embed_batch(texts, input_type="passage")
    return [dict(r, _vector=v, _text=t)
            for r, v, t in zip(rows, vecs, texts)]
```

`nv-embedqa-e5-v5` is an **asymmetric** embedding model — corpus
documents must be embedded with `input_type="passage"`, queries with
`input_type="query"`. Mix them up and the cosine scores collapse. The
Next.js Retrieval Lab burned the same point home in Lesson 11.

The index is cached in a module-level global so we only POST to
`/v1/embeddings` *once* per process — building the full 12-row index
costs a single round trip to NVIDIA Build.

```python
def similar_cves(query: str, *, k: int = 3, corpus_path: str | None = None):
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index(Path(corpus_path or DEFAULT))

    q_vec  = _embed_batch([query], input_type="query")[0]
    scored = [(_cosine(q_vec, row["_vector"]), row) for row in _INDEX]
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"query": query, "k": k,
            "matches": [{"cve_id":   r["cve_id"],
                         "package":  r["package"],
                         "summary":  r["summary"],
                         "patterns": r.get("patterns", []),
                         "guidance": r.get("guidance", ""),
                         "score":    round(s, 3)}
                        for s, r in scored[:k]]}
```

`_cosine` is six lines of stdlib math. No FAISS, no Chroma, no
Pinecone — the corpus is tiny enough that the entire similarity scan
runs in microseconds. **In a real deployment** you'd swap `_INDEX +
_cosine` for a proper vector DB; the rest of the file does not change.

---

## 4. 🧩 Step 2 — Register the tool with the agent

Inside [`triage_rag.py`](../../edgeLLM/vuln-triage/triage_rag.py) the
tool table grows by exactly one entry compared to 12c:

```python
TOOLS = {
    "lookup_cve":         lookup_cve,
    "pip_audit_findings": pip_audit_findings,
    "search_usage":       search_usage,
    "read_file":          read_file,
    "similar_cves":       similar_cves,        # ← new
}
```

…and the system prompt mentions it *first*, with an explicit hint that
the agent should call it **before** anything else:

```python
Available tools:
  similar_cves(query, k=3)
      → Retrieve up to k semantically similar CVE notes from the
        internal corpus. Each note carries a `patterns` array (code
        signatures that make the CVE reachable) and `guidance`
        (a one-paragraph triage rule of thumb). Call this FIRST so
        you know what to grep for.
  lookup_cve(cve_id)
      → Authoritative NVD record (description, CVSS, CWE, affected
        versions).
  pip_audit_findings(requirements_path)
  search_usage(pattern, project_dir, is_regex=false)
  read_file(path, project_dir, start=1, end=null)
```

Notice the wording: *"Call this FIRST so you know what to grep for."*
That single line changes the agent's behaviour materially — it switches
from "blindly search for the package name" to "first ask: what's the
canonical danger pattern for this kind of CVE?"

---

## 5. ▶️ Run it

```bash
cd ~/vuln-triage
python3 triage_rag.py --project sample_project --cve CVE-2020-1747
```

Verified output on Jetson Orin Nano. CVE-2020-1747 is the
"declared-but-never-imported" pyyaml case — the agent must recognize
that the *package* is in `requirements.txt` but no *code* invokes its
vulnerable APIs:

```text
⚙  building CVE embedding index (one call) …
   index ready.

[1/1] pyyaml 5.3 — CVE-2020-1747
  ─ step 1: calling model … 4.0s  (p=594 c=82)
      │ Thought: I need to triage CVE-2020-1747 related to pyyaml version 5.3. I will first
      │ look for similar CVEs in the corpus to understand the pattern and guidance, then
      │ check the project for actual usage.
      │ Action: similar_cves({"query": "CVE-2020-1747", "k": 3})
      │ Observation: {"matches":[{"cve_id":"CVE-2020-14343","patterns":["yaml.full_load","yaml.FullLoader","yaml.load"],"guidance":"Same shape as CVE-2020-1747: requires .load() with the unsafe loader on attacker-controlled YAML…"},…
  ─ step 2: calling model … 3.7s  (p=1081 c=111)
      │ Thought: I found that CVE-2020-1747 is related to PyYAML's unsafe loading … The similar
      │ CVEs show that this vulnerability typically involves yaml.load() with unsafe loaders.
      │ Action: pip_audit_findings({"requirements_path": ".../sample_project/requirements.txt"})
      │ Observation: {…pyyaml 5.3 → CVE-2020-1747…}
  ─ step 3: calling model … 9.1s
      │ Thought: Let me check for the vulnerable patterns directly.
      │ Action: search_usage({"pattern": "yaml.load", "project_dir": ".../sample_project"})
      │ Observation: {"pattern": "yaml.load", "hits": [], "truncated": false}
  ─ step 4: calling model … 3.2s
      │ Thought: I don't see any usage of yaml.load. Let me check for other vulnerable patterns.
      │ Action: search_usage({"pattern": "yaml.full_load", "project_dir": ".../sample_project"})
      │ Observation: {"pattern": "yaml.full_load", "hits": [], "truncated": false}
  ─ step 5: calling model … 5.9s
      │ Thought: Let me check if pyyaml is even imported.
      │ Action: search_usage({"pattern": "import", "project_dir": ".../sample_project", "is_regex": "true"})
      │ Observation: {"hits": [{"file": "app.py", "snippet": "- `jinja2` is imported, but the template string is …"},…
  ─ step 6: calling model … 9.6s
      │ Thought: I have enough evidence to make a determination. …
      │ Action: lookup_cve({"cve_id": "CVE-2020-1747"})
      │ Observation: {"description": "A vulnerability was discovered in the PyYAML library …
  ─ step 7: calling model … 9.2s
      │ Thought: I have confirmed that CVE-2020-1747 affects PyYAML versions before 5.3.1, …
      │ However, in this project, pyyaml is listed in requirements.txt but is never actually
      │ imported or used. The project does not contain any usage of yaml.load or yaml.full_load.
      │ Final Answer: {"exploitable_here": false, "confidence": "high",
      │                "justification": "The project's requirements.txt lists pyyaml 5.3 but
      │                the package is never imported or used in the codebase…",
      │                "recommended_action": "Remove unused pyyaml dependency from requirements.txt
      │                to reduce attack surface."}

┌─ CVE-2020-1747  [pyyaml 5.3]
│  verdict   : NOT EXPLOITABLE HERE   (confidence: high)
│  reason    : The project's requirements.txt lists pyyaml 5.3 but the package is
│              never imported or used in the codebase. Code searches show no
│              usage of vulnerable functions like yaml.load or yaml.full_load.
│  action    : Remove unused pyyaml dependency from requirements.txt to reduce
│              attack surface.
└──
```

**Notice how step 1's observation changed step 3's behaviour.** The
agent searched for `yaml.load` and `yaml.full_load` *specifically* —
not just "pyyaml". That precision came from the corpus's `patterns`
array. Without the retrieval step, the coder model would usually search for
the package name first and waste an extra round figuring out which
specific function it should grep for.

The final recommendation — *"Remove unused pyyaml dependency"* — is the
exact action a security engineer would write in the JIRA ticket. That's
not in any prompt; it's emergent from the corpus guidance ("Move to
`yaml.safe_load` regardless") combined with the evidence the agent
itself gathered.

---

## 6. 🤔 Pattern: retrieval as *one of many* tools

This is structurally different from the "classic" RAG pipeline:

| Classic RAG                          | Agentic RAG (this lab)              |
|--------------------------------------|--------------------------------------|
| Retrieve → stuff into context → LLM. | LLM decides *if* and *what* to retrieve. |
| Retrieval happens always.            | Retrieval is one tool among many.    |
| Caller controls the embedding query. | The model writes its own query.      |
| Fixed top-k stuffed into the prompt. | Model can pull multiple k=3 batches if needed. |

The agentic variant adapts to the **complexity of the question**.
For a clear-cut CVE ("requests is called directly, exploitable"), the
agent often *doesn't* bother retrieving — it skips straight to
`search_usage`. For a fuzzy class like pyyaml or jinja2, it leans on
the corpus heavily. We pay only for the calls we need.

---

## 7. 🧪 Try in class

1. **Add a new triage note.** Append a JSON line for `CVE-2024-3651`
   (idna ReDoS) to `cve_corpus.jsonl` with a tight `patterns` and
   `guidance`. Re-run the agent on `--cve CVE-2024-3651` and watch the
   first step pull *your* note into context.
2. **Compare with retrieval disabled.** Run the same case through
   `triage_react.py` (lesson 12c). Count the steps. Did the model
   eventually find the right patterns on its own? How much later?
3. **Shrink the corpus to one row.** Delete every note except for
   pyyaml. Now run on a `requests` CVE. The retrieval will return
   irrelevant guidance — does the agent ignore it, or does it derail?
4. **Switch the embedding model.** Set
   `TRIAGE_EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2`. Compare top-3
   ordering — embedding choice affects retrieval quality at the same
   cosine threshold.
5. **Wire the verdicts into CI.** Pipe `triage_rag.py --quiet` JSON
   output into a GitHub Actions job that fails the build only when
   `exploitable_here: true && confidence: high`. The whole point of a
   triage agent is to *be* the noise filter.

---

## 8. 📚 What you can build next

You now have the three core agent patterns in your pocket:

1. **Provider tool-calling** (12b) — fastest path on supported APIs.
2. **Text ReAct** (12c) — portable to any chat endpoint.
3. **Agentic RAG** (12d) — adds memory without a vector DB.

The same three patterns build basically every other tool-using LLM
application: a code review bot, a deploy assistant, a customer-support
agent, an oncall summarizer. Swap the tools, keep the loop.

**Where to go next:**

- 🤖 **A full CI job.** Replace `triage_rag.py`'s pretty-printed output
  with strict JSON, then call it from a `.github/workflows/triage.yml`
  on every dependency PR. Block merges where the verdict is
  `exploitable_here: true && confidence: high`.
- 🛡️ **Defender-in-the-loop.** Add a `propose_patch(package, pin)` tool
  that drafts the `requirements.txt` diff. The model becomes a
  one-loop dependency-update bot.
- 🧠 **Self-distillation.** Save every (CVE, verdict, evidence) tuple
  the agent produces. After a hundred runs you have a labelled
  triage dataset — fine-tune a smaller local model on it and serve from
  `ollama` on the Jetson.
- 📦 **Beyond Python.** Swap `pip-audit` for `npm audit`, `cargo audit`,
  or `osv-scanner`. The shape of the agent doesn't change at all —
  only the toolbox.

---

**Source folder:** [`edgeLLM/vuln-triage/`](../../edgeLLM/vuln-triage/)
**Tested on:** Jetson Orin Nano (Ubuntu 22.04, aarch64) with Python 3.10,
`openai 2.37.0`, `httpx 0.28.1`, `pip-audit 2.10.0`. Original verified
run used `qwen/qwen3-coder-480b-a35b-instruct` (chat) +
`nvidia/nv-embedqa-e5-v5` (embeddings); since the qwen model reached
EOL on 2026-06-11, current recommended defaults are
**`minimaxai/minimax-m2.7`** or **`z-ai/glm-5.1`**. The embedding model
is unaffected.
