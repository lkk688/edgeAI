# vuln-triage — AI-powered CVE triage for the Jetson Orin Nano

A drastically simplified take on
[NVIDIA's vulnerability-analysis blueprint](https://github.com/NVIDIA-AI-Blueprints/vulnerability-analysis),
shrunk to run from a single Jetson with one Python process and one
NVIDIA Build API key.

The companion lessons are [`12_*`](../../docs/curriculum/) — four
markdown files that build up to the same three Python scripts here:

| Tutorial | What it teaches | Script |
|---|---|---|
| [`12_vulnerability_triage_intro.md`](../../docs/curriculum/12_vulnerability_triage_intro.md) | Why scanner output needs a human (or an LLM) in the loop; how the sample project is wired up | — |
| [`12b_basic_tool_calling_triage.md`](../../docs/curriculum/12b_basic_tool_calling_triage.md) | Single-turn OpenAI tool-calling against a coding model | [`triage_basic.py`](triage_basic.py) |
| [`12c_react_loop_triage.md`](../../docs/curriculum/12c_react_loop_triage.md) | Manual ReAct loop with a regex parser (zero framework) | [`triage_react.py`](triage_react.py) |
| [`12d_rag_cve_triage.md`](../../docs/curriculum/12d_rag_cve_triage.md) | Adds `similar_cves` as one more tool, backed by NVIDIA embeddings | [`triage_rag.py`](triage_rag.py) |

## Layout

```
vuln-triage/
├── requirements.txt          # openai, httpx, pip-audit
├── .env.example
├── shared.py                 # OpenAI client, Verdict dataclass, prompts
├── tools/
│   ├── cve_lookup.py         # NVD JSON lookup (disk-cached)
│   ├── code_search.py        # grep + read_file, project-rooted
│   ├── pip_audit_runner.py   # subprocess pip-audit + JSON parse
│   └── embedding_search.py   # nv-embedqa-e5-v5 + cosine search
├── triage_basic.py           # §12b — OpenAI tool-calling loop
├── triage_react.py           # §12c — manual ReAct
├── triage_rag.py             # §12d — ReAct + retrieval tool
├── sample_project/
│   ├── requirements.txt      # intentionally vulnerable deps
│   ├── app.py                # uses some of them, not others
│   └── README.md
└── examples/
    └── cve_corpus.jsonl      # ~12 hand-written triage notes for RAG
```

## Quick start

```bash
ssh jetsonorin
source ~/.venv/bin/activate

# Install once (the venv has no own pip; use /usr/bin/python3 -m pip):
/usr/bin/python3 -m pip install \
    --target ~/.venv/lib/python3.10/site-packages \
    -r vuln-triage/requirements.txt

export NVIDIA_API_KEY=nvapi-...

cd vuln-triage
python triage_basic.py --project sample_project --limit 2     # §12b
python triage_react.py --project sample_project --limit 2     # §12c
python triage_rag.py   --project sample_project --limit 2     # §12d
```

Each script prints a per-CVE verdict block and a summary line.
