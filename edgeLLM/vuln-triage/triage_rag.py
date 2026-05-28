"""triage_rag.py — vulnerability triage with **embedding-based retrieval**.

This is the same ReAct loop as `triage_react.py`, plus one extra tool:

    similar_cves(query, k=3)

The model can use it to ground its reasoning in a small corpus of past
triage notes (see `examples/cve_corpus.jsonl`). It is the simplest
possible "agentic RAG" pattern — retrieval is *one tool among many*, not
a hard-coded preprocessing step.

The corpus is intentionally tiny (≈12 hand-curated notes) so it fits
in memory and embeds in a single NVIDIA-Build call. In a real
deployment you would index thousands of CVEs and use a proper vector
DB; the tutorial uses cosine similarity in pure Python to keep the
moving parts visible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (  # noqa: E402
    DEFAULT_CODER_MODEL,
    VERDICT_SCHEMA_HINT,
    Verdict,
    make_client,
    parse_action_args,
    parse_verdict,
)
from tools import (  # noqa: E402
    lookup_cve,
    pip_audit_findings,
    read_file,
    search_usage,
)
from tools.embedding_search import similar_cves  # noqa: E402


TOOLS = {
    "lookup_cve":         lookup_cve,
    "pip_audit_findings": pip_audit_findings,
    "search_usage":       search_usage,
    "read_file":          read_file,
    "similar_cves":       similar_cves,
}

MAX_STEPS = 8


REACT_SYSTEM = textwrap.dedent(
    """
    You are a security analyst triaging one CVE against a small Python
    project on an NVIDIA Jetson edge device. You work in a strict
    ReAct loop:

        Thought: <short reasoning>
        Action: <tool_name>({"arg": "value", ...})

    After your Action you will receive:

        Observation: <JSON tool result>

    When you have enough evidence, reply with:

        Thought: <final reasoning>
        Final Answer: <single-line JSON>

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

    Examples of valid Action lines (use the JSON-object form):
      Action: similar_cves({"query": "yaml load arbitrary code", "k": 3})
      Action: lookup_cve({"cve_id": "CVE-2020-1747"})
      Action: search_usage({"pattern": "yaml.load", "project_dir": "/path/to/proj"})

    Rules:
      - One Action per turn. Arguments are JSON on the Action line.
      - Cite at least one search_usage or read_file observation in your
        Final Answer's `justification` field.
      - Final Answer is the ONLY line allowed to contain raw JSON braces
        outside of Action arguments.
    """
).strip()


RE_ACTION  = re.compile(r"^\s*\**\s*Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
RE_FINAL   = re.compile(r"^\s*\**\s*Final Answer\s*:\s*(.+)$", re.IGNORECASE)
RE_THOUGHT = re.compile(r"^\s*\**\s*Thought\s*:\s*(.+)$",      re.IGNORECASE)


def _parse_step(text: str):
    thought = action = args_str = final = None
    for line in text.splitlines():
        if m := RE_THOUGHT.match(line):
            thought = m.group(1).strip()
        elif m := RE_ACTION.match(line):
            action = m.group(1).strip()
            args_str = m.group(2).strip()
        elif m := RE_FINAL.match(line):
            final = m.group(1).strip()
    return thought, action, args_str, final


def _run_tool(name: str, args_str: str) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        args = parse_action_args(args_str, fn)
    except ValueError as exc:
        return json.dumps({"error": f"bad arguments: {exc}"})
    try:
        result = fn(**args)
    except TypeError as exc:
        return json.dumps({"error": f"bad signature for {name}: {exc}"})
    except Exception as exc:  # pylint: disable=broad-except
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    out = json.dumps(result, default=str)
    return out[:4500] + ("  …[truncated]" if len(out) > 4500 else "")


def rag_triage(
    client,
    *,
    finding: dict,
    project_dir: str,
    requirements_path: str,
    model: str,
    verbose: bool = True,
) -> Verdict:
    cve_id = finding.get("primary_cve") or (finding.get("cve_ids") or ["UNKNOWN"])[0]
    package = finding["package"]
    version = finding["version"]

    initial = textwrap.dedent(
        f"""
        Triage one finding:
            CVE         : {cve_id}
            Package     : {package}
            Version     : {version}
            Requirements: {requirements_path}
            Project dir : {project_dir}

        Start with similar_cves to retrieve corpus guidance, then verify
        against the actual code. {VERDICT_SCHEMA_HINT}
        """
    ).strip()

    messages: list[dict] = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": initial},
    ]

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"  ─ step {step}: calling model …", end="", flush=True)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stop=["\nObservation:", "Observation:"],
            temperature=0.1,
            max_tokens=1024,
        )
        dt = time.perf_counter() - t0
        reply = (resp.choices[0].message.content or "").strip()
        if verbose:
            print(f" {dt:.1f}s  (p={resp.usage.prompt_tokens} c={resp.usage.completion_tokens})")
            for ln in reply.splitlines():
                print(f"      │ {ln}")
        messages.append({"role": "assistant", "content": reply})

        _, action, args_str, final = _parse_step(reply)
        if final is not None:
            return parse_verdict(
                final, cve_id=cve_id, package=package, version=version
            )
        if action is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "I did not see an Action line. Continue with "
                        "`Action: tool(...)` or finish with `Final Answer: <json>`."
                    ),
                }
            )
            continue
        observation = _run_tool(action, args_str or "{}")
        if verbose:
            short = observation[:140].replace("\n", " ")
            print(f"      │ Observation: {short}{' …' if len(observation) > 140 else ''}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    messages.append(
        {
            "role": "user",
            "content": (
                "Out of steps. Reply with ONLY `Final Answer: <json>`. "
                f"{VERDICT_SCHEMA_HINT}"
            ),
        }
    )
    forced = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=512
    )
    text = forced.choices[0].message.content or ""
    _, _, _, final = _parse_step(text)
    return parse_verdict(final or text, cve_id=cve_id, package=package, version=version)


def main() -> int:
    p = argparse.ArgumentParser(
        description="ReAct triage with embedding-based corpus retrieval."
    )
    p.add_argument("--project", default="./sample_project")
    p.add_argument("--requirements", default=None)
    p.add_argument("--model", default=DEFAULT_CODER_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cve", action="append", default=[])
    p.add_argument("--package", action="append", default=[])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    project_dir = os.path.abspath(args.project)
    requirements_path = args.requirements or os.path.join(project_dir, "requirements.txt")
    if not os.path.isfile(requirements_path):
        print(f"[error] {requirements_path}: no such file", file=sys.stderr)
        return 2

    print(f"⚙  project        : {project_dir}")
    print(f"⚙  requirements   : {requirements_path}")
    print(f"⚙  model          : {args.model}  (ReAct + RAG mode)")
    print("⚙  building CVE embedding index (one call) …")

    # Warm up the embedding index so the first step's timing is honest.
    similar_cves("warmup", k=1)
    print("   index ready.\n")

    audit = pip_audit_findings(requirements_path)
    findings = audit["findings"]
    if not findings:
        print("no findings — nothing to triage.")
        return 0
    cve_filter = {c.upper() for c in (args.cve or [])}
    pkg_filter = {p.lower() for p in (args.package or [])}
    seen = set()
    queue: list[dict] = []
    for f in findings:
        key = (f["package"], f.get("primary_cve"))
        if key in seen:
            continue
        seen.add(key)
        if cve_filter and str(f.get("primary_cve") or "").upper() not in cve_filter:
            continue
        if pkg_filter and f["package"].lower() not in pkg_filter:
            continue
        queue.append(f)
    if args.limit > 0:
        queue = queue[: args.limit]

    client = make_client()
    verdicts: list[Verdict] = []
    for i, f in enumerate(queue, 1):
        print(f"\n[{i}/{len(queue)}] {f['package']} {f['version']} — {f.get('primary_cve')}")
        v = rag_triage(
            client,
            finding=f,
            project_dir=project_dir,
            requirements_path=requirements_path,
            model=args.model,
            verbose=not args.quiet,
        )
        verdicts.append(v)
        print(v.pretty())

    print("\n=== SUMMARY ===")
    for v in verdicts:
        flag = (
            "EXPLOITABLE" if v.exploitable_here
            else "ok"       if v.exploitable_here is False
            else "??"
        )
        print(f"  {flag:11}  {v.cve_id}  {v.package} {v.version}  [{v.confidence}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
