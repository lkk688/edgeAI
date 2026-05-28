"""triage_basic.py — single-turn tool-calling vulnerability triage agent.

What this script does
---------------------
1. Runs `pip-audit` against `--requirements` to get a list of CVE findings.
2. For each finding, hands the CVE id + package + version to a coding-grade
   LLM on NVIDIA Build (default: `qwen/qwen3-coder-480b-a35b-instruct`).
3. The model is given four tools it can call iteratively:
       lookup_cve(cve_id)
       pip_audit_findings(requirements_path)
       search_usage(pattern, project_dir)
       read_file(path, project_dir)
4. The agent loop forwards every tool call to the local Python
   implementation, feeds the result back to the model, and stops when the
   model emits a final assistant message containing a JSON verdict.

This is the simplest useful pattern — one OpenAI-style tool-calling loop,
no framework. The §12c tutorial shows the same triage rewritten as a
manual ReAct loop, and §12d adds embedding-based retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time

from openai.types.chat import ChatCompletionMessage

# Local imports — see shared.py and tools/__init__.py for the rest.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (  # noqa: E402
    DEFAULT_CODER_MODEL,
    SYSTEM_PROMPT_BASIC,
    VERDICT_SCHEMA_HINT,
    Verdict,
    make_client,
    parse_verdict,
)
from tools import (  # noqa: E402
    lookup_cve,
    pip_audit_findings,
    read_file,
    search_usage,
)

# ---------------------------------------------------------------------------
# Tool schemas — exact OpenAI tool-calling JSON. The names must match the
# keys in TOOL_IMPL below.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_cve",
            "description": (
                "Fetch the official NVD record for a CVE id and return its "
                "description, CVSS score, CWE ids, and affected packages."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "pip_audit_findings",
            "description": (
                "Run pip-audit against a requirements file and return the "
                "full list of vulnerable (package, version, cve_ids) tuples. "
                "Useful for orienting yourself in the codebase before "
                "drilling into one CVE."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requirements_path": {"type": "string"},
                },
                "required": ["requirements_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_usage",
            "description": (
                "Search the project source tree for a substring (or regex) "
                "and return up to 30 file/line/snippet matches. Use this to "
                "see whether a vulnerable function is actually called."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "project_dir": {"type": "string"},
                    "is_regex": {"type": "boolean", "default": False},
                },
                "required": ["pattern", "project_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a slice of a source file from the project tree. "
                "Use after search_usage finds an interesting line."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "project_dir": {"type": "string"},
                    "start": {"type": "integer", "default": 1},
                    "end": {"type": "integer"},
                },
                "required": ["path", "project_dir"],
            },
        },
    },
]

TOOL_IMPL = {
    "lookup_cve": lookup_cve,
    "pip_audit_findings": pip_audit_findings,
    "search_usage": search_usage,
    "read_file": read_file,
}

# Soft cap on how many tool calls the model can make per CVE — keeps
# runaway loops from burning quota.
MAX_TOOL_ROUNDS = 6


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def _dispatch_tool(name: str, arguments_json: str) -> str:
    """Run one tool call and stringify the result for the model."""
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
    except Exception as exc:  # pylint: disable=broad-except
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(result, default=str)[:6000]


def triage_one(
    client,
    *,
    finding: dict,
    project_dir: str,
    requirements_path: str,
    model: str,
    verbose: bool = True,
) -> Verdict:
    """Triage one (cve, package, version) tuple and return a Verdict."""
    cve_id = finding.get("primary_cve") or (
        finding.get("cve_ids") or ["UNKNOWN"]
    )[0]
    package = finding["package"]
    version = finding["version"]

    user_prompt = textwrap.dedent(
        f"""
        Triage this finding:
            CVE         : {cve_id}
            Package     : {package}
            Version     : {version}
            Requirements: {requirements_path}
            Project dir : {project_dir}

        Decide whether this codebase actually exposes the vulnerable code
        path. Call tools as needed; reply with the JSON verdict when done.

        {VERDICT_SCHEMA_HINT}
        """
    ).strip()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_BASIC},
        {"role": "user", "content": user_prompt},
    ]

    for round_idx in range(MAX_TOOL_ROUNDS):
        if verbose:
            print(f"  · round {round_idx + 1}: calling model …", end="", flush=True)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=4096,
        )
        dt = time.perf_counter() - t0
        msg: ChatCompletionMessage = resp.choices[0].message
        if verbose:
            print(
                f" {dt:.1f}s  (tokens: prompt={resp.usage.prompt_tokens}, "
                f"completion={resp.usage.completion_tokens})"
            )

        # Append the assistant turn to history (preserving tool_calls).
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ]
                or None,
            }
        )

        if not msg.tool_calls:
            return parse_verdict(
                msg.content or "",
                cve_id=cve_id,
                package=package,
                version=version,
            )

        # Execute each requested tool and reply with the result.
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = tc.function.arguments
            if verbose:
                short_args = (tool_args or "")[:120]
                print(f"      → {tool_name}({short_args})")
            result_str = _dispatch_tool(tool_name, tool_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": result_str,
                }
            )

    # Out of budget — ask for a verdict with no further tool use.
    messages.append(
        {
            "role": "user",
            "content": (
                "You have used your tool budget. Reply now with ONLY the JSON "
                f"verdict. {VERDICT_SCHEMA_HINT}"
            ),
        }
    )
    final = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return parse_verdict(
        final.choices[0].message.content or "",
        cve_id=cve_id,
        package=package,
        version=version,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Single-turn tool-calling vulnerability triage agent."
    )
    p.add_argument(
        "--project",
        default="./sample_project",
        help="Path to the Python project to triage (default: ./sample_project)",
    )
    p.add_argument(
        "--requirements",
        default=None,
        help="Path to requirements.txt (default: <project>/requirements.txt)",
    )
    p.add_argument("--model", default=DEFAULT_CODER_MODEL)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only triage the first N findings (0 = all)",
    )
    p.add_argument(
        "--cve",
        action="append",
        default=[],
        help="Triage only this CVE id (may be passed multiple times)",
    )
    p.add_argument(
        "--package",
        action="append",
        default=[],
        help="Triage only findings for this package (may be passed multiple times)",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    project_dir = os.path.abspath(args.project)
    requirements_path = args.requirements or os.path.join(
        project_dir, "requirements.txt"
    )
    if not os.path.isfile(requirements_path):
        print(f"[error] {requirements_path}: no such file", file=sys.stderr)
        return 2

    print(f"⚙  project        : {project_dir}")
    print(f"⚙  requirements   : {requirements_path}")
    print(f"⚙  model          : {args.model}")

    # 1) Discover findings ----------------------------------------------------
    print("\n→ running pip-audit …")
    audit = pip_audit_findings(requirements_path)
    findings = audit["findings"]
    if not findings:
        print("  no vulnerable dependencies found — nothing to triage.")
        return 0
    print(f"  pip-audit reported {len(findings)} finding(s).")

    # Group by (package, primary_cve) — pip-audit may report the same CVE
    # twice with different alias ids; we only triage each unique CVE once.
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

    # 2) Triage each finding --------------------------------------------------
    client = make_client()
    verdicts: list[Verdict] = []
    for i, f in enumerate(queue, 1):
        print(
            f"\n[{i}/{len(queue)}] {f['package']} {f['version']} — {f.get('primary_cve')}"
        )
        v = triage_one(
            client,
            finding=f,
            project_dir=project_dir,
            requirements_path=requirements_path,
            model=args.model,
            verbose=not args.quiet,
        )
        verdicts.append(v)
        print(v.pretty())

    # 3) Summary --------------------------------------------------------------
    print("\n=== SUMMARY ===")
    for v in verdicts:
        flag = (
            "EXPLOITABLE"
            if v.exploitable_here
            else "ok"
            if v.exploitable_here is False
            else "??"
        )
        print(f"  {flag:11}  {v.cve_id}  {v.package} {v.version}  [{v.confidence}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
