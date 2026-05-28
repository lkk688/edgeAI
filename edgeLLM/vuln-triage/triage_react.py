"""triage_react.py — ReAct-style triage WITHOUT any agent framework.

`triage_basic.py` (§12b) used OpenAI's structured `tools` parameter — the
provider does all the parsing for you. This script does the same job
the other way: we feed the model a textual ReAct protocol and parse its
Thought / Action / Observation lines ourselves.

Why bother? Two reasons:

1. **Portability.** ReAct works against any chat endpoint, even ones
   without `tools` support (e.g. a self-hosted vLLM serving a base
   model). The whole protocol is just text.
2. **Visibility.** Students literally see the model's chain of thought
   between actions. Great pedagogy, easy to debug.

The "loop" is a 50-line state machine — no LangChain, no AutoGen, no
agent classes. Just `while not done: chat(); parse(); execute()`.
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

# Same tool set as the basic agent — minus the OpenAI schema wrappers.
TOOLS = {
    "lookup_cve":         lookup_cve,
    "pip_audit_findings": pip_audit_findings,
    "search_usage":       search_usage,
    "read_file":          read_file,
}

# Hard ceiling: keeps a runaway model from looping forever.
MAX_STEPS = 8


REACT_SYSTEM = textwrap.dedent(
    """
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
      - Final Answer is the ONLY line allowed to contain raw JSON braces
        besides Action arguments.
    """
).strip()


# Robust line-oriented parser. The model sometimes emits markdown bold
# (`**Thought:**`) or extra blank lines — both fine.
RE_ACTION  = re.compile(r"^\s*\**\s*Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
RE_FINAL   = re.compile(r"^\s*\**\s*Final Answer\s*:\s*(.+)$", re.IGNORECASE)
RE_THOUGHT = re.compile(r"^\s*\**\s*Thought\s*:\s*(.+)$",      re.IGNORECASE)


def _parse_step(text: str):
    """Pull (thought, action_name, action_args, final_answer) out of one
    model reply. Any field may be None.
    """
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
    """Execute one tool and return a JSON string the model can ingest."""
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


def react_triage(
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

    initial_user = textwrap.dedent(
        f"""
        Triage one finding:
            CVE         : {cve_id}
            Package     : {package}
            Version     : {version}
            Requirements: {requirements_path}
            Project dir : {project_dir}

        Use the ReAct loop. {VERDICT_SCHEMA_HINT}
        """
    ).strip()

    messages: list[dict] = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": initial_user},
    ]

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"  ─ step {step}: calling model …", end="", flush=True)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            # Stop the model after the action line — anything past it is
            # the model trying to fake an Observation, which we forbid.
            stop=["\nObservation:", "Observation:"],
            temperature=0.1,
            max_tokens=1024,
        )
        dt = time.perf_counter() - t0
        reply = (resp.choices[0].message.content or "").strip()
        if verbose:
            usage = resp.usage
            print(f" {dt:.1f}s  (p={usage.prompt_tokens} c={usage.completion_tokens})")
            for ln in reply.splitlines():
                print(f"      │ {ln}")

        messages.append({"role": "assistant", "content": reply})

        thought, action, args_str, final = _parse_step(reply)

        if final is not None:
            return parse_verdict(
                final, cve_id=cve_id, package=package, version=version
            )

        if action is None:
            # Model produced only a thought (or junk) — nudge it.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "I did not see an `Action:` line. Either continue "
                        "with `Action: tool(...)` or finish with "
                        "`Final Answer: <json>`."
                    ),
                }
            )
            continue

        observation = _run_tool(action, args_str or "{}")
        if verbose:
            short = observation[:140].replace("\n", " ")
            print(f"      │ Observation: {short}{' …' if len(observation) > 140 else ''}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # Out of steps — force a final answer.
    messages.append(
        {
            "role": "user",
            "content": (
                "You have used all allowed ReAct steps. Reply with ONLY "
                f"`Final Answer: <json>`. {VERDICT_SCHEMA_HINT}"
            ),
        }
    )
    forced = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=512
    )
    text = forced.choices[0].message.content or ""
    _, _, _, final = _parse_step(text)
    return parse_verdict(
        final or text, cve_id=cve_id, package=package, version=version
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="ReAct-style vulnerability triage (no framework)."
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
    print(f"⚙  model          : {args.model}  (ReAct mode)")

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
        v = react_triage(
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
