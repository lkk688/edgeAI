"""End-to-end test and benchmark for the agent, driven entirely from the CLI.

The agent's tools only fail in interesting ways when a real model drives them,
so this does not mock anything: it starts from a scratch workspace, asks the
sidecar to perform tasks that each require a specific tool, and then checks the
*filesystem*, not the model's prose, to decide whether the tool actually ran.

    gputool-agent-test --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.5-4B

Exit status is non-zero if any scenario fails, so it drops straight into CI.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# ── fixtures ──────────────────────────────────────────────────────────────
# A small tree with content the scenarios below can look for. Deliberately
# boring: the point is to test the tools, not the model's reading comprehension.
FIXTURES = {
    "README.md": (
        "# Demo Project\n\n"
        "A tiny fixture tree used by the gputool agent tests.\n"
        "The magic token is BLUEBIRD-42.\n"
    ),
    "src/app.py": (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def divide(a, b):\n"
        "    return a / b   # TODO: guard against zero\n"
    ),
    "src/util.py": (
        "import os\n\n"
        "def read_config(path):\n"
        "    with open(path) as f:\n"
        "        return f.read()\n"
    ),
    "docs/notes.txt": "Deployment notes. Remember BLUEBIRD-42 before shipping.\n",
}


@dataclass
class Result:
    name: str
    tool: str
    ok: bool
    seconds: float
    steps: int
    detail: str = ""
    tools_seen: list = field(default_factory=list)


def build_workspace(root: str) -> None:
    for rel, body in FIXTURES.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)


def run_task(url: str, payload: dict, timeout: int):
    """POST /run and consume the SSE stream. Returns (steps, tools, final, error)."""
    req = urllib.request.Request(
        url.rstrip("/") + "/run",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    steps, tools, final, error = 0, [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "step":
                steps += 1
                if ev.get("action"):
                    tools.append(ev["action"])
            elif kind == "final":
                final = ev.get("answer")
            elif kind == "error":
                error = ev.get("message")
    return steps, tools, final, error


# ── scenarios ─────────────────────────────────────────────────────────────
# Each returns (task, verifier). The verifier inspects the workspace afterwards
# and returns (ok, detail) — so a model that *claims* success without touching
# a file still fails.
def scenarios():
    def v_read(root, final, tools):
        hit = final and "BLUEBIRD-42" in final
        return hit, "found the token in the answer" if hit else "token not present in answer"

    def v_grep(root, final, tools):
        hit = final and ("app.py" in final or "divide" in final)
        return hit, "located the TODO" if hit else "did not name the file/function"

    def v_search(root, final, tools):
        hit = final and ("util.py" in final or "read_config" in final)
        return hit, "found the helper" if hit else "did not identify util.py"

    def v_write(root, final, tools):
        p = os.path.join(root, "OUTPUT.md")
        if not os.path.exists(p):
            return False, "OUTPUT.md was not created"
        body = open(p, encoding="utf-8").read()
        return (len(body.strip()) > 0), f"OUTPUT.md written ({len(body)} bytes)"

    def v_edit(root, final, tools):
        p = os.path.join(root, "src/app.py")
        body = open(p, encoding="utf-8").read()
        changed = "ZeroDivisionError" in body or "if b == 0" in body or "b != 0" in body
        return changed, "guard added to divide()" if changed else "divide() unchanged on disk"

    def v_shell(root, final, tools):
        p = os.path.join(root, "shell_marker.txt")
        if not os.path.exists(p):
            return False, "shell_marker.txt not created"
        return "AGENTWASHERE" in open(p, encoding="utf-8").read(), "marker written by the shell"

    def v_test(root, final, tools):
        # Only a real pytest run can report the pass count.
        hit = final and ("1 passed" in final or "passed" in final.lower())
        return hit, "reported the test result" if hit else "did not report a pytest result"

    return [
        ("read_file", "read_file",
         "Read README.md and tell me the magic token it contains.", v_read),
        ("grep", "grep",
         "Search the code for a TODO comment and tell me which file and function it is in.", v_grep),
        ("search_files", "search_files",
         "Find which file defines a function called read_config.", v_search),
        ("write_file", "write_file",
         "Create a new file OUTPUT.md containing a one-line summary of this project.", v_write),
        ("edit_file", "edit_file",
         "Edit src/app.py so that divide() raises a clear error when b is zero.", v_edit),
        # --- shell scenarios: skipped unless --terminal is passed ---
        ("run_terminal", "run_terminal",
         "Using the shell, write the text AGENTWASHERE into a file called shell_marker.txt.", v_shell),
        ("pytest", "run_terminal",
         "Write a pytest test for the add() function in src/app.py, then run pytest and "
         "tell me how many tests passed.", v_test),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(prog="gputool-agent-test")
    ap.add_argument("--agent-url", default=os.environ.get("AGENT_URL", "http://127.0.0.1:8002"))
    # Both default to empty so --provider decides. A hardcoded default here
    # silently overrides the provider table on the server side, which turns
    # "--provider minimax" into a request aimed at the local vLLM port.
    ap.add_argument("--base-url", default="",
                    help="OpenAI-compatible endpoint (default: from --provider)")
    ap.add_argument("--model", default="", help="model id (default: from --provider)")
    # Default to sending NO key, so the sidecar resolves one from its own
    # environment (MINIMAX_API_KEY and friends). Sending the placeholder
    # "EMPTY" would override that and turn every call into a 401.
    ap.add_argument("--api-key", default="", help="override the key the sidecar would resolve")
    ap.add_argument("--provider", default="", help="provider name, e.g. local or minimax")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--repeat", type=int, default=1, help="runs per scenario, for timing")
    ap.add_argument("--only", default="", help="run just this scenario name")
    ap.add_argument("--keep", action="store_true", help="keep the scratch workspace")
    ap.add_argument("--terminal", action="store_true",
                    help="enable run_terminal and include the shell scenarios")
    ap.add_argument("--backend", default="host", choices=["host", "docker"],
                    help="where shell commands run (docker needs a working docker)")
    ap.add_argument("--image", default="", help="container image for --backend docker")
    args = ap.parse_args()

    # Fail fast with a useful message rather than a stack trace.
    try:
        with urllib.request.urlopen(args.agent_url.rstrip("/") + "/health", timeout=10) as r:
            health = json.loads(r.read())
        print(f"agent    : {args.agent_url}  tools={','.join(health.get('tools', []))}")
    except Exception as e:
        print(f"ERROR: agent sidecar not reachable at {args.agent_url} ({e})", file=sys.stderr)
        print("  start it with:  gputool agent start", file=sys.stderr)
        return 2
    print(f"model    : {args.model or '(provider default)'} via "
          f"{args.base_url or args.provider or 'local'}")

    results: list[Result] = []
    SHELL_ONLY = {"run_terminal", "pytest"}
    for name, tool, task, verify in scenarios():
        if args.only and args.only != name:
            continue
        if name in SHELL_ONLY and not args.terminal:
            continue
        times = []
        last = Result(name, tool, False, 0.0, 0)
        for _ in range(max(1, args.repeat)):
            root = tempfile.mkdtemp(prefix=f"agenttest-{name}-")
            build_workspace(root)
            t0 = time.perf_counter()
            try:
                steps, tools_seen, final, error = run_task(
                    args.agent_url,
                    {"task": task, "root": root,
                     **({"base_url": args.base_url} if args.base_url else {}),
                     **({"model": args.model} if args.model else {}),
                     "max_steps": args.max_steps, "temperature": 0.1,
                     "terminal": args.terminal, "backend": args.backend,
                     "image": args.image,
                     **({"api_key": args.api_key} if args.api_key else {}),
                     **({"provider": args.provider} if args.provider else {})},
                    args.timeout,
                )
                dt = time.perf_counter() - t0
                if error:
                    last = Result(name, tool, False, dt, steps, f"agent error: {error[:70]}", tools_seen)
                else:
                    ok, detail = verify(root, final or "", tools_seen)
                    last = Result(name, tool, ok, dt, steps, detail, tools_seen)
            except Exception as e:
                dt = time.perf_counter() - t0
                last = Result(name, tool, False, dt, 0, f"{type(e).__name__}: {str(e)[:60]}")
            times.append(dt)
            if not args.keep:
                shutil.rmtree(root, ignore_errors=True)
            else:
                print(f"    workspace kept: {root}")
        last.seconds = statistics.median(times)
        results.append(last)
        mark = "PASS" if last.ok else "FAIL"
        print(f"  [{mark}] {name:<13} {last.seconds:6.1f}s  steps={last.steps:<2} "
              f"tools={','.join(last.tools_seen) or '-':<28} {last.detail}")

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print()
    print(f"{passed}/{total} scenarios passed"
          f" | median {statistics.median([r.seconds for r in results]):.1f}s"
          f" | total {sum(r.seconds for r in results):.1f}s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
