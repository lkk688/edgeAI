"""tool_calling.py — the simplest useful agent: one OpenAI tool-calling loop.

Same five file tools as `react_loop.py`, but here the model uses the provider's
**native** `tools=` field (structured function-calling) instead of the ReAct
text protocol. The loop is tiny:

    1. send the task + the tool schemas
    2. if the model returns tool_calls -> run them, append the results, repeat
    3. if it returns plain content -> that's the final answer

Native tool-calling is cleaner when the backend supports it (NVIDIA Build,
OpenAI, Anthropic). When it does not (a base model, a bare llama.cpp), use the
ReAct loop instead — see `react_loop.py`. Both share `agent_tools.py`.

It shares the SAME backends and keys as `chat.py` (read from `~/.env.local`):
NVIDIA_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY.

Run it:
    pip install openai
    python3 tool_calling.py "What does app.py do? Read it." --dir ./sample_project
Or import `run_tool_calling()` in your own scripts / tutorials.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .tools import OPENAI_SCHEMAS, Tools


def run_tool_calling(client, model, task, *, root=".", max_rounds=8, log=print):
    """Run a native tool-calling loop until the model gives a final answer."""
    tools = Tools(root)
    messages = [
        {"role": "system", "content":
         "You are a coding agent. Use the file tools to inspect or modify the "
         "project, then reply with a concise final answer."},
        {"role": "user", "content": task},
    ]

    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = resp.choices[0].message
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None,
        })

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tools.dispatch(tc.function.name, args)
            log("  → %s(%s)\n     %s" % (
                tc.function.name, json.dumps(args)[:80],
                result[:200].replace("\n", " ")))
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "name": tc.function.name, "content": result,
            })

    return "(reached max_rounds without a final answer)"


def _make_client():
    """Build an OpenAI-compatible client from the same env keys chat.py uses."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("This demo needs the OpenAI client:  pip install openai")
    base = os.environ.get("AGENT_BASE_URL") or "https://integrate.api.nvidia.com/v1"
    key = (os.environ.get("NVIDIA_API_KEY")
           or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("ANTHROPIC_API_KEY") or "")
    if not key:
        sys.exit("No API key found. Set NVIDIA_API_KEY (or OPENAI/ANTHROPIC) in ~/.env.local.")
    return OpenAI(base_url=base, api_key=key)


def main():
    ap = argparse.ArgumentParser(description="Basic OpenAI tool-calling file agent.")
    ap.add_argument("task", nargs="+", help="what you want the agent to do")
    ap.add_argument("--dir", default=".", help="project root the agent may read/edit")
    ap.add_argument("--model", default=os.environ.get(
        "AGENT_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"))
    args = ap.parse_args()

    answer = run_tool_calling(_make_client(), args.model, " ".join(args.task), root=args.dir)
    print("\n=== final answer ===\n" + answer)


if __name__ == "__main__":
    raise SystemExit(main())
