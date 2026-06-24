"""react_loop.py — the ReAct loop: the core of a tool-using AI agent.

**ReAct = Reason + Act.** Instead of answering in one shot, the model is asked to
*think* and *act* in a strict text protocol, one step at a time:

    Thought: <reasoning about what to do next>
    Action: <one tool name>
    Action Input: {"json": "arguments"}

We parse the Action, run the tool, and feed the result back:

    Observation: <tool output>

…and loop until the model decides it is done:

    Thought: I now know the answer
    Final Answer: <answer for the user>

Because the whole protocol is plain text, it works against **any** chat endpoint —
even a model with no native tool-calling. That is why it is the common foundation:

  - `chat.py` uses ReActAgent for its `--agent` / `/agent on` mode.
  - The vuln-triage lessons (12c) use the same loop shape for CVE triage.
  - `tool_calling.py` is the alternative when the provider supports native tools.

This module is deliberately decoupled from any HTTP client: you pass in a
`complete(messages) -> str` callable, so the same loop runs on llama.cpp,
NVIDIA Build, OpenAI, or Anthropic.

Real models don't always emit the exact format, so the parser is tolerant: it
accepts both `Action: tool\nAction Input: {json}` and the function-call style
`Action: tool(arg="value", n=3)` that Qwen/GPT-style models often produce.
"""
from __future__ import annotations

import json
import re

REACT_SYSTEM = """You are a coding agent that solves the user's task by using tools.

Tools available:
{tools}

Work in a strict loop — output ONE step per reply, then STOP and wait:

Thought: think about what to do next
Action: the tool name (one of: {names})
Action Input: a single-line JSON object of arguments

You will then be given:
Observation: the tool result

When you have enough information, reply with EXACTLY this (no Action):
Thought: I now know the answer
Final Answer: <your answer to the user>

Rules:
- Output ONE Action (or ONE Final Answer) per reply, then stop. NEVER write your own Observation.
- `Action` is just the tool name; put all arguments in `Action Input` as one-line JSON.
  Example:
      Action: read_file
      Action Input: {{"path": "app.py", "end": 40}}
- After you receive an Observation, do NOT repeat the same Action — use what you learned.
"""

_FINAL = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.S | re.I)
_ACTION_LINE = re.compile(r"Action\s*:\s*(.+)", re.I)
_INPUT_ANY = re.compile(r"Action\s*Input\s*:\s*(.+)", re.I)
_CALL = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.S)
_KWARG = re.compile(
    r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|true|false|null|-?\d+\.?\d*|[^,)]+)',
    re.I,
)
# Qwen native tool-call format: <tool_call>{"name":"…","arguments":{…}}</tool_call>
_TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S | re.I)


def _coerce(token):
    t = token.strip()
    if (t[:1] == '"' and t[-1:] == '"') or (t[:1] == "'" and t[-1:] == "'"):
        return t[1:-1]
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _kwargs(inner):
    """Parse `key="v", n=3` argument text into a dict."""
    return {m.group(1): _coerce(m.group(2)) for m in _KWARG.finditer(inner)}


def _balanced_json(text, start_at=0):
    """Find the first balanced {...} block starting at or after `start_at`.

    Returns the JSON substring (including braces). If no `{` opens any block
    that closes within `text`, falls back to the *first unclosed* opener and
    returns the substring from there to end-of-text — `_try_json` then closes
    it on a best-effort basis (so models that hit max_tokens still parse).

    String-aware: a `{` inside a string literal does not increase the depth.
    """
    first_unclosed = None
    i = text.find("{", start_at)
    while i >= 0:
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
        # Reached end of text without balancing — remember this opener as a
        # fallback in case nothing later balances either.
        if first_unclosed is None and (depth > 0 or in_str):
            first_unclosed = text[i:]
        i = text.find("{", i + 1)
    return first_unclosed


def _try_json(blob):
    """`json.loads` with one fallback: close a trailing unclosed string + brace.

    Models that hit max_tokens often cut the JSON mid-string; the heuristic
    tries to keep the agent's loop alive instead of dropping the args.
    """
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    # Best-effort: count unmatched braces / unterminated strings and close them.
    in_str = False
    esc = False
    depth = 0
    for c in blob:
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
    fixed = blob + ('"' if in_str else "") + ("}" * max(0, depth))
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def parse_step(text):
    """Return ('final', answer) | ('action', name, args) | None for a model reply.

    Accepts a wide range of dialects models actually produce:

      * Plain ReAct                : `Action: foo\\nAction Input: {"k": 1}`
      * Function-call style        : `Action: foo({"k": 1})` or `Action: foo(k=1)`
      * Code-fenced JSON           : ```Action Input:\\n```json\\n{...}\\n``` ```
      * Markdown-bolded labels     : `**Action:** foo`
      * Qwen native tool calls     : `<tool_call>{"name": "foo", "arguments": {...}}</tool_call>`
      * Truncated JSON             : missing `"` / `}` (max_tokens cutoff) — best-effort closure.
      * Bare `{...}` after Action  : no `Action Input:` label, just a JSON object on a new line.
    """
    # 1) Final answer wins (model decided it's done).
    fin = _FINAL.search(text)
    if fin:
        return ("final", fin.group(1).strip())

    # 2) Qwen's native <tool_call>{"name":...,"arguments":{...}}</tool_call>.
    tc = _TOOLCALL.search(text)
    if tc:
        obj = _try_json(tc.group(1))
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("function") or ""
            args = obj.get("arguments")
            if args is None:
                args = obj.get("args") or obj.get("parameters") or {}
            if isinstance(args, str):
                args = _try_json(args) or {}
            if isinstance(name, str) and name and isinstance(args, dict):
                return ("action", name, args)

    # 3) Standard ReAct `Action:` line.
    am = _ACTION_LINE.search(text)
    if not am:
        return None
    action = am.group(1).strip().strip("`").strip("*").strip()

    # 3a) Function-call style: `Action: name(...)`.
    call = _CALL.match(action)
    if call:
        name, inner = call.group(1), call.group(2).strip()
        if inner.startswith("{"):
            obj = _try_json(inner)
            if isinstance(obj, dict):
                return ("action", name, obj)
        return ("action", name, _kwargs(inner))

    # 3b) Plain style: `Action: name`  (+ a separate Action Input line).
    name = action.split()[0] if action else ""

    # 3c) Prefer a JSON object — find one anywhere AFTER the Action: line.
    # That lets us tolerate code fences, missing Action Input: labels, and
    # the model writing the JSON on its own block. We search the text
    # immediately after the Action line, not the whole text, so a JSON
    # object embedded inside an earlier Observation is not mistaken for args.
    args = {}
    rest = text[am.end():]
    blob = _balanced_json(rest)
    if blob is not None:
        obj = _try_json(blob)
        if isinstance(obj, dict):
            args = obj

    # 3d) Last resort: `Action Input: key=val` (rare, but seen with smaller models).
    if not args:
        im = _INPUT_ANY.search(text)
        if im and "=" in im.group(1):
            args = _kwargs(im.group(1))

    return ("action", name, args)


class ReActAgent:
    """Drive a reason-and-act loop over a set of tools.

    Parameters
    ----------
    complete : callable(messages) -> str
        Sends an OpenAI-style message list to a model, returns the reply text.
    tools : edge_agent.Tools
        Anything with `.dispatch(name, args_dict) -> str`.
    max_steps : int
        Safety cap on tool rounds (prevents runaway loops / quota burn).
    log : callable(str)
        Where to print the live trace (defaults to print).
    """

    def __init__(self, complete, tools, *, max_steps=8, log=print):
        self.complete = complete
        self.tools = tools
        self.max_steps = max_steps
        self.log = log

    def run(self, task):
        # Re-resolve tool docs and names at run time so a fresh SERPAPI_API_KEY
        # in the process env enables web_search without a re-import.
        from .tools import tool_docs, tool_names

        names = tool_names()
        system = REACT_SYSTEM.format(tools=tool_docs(), names=", ".join(names))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            text = (self.complete(messages) or "").strip()
            messages.append({"role": "assistant", "content": text})
            self.log(self._trace(text, step))

            parsed = parse_step(text)
            if parsed and parsed[0] == "final":
                return parsed[1]

            if not parsed:
                # No Action and no Final Answer — feed the rule back as an Observation.
                messages.append({"role": "user", "content":
                                 "Observation: ERROR: no Action found. Reply with either an Action + "
                                 "Action Input (JSON), or a Final Answer."})
                continue

            _, name, args = parsed
            observation = self.tools.dispatch(name, args if isinstance(args, dict) else {})
            self.log("   Observation: " + observation.replace("\n", "\n   ")[:1500])
            messages.append({"role": "user", "content": "Observation: " + observation})

        return "(stopped: reached max_steps=%d without a Final Answer)" % self.max_steps

    @staticmethod
    def _trace(text, step):
        keep = [ln for ln in text.splitlines()
                if ln.strip().startswith(("Thought:", "Action:", "Action Input:", "Final Answer:"))]
        body = "\n".join(keep) if keep else (text[:400] + ("…" if len(text) > 400 else ""))
        return "\n[step %d]\n%s" % (step, body)
