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

_FINAL = re.compile(r"Final\s*Answer:\s*(.*)", re.S | re.I)
_ACTION_LINE = re.compile(r"Action:\s*(.+)", re.I)
_INPUT_JSON = re.compile(r"Action\s*Input:\s*(\{.*\})", re.S | re.I)
_INPUT_ANY = re.compile(r"Action\s*Input:\s*(.+)", re.I)
_CALL = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.S)
_KWARG = re.compile(
    r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|true|false|null|-?\d+\.?\d*|[^,)]+)',
    re.I,
)


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


def parse_step(text):
    """Return ('final', answer) | ('action', name, args) | None for a model reply."""
    fin = _FINAL.search(text)
    if fin:
        return ("final", fin.group(1).strip())

    am = _ACTION_LINE.search(text)
    if not am:
        return None
    action = am.group(1).strip().strip("`").strip()

    # Function-call style:  name(args)
    call = _CALL.match(action)
    if call:
        name, inner = call.group(1), call.group(2).strip()
        if inner[:1] == "{":
            try:
                return ("action", name, json.loads(inner))
            except json.JSONDecodeError:
                pass
        return ("action", name, _kwargs(inner))

    # Plain style:  Action: name   (+ a separate Action Input line)
    name = action.split()[0] if action else ""
    args = {}
    jm = _INPUT_JSON.search(text)
    if jm:
        try:
            args = json.loads(jm.group(1))
        except json.JSONDecodeError:
            args = {}
    else:
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
