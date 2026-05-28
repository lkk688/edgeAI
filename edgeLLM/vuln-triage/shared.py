"""Shared helpers for the vulnerability-triage tutorials (lessons 12*).

Everything that does *not* depend on a specific agent style (basic
tool-calling vs ReAct vs RAG) lives here so the three agent scripts can
stay small and readable.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)

# Coding models on NVIDIA Build — either one accepts the OpenAI tool-call schema.
DEFAULT_CODER_MODEL = os.environ.get(
    "TRIAGE_CODER_MODEL", "qwen/qwen3-coder-480b-a35b-instruct"
)
DEFAULT_REASONER_MODEL = os.environ.get(
    "TRIAGE_REASONER_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"
)

# Embedding model + endpoint for the RAG variant (12d).
DEFAULT_EMBED_MODEL = os.environ.get(
    "TRIAGE_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"
)


def make_client(timeout: float = 180.0) -> OpenAI:
    """Build an OpenAI-compatible client pointed at NVIDIA Build.

    Defaults to ignoring system proxies (so corporate proxies don't
    intercept HTTPS to integrate.api.nvidia.com), and uses the API key
    from `$NVIDIA_API_KEY`.
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print(
            "[error] NVIDIA_API_KEY is not set.\n"
            "        export NVIDIA_API_KEY=nvapi-...  and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        timeout=timeout,
        max_retries=2,
        http_client=httpx.Client(trust_env=False),
    )


# ---------------------------------------------------------------------------
# JSON parsing — the agent is asked to return a verdict object. Models
# sometimes wrap it in code fences or chat decoration, so we extract
# defensively rather than calling `json.loads` directly.
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    cve_id: str
    package: str
    version: str
    exploitable_here: bool | None
    confidence: str
    justification: str
    recommended_action: str
    raw_text: str = ""

    def pretty(self) -> str:
        head = f"{self.cve_id}  [{self.package} {self.version}]"
        flag = (
            "EXPLOITABLE HERE"
            if self.exploitable_here
            else "NOT EXPLOITABLE HERE"
            if self.exploitable_here is False
            else "INCONCLUSIVE"
        )
        return (
            f"\n┌─ {head}\n"
            f"│  verdict   : {flag}   (confidence: {self.confidence})\n"
            f"│  reason    : {textwrap.fill(self.justification, 70, subsequent_indent='│              ')}\n"
            f"│  action    : {textwrap.fill(self.recommended_action, 70, subsequent_indent='│              ')}\n"
            f"└──"
        )


def extract_json_block(text: str) -> dict[str, Any]:
    """Return the first JSON object found in `text`, tolerant of code fences."""
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if "```" in text:
        parts = text.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue

    # Fallback: take everything between the first `{` and the last `}`.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"could not find a JSON object in:\n{text[:400]}")


def parse_verdict(text: str, *, cve_id: str, package: str, version: str) -> Verdict:
    """Map the model's final JSON into a Verdict dataclass."""
    obj = extract_json_block(text)
    return Verdict(
        cve_id=cve_id,
        package=package,
        version=version,
        exploitable_here=obj.get("exploitable_here"),
        confidence=str(obj.get("confidence", "unknown")),
        justification=str(obj.get("justification", "")).strip(),
        recommended_action=str(obj.get("recommended_action", "")).strip(),
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Output schema — used as the system prompt's "respond in this exact shape"
# instruction across all three agents.
# ---------------------------------------------------------------------------

VERDICT_SCHEMA_HINT = textwrap.dedent(
    """
    Return your final answer as a single JSON object on the very last line,
    with EXACTLY these keys and no additional fields:

      {
        "exploitable_here":    true | false,
        "confidence":          "low" | "medium" | "high",
        "justification":       "one to three sentences citing the evidence",
        "recommended_action":  "what the maintainer should do, in one sentence"
      }

    Do NOT wrap the JSON in markdown fences. Output the JSON on its own line.
    """
).strip()


def parse_action_args(args_str: str, fn: Callable) -> dict[str, Any]:
    """Parse the argument blob inside `Action: tool(...)` into a kwargs dict.

    Models call tools in *all* of these shapes — we accept them all and
    map positionals into kwargs using the function's signature:

        {"cve_id": "CVE-2024-1234"}              # canonical JSON object
        cve_id="CVE-2024-1234"                   # Python kwargs
        "CVE-2024-1234"                          # Python positional
        "yaml load", 3                           # multiple positionals
        "yaml load", k=3                         # mixed

    Strings can be single- or double-quoted Python literals. Anything
    that is not a `ast.literal_eval`-safe value raises ValueError.
    """
    args_str = (args_str or "").strip()
    if not args_str:
        return {}

    # Fast path: JSON object (our preferred form, what the prompt asks for).
    if args_str.startswith("{"):
        try:
            obj = json.loads(args_str)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass  # fall through to ast-based parser

    # General path: treat the string as the argument list of a Python call.
    try:
        node = ast.parse(f"_({args_str})", mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"could not parse arguments: {exc}") from exc
    if not isinstance(node, ast.Call):
        raise ValueError("Action arguments are not a function call form")

    sig = inspect.signature(fn)
    param_names = [
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    out: dict[str, Any] = {}
    for i, arg in enumerate(node.args):
        if i >= len(param_names):
            raise ValueError(
                f"too many positional arguments for {len(param_names)}-arg function"
            )
        out[param_names[i]] = ast.literal_eval(arg)
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs unpacking is not supported in Actions")
        out[kw.arg] = ast.literal_eval(kw.value)
    return out


SYSTEM_PROMPT_BASIC = textwrap.dedent(
    """
    You are a security analyst triaging a CVE against the source code of a
    small Python project running on an NVIDIA Jetson Orin Nano edge device.

    A vulnerability scanner has flagged a package as having a known CVE.
    Your job is to decide whether the vulnerable code path is actually
    reachable from this codebase, and to recommend an action.

    You have a small set of tools. Call them as needed. After you have
    enough evidence, stop calling tools and reply with the JSON verdict.
    Prefer to call tools over guessing. Three to five tool calls is
    usually plenty for a single CVE.
    """
).strip()
