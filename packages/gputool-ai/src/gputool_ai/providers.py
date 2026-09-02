"""Model endpoints and the prompt style that suits each.

Two things vary together and are easy to get wrong separately:

  * **where the model lives** — a local vLLM server, or a hosted
    OpenAI-compatible API such as MiniMax or NVIDIA Build;
  * **how much you should ask of it** — a 4B model on a teaching bench is good
    at "which command lists open ports", and bad at multi-line scripts. A large
    hosted model is the reverse.

Keeping them in one table means picking a provider also picks a sensible
default style, while either can still be overridden.
"""
from __future__ import annotations

import os

_ENV_LOADED = False


def load_env_files(force: bool = False) -> list:
    """Populate os.environ from ~/.env.local and friends, once per process.

    Keys belong in a file with restrictive permissions, not on a command line
    where they end up in shell history and `ps` output. Existing environment
    variables always win, so an explicit export still overrides the file.

    Returns the list of files that were actually read.
    """
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return []
    _ENV_LOADED = True

    from ._paths import env_file_candidates

    read = []
    for path in env_file_candidates():
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # First file wins, and a real environment variable beats both.
                if k and k not in os.environ:
                    os.environ[k] = v
            read.append(str(path))
        except OSError:
            continue
    return read


# name -> (base_url, api-key env var, default model, default style)
PROVIDERS: dict[str, dict] = {
    "local": {
        "base_url": "http://127.0.0.1:8000/v1",
        "key_env": "",                       # vLLM/llama.cpp usually need none
        "model": "Qwen/Qwen3.5-4B",
        "style": "simple",
    },
    "llamacpp": {
        "base_url": "http://127.0.0.1:8080/v1",
        "key_env": "",
        "model": "local",
        "style": "simple",
    },
    # MiniMax runs separate international and mainland-China endpoints, and a key
    # issued for one 401s on the other. The China host is the default here
    # because that is where the lab's key is valid; use "minimax-intl" otherwise.
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "key_env": "MINIMAX_API_KEY",
        "model": "MiniMax-M2.7",
        "style": "full",
    },
    "minimax-intl": {
        "base_url": "https://api.minimax.io/v1",
        "key_env": "MINIMAX_API_KEY",
        "model": "MiniMax-M2.7",
        "style": "full",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "style": "full",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "model": "meta/llama-3.3-70b-instruct",
        "style": "full",
    },
}


def resolve(provider: str = "", base_url: str = "", model: str = "",
            api_key: str = "", style: str = "") -> dict:
    """Work out the endpoint to call and the style to ask for.

    Explicit arguments always win; anything omitted comes from the provider
    entry, then the environment. Returns a dict with base_url, model, api_key
    and style, plus `missing_key` when a hosted provider has no key set — the
    caller can then say so plainly instead of failing with a 401.
    """
    load_env_files()
    name = (provider or os.environ.get("GPUTOOL_PROVIDER", "local")).lower()
    entry = PROVIDERS.get(name, PROVIDERS["local"])

    key = api_key
    if not key and entry["key_env"]:
        key = os.environ.get(entry["key_env"], "")
    if not key:
        key = os.environ.get("GPUTOOL_API_KEY", "") or "EMPTY"

    return {
        "provider": name,
        "base_url": base_url or os.environ.get("GPUTOOL_BASE_URL", "") or entry["base_url"],
        "model": model or os.environ.get("GPUTOOL_MODEL", "") or entry["model"],
        "api_key": key,
        "style": (style or os.environ.get("GPUTOOL_STYLE", "") or entry["style"]).lower(),
        "missing_key": bool(entry["key_env"]) and key == "EMPTY",
        "key_env": entry["key_env"],
    }


# ── prompt styles ─────────────────────────────────────────────────────────
# Appended to the ReAct system prompt. `simple` is written for a small local
# model driving a Linux class: one short command per step, the ordinary
# utilities, no scripting. That is both what a 4B model can do reliably and
# what a student should be reading.
SIMPLE_STYLE = """
Command style — IMPORTANT:
- Run ONE short command per step. No multi-line scripts, no heredocs, no `for` loops.
- Prefer the common utilities a student should learn: ls, cd, pwd, cat, head, tail,
  grep, find, wc, sort, uniq, cut, sed, awk, chmod, chown, ps, df, du, tar, curl.
- Write the command the way a person would type it. If a one-liner is getting long,
  split it across two steps instead.
- Never use sudo or any command needing root; report what is needed instead.
- After the command, say in one plain sentence what it did and why.
"""

FULL_STYLE = """
Command style:
- Prefer simple, readable commands; combine steps only when it genuinely helps.
- Scripts are allowed when a task needs them, but keep them short and commented.
- Never use sudo or any command needing root; report what is needed instead.
"""


def style_text(style: str) -> str:
    return SIMPLE_STYLE if (style or "simple").lower() == "simple" else FULL_STYLE


def describe(cfg: dict) -> str:
    bits = [f"provider={cfg['provider']}", f"model={cfg['model']}", f"style={cfg['style']}"]
    if cfg.get("missing_key"):
        bits.append(f"NO KEY (set {cfg['key_env']})")
    return "  ".join(bits)
