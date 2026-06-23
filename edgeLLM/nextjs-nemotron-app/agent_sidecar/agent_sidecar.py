"""agent_sidecar.py — FastAPI bridge that streams the edge_agent ReAct loop.

The Next.js Agent Lab can't run the agent's `for` loop in a route handler
without either reimplementing edge_agent in JavaScript or shelling out per
turn. Instead, this small FastAPI service:

  1. Hosts the `edge_agent` Python package (read_file / grep / search_files /
     write_file / edit_file, plus optional web_search via SerpAPI).
  2. Runs the ReAct loop on the server.
  3. Streams **one Server-Sent Event per step** to the browser:
        data: {"type":"step",        "n":1, "thought":"…", "action":"grep", "input":{…}}
        data: {"type":"observation", "n":1, "text":"…"}
        data: {"type":"final",       "n":4, "answer":"…"}
        data: {"type":"error",       "message":"…"}
        data: [DONE]

The actual LLM call is made over the same OpenAI-compatible endpoint the
chat lab uses — NVIDIA Build by default, but the route accepts any model
id and uses its API key from env. The provider is resolved on the
*Next.js* side and forwarded as a base_url + key pair.

Endpoints
---------
POST /run    — body JSON, response is text/event-stream
GET  /health — {"ok": true, "tools": [...], "web_search": true|false}
GET  /docs   — Swagger UI (FastAPI default)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# Make the edge_agent package importable when running this file directly.
HERE = Path(__file__).resolve().parent
EDGE_AGENT_SRC = (HERE.parent.parent / "edge_agent" / "src").resolve()
if EDGE_AGENT_SRC.is_dir():
    sys.path.insert(0, str(EDGE_AGENT_SRC))

import edge_agent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent-sidecar")

PORT = int(os.environ.get("AGENT_SIDECAR_PORT", "8002"))
# Default workspace the agent operates on if the request does not override it.
DEFAULT_ROOT = os.environ.get(
    "AGENT_WORKSPACE",
    str((HERE / "workspace").resolve()),
)
# Hard ceiling — the request can request fewer steps but never more.
MAX_STEPS_HARD = int(os.environ.get("AGENT_MAX_STEPS", "12"))

app = FastAPI(
    title="Edge Agent Sidecar",
    description=(
        "Streams the edge_agent ReAct loop (file tools + optional SerpAPI "
        "web search) over Server-Sent Events for the Next.js Agent Lab."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "tools": edge_agent.tool_names(),
        "web_search": edge_agent.web_search_available(),
        "default_root": DEFAULT_ROOT,
        "max_steps_hard": MAX_STEPS_HARD,
    }


def _make_complete(base_url: str, api_key: str, model: str, temperature: float):
    """Build a `complete(messages) -> str` callable.

    Lazy-imports openai so a missing dependency surfaces as a clear SSE error
    rather than a sidecar boot failure.
    """
    from openai import OpenAI  # noqa: PLC0415  (lazy on purpose)

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0, max_retries=2)

    def complete(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
            stop=["\nObservation:", "Observation:"],
        )
        return (resp.choices[0].message.content or "").strip()

    return complete


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


@app.post("/run")
async def run(request: Request) -> StreamingResponse:
    """Run the agent on `task` and stream the trace.

    Body (JSON):
        {
          "task":        "…",                 # required
          "root":        "/abs/path",         # optional, default workspace
          "model":       "qwen/qwen3-coder-…",
          "base_url":    "https://integrate.api.nvidia.com/v1",
          "api_key":     "nvapi-…",
          "temperature": 0.1,
          "max_steps":   8
        }

    The `base_url` + `api_key` come from the Next.js side, which already
    knows how to resolve them per model (NVIDIA / OpenAI / Anthropic).
    """
    try:
        body = await request.json()
    except Exception as exc:
        return StreamingResponse(
            iter([_sse({"type": "error", "message": f"bad JSON body: {exc}"}), _sse_done()]),
            media_type="text/event-stream",
            status_code=400,
        )

    task = (body.get("task") or "").strip()
    if not task:
        return StreamingResponse(
            iter([_sse({"type": "error", "message": "`task` is required"}), _sse_done()]),
            media_type="text/event-stream",
            status_code=400,
        )

    root = os.path.abspath(body.get("root") or DEFAULT_ROOT)
    if not os.path.isdir(root):
        return StreamingResponse(
            iter([_sse({"type": "error", "message": f"workspace does not exist: {root}"}), _sse_done()]),
            media_type="text/event-stream",
            status_code=400,
        )

    # api_key may be missing if the caller picked a key-less backend (local
    # llama.cpp, an open OpenAI-compat server). The OpenAI client requires a
    # non-empty string though, so we fall back to a placeholder.
    api_key = (
        body.get("api_key")
        or os.environ.get("NVIDIA_API_KEY")
        or "EMPTY"
    )

    model = body.get("model") or "minimaxai/minimax-m2.7"
    base_url = body.get("base_url") or "https://integrate.api.nvidia.com/v1"
    temperature = float(body.get("temperature", 0.1))
    max_steps = min(int(body.get("max_steps") or 8), MAX_STEPS_HARD)

    log.info(
        "agent run task=%r root=%s model=%s max_steps=%s",
        task[:80] + ("…" if len(task) > 80 else ""),
        root, model, max_steps,
    )

    def event_stream() -> Iterator[str]:
        # Build the LLM completer + the file tools confined to `root`.
        try:
            complete = _make_complete(base_url, api_key, model, temperature)
        except Exception as exc:
            yield _sse({"type": "error",
                        "message": f"could not init OpenAI client: {exc}"})
            yield _sse_done()
            return

        tools = edge_agent.Tools(root=root)
        names = edge_agent.tool_names()
        # Reuse the package's REACT_SYSTEM but interpolate the current tool set.
        system = edge_agent.REACT_SYSTEM.format(
            tools=edge_agent.tool_docs(),
            names=", ".join(names),
        )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        yield _sse({
            "type": "start",
            "root": root,
            "model": model,
            "tools": names,
            "max_steps": max_steps,
        })

        t0 = time.time()
        for step in range(1, max_steps + 1):
            try:
                reply = complete(messages)
            except Exception as exc:
                yield _sse({"type": "error",
                            "message": f"model call failed at step {step}: {exc}"})
                yield _sse_done()
                return
            messages.append({"role": "assistant", "content": reply})

            parsed = edge_agent.react_loop.parse_step(reply)
            if parsed and parsed[0] == "final":
                yield _sse({
                    "type": "final",
                    "n": step,
                    "answer": parsed[1],
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "raw": reply,
                })
                yield _sse_done()
                return

            if not parsed:
                # No Action — nudge the model.
                yield _sse({"type": "nudge", "n": step, "raw": reply})
                messages.append({
                    "role": "user",
                    "content": (
                        "Observation: ERROR: no Action found. Reply with either "
                        "an Action + Action Input (JSON), or a Final Answer."
                    ),
                })
                continue

            _, name, args = parsed
            # Extract a concise Thought line for the UI.
            thought = ""
            for ln in reply.splitlines():
                if ln.strip().lower().startswith("thought:"):
                    thought = ln.split(":", 1)[1].strip()
                    break

            yield _sse({
                "type": "step",
                "n": step,
                "thought": thought,
                "action": name,
                "input": args if isinstance(args, dict) else {},
                "raw": reply,
            })

            obs = tools.dispatch(name, args if isinstance(args, dict) else {})
            yield _sse({
                "type": "observation",
                "n": step,
                "text": obs,
            })
            messages.append({"role": "user", "content": "Observation: " + obs})

        # Out of steps without a Final Answer — try once more with a forced
        # "stop using tools" prompt so the user gets *some* answer.
        messages.append({
            "role": "user",
            "content": ("You have used all allowed steps. Reply with ONLY "
                        "`Final Answer: <one-paragraph summary>`."),
        })
        try:
            forced = complete(messages)
            parsed = edge_agent.react_loop.parse_step(forced)
            answer = (parsed[1] if parsed and parsed[0] == "final"
                      else "(no final answer — agent ran out of steps)")
            yield _sse({
                "type": "final",
                "n": max_steps + 1,
                "answer": answer,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "raw": forced,
                "exhausted": True,
            })
        except Exception as exc:
            yield _sse({"type": "error",
                        "message": f"forced final call failed: {exc}"})
        yield _sse_done()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    log.info(
        "starting edge-agent sidecar on 0.0.0.0:%s — docs at /docs "
        "(workspace=%s, web_search=%s)",
        PORT, DEFAULT_ROOT, edge_agent.web_search_available(),
    )
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
