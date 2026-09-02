"""Terminal-Bench adapter: run the gputool agent against benchmark tasks.

    harbor run -d terminal-bench/terminal-bench-2-1 \\
      --agent-import-path gputool_ai.tbench_adapter:GputoolAgent -k 5

Terminal-Bench hands the agent a task container and a `TmuxSession`, not a
shell it can call. So instead of the host or docker backend, `run_terminal` is
routed through a third backend that types into that session and reads the pane
back — the agent's own code is unchanged, only where its commands land.

Getting an exit code out of a tmux pane needs care: a pane is a screen, not a
pipe, so there is no return value to read. Each command is therefore wrapped
with a unique sentinel that echoes `$?`, and the output is whatever appeared
between the command and the sentinel. That is the same trick interactive
harnesses use, and it is why commands must not be interactive themselves.
"""
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

import edge_agent

TBENCH_TERMINAL_DOC = (
    "- run_terminal(command, timeout=120, cwd=None) - run a shell command in the "
    "task terminal and return its exit code and output. This is the only tool that "
    "affects the task; the file tools act on scratch space. Use non-interactive "
    "flags, nothing can answer a prompt."
)

from .agent_tools import ShellTools
from .providers import load_env_files, resolve as resolve_provider, style_text
from .terminal import TerminalMixin, _clip, is_privileged, _PRIVILEGED_MSG


class TmuxBackend:
    """Run commands in a Terminal-Bench task container via its tmux session."""

    name = "tmux"

    def __init__(self, session: TmuxSession):
        self.session = session

    def describe(self) -> str:
        return "terminal-bench tmux session"

    def run(self, command: str, timeout: int, cwd: str):
        t0 = time.perf_counter()
        marker = f"__TB_{uuid.uuid4().hex[:10]}__"
        # `cd` is folded in so the agent's persistent cwd is honoured inside the
        # container, where our host-side path means nothing.
        wrapped = f"{command}\necho {marker}$?"
        try:
            self.session.send_keys(
                [wrapped, "Enter"],
                block=True,
                max_timeout_sec=float(timeout),
            )
        except Exception as e:  # a hung command surfaces as a timeout here
            out = self._read_until(marker)
            return 124, out or f"{type(e).__name__}: {e}", time.perf_counter() - t0, True

        raw = self._read_until(marker)
        code, body = self._split(raw, marker)
        return code, body, time.perf_counter() - t0, False

    def _read_until(self, marker: str) -> str:
        try:
            return self.session.get_incremental_output()
        except Exception:
            try:
                return self.session.capture_pane(capture_entire=True)
            except Exception:
                return ""

    @staticmethod
    def _split(raw: str, marker: str):
        """Pull the exit code out of the sentinel and strip it from the output."""
        m = re.search(re.escape(marker) + r"(\d+)", raw or "")
        code = int(m.group(1)) if m else 0
        body = re.sub(re.escape(marker) + r"\d*", "", raw or "")
        # Drop the echoed command line itself; it is noise the model already knows.
        body = "\n".join(l for l in body.splitlines() if marker not in l)
        return code, body.strip()


class TmuxShellTools(TerminalMixin, edge_agent.Tools):
    """File tools + run_terminal, with the shell pointed at the task container.

    The file tools still act on the harness-side workspace, which is only used
    for scratch; everything that matters happens through run_terminal inside the
    container, which is what the benchmark grades.
    """

    def __init__(self, root, session: TmuxSession):
        edge_agent.Tools.__init__(self, root=root)
        self._term_init(TmuxBackend(session))

    def dispatch(self, name, args):
        if name == "run_terminal":
            if not isinstance(args, dict):
                return "ERROR: arguments must be a JSON object."
            try:
                return str(self.run_terminal(**args))[:14000]
            except Exception as e:
                return "ERROR: %s: %s" % (type(e).__name__, e)
        return edge_agent.Tools.dispatch(self, name, args)


class GputoolAgent(BaseAgent):
    """The gputool ReAct agent, driving a Terminal-Bench task container.

    Model and endpoint come from the environment so the same adapter benchmarks
    a local vLLM server or a hosted API without a code change:

        GPUTOOL_TB_BASE_URL   default http://127.0.0.1:8000/v1
        GPUTOOL_TB_MODEL      default Qwen/Qwen3.5-4B
        GPUTOOL_TB_API_KEY    default EMPTY
        GPUTOOL_TB_MAX_STEPS  default 20
    """

    @staticmethod
    def name() -> str:
        return "gputool-agent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keys live in ~/.env.local, not on the command line.
        load_env_files()
        # Same provider table the sidecar uses, so "-k provider=minimax" or
        # GPUTOOL_TB_PROVIDER picks endpoint, model, key and style together.
        cfg = resolve_provider(
            provider=kwargs.get("provider") or os.environ.get("GPUTOOL_TB_PROVIDER", ""),
            base_url=kwargs.get("base_url") or os.environ.get("GPUTOOL_TB_BASE_URL", ""),
            model=kwargs.get("model") or os.environ.get("GPUTOOL_TB_MODEL", ""),
            api_key=kwargs.get("api_key") or os.environ.get("GPUTOOL_TB_API_KEY", ""),
            style=kwargs.get("style") or os.environ.get("GPUTOOL_TB_STYLE", ""),
        )
        self.cfg = cfg
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]
        self.api_key = cfg["api_key"]
        self.style = cfg["style"]
        self.max_steps = int(kwargs.get("max_steps") or os.environ.get("GPUTOOL_TB_MAX_STEPS", "20"))
        # Keep well clear of a small context window; 1500 was too greedy on an
        # 8k server once observations accumulated.
        self.max_tokens = int(kwargs.get("max_tokens") or os.environ.get("GPUTOOL_TB_MAX_TOKENS", "800"))
        self._in_tokens = 0
        self._out_tokens = 0

    # ── context management ────────────────────────────────────────────────
    # A ReAct loop grows its own prompt: every observation is appended, and
    # terminal output is long. Against an 8k-context server the conversation
    # blows the window a few steps in and every later call returns HTTP 400 —
    # the agent then looks like it "failed the task" when it never got to try.
    # Keep the system prompt and the original instruction (both load-bearing),
    # keep the most recent exchanges (where the work is), and drop the middle.
    KEEP_RECENT = 8

    def _trim(self, messages: list) -> list:
        if len(messages) <= self.KEEP_RECENT + 2:
            return messages
        head = messages[:2]                      # system + the task itself
        tail = messages[-self.KEEP_RECENT:]
        dropped = len(messages) - len(head) - len(tail)
        note = {"role": "user", "content":
                f"[{dropped} earlier steps omitted to stay within the context window. "
                f"Re-run a command if you need its output again.]"}
        return head + [note] + tail

    def _completer(self):
        """Return a callable(messages) -> str backed by an OpenAI-compatible API.

        Retries once with a smaller reply budget on a context-length error, so a
        long observation costs one step rather than ending the run.
        """
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        def complete(messages):
            for max_tok in (self.max_tokens, 512):
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=self._trim(messages),
                        temperature=0.1,
                        max_tokens=max_tok,
                    )
                except Exception as e:
                    if "context length" in str(e).lower() and max_tok != 512:
                        continue        # retry with a smaller reply budget
                    raise
                u = getattr(resp, "usage", None)
                if u:
                    self._in_tokens += getattr(u, "prompt_tokens", 0) or 0
                    self._out_tokens += getattr(u, "completion_tokens", 0) or 0
                return resp.choices[0].message.content or ""
            return ""

        return complete

    
    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """Run the ReAct loop with the shell pointed at the task container.

        This does not call `ReActAgent.run()`: that helper re-resolves the tool
        list from `edge_agent.tools` at run time, which would drop `run_terminal`
        from the system prompt and leave the model unaware of the only tool that
        can affect the task. The loop below mirrors the sidecar's, which builds
        the prompt from the tool set actually in use.
        """
        markers: list[tuple[float, str]] = []
        trace: list[str] = []
        scratch = str(logging_dir or "/tmp")
        try:
            tools = TmuxShellTools(root=scratch, session=session)
            docs = edge_agent.tool_docs() + "\n" + TBENCH_TERMINAL_DOC
            names = list(edge_agent.tool_names()) + ["run_terminal"]
            system = (edge_agent.REACT_SYSTEM.format(tools=docs, names=", ".join(names))
                      + style_text(self.style))
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ]
            complete = self._completer()

            for step in range(1, self.max_steps + 1):
                reply = complete(messages)
                trace.append(f"--- step {step} ---\n{reply}")
                parsed = edge_agent.react_loop.parse_step(reply)
                if parsed and parsed[0] == "final":
                    trace.append(f"FINAL: {parsed[1]}")
                    break
                if not parsed:
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content":
                                     "Reply with either 'Action:' and 'Action Input:' or "
                                     "'Final Answer:'."})
                    continue
                _, name, args = parsed
                observation = tools.dispatch(name, args or {})
                trace.append(f"OBS: {observation[:400]}")
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            try:
                markers.append((session.get_asciinema_timestamp(), "gputool-agent finished"))
            except Exception:
                pass
            self._write_trace(logging_dir, trace)
            return AgentResult(
                total_input_tokens=self._in_tokens,
                total_output_tokens=self._out_tokens,
                failure_mode=FailureMode.NONE,
                timestamped_markers=markers,
            )
        except Exception as e:
            self._write_trace(logging_dir, trace + [f"ERROR {type(e).__name__}: {e}"])
            return AgentResult(
                total_input_tokens=self._in_tokens,
                total_output_tokens=self._out_tokens,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
                timestamped_markers=markers + [(0.0, f"{type(e).__name__}: {e}"[:200])],
            )

    @staticmethod
    def _write_trace(logging_dir, lines):
        if not logging_dir:
            return
        try:
            (Path(logging_dir) / "gputool-agent.log").write_text(
                "\n".join(lines), encoding="utf-8")
        except Exception:
            pass