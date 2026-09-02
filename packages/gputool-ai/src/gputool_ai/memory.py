"""Long-term memory: every session written to plain Markdown on disk.

Markdown rather than a database, deliberately. A student can open the file,
a grader can diff it, `grep` finds things, and it survives the tool being
uninstalled. Nothing here needs a server or a schema migration.

Layout under ~/.gputool/memory (override with GPUTOOL_MEMORY_DIR):

    memory/
      INDEX.md                     one line per session, newest last
      commands.md                  every shell command ever run, with exit codes
      sessions/2026-08-29-1a2b.md  one file per session: task, steps, outcome

`commands.md` is the one students actually want: a searchable history of the
commands that worked, which is the point of a Linux course.
"""
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

_MAX_OBS = 1200  # per-observation cap; the .md is for reading, not archiving


def memory_dir() -> Path:
    d = Path(os.environ.get(
        "GPUTOOL_MEMORY_DIR", Path.home() / ".gputool" / "memory")).expanduser()
    (d / "sessions").mkdir(parents=True, exist_ok=True)
    return d


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fence(text: str, lang: str = "") -> str:
    """Fence a block, defending against text that contains a fence itself."""
    body = (text or "").rstrip()
    ticks = "```"
    while ticks in body:
        ticks += "`"
    return f"{ticks}{lang}\n{body}\n{ticks}"


class SessionMemory:
    """Records one agent session, appending as it goes.

    Written incrementally rather than at the end, so a crash or a timeout still
    leaves a usable record — which is exactly when you want one.
    """

    def __init__(self, task: str, model: str = "", backend: str = "", root: str = ""):
        self.dir = memory_dir()
        self.id = time.strftime("%Y-%m-%d-") + uuid.uuid4().hex[:4]
        self.path = self.dir / "sessions" / f"{self.id}.md"
        self.task = task
        self.model = model
        self.started = time.time()
        self.n_steps = 0
        self.commands: list[tuple[str, int]] = []
        self._write_header(backend, root)

    # -- writing ---------------------------------------------------------
    def _append(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")

    def _write_header(self, backend: str, root: str) -> None:
        self._append(
            f"# Session {self.id}\n\n"
            f"- **When**: {_stamp()}\n"
            f"- **Model**: {self.model or 'unknown'}\n"
            f"- **Backend**: {backend or 'n/a'}\n"
            f"- **Workspace**: `{root or os.getcwd()}`\n\n"
            f"## Task\n\n{self.task}\n\n---"
        )

    def step(self, n: int, thought: str = "", action: str = "",
             args: dict | None = None, observation: str = "") -> None:
        self.n_steps = max(self.n_steps, n)
        parts = [f"### Step {n} — `{action or 'think'}`"]
        if thought:
            parts.append(f"*{thought.strip()[:400]}*")
        if action == "run_terminal" and args and args.get("command"):
            cmd = str(args["command"])
            parts.append(_fence(cmd, "bash"))
            code = self._exit_code(observation)
            self.commands.append((cmd, code))
        elif args:
            shown = {k: (str(v)[:120]) for k, v in args.items()}
            parts.append(_fence(str(shown), "json"))
        if observation:
            obs = observation.strip()
            if len(obs) > _MAX_OBS:
                obs = obs[:_MAX_OBS] + f"\n... [{len(observation) - _MAX_OBS} chars elided]"
            parts.append(_fence(obs))
        self._append("\n\n".join(parts))

    @staticmethod
    def _exit_code(observation: str) -> int:
        m = re.search(r"exit=(\d+)", observation or "")
        return int(m.group(1)) if m else -1

    def finish(self, answer: str = "", error: str = "") -> Path:
        secs = time.time() - self.started
        if error:
            self._append(f"## Outcome\n\n**Error:** {error}\n\n_{self.n_steps} steps, {secs:.1f}s_")
        else:
            self._append(f"## Answer\n\n{answer or '(none)'}\n\n_{self.n_steps} steps, {secs:.1f}s_")
        self._index(answer, error, secs)
        self._commands_log()
        return self.path

    def _index(self, answer: str, error: str, secs: float) -> None:
        idx = self.dir / "INDEX.md"
        if not idx.exists():
            idx.write_text("# Session index\n\n"
                           "| When | Session | Model | Steps | Task |\n"
                           "|---|---|---|---|---|\n", encoding="utf-8")
        one_line = " ".join((self.task or "").split())[:90]
        status = "error" if error else "ok"
        with open(idx, "a", encoding="utf-8") as f:
            f.write(f"| {_stamp()} | [{self.id}](sessions/{self.id}.md) | "
                    f"{self.model or '-'} | {self.n_steps} ({status}) | {one_line} |\n")

    def _commands_log(self) -> None:
        """Append this session's commands to the shared, greppable history."""
        if not self.commands:
            return
        log = self.dir / "commands.md"
        if not log.exists():
            log.write_text(
                "# Command history\n\n"
                "Every shell command the agent has run, newest at the bottom.\n"
                "Search it: `grep -n 'tar' ~/.gputool/memory/commands.md`\n\n",
                encoding="utf-8")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n## {self.id} — {_stamp()}\n\n")
            for cmd, code in self.commands:
                mark = "ok" if code == 0 else (f"exit {code}" if code >= 0 else "?")
                f.write(f"- `{cmd}`  <sub>{mark}</sub>\n")


# ── recall ────────────────────────────────────────────────────────────────
def recent_commands(limit: int = 12) -> list[str]:
    """Successful commands from previous sessions, newest first.

    Fed back into the system prompt so the agent reuses what already worked on
    this machine instead of rediscovering it — the cheapest form of memory that
    actually changes behaviour.
    """
    log = memory_dir() / "commands.md"
    if not log.exists():
        return []
    out: list[str] = []
    for line in reversed(log.read_text(encoding="utf-8").splitlines()):
        m = re.match(r"- `(.+)`\s+<sub>ok</sub>", line.strip())
        if m:
            cmd = m.group(1)
            if cmd not in out:
                out.append(cmd)
        if len(out) >= limit:
            break
    return out


def memory_note(limit: int = 12) -> str:
    """A short prompt fragment describing what worked before. '' when empty."""
    cmds = recent_commands(limit)
    if not cmds:
        return ""
    listed = "\n".join(f"  {c}" for c in cmds)
    return ("\nCommands that already worked on this machine — prefer them when relevant:\n"
            + listed + "\n")
