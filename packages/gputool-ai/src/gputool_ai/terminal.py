"""A terminal tool for the agent, with a host backend and a Docker backend.

The file tools alone cannot do most real work — building, running tests, calling
git, inspecting a GPU. This adds one tool, `run_terminal`, modelled on how
Claude Code and Codex expose a shell:

  * **state persists between calls** — `cd build` then `make` behaves the way a
    person expects, and the agent is told where it is on every result;
  * **output is capped, from both ends** — a 50k-line build log is useless to a
    model, but the head and the tail are exactly what matters, so the middle is
    elided rather than the tail truncated;
  * **timeouts return partial output** instead of nothing, because the last
    lines before a hang are the diagnostic;
  * **the exit code is always reported**, so the model can tell "ran, failed"
    from "did not run".

Two backends:

  host   — subprocess on this machine, confined to the workspace by default.
  docker — exec into one long-lived container (e.g. a PyTorch image, optionally
           with `--gpus all`). The workspace is bind-mounted, so files created
           by the file tools and by the shell are the same files.

Choosing a backend is a real decision, not a detail. `host` gives the agent
whatever the invoking user can do on that machine, with no isolation; `docker`
confines it to a container that can be thrown away. Nothing here is enabled
unless the caller asks for it — see `TerminalTools`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import uuid

DEFAULT_TIMEOUT = 120
MAX_OUTPUT_CHARS = 12000
HEAD_CHARS = 7000  # keep more of the head: errors usually announce themselves early


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Cap output while keeping both ends — the middle is what you can spare."""
    if len(text) <= limit:
        return text
    head = text[:HEAD_CHARS]
    tail = text[-(limit - HEAD_CHARS):]
    dropped = len(text) - HEAD_CHARS - (limit - HEAD_CHARS)
    return f"{head}\n\n... [{dropped} characters elided] ...\n\n{tail}"


# ── privileged commands ───────────────────────────────────────────────────
# The agent never runs anything privileged. Package installs go into its own
# virtualenv (see HostBackend), so the usual reason to reach for sudo does not
# arise; anything genuinely needing root is a decision for a person, on a shared
# teaching cluster especially. The refusal is phrased so the model reports back
# rather than hunting for a workaround.
_PRIVILEGED = re.compile(
    r"(?:^|[;&|]|\$\(|`)\s*(?:sudo|su|pkexec|doas|runuser)\b", re.IGNORECASE)

_PRIVILEGED_MSG = "\n".join([
    "REFUSED: this command needs elevated privileges, and the agent does not run them.",
    "If a system package is genuinely required, name it in your final answer and a",
    "human will install it. For Python packages, use the agent's own environment:",
    "  uv pip install <package>        (or: pip install <package>)",
])


def is_privileged(command: str) -> bool:
    return bool(_PRIVILEGED.search(command or ""))


class HostBackend:
    """Run commands directly on this machine, in the agent's own virtualenv.

    No container, so nothing isolates the filesystem — but package installs are
    isolated, which is the failure that actually bites. An agent told to "install
    pytest and run the tests" on a shared teaching node would otherwise mutate
    whatever conda environment the sidecar happened to start in, and break
    somebody else's work. Instead it gets a uv-managed virtualenv of its own,
    created with --system-site-packages so heavy things already present (torch,
    CUDA bindings) are visible without being copied.
    """

    name = "host"

    def __init__(self, cwd: str):
        self.cwd = os.path.abspath(cwd)
        self.venv_bin = self._ensure_agent_venv()
        self.env = self._shell_env(self.venv_bin)

    # -- the agent's virtualenv ------------------------------------------
    @staticmethod
    def _bootstrap_uv() -> str:
        """Get a uv binary without sudo. Returns its path, or '' if unavailable."""
        found = shutil.which("uv")
        if found:
            return found
        # pip-installing uv lands it beside the current interpreter, which needs
        # no privileges and no network beyond PyPI.
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"],
                           capture_output=True, timeout=300)
        except Exception:
            return ""
        cand = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "uv")
        return cand if os.path.exists(cand) else (shutil.which("uv") or "")

    def _ensure_agent_venv(self) -> str:
        """Create or reuse the agent's virtualenv. Returns its bin dir, or ''."""
        venv = os.environ.get(
            "GPUTOOL_AGENT_VENV", os.path.join(os.path.expanduser("~"), ".gputool", "agent-venv"))
        bindir = os.path.join(venv, "bin")
        if os.path.isfile(os.path.join(bindir, "python")):
            return bindir
        uv = self._bootstrap_uv()
        if not uv:
            return ""
        try:
            os.makedirs(os.path.dirname(venv), exist_ok=True)
            r = subprocess.run(
                # --seed installs pip into the venv, so a bare `pip install` from the
                # agent lands here instead of silently mutating the conda env the
                # sidecar happens to run in.
                [uv, "venv", "--seed", "--system-site-packages", "--python", sys.executable, venv],
                capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                return ""
        except Exception:
            return ""
        return bindir if os.path.isdir(bindir) else ""

    def _shell_env(self, venv_bin: str = "") -> dict:
        """PATH the agent's shell should see.

        `bash -lc` inherits the PATH the service started with, and on these nodes
        conda goes on PATH late in .bashrc, after the guard that returns early
        for non-interactive shells — so without help the agent gets a shell with
        no pip and no pytest and burns its steps rediscovering that.

        Order: the agent's own venv first (so `pip install` is isolated), then
        the interpreter running the sidecar (so torch and friends resolve).
        """
        env = dict(os.environ)
        sidecar_bin = os.path.dirname(os.path.abspath(sys.executable))
        head = [p for p in (venv_bin, sidecar_bin) if p]
        rest = [p for p in env.get("PATH", "").split(os.pathsep) if p and p not in head]
        env["PATH"] = os.pathsep.join(head + rest)
        env.setdefault("PYTHONUNBUFFERED", "1")
        if venv_bin:
            # Makes bare `pip` and `uv pip` target the venv rather than guessing.
            env["VIRTUAL_ENV"] = os.path.dirname(venv_bin)
            env.pop("PYTHONHOME", None)
        return env

    def describe(self) -> str:
        return f"host, venv={os.path.dirname(self.venv_bin) if self.venv_bin else 'none'}"

    def run(self, command: str, timeout: int, cwd: str):
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                ["bash", "-lc", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
                env=self.env,
            )
            out = (p.stdout or "") + (p.stderr or "")
            return p.returncode, out, time.perf_counter() - t0, False
        except subprocess.TimeoutExpired as e:
            partial = ""
            for chunk in (e.stdout, e.stderr):
                if chunk:
                    partial += chunk.decode(errors="replace") if isinstance(chunk, bytes) else chunk
            return 124, partial, time.perf_counter() - t0, True
        except FileNotFoundError:
            return 127, "bash not found on this system", time.perf_counter() - t0, False


class DockerBackend:
    """Run commands inside one long-lived container.

    The container is created on first use and reused after that, so state built
    up by earlier commands (installed packages, background servers) survives —
    which is what makes multi-step tasks work at all.
    """

    name = "docker"

    def __init__(self, cwd: str, image: str, gpus: bool = True,
                 container: str | None = None, workspace_mount: str = "/workspace"):
        self.host_cwd = os.path.abspath(cwd)
        self.image = image
        self.gpus = gpus
        self.mount = workspace_mount
        self.container = container or f"gputool-agent-{uuid.uuid4().hex[:8]}"
        self._started = False

    def _docker(self, *args, timeout=60):
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              timeout=timeout, errors="replace")

    def _alive(self) -> bool:
        r = self._docker("inspect", "-f", "{{.State.Running}}", self.container)
        return r.returncode == 0 and r.stdout.strip() == "true"

    def ensure_started(self) -> tuple[bool, str]:
        if self._alive():
            self._started = True
            return True, f"reusing container {self.container}"
        args = ["run", "-d", "--name", self.container,
                "-v", f"{self.host_cwd}:{self.mount}",
                "-w", self.mount]
        if self.gpus:
            args += ["--gpus", "all"]
        args += [self.image, "sleep", "infinity"]
        r = self._docker(*args, timeout=300)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout).strip()
            # A GPU request fails on a host without the NVIDIA runtime; say so
            # plainly rather than letting every later command fail mysteriously.
            if self.gpus and "nvidia" in msg.lower():
                return False, f"could not start container with --gpus all: {msg[:300]}"
            return False, f"could not start container: {msg[:300]}"
        self._started = True
        return True, f"started container {self.container} from {self.image}"

    def run(self, command: str, timeout: int, cwd: str):
        if not self._started:
            ok, msg = self.ensure_started()
            if not ok:
                return 125, msg, 0.0, False
        # Translate the host-side cwd into its path inside the container.
        rel = os.path.relpath(cwd, self.host_cwd)
        inner = self.mount if rel in (".", "") else os.path.join(self.mount, rel)
        t0 = time.perf_counter()
        try:
            p = self._docker("exec", "-w", inner, self.container,
                             "bash", "-lc", command, timeout=timeout)
            return p.returncode, (p.stdout or "") + (p.stderr or ""), time.perf_counter() - t0, False
        except subprocess.TimeoutExpired:
            return 124, "", time.perf_counter() - t0, True

    def cleanup(self):
        if self._started:
            self._docker("rm", "-f", self.container, timeout=60)
            self._started = False


# cu12.8 ships sm_120 kernels, which Blackwell needs. The cu12.4 images stop at
# sm_90: on a 5080 they enumerate the device happily and then fail every launch
# with "no kernel image is available for execution on the device", which reads
# like a driver fault and is not one. Verified computing on both 5080 (sm_120,
# 20.2 TFLOPS) and 4090 (sm_89 via the sm_86 binaries, 35.4 TFLOPS).
DEFAULT_GPU_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"


def make_backend(mode: str, cwd: str, image: str = "", gpus: bool = True):
    if mode == "docker":
        return DockerBackend(cwd, image or os.environ.get(
            "GPUTOOL_AGENT_IMAGE", DEFAULT_GPU_IMAGE), gpus=gpus)
    return HostBackend(cwd)


class TerminalMixin:
    """Adds `run_terminal` to an edge_agent.Tools subclass.

    Kept as a mixin so the file tools stay exactly as they are; a deployment
    that does not want shell access simply does not mix this in.
    """

    def _term_init(self, backend):
        self._backend = backend
        self._term_cwd = os.path.abspath(self.root)

    def run_terminal(self, command, timeout=DEFAULT_TIMEOUT, cwd=None):
        """Run a shell command. Returns exit code, cwd and combined output."""
        if not isinstance(command, str) or not command.strip():
            return "ERROR: `command` must be a non-empty string."
        timeout = max(1, min(int(timeout), 900))

        if is_privileged(command):
            return _PRIVILEGED_MSG

        # An explicit cwd is resolved against the workspace and confined to it,
        # reusing the same guard the file tools use.
        if cwd:
            try:
                work = self._resolve(cwd)
            except ValueError as e:
                return f"ERROR: {e}"
        else:
            work = self._term_cwd
        if not os.path.isdir(work):
            return f"ERROR: working directory does not exist: {work}"

        code, out, secs, timed_out = self._backend.run(command, timeout, work)

        # `cd` has to survive between calls or multi-step work is impossible.
        # Ask the shell where it ended up rather than trying to parse the command.
        if not timed_out and code == 0 and "cd " in command:
            probe = self._backend.run("pwd", 10, work)
            newp = (probe[1] or "").strip().splitlines()
            if newp and os.path.isdir(newp[-1]):
                self._term_cwd = newp[-1]

        rel = os.path.relpath(work, self.root)
        header = (f"[{self._backend.name}] exit={code} cwd={'.' if rel == '.' else rel} "
                  f"time={secs:.1f}s" + ("  TIMED OUT" if timed_out else ""))
        body = _clip((out or "").rstrip())
        if timed_out:
            body += f"\n\n(command exceeded {timeout}s and was killed; output above is partial)"
        return f"{header}\n{body}" if body else header


TERMINAL_TOOL_DOCS = """\
- run_terminal(command, timeout=120, cwd=None) — run a shell command and return
  its exit code and combined stdout/stderr. State persists: `cd` in one call
  affects the next. Use it for builds, tests, git, package installs and anything
  the file tools cannot do. Prefer non-interactive flags (-y, --no-input);
  nothing can answer a prompt. Long output is elided in the middle."""


def terminal_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": ("Run a shell command in the workspace and return exit code "
                            "and combined output. Working directory persists between calls."),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120, max 900)."},
                    "cwd": {"type": "string", "description": "Directory relative to the workspace root."},
                },
                "required": ["command"],
            },
        },
    }
