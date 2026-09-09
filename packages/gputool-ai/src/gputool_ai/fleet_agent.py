"""Node-side fleet agent: report status up, accept a narrow set of commands down.

Runs as a thread inside `gputool monitor`, needs no root, and holds no open
listening port of its own by default — it dials out to the hub. That matters
here because the machines span two subnets (10.31.96.x and 10.31.81.x) where
mDNS does not cross, and because none of the lab accounts can open a firewall
port anyway.

Two directions, deliberately asymmetric:

  **up**    a status snapshot every few minutes: GPU, driver health, disk,
            conda, what is serving. Cheap, read-only, no privileges.
  **down**  commands, but only from a fixed allowlist of gputool subcommands.
            There is no "run this shell string" path, by design: a hub that can
            run arbitrary commands on twelve machines is a much bigger thing to
            secure than one that can call `gputool device`.

The allowlist is the security boundary. Everything else — the token, TLS if you
put a proxy in front — is defence in depth around it.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import uuid
import time
import urllib.error
import urllib.request
from pathlib import Path

from ._paths import state_dir

# ── the allowlist ─────────────────────────────────────────────────────────
# Only these gputool subcommands can be triggered remotely. Read-only ones are
# safe to run unattended; the others change state on the node, so they are
# listed explicitly rather than pattern-matched.
SAFE_COMMANDS = {
    "device":        [],                 # full device report
    "profile":       [],                 # serving profile
    "hf-cache":      ["status", "mount", "unmount"],
    "serve-vllm":    ["status", "stop"],
    "serve-llamacpp": ["status", "stop"],
    "agent":         ["status", "stop"],
    "check":         [],
}
# Commands that start long-running work. Allowed, but flagged so the hub can
# require a deliberate confirmation rather than firing them from a dashboard click.
HEAVY_COMMANDS = {"serve-vllm": {"start"}, "serve-llamacpp": {"start"}, "agent": {"start"}}

# A model id or file name may be passed to a start command. Anything outside
# this is refused: no spaces, no shell metacharacters, no path traversal.
SAFE_ARG = re.compile(r"^[A-Za-z0-9._/@:+-]{1,120}$")

POLL_SECONDS = int(os.environ.get("GPUTOOL_FLEET_INTERVAL", "180"))


# ── identity and reachability ─────────────────────────────────────────────
# mDNS is link-local multicast. It does not cross a router, and it does not
# cross a WireGuard overlay at all — so on a fleet that spans two campus
# subnets, office wifi, and Jetsons dialling in to a self-hosted Headscale
# from a cloud VM, it discovers nothing. Tailscale is the substrate that
# actually spans all of that: every node gets a stable 100.x address and a
# stable name regardless of which physical network it woke up on.
#
# Two consequences for this agent:
#   * identity cannot be the hostname (it changes, and it collides — we have
#     several "orin-nano"), so each node keeps a UUID on disk;
#   * the hub is reached by trying candidate addresses in order, because the
#     same node may need a LAN address at one time and a Tailscale one later.

def node_id() -> str:
    """A stable id for this node, independent of hostname and network."""
    p = state_dir() / "node-id"
    try:
        if p.is_file():
            v = p.read_text(encoding="utf-8").strip()
            if v:
                return v
    except OSError:
        pass
    v = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    try:
        p.write_text(v + "\n", encoding="utf-8")
    except OSError:
        pass
    return v


def tailscale_bin() -> str:
    """Userspace tailscale from gputool if present, else a system one."""
    ts = Path.home() / ".gputool" / "tailscale" / "tailscale"
    if ts.exists():
        return str(ts)
    return shutil.which("tailscale") or ""


def tailscale_ip() -> str:
    ts = tailscale_bin()
    if not ts:
        return ""
    args = [ts]
    sock = Path.home() / ".gputool" / "tailscaled.sock"
    if sock.exists() and str(sock) in ts or (Path.home() / ".gputool" / "tailscale" / "tailscale").exists():
        args += [f"--socket={sock}"]
    code, out = _run(args + ["ip", "-4"], timeout=15)
    if code != 0:
        code, out = _run([ts, "ip", "-4"], timeout=15)
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("100."):
            return line
    return ""


def lan_ips() -> list:
    """Non-loopback IPv4 addresses, Tailscale excluded (reported separately)."""
    out = []
    code, txt = _run(["hostname", "-I"], timeout=10)
    if code == 0:
        for a in txt.split():
            if a.count(".") == 3 and not a.startswith(("127.", "100.")):
                out.append(a)
    return out


def addresses() -> list:
    """Ranked ways to reach this node — most portable first.

    Tailscale leads because it is the only one that works from everywhere:
    a campus node, an office laptop on wifi, and a Jetson behind CGNAT all
    reach each other over it, and none of them reach each other by LAN IP.
    """
    out = []
    ts = tailscale_ip()
    if ts:
        out.append({"kind": "tailscale", "addr": ts})
    for a in lan_ips():
        out.append({"kind": "lan", "addr": a})
    out.append({"kind": "hostname", "addr": socket.gethostname()})
    return out


def hub_candidates(explicit: str = "") -> list:
    """Hub URLs to try, in order. First one that answers wins.

    Order matters: an explicit flag is a deliberate choice, the environment is
    the deployed default, and the Tailscale name is the fallback that keeps
    working when a node moves between networks.
    """
    cands = []
    for v in (explicit, os.environ.get("GPUTOOL_FLEET_HUB", "")):
        if v:
            cands.extend(u.strip() for u in v.split(",") if u.strip())
    # A MagicDNS name resolves for every node on the tailnet, whatever network
    # it is physically on. Costs nothing to try and rescues a moved node.
    name = os.environ.get("GPUTOOL_FLEET_HUB_NAME", "fleet-hub")
    port = os.environ.get("GPUTOOL_FLEET_PORT", "8010")
    if name:
        cands.append(f"http://{name}:{port}")
    seen, out = set(), []
    for c in cands:
        c = c.rstrip("/")
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def reachable_hub(candidates: list, token: str, timeout: int = 8) -> str:
    """First candidate whose /api/nodes answers. '' if none do."""
    for url in candidates:
        try:
            req = urllib.request.Request(
                f"{url}/api/nodes",
                headers={"Authorization": f"Bearer {token}"} if token else {})
            with urllib.request.urlopen(req, timeout=timeout):
                return url
        except urllib.error.HTTPError as e:
            # 401 still proves something is listening and speaking our protocol.
            if e.code in (401, 403):
                return url
        except Exception:
            continue
    return ""


def _run(cmd: list, timeout: int = 60) -> tuple:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except FileNotFoundError:
        return 127, "not found"


def _nvidia_smi(query: str) -> list:
    code, out = _run(["nvidia-smi", f"--query-gpu={query}",
                      "--format=csv,noheader,nounits"], timeout=20)
    if code != 0:
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def gputool_path() -> str:
    return os.environ.get("GPUTOOL_BIN") or str(Path.home() / ".local" / "bin" / "gputool")


def collect_status() -> dict:
    """A snapshot of this node. Read-only, no privileges, a second or two."""
    gpus = []
    for line in _nvidia_smi("name,compute_cap,memory.total,memory.used,temperature.gpu,utilization.gpu"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({
                "name": parts[0], "compute_cap": parts[1],
                "memory_total_mb": _int(parts[2]), "memory_used_mb": _int(parts[3]),
                "temperature_c": _int(parts[4]), "utilization_pct": _int(parts[5]),
            })

    driver = ""
    d = _nvidia_smi("driver_version")
    if d:
        driver = d[0]

    # Driver health: the four facts that predict whether this node survives the
    # next kernel upgrade (see the runbook's section on DKMS).
    nmods = 0
    try:
        code, out = _run(["lsmod"], timeout=10)
        nmods = len([l for l in out.splitlines() if l.startswith("nvidia")])
    except Exception:
        pass
    kernel_gcc = ""
    try:
        m = re.search(r"gcc-(\d+)", Path("/proc/version").read_text())
        kernel_gcc = m.group(1) if m else ""
    except Exception:
        pass

    total, used, free = shutil.disk_usage("/")
    serving = _detect_serving()

    return {
        "node": socket.gethostname(),
        "node_id": node_id(),
        "addresses": addresses(),
        "reported_at": time.time(),
        "os": _os_pretty(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "ram_gb": _ram_gb(),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "driver": driver,
        "driver_health": {
            "modules_loaded": nmods,
            "kernel_gcc": kernel_gcc,
            "kernel_gcc_present": bool(kernel_gcc) and Path(f"/usr/bin/gcc-{kernel_gcc}").exists(),
            "nvidia_smi_ok": bool(gpus),
        },
        "disk": {"total_gb": total // 2**30, "free_gb": free // 2**30,
                 "used_pct": round(100 * used / total) if total else 0},
        "conda": _conda_info(),
        "serving": serving,
        "gputool": _gputool_version(),
        "agent_version": 1,
    }


def _int(s):
    try:
        return int(float(s))
    except Exception:
        return None


def _os_pretty() -> str:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return platform.platform()


def _ram_gb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024 // 1024
    except Exception:
        pass
    return 0


def _conda_info() -> dict:
    for root in (Path.home() / "miniconda3", Path.home() / "miniconda",
                 Path.home() / "anaconda3", Path("/opt/conda")):
        if (root / "bin" / "conda").exists():
            envs = sorted(p.name for p in (root / "envs").glob("*")) if (root / "envs").is_dir() else []
            return {"present": True, "root": str(root), "envs": envs}
    return {"present": False, "root": "", "envs": []}


def _detect_serving() -> dict:
    """Which inference servers are up, by probing their local ports."""
    out = {}
    for name, port, path in (("vllm", 8000, "/v1/models"),
                             ("llamacpp", 8080, "/health"),
                             ("agent", 8002, "/health")):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
                out[name] = {"up": True, "port": port,
                             "detail": r.read(400).decode("utf-8", "ignore")[:200]}
        except Exception:
            out[name] = {"up": False, "port": port}
    return out


def _gputool_version() -> str:
    code, out = _run([gputool_path(), "version"], timeout=20)
    return out.strip().split()[-1] if code == 0 and out.strip() else ""


# ── commands down ─────────────────────────────────────────────────────────
def validate(cmd: str, args: list) -> tuple:
    """Return (ok, reason). The single place that decides what may run."""
    if cmd not in SAFE_COMMANDS:
        return False, f"command not allowed: {cmd!r}"
    allowed_subs = SAFE_COMMANDS[cmd]
    heavy = HEAVY_COMMANDS.get(cmd, set())
    if args:
        sub = args[0]
        if allowed_subs and sub not in allowed_subs and sub not in heavy:
            return False, f"subcommand not allowed for {cmd}: {sub!r}"
        for a in args:
            if not SAFE_ARG.match(str(a)):
                return False, f"argument rejected: {a!r}"
            if ".." in str(a):
                return False, "path traversal in argument"
    elif allowed_subs:
        return False, f"{cmd} needs one of: {', '.join(allowed_subs)}"
    return True, "ok"


def execute(cmd: str, args: list, timeout: int = 300) -> dict:
    ok, reason = validate(cmd, args)
    if not ok:
        _audit(cmd, args, "REFUSED", reason)
        return {"ok": False, "refused": True, "reason": reason, "output": ""}
    full = [gputool_path(), cmd, *[str(a) for a in args]]
    code, out = _run(full, timeout=timeout)
    _audit(cmd, args, f"exit={code}", out[:200])
    return {"ok": code == 0, "exit_code": code, "output": out[-6000:],
            "command": " ".join(full)}


def _audit(cmd, args, result, detail) -> None:
    """Every remote command lands in a local log, whatever the hub says happened."""
    try:
        line = json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd,
                           "args": args, "result": result, "detail": detail[:200]})
        with open(state_dir() / "fleet-audit.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── the loop ──────────────────────────────────────────────────────────────
def _post(url: str, payload: dict, token: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def run_forever(hub: str, token: str, interval: int = POLL_SECONDS,
                once: bool = False, log=print) -> None:
    cands = hub_candidates(hub)
    hub = reachable_hub(cands, token)
    if not hub:
        log(f"[fleet] no hub reachable among: {', '.join(cands) or '(none configured)'}")
        if once:
            return
    else:
        log(f"[fleet] hub: {hub}")
    while True:
        # A node that moved networks needs to re-find the hub rather than fail
        # forever against an address that was correct on a different wifi.
        if not hub:
            hub = reachable_hub(hub_candidates(""), token)
            if hub:
                log(f"[fleet] hub found: {hub}")
            else:
                time.sleep(max(30, interval))
                continue
        try:
            status = collect_status()
            reply = _post(f"{hub}/api/report", status, token)
            pending = reply.get("commands") or []
            if pending:
                log(f"[fleet] {len(pending)} command(s) from hub")
            results = []
            for c in pending:
                res = execute(c.get("cmd", ""), c.get("args") or [])
                res["id"] = c.get("id")
                results.append(res)
            if results:
                _post(f"{hub}/api/results", {"node": status["node"], "results": results}, token)
            log(f"[fleet] reported {status['node']}: "
                f"{status['gpu_count']} gpu, {status['disk']['free_gb']}G free")
        except urllib.error.HTTPError as e:
            log(f"[fleet] hub rejected report: HTTP {e.code}")
        except Exception as e:
            log(f"[fleet] report failed: {type(e).__name__}: {e}")
            hub = ""   # re-discover on the next pass
        if once:
            return
        time.sleep(max(30, interval))


def start_thread(hub: str, token: str, interval: int = POLL_SECONDS) -> threading.Thread:
    t = threading.Thread(target=run_forever, args=(hub, token, interval),
                         daemon=True, name="gputool-fleet")
    t.start()
    return t


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="gputool-fleet-agent")
    ap.add_argument("--hub", default=os.environ.get("GPUTOOL_FLEET_HUB", ""))
    ap.add_argument("--token", default=os.environ.get("GPUTOOL_FLEET_TOKEN", ""))
    ap.add_argument("--interval", type=int, default=POLL_SECONDS)
    ap.add_argument("--once", action="store_true", help="report once and exit")
    ap.add_argument("--show", action="store_true", help="print the status payload and exit")
    a = ap.parse_args()
    if a.show:
        print(json.dumps(collect_status(), indent=2))
        return 0
    if not a.hub:
        print("ERROR: --hub or GPUTOOL_FLEET_HUB is required")
        return 2
    run_forever(a.hub, a.token, a.interval, once=a.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
