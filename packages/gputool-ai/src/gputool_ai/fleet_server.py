"""Fleet hub: a small FastAPI service the nodes report into.

Nodes dial out to this; the hub never connects to a node. That is what makes it
work across the two lab subnets without firewall changes, and it means a node
needs no listening port and no root.

    gputool-fleet-server --token <shared-secret>
    # then on each node:
    gputool monitor start --hub http://hub-host:8010 --token <shared-secret>

Control is queued, not pushed: an operator asks for a command, it sits in the
node's queue, and the node picks it up on its next check-in. So the worst case
for a wedged node is a command that never runs — not a hub that hangs waiting
for it.

The node decides what it will accept (see fleet_agent.SAFE_COMMANDS). The hub
mirrors that list for the UI, but the node's copy is the one that counts: a
compromised hub still cannot make a node run something outside its allowlist.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .fleet_agent import HEAVY_COMMANDS, SAFE_COMMANDS

STALE_AFTER = int(os.environ.get("GPUTOOL_FLEET_STALE", "600"))  # seconds
STATE_PATH = Path(os.environ.get(
    "GPUTOOL_FLEET_STATE", Path.home() / ".gputool" / "fleet-state.json"))

app = FastAPI(title="gputool fleet hub", version="1.0")

# node -> last status dict;  node -> [queued commands];  node -> [recent results]
NODES: dict = {}
QUEUE: dict = {}
RESULTS: dict = {}


def _token() -> str:
    return os.environ.get("GPUTOOL_FLEET_TOKEN", "")


def _auth(header: str | None) -> None:
    want = _token()
    if not want:
        return  # no token configured: open mode, intended for a trusted LAN only
    got = (header or "").removeprefix("Bearer ").strip()
    # Constant-time compare: a token check that leaks timing is worth fixing
    # even on a lab network, because it costs one function call to avoid.
    if not secrets.compare_digest(got, want):
        raise HTTPException(status_code=401, detail="bad or missing token")


def _persist() -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({"nodes": NODES}, indent=1), encoding="utf-8")
    except OSError:
        pass


def _load() -> None:
    try:
        if STATE_PATH.is_file():
            NODES.update(json.loads(STATE_PATH.read_text(encoding="utf-8")).get("nodes", {}))
    except Exception:
        pass


_load()


def _age(node: dict) -> float:
    return time.time() - float(node.get("reported_at") or 0)


def _health(node: dict) -> tuple:
    """(state, reasons) — a short verdict for the dashboard."""
    reasons = []
    if _age(node) > STALE_AFTER:
        return "stale", [f"no report for {int(_age(node) // 60)} min"]
    dh = node.get("driver_health") or {}
    if not dh.get("nvidia_smi_ok"):
        reasons.append("nvidia-smi cannot reach the driver")
    if dh.get("modules_loaded", 0) == 0:
        reasons.append("no nvidia kernel module loaded")
    if dh.get("kernel_gcc") and not dh.get("kernel_gcc_present"):
        reasons.append(f"gcc-{dh['kernel_gcc']} missing — DKMS will fail on the next kernel")
    free = (node.get("disk") or {}).get("free_gb", 999)
    if free < 15:
        reasons.append(f"only {free} GB free — apt will fail mid-transaction")
    elif free < 50:
        reasons.append(f"{free} GB free")
    if any(r for r in reasons if "cannot reach" in r or "no nvidia" in r or "apt will fail" in r):
        return "critical", reasons
    return ("warn", reasons) if reasons else ("ok", [])


# ── node-facing ───────────────────────────────────────────────────────────
@app.post("/api/report")
def report(payload: dict = Body(...), authorization: str | None = Header(None)):
    _auth(authorization)
    # Key on the stable id, not the hostname: several Orin Nanos ship with the
    # same hostname, and a node that moves networks keeps its identity.
    name = payload.get("node_id") or payload.get("node")
    if not name:
        raise HTTPException(status_code=400, detail="payload needs a node id or name")
    payload["reported_at"] = time.time()
    NODES[name] = payload
    _persist()
    pending = QUEUE.pop(name, [])
    return {"ok": True, "commands": pending}


@app.post("/api/results")
def results(payload: dict = Body(...), authorization: str | None = Header(None)):
    _auth(authorization)
    name = payload.get("node", "?")
    RESULTS.setdefault(name, [])
    for r in payload.get("results", []):
        r["at"] = time.time()
        RESULTS[name].append(r)
    RESULTS[name] = RESULTS[name][-20:]
    return {"ok": True}


# ── operator-facing ───────────────────────────────────────────────────────
@app.get("/api/nodes")
def list_nodes(authorization: str | None = Header(None)):
    _auth(authorization)
    out = []
    for name, n in sorted(NODES.items()):
        state, reasons = _health(n)
        out.append({"node": n.get("node", name), "node_id": name,
                    "addresses": n.get("addresses", []),
                    "state": state, "reasons": reasons,
                    "age_s": round(_age(n)), "gpus": n.get("gpus", []),
                    "driver": n.get("driver"), "disk": n.get("disk"),
                    "serving": n.get("serving"), "os": n.get("os"),
                    "arch": n.get("arch"), "conda": n.get("conda")})
    return {"nodes": out, "count": len(out)}


@app.get("/api/nodes/{name}")
def get_node(name: str, authorization: str | None = Header(None)):
    _auth(authorization)
    if name not in NODES:
        raise HTTPException(status_code=404, detail="unknown node")
    n = dict(NODES[name])
    state, reasons = _health(n)
    n["state"], n["reasons"] = state, reasons
    n["queued"] = QUEUE.get(name, [])
    n["recent_results"] = RESULTS.get(name, [])[-5:]
    return n


@app.post("/api/nodes/{name}/command")
def queue_command(name: str, payload: dict = Body(...),
                  authorization: str | None = Header(None)):
    _auth(authorization)
    if name not in NODES:
        raise HTTPException(status_code=404, detail="unknown node")
    cmd = payload.get("cmd", "")
    args = payload.get("args") or []
    if cmd not in SAFE_COMMANDS:
        raise HTTPException(status_code=400,
                            detail=f"not an allowed command: {cmd!r}")
    heavy = cmd in HEAVY_COMMANDS and args and args[0] in HEAVY_COMMANDS[cmd]
    if heavy and not payload.get("confirm"):
        raise HTTPException(
            status_code=409,
            detail=f"{cmd} {args[0]} starts long-running work; resend with confirm=true")
    item = {"id": secrets.token_hex(4), "cmd": cmd, "args": args,
            "queued_at": time.time()}
    QUEUE.setdefault(name, []).append(item)
    return {"ok": True, "queued": item,
            "note": "runs on the node's next check-in"}


@app.get("/api/allowed")
def allowed(authorization: str | None = Header(None)):
    _auth(authorization)
    return {"commands": {k: v for k, v in SAFE_COMMANDS.items()},
            "heavy": {k: sorted(v) for k, v in HEAVY_COMMANDS.items()},
            "note": "the node enforces this list; the hub only mirrors it"}


# ── PAIR interop ──────────────────────────────────────────────────────────
# NVIDIA's Personal AI Router (Apache-2.0) discovers peers over mDNS and then
# polls each one's /v1/node-info. mDNS does not cross our two subnets, but PAIR
# also accepts manually-entered nodes — so serving this shape lets PAIR consume
# the fleet without us implementing pairing or mTLS.
@app.get("/v1/node-info")
def node_info(node: str = Query("", description="node name; omit for an aggregate")):
    if node:
        if node not in NODES:
            raise HTTPException(status_code=404, detail="unknown node")
        return _pair_shape(NODES[node])
    return {"nodes": [_pair_shape(n) for n in NODES.values()]}


def _pair_shape(n: dict) -> dict:
    gpus = n.get("gpus") or []
    return {
        "hostname": n.get("node"),
        "node_id": n.get("node_id"),
        # PAIR expects a ranked IP list; ours is Tailscale-first because that
        # is the only address that works from every network in this fleet.
        "addresses": [a.get("addr") for a in (n.get("addresses") or [])],
        "os": n.get("os"),
        "arch": n.get("arch"),
        "cpu": {"cores": n.get("cpu_count")},
        "memory": {"total_gb": n.get("ram_gb")},
        "gpus": [{"name": g.get("name"),
                  "memory_total_mb": g.get("memory_total_mb"),
                  "memory_used_mb": g.get("memory_used_mb"),
                  "utilization_pct": g.get("utilization_pct"),
                  "compute_capability": g.get("compute_cap")} for g in gpus],
        # PAIR's scheduler wants coarse pressure, thresholded at 40/70/85%.
        "pressure": _pressure(gpus),
        "driver_version": n.get("driver"),
        "endpoints": {k: v for k, v in (n.get("serving") or {}).items() if v.get("up")},
        "stale": _age(n) > STALE_AFTER,
    }


def _pressure(gpus: list) -> int:
    if not gpus:
        return 0
    top = max((g.get("utilization_pct") or 0) for g in gpus)
    return 3 if top >= 85 else 2 if top >= 70 else 1 if top >= 40 else 0


# ── dashboard ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    rows = []
    for name, n in sorted(NODES.items()):
        state, reasons = _health(n)
        colour = {"ok": "#2C6B4E", "warn": "#97621B",
                  "critical": "#9E3A2B", "stale": "#64736F"}[state]
        gpus = n.get("gpus") or []
        gpu_txt = "—"
        if gpus:
            gpu_txt = f"{len(gpus)}&times; {gpus[0].get('name','?')}"
            used = sum(g.get("memory_used_mb") or 0 for g in gpus)
            tot = sum(g.get("memory_total_mb") or 0 for g in gpus)
            if tot:
                gpu_txt += f" &middot; {used}/{tot} MiB"
        disk = n.get("disk") or {}
        up = ", ".join(k for k, v in (n.get("serving") or {}).items() if v.get("up")) or "—"
        rows.append(
            f"<tr><td><b>{name}</b></td>"
            f"<td style='color:{colour}'><b>{state}</b></td>"
            f"<td>{gpu_txt}</td><td>{n.get('driver') or '—'}</td>"
            f"<td>{disk.get('free_gb','?')} GB</td><td>{up}</td>"
            f"<td>{int(_age(n))}s ago</td>"
            f"<td style='color:#97621B'>{'; '.join(reasons)}</td></tr>")
    body = "".join(rows) or "<tr><td colspan=8>no nodes have reported yet</td></tr>"
    return f"""<!doctype html><meta charset=utf-8>
<title>gputool fleet</title><meta http-equiv=refresh content=30>
<style>
 body{{font:14px ui-monospace,Menlo,monospace;margin:2rem;background:#F5F7F6;color:#141D1B}}
 h1{{font:600 1.3rem ui-sans-serif,system-ui;margin:0 0 .3rem}}
 p.sub{{color:#64736F;margin:0 0 1.2rem}}
 table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #DCE3E1}}
 th{{text-align:left;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;
     color:#64736F;background:#ECF0EF;padding:.5rem .7rem;border-bottom:1px solid #C3CECB}}
 td{{padding:.5rem .7rem;border-bottom:1px solid #DCE3E1;white-space:nowrap}}
 @media(prefers-color-scheme:dark){{body{{background:#0D1413;color:#E3EBE9}}
   table{{background:#141D1B;border-color:#253230}} th{{background:#1A2523;color:#8B9B97;border-color:#33433F}}
   td{{border-color:#253230}}}}
</style>
<h1>gputool fleet</h1>
<p class=sub>{len(NODES)} node(s) &middot; refreshes every 30s &middot;
 nodes report in, the hub never dials out</p>
<table><tr><th>node</th><th>state</th><th>gpu</th><th>driver</th><th>free</th>
<th>serving</th><th>last seen</th><th>notes</th></tr>{body}</table>"""


def main() -> int:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(prog="gputool-fleet-server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GPUTOOL_FLEET_PORT", "8010")))
    ap.add_argument("--token", default="", help="shared secret; nodes must send it")
    a = ap.parse_args()
    if a.token:
        os.environ["GPUTOOL_FLEET_TOKEN"] = a.token
    if not os.environ.get("GPUTOOL_FLEET_TOKEN"):
        print("WARNING: no token set — any host that can reach this port can "
              "report and queue commands. Fine on a trusted LAN, not otherwise.")
    print(f"fleet hub on http://{a.host}:{a.port}  (dashboard at /)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
