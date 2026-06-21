#!/usr/bin/env python3
"""
generate_nginx.py  —  dynamic SSH reverse-proxy (TCP stream) generator for edgecloud.

Discovers Headscale nodes and writes nginx `stream` server blocks so each node's
SSH (:22) is reachable from the public edge on a DETERMINISTIC port:

    ssh sjsujetson@headscale.forgengi.org -p <port>   ->   <node-tailscale-ip>:22

Port scheme — derived from the headscale NAME's end number, never from the IP:

    sjsujetson-NN      -> 200NN     jetson fleet          (20001..200xx)
    coe-cmpe-CCC-NN    -> CCCNN     cmpe GPU devices       (e.g. 288 -> 28801..288xx)
    <anything else>    -> 201xx     auto-assigned (20101, 20102, ...) and PRINTED below

The node NAME (headscale's authoritative 3rd column) is used, not the OS hostname:
some boards report a bare "sjsujetson" hostname but are named "sjsujetson-61".

Run on edgecloud (the Headscale host):
    sudo python3 generate_nginx.py             # write + nginx -t + reload
    sudo python3 generate_nginx.py --dry-run   # just print the plan

Writes the full `stream { ... }` block to OUTPUT_FILE, which nginx.conf includes
at the main context:   include /etc/nginx/headscale-stream.conf;

Companion to generate_llm_nginx.py (HTTPS/LLM gateway on llm.forgengi.org).
"""
import re
import sys
import subprocess

OUTPUT_FILE = "/etc/nginx/headscale-stream.conf"
SSH_PORT = 22
JETSON_BASE = 20000      # sjsujetson-NN -> 20000 + NN
OTHER_BASE = 20100       # everything else -> 20101, 20102, ... (first free)
PORT_OVERRIDES = {}      # pin an awkward node: { "<name-or-hostname>": <port> }

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def list_nodes():
    """Return [(ident, hostname, ip, online)] from `headscale nodes list`.
    ident = authoritative Name column (falls back to Hostname)."""
    out = subprocess.run(["sudo", "headscale", "nodes", "list"],
                         capture_output=True, text=True).stdout
    nodes = []
    for line in ANSI.sub("", out).splitlines():
        if "|" not in line or "Hostname" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 11:
            continue
        hostname, name, ips, connected = parts[1], parts[2], parts[6], parts[10]
        m = re.search(r"(100\.64\.\d+\.\d+)", ips)
        if m:
            nodes.append((name or hostname, hostname, m.group(1),
                          connected.lower() == "online"))
    return nodes


def classify(ident):
    """Return (port|None, category). None means 'other' (assigned a 201xx later)."""
    if ident in PORT_OVERRIDES:
        return PORT_OVERRIDES[ident], "override"
    m = re.search(r"sjsujetson-(\d+)", ident)
    if m:
        return JETSON_BASE + int(m.group(1)), "jetson"
    m = re.search(r"coe-cmpe-(\d+)-(\d+)", ident)
    if m:
        return int("%s%02d" % (m.group(1), int(m.group(2)))), "cmpe"
    return None, "other"


def plan_ports(nodes):
    """Assign every node a stable port. Returns rows sorted by port."""
    rows, used, others = [], set(PORT_OVERRIDES.values()), []
    for ident, hostname, ip, online in nodes:
        port, cat = classify(ident)
        if port is None:
            others.append((ident, hostname, ip, online))
            continue
        used.add(port)
        rows.append(dict(ident=ident, hostname=hostname, ip=ip,
                         port=port, cat=cat, online=online))
    # "other" devices -> next free 201xx, deterministic by name
    nxt = OTHER_BASE + 1
    for ident, hostname, ip, online in sorted(others):
        while nxt in used:
            nxt += 1
        used.add(nxt)
        rows.append(dict(ident=ident, hostname=hostname, ip=ip,
                         port=nxt, cat="other", online=online))
        nxt += 1
    return sorted(rows, key=lambda a: a["port"])


def server_block(a):
    off = "" if a["online"] else "  [offline at generation]"
    return """
    # {ident} ({hostname}, {cat}){off}
    server {{
        listen {port};
        proxy_pass {ip}:{ssh};
        proxy_timeout 1h;
    }}""".format(ident=a["ident"], hostname=a["hostname"], cat=a["cat"],
                 off=off, port=a["port"], ip=a["ip"], ssh=SSH_PORT)


def main():
    dry = "--dry-run" in sys.argv
    nodes = list_nodes()
    if not nodes:
        print("No Headscale nodes found (run with sudo on the Headscale host).")
        return
    rows = plan_ports(nodes)

    print("%-18s %-15s %-6s %-8s %s" % ("NAME", "IP", "PORT", "CATEGORY", "ONLINE"))
    for a in rows:
        print("%-18s %-15s %-6d %-8s %s" %
              (a["ident"], a["ip"], a["port"], a["cat"], "yes" if a["online"] else "NO"))

    others = [a for a in rows if a["cat"] == "other"]
    if others:
        print("\n*** OTHER DEVICES (auto-assigned 201xx ports) ***")
        for a in others:
            print("  ssh ...@headscale.forgengi.org -p %d   -> %s (%s)" %
                  (a["port"], a["ident"], a["ip"]))

    config = "stream {\n" + "\n".join(server_block(a) for a in rows) + "\n}\n"

    if dry:
        print("\n--dry-run: %d block(s) planned, nothing written." % len(rows))
        return

    with open(OUTPUT_FILE, "w") as f:
        f.write(config)
    print("\nWrote %d server block(s) to %s" % (len(rows), OUTPUT_FILE))

    if subprocess.run(["nginx", "-t"]).returncode == 0:
        subprocess.run(["systemctl", "reload", "nginx"])
        print("✅ nginx reloaded.")
    else:
        print("❌ nginx config test failed — NOT reloaded. Check %s" % OUTPUT_FILE)


if __name__ == "__main__":
    main()
