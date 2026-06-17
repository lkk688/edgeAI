#!/usr/bin/env python3
"""
generate_llm_nginx.py  —  dynamic LLM reverse-proxy generator for edgecloud.

Auto-discovers Headscale nodes that are running an OpenAI-compatible LLM server
on port 8080 (e.g. gputool/sjsujetsontool llama.cpp) and writes nginx `location`
blocks into the llm.forgengi.org vhost, so each node is reachable by a friendly
name over HTTPS:

    https://llm.forgengi.org/<name>/v1   ->   http://<node-tailscale-ip>:8080/v1

Run on edgecloud (the Headscale host):   sudo python3 generate_llm_nginx.py
It edits only the region between the LLM NODES markers, then reloads nginx.

Companion to generate_nginx.py (which maps SSH ports). Naming:
    coe-cmpe-288-05  -> node05      sjsujetson-01 -> jetson01
Override any name with NAME_OVERRIDES below.
"""
import re
import json
import subprocess
import urllib.request

SITE_FILE = "/etc/nginx/sites-available/llm"
LLM_PORT = 8080
PROBE_TIMEOUT = 2.0
START = "# >>> LLM NODES START (auto-generated; do not edit between markers) <<<"
END = "# >>> LLM NODES END <<<"

# Optional explicit name overrides: { "<hostname>": "<friendly-name>" }
NAME_OVERRIDES = {}

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def friendly_name(hostname):
    if hostname in NAME_OVERRIDES:
        return NAME_OVERRIDES[hostname]
    m = re.search(r"coe-cmpe-\d+-(\d+)", hostname)
    if m:
        return "node%s" % m.group(1).zfill(2)
    m = re.search(r"sjsujetson-(\d+)", hostname)
    if m:
        return "jetson%s" % m.group(1).zfill(2)
    if hostname == "sjsujetson":
        return None  # ambiguous unnumbered node — skip (set an override to include)
    return re.sub(r"[^a-z0-9-]", "-", hostname.lower())


def list_nodes():
    """Return [(hostname, ip)] for online nodes from `headscale nodes list`."""
    out = subprocess.run(["headscale", "nodes", "list"],
                         capture_output=True, text=True).stdout
    nodes = []
    for line in ANSI.sub("", out).splitlines():
        if "|" not in line or "Hostname" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 11:
            continue
        hostname, ips, connected = parts[1], parts[6], parts[10]
        if connected.lower() != "online":
            continue
        m = re.search(r"(100\.64\.\d+\.\d+)", ips)
        if m:
            nodes.append((hostname, m.group(1)))
    return nodes


def has_llm(ip):
    """True if the node answers on the OpenAI-compatible /v1/models endpoint."""
    url = "http://%s:%d/v1/models" % (ip, LLM_PORT)
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as r:
            json.loads(r.read().decode("utf-8", "ignore"))
        return True
    except Exception:
        return False


def block(name, ip):
    return """    # Node: {name} -> {ip}
    location /{name}/ {{
        proxy_pass http://{ip}:{port}/;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }}""".format(name=name, ip=ip, port=LLM_PORT)


def main():
    print("Scanning Headscale nodes for LLM servers on :%d ..." % LLM_PORT)
    seen, blocks = set(), []
    for hostname, ip in list_nodes():
        name = friendly_name(hostname)
        if not name or name in seen:
            continue
        if has_llm(ip):
            seen.add(name)
            blocks.append(block(name, ip))
            print("  + %-16s -> %s (%s)" % (name, hostname, ip))
        else:
            print("  - %-16s   %s (no LLM on :%d)" % (name, hostname, LLM_PORT))

    if not blocks:
        print("No LLM nodes found. Leaving config unchanged.")
        return

    with open(SITE_FILE) as f:
        cfg = f.read()
    new_region = START + "\n" + "\n\n".join(blocks) + "\n    " + END
    cfg = re.sub(re.escape(START) + r".*?" + re.escape(END), new_region, cfg, flags=re.S)
    with open(SITE_FILE, "w") as f:
        f.write(cfg)

    if subprocess.run(["nginx", "-t"]).returncode == 0:
        subprocess.run(["systemctl", "reload", "nginx"])
        print("\n✅ Wrote %d node(s) and reloaded nginx." % len(blocks))
    else:
        print("\n❌ nginx config test failed — NOT reloaded. Check %s" % SITE_FILE)


if __name__ == "__main__":
    main()
