# edgecloud — Headscale + LLM Gateway Operations Guide

**Host:** `edgecloud` (`ip-172-26-0-224`, AWS) · public IP `34.219.120.177` · Tailscale IP `100.64.0.2`
**Role:** Headscale control server + nginx gateway (SSH port-forwards **and** HTTPS LLM reverse-proxy)
**Domain:** `forgengi.org` on Cloudflare (**DNS-only / grey-cloud** — Cloudflare is not in the data path)

This document summarizes everything configured on this box and the day-to-day commands to operate it.

---

## 1. Architecture at a glance

```
                        Cloudflare DNS (grey/DNS-only, just name->IP)
                                        │
   any client / Jetson  ──HTTPS──▶  edgecloud nginx  ──Tailscale (WireGuard)──▶  GPU node :8080
   https://llm.forgengi.org/node05/v1        (TLS term)        100.64.0.x            llama.cpp

   ssh -p 28805 user@headscale.forgengi.org ──▶ nginx stream ──Tailscale──▶ node:22
```

Two independent gateways run on this host, both driven by the live Headscale node list:

| Gateway | Purpose | nginx layer | Config file | Generator |
|---|---|---|---|---|
| **SSH ports** | `ssh -p <port>` to any node | `stream {}` (L4 TCP) | `/etc/nginx/headscale-stream.conf` | `~/generate_nginx.py` |
| **LLM HTTPS** | `https://llm.forgengi.org/<name>/v1` | `http {}` (L7 TLS) | `/etc/nginx/sites-available/llm` | `~/generate_llm_nginx.py` |

- **SSH port scheme** (by headscale **Name** end-number, not IP): `sjsujetson-NN → 200NN` (jetson-01 → `20001`), `coe-cmpe-CCC-NN → CCCNN` (coe-cmpe-288-05 → `28805`). Any **other** device is auto-assigned `201xx` (20101, 20102, …) and printed at the end of the generator run. The generator uses the Name column, so a board whose hostname is the bare `sjsujetson` but whose Name is `sjsujetson-61` still maps to `20061`. It includes all known nodes and runs `nginx -t` + reload itself.
- **LLM name scheme:** `coe-cmpe-288-05 → node05`, `sjsujetson-NN → jetsonNN`.

---

## 2. The LLM HTTPS gateway (`llm.forgengi.org`)  ← new

Lets any node use a shared GPU model **by name over HTTPS**, no IPs:

```
https://llm.forgengi.org/<name>/v1   ->   http://<node-tailscale-ip>:8080/v1
```
Currently published: **`node05`** → `coe-cmpe-288-05` (`100.64.0.44:8080`, Qwen3.5-9B).

### How it was built (one-time)
1. **DNS (Cloudflare):** A record `llm` → `34.219.120.177`, **Proxy status = DNS only (grey)**.
2. **TLS cert:**
   ```bash
   sudo certbot --nginx -d llm.forgengi.org
   ```
3. **nginx vhost:** `/etc/nginx/sites-available/llm` (symlinked into `sites-enabled/`). Per-node routes live between the markers:
   ```
   # >>> LLM NODES START (auto-generated; do not edit between markers) <<<
   ...location /node05/ { proxy_pass http://100.64.0.44:8080/; ... }
   # >>> LLM NODES END <<<
   ```
   Key proxy settings per node: `proxy_buffering off` (token streaming), `proxy_read_timeout 86400`, trailing slash on `proxy_pass` (strips the `/node05/` prefix).
4. **Auth:** the **node's own** `--api-key` (e.g. `sjsugputool`) still gates `/v1/chat/completions`. nginx just forwards the `Authorization: Bearer …` header.

### Add / refresh LLM nodes (the dynamic part)
The generator auto-discovers any online Headscale node answering on `:8080` and rewrites the route blocks:
```bash
sudo python3 ~/generate_llm_nginx.py     # scans nodes, writes routes, tests + reloads nginx
```
Friendly names come from the hostname (`coe-cmpe-288-05 → node05`). To force a name, edit `NAME_OVERRIDES` at the top of `~/generate_llm_nginx.py`.

### Test it
```bash
# model list (no auth needed)
curl https://llm.forgengi.org/node05/v1/models

# chat completion (auth required)
curl https://llm.forgengi.org/node05/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sjsugputool" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```
From a Jetson: `sjsujetsontool chat --server` (URL `https://llm.forgengi.org/node05/v1`).

---

## 3. The SSH port gateway (existing)

`~/generate_nginx.py` reads `headscale nodes list` and writes a `stream {}` block per node mapping a port → `node-ip:22`, into `/etc/nginx/headscale-stream.conf` (included from `nginx.conf`).

```bash
sudo python3 ~/generate_nginx.py      # regenerate SSH port map
sudo nginx -t && sudo systemctl reload nginx
```
Then: `ssh -p 28805 010796032@headscale.forgengi.org` (cmpe-288-05), `ssh -p 20001 sjsujetson@headscale.forgengi.org` (jetson-01).

> ⚠️ **Known sharp edge:** if two nodes map to the **same port** (e.g. a stale duplicate node, or the unnumbered `sjsujetson` host colliding with `sjsujetson-NN`), nginx fails its config test with `duplicate "0.0.0.0:<port>"` and **won't reload**. Fix the duplicate node first (Section 4), then regenerate. A backup of the last-known-good file was saved as `/etc/nginx/headscale-stream.conf.bak.<timestamp>`.

---

## 4. Headscale node management

All `headscale` commands need `sudo`.

```bash
# list every node (ID, hostname, given-name, IP, online/offline)
sudo headscale nodes list

# delete a stale / duplicate node
sudo headscale nodes delete -i <ID> --force

# rename a node's given-name (e.g. drop a random -xxxxxxxx collision suffix)
sudo headscale nodes rename <new-name> -i <ID> --force

# pre-auth keys (used by gputool/sjsujetsontool to join)
sudo headscale preauthkeys list  --user labs
sudo headscale preauthkeys create --user labs --reusable --expiration 720h
```

### Why duplicate nodes appear (and how to avoid them)
When a node re-registers with a **new machine key** (its Tailscale state was wiped) while its old entry still exists, Headscale keeps both and appends a random suffix to the newcomer's name (e.g. `coe-cmpe-288-05-oondevrq`). To clean up:
```bash
sudo headscale nodes delete -i <stale-offline-ID> --force      # remove the old one
sudo headscale nodes rename <clean-name> -i <live-ID> --force  # reclaim the name
sudo python3 ~/generate_nginx.py        # refresh SSH ports
sudo python3 ~/generate_llm_nginx.py    # refresh LLM routes
```
**Prevention:** don't wipe a node's Tailscale state. The clients preserve it —
`/var/lib/tailscale/tailscaled.state` (sudo Jetsons) and `~/.gputool/tailscaled.state` (userspace gputool); `down`/`up` reuse the same identity and won't create duplicates.

---

## 5. nginx & TLS cheat sheet

```bash
sudo nginx -t                      # validate ALL config (must pass before reload)
sudo systemctl reload nginx        # apply changes (graceful)
sudo systemctl status nginx

# config locations
/etc/nginx/sites-enabled/headscale     # https://headscale.forgengi.org -> 127.0.0.1:8080 (headscale)
/etc/nginx/sites-enabled/llm           # https://llm.forgengi.org/<name> -> node:8080  (LLM gateway)
/etc/nginx/sites-enabled/default
/etc/nginx/headscale-stream.conf       # SSH port stream map (included by nginx.conf)

# TLS / Let's Encrypt
sudo certbot certificates              # list certs + expiry
sudo certbot renew --dry-run           # test auto-renewal
# certs: /etc/letsencrypt/live/{headscale,llm}.forgengi.org/
```

> **Cloudflare note:** the zone is **DNS-only (grey cloud)**, so the SSL/TLS "encryption mode" (Full / Automatic / …) has **no effect** here — clients connect straight to this origin's Let's Encrypt certs. Leave it as-is. (It would only matter if a record were switched to proxied/orange.)

---

## 6. Add a brand-new LLM node — end-to-end

1. On the GPU node: serve a model and join Headscale
   ```bash
   gputool serve-llamacpp start <model.gguf> 8080 --api-key <token>   # binds 0.0.0.0
   gputool tailscale up                                               # joins headscale
   ```
2. On **edgecloud**: publish it
   ```bash
   sudo python3 ~/generate_llm_nginx.py     # auto-detects the new :8080 node, adds a route
   ```
3. Use it: `https://llm.forgengi.org/<name>/v1` (name printed by the generator).

To force a specific name, add to `NAME_OVERRIDES` in `~/generate_llm_nginx.py`:
```python
NAME_OVERRIDES = {"coe-cmpe-288-05": "node05"}
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nginx -t` → `duplicate "0.0.0.0:<port>"` | stale/duplicate Headscale node, or name collision | delete stale node (§4), `sudo python3 ~/generate_nginx.py`, reload |
| `https://llm.forgengi.org/<name>` → 502 | node offline / LLM not running / wrong tailscale IP | check `sudo headscale nodes list`; re-run `~/generate_llm_nginx.py`; from edgecloud `curl http://<node-ip>:8080/v1/models` |
| LLM completion → 401 | missing/wrong bearer token | send `-H "Authorization: Bearer <token>"` (the node's `--api-key`) |
| streaming feels buffered/laggy | `proxy_buffering` on | generator already sets `proxy_buffering off`; ensure the node route came from it |
| cert expiring | — | `sudo certbot renew` (auto-renew timer is installed) |
| node shows a `-xxxxxxxx` suffix | re-registered with new key | §4 delete+rename, then regenerate both gateways |

---

## 8. File inventory (this host)

```
~/generate_nginx.py                         # SSH port stream generator (existing)
~/generate_llm_nginx.py                     # LLM HTTPS route generator (new)
/etc/nginx/sites-available/llm              # LLM gateway vhost (+ symlink in sites-enabled/)
/etc/nginx/headscale-stream.conf            # SSH port map (generated)
/etc/nginx/headscale-stream.conf.bak.*      # backups (a stale dup-28805 block was removed)
/etc/letsencrypt/live/llm.forgengi.org/     # LLM gateway TLS cert
/etc/letsencrypt/live/headscale.forgengi.org/  # headscale TLS cert
```

Both generators live in the edgeAI repo too: `jetson/generate_llm_nginx.py` (and the original on this host). Keep `~/generate_llm_nginx.py` in sync if you update it upstream.

---
*Generated 2026-06-17. Companion docs in the edgeAI repo: `docs/curriculum/00c_gputool_guide.md` (serving + gateway) and `docs/curriculum/00_sjsujetsontool_guide.md` (`sjsujetsontool chat`).*
