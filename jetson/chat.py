#!/usr/bin/env python3
# gputool chat — terminal client for an OpenAI-compatible LLM server (e.g. llama.cpp).
#
# This file is the single source of truth for the `gputool chat` command. It is
# fetched to ~/.gputool/chat.py automatically by `gputool install` and
# `gputool update`, so new devices get it without any manual copying.
# Pure stdlib (no pip deps) so it runs even on locked-down machines without curl.
import sys, os, json, time, argparse, threading, itertools
import urllib.request, urllib.error

C = {}
def setup_colors(on):
    names = dict(reset="\033[0m", bold="\033[1m", dim="\033[2m",
                 red="\033[31m", green="\033[32m", yellow="\033[33m",
                 blue="\033[34m", magenta="\033[35m", cyan="\033[36m")
    for k, v in names.items():
        C[k] = v if on else ""

class Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    def __init__(self, text="thinking"):
        self.text = text; self._stop = False; self._t = None
    def start(self):
        if not sys.stdout.isatty():
            return
        def run():
            for ch in itertools.cycle(self.FRAMES):
                if self._stop:
                    break
                sys.stdout.write("\r%s%s %s...%s" % (C['dim'], ch, self.text, C['reset']))
                sys.stdout.flush(); time.sleep(0.08)
        self._t = threading.Thread(target=run, daemon=True); self._t.start()
    def end(self):
        self._stop = True
        if self._t:
            self._t.join(timeout=0.3)
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 48 + "\r"); sys.stdout.flush()

def fetch_model(base, headers):
    try:
        req = urllib.request.Request(base + "/models", headers=headers)
        d = json.load(urllib.request.urlopen(req, timeout=5))
        return (d.get("data") or [{}])[0].get("id")
    except Exception:
        return None

def request(endpoint, headers, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=600)

def err_for(e):
    if isinstance(e, urllib.error.HTTPError):
        body = e.read().decode("utf-8", "ignore")
        if e.code == 401:
            return ("%s✖ Authentication failed (401).%s Server requires an API key — "
                    "pass --api-key or set GPUTOOL_LLAMA_API_KEY." % (C['red'], C['reset']))
        return "%s✖ HTTP %s%s: %s" % (C['red'], e.code, C['reset'], body[:300])
    if isinstance(e, urllib.error.URLError):
        return ("%s✖ Cannot connect:%s %s. Is the server running and reachable at the given host/port?"
                % (C['red'], C['reset'], getattr(e, 'reason', e)))
    return "%s✖ %s%s" % (C['red'], e, C['reset'])

def run_turn(endpoint, headers, payload, show_think):
    """Send one chat request. Returns (assistant_text, usage) or (None, None) on error."""
    label = "%s%sAssistant ▸ %s" % (C['blue'], C['bold'], C['reset'])
    if not payload.get("stream"):
        spin = Spinner(); spin.start()
        try:
            resp = request(endpoint, headers, payload)
            obj = json.load(resp)
        except Exception as e:
            spin.end(); print(err_for(e)); return None, None
        spin.end()
        msg = (obj.get("choices") or [{}])[0].get("message", {})
        text = msg.get("content") or ""
        if show_think and msg.get("reasoning_content"):
            print("%s💭 %s%s" % (C['dim'], msg["reasoning_content"].strip(), C['reset']))
        print(label + text)
        return text, obj.get("usage")

    spin = Spinner(); spin.start()
    try:
        resp = request(endpoint, headers, payload)
    except Exception as e:
        spin.end(); print(err_for(e)); return None, None

    first = True; in_think = False; parts = []; usage = None
    try:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            rc = delta.get("reasoning_content")
            ct = delta.get("content")
            if rc and show_think:
                if first:
                    spin.end(); sys.stdout.write(label); first = False
                if not in_think:
                    sys.stdout.write("%s💭 " % C['dim']); in_think = True
                sys.stdout.write(rc); sys.stdout.flush()
            if ct:
                if first:
                    spin.end(); sys.stdout.write(label); first = False
                if in_think:
                    sys.stdout.write("%s\n           " % C['reset']); in_think = False
                sys.stdout.write(ct); sys.stdout.flush()
                parts.append(ct)
    except KeyboardInterrupt:
        sys.stdout.write("%s [interrupted]%s" % (C['yellow'], C['reset']))
    if first:
        spin.end()
    sys.stdout.write("\n")
    return "".join(parts), usage

def stats_line(usage, dt):
    if not usage:
        return "%s(%.1fs)%s" % (C['dim'], dt, C['reset'])
    ct = usage.get("completion_tokens", 0)
    tps = (ct / dt) if dt > 0 else 0
    return "%s(%d prompt + %d completion tokens · %.1f tok/s · %.1fs)%s" % (
        C['dim'], usage.get("prompt_tokens", 0), ct, tps, dt, C['reset'])

def main():
    ap = argparse.ArgumentParser(prog="gputool chat", add_help=True,
        description="Chat with a local OpenAI-compatible LLM server (e.g. gputool's llama.cpp).")
    ap.add_argument("message", nargs="*", help="one-shot message; omit to start an interactive session")
    ap.add_argument("--host", default="127.0.0.1", help="server host/IP (default: 127.0.0.1)")
    ap.add_argument("--port", default="8080", help="server port (default: 8080)")
    ap.add_argument("--url", default=None, help="full base URL override, e.g. http://10.31.96.155:8080/v1")
    ap.add_argument("--api-key", default=os.environ.get("GPUTOOL_LLAMA_API_KEY", ""),
                    help="bearer token (default: $GPUTOOL_LLAMA_API_KEY)")
    ap.add_argument("--model", default=None, help="model name (default: auto-detected from /v1/models)")
    ap.add_argument("--system", default=None, help="system prompt")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--think", action="store_true", help="enable model reasoning/thinking output")
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="disable token streaming")
    ap.add_argument("--no-color", dest="color", action="store_false", help="disable colored output")
    args = ap.parse_args()

    setup_colors(args.color and sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb"))
    base = (args.url or ("http://%s:%s/v1" % (args.host, args.port))).rstrip("/")
    endpoint = base + "/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key

    think = {"enabled": args.think}
    model = args.model or fetch_model(base, headers) or "local-model"

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def build_payload():
        p = {"model": model, "messages": messages, "max_tokens": args.max_tokens,
             "temperature": args.temperature, "stream": args.stream}
        if args.stream:
            p["stream_options"] = {"include_usage": True}
        if not think["enabled"]:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        return p

    def ask():
        messages.append({"role": "user", "content": user})
        t = time.time()
        text, usage = run_turn(endpoint, headers, build_payload(), think["enabled"])
        if text is None:
            messages.pop()  # drop the user turn so history stays consistent on error
            return False
        messages.append({"role": "assistant", "content": text})
        print(stats_line(usage, time.time() - t))
        return True

    # One-shot mode
    if args.message:
        user = " ".join(args.message)
        sys.exit(0 if ask() else 1)

    # Interactive mode
    print("%s%s" % (C['cyan'], C['bold']))
    print("══════════════════════════════════════════════════")
    print(" 🦙 gputool chat — local OpenAI-compatible LLM")
    print("══════════════════════════════════════════════════%s" % C['reset'])
    print("%s  Endpoint : %s" % (C['dim'], endpoint))
    print("  Model    : %s" % model)
    print("  Auth     : %s   Streaming: %s   Thinking: %s" % (
        "on" if args.api_key else "off", "on" if args.stream else "off",
        "on" if think["enabled"] else "off"))
    print("  Commands : /exit  /reset  /system <text>  /think on|off  /help%s" % C['reset'])
    print()

    while True:
        try:
            user = input("%s%sYou ▸ %s" % (C['green'], C['bold'], C['reset'])).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n%sBye!%s" % (C['dim'], C['reset'])); break
        if not user:
            continue
        low = user.lower()
        if low in ("/exit", "/quit", "/q"):
            print("%sBye!%s" % (C['dim'], C['reset'])); break
        if low in ("/reset", "/clear"):
            keep = [m for m in messages if m["role"] == "system"]
            messages.clear(); messages.extend(keep)
            print("%s↺ Conversation reset.%s" % (C['yellow'], C['reset'])); continue
        if low in ("/help", "/?"):
            print("%s  /exit  quit    |  /reset  clear history    |  /system <text>  set system prompt\n"
                  "  /think on|off   toggle reasoning output%s" % (C['dim'], C['reset'])); continue
        if low.startswith("/system"):
            sp = user[len("/system"):].strip()
            messages[:] = [m for m in messages if m["role"] != "system"]
            if sp:
                messages.insert(0, {"role": "system", "content": sp})
                print("%s✎ System prompt set.%s" % (C['yellow'], C['reset']))
            else:
                print("%s✎ System prompt cleared.%s" % (C['yellow'], C['reset']))
            continue
        if low.startswith("/think"):
            arg = low[len("/think"):].strip()
            think["enabled"] = (arg == "on") if arg in ("on", "off") else (not think["enabled"])
            print("%s🧠 Thinking %s.%s" % (C['yellow'], "enabled" if think["enabled"] else "disabled", C['reset']))
            continue
        ask()

if __name__ == "__main__":
    main()
