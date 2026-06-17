#!/usr/bin/env python3
# gputool chat — terminal client for an OpenAI-compatible LLM server (e.g. llama.cpp).
#
# This file is the single source of truth for the `gputool chat` command. It is
# fetched to ~/.gputool/chat.py automatically by `gputool install` and
# `gputool update`, so new devices get it without any manual copying.
#
# UI: uses the `rich` library for a Claude-Code-style experience (streaming
# Markdown, syntax-highlighted code, panels) when it is installed, and falls
# back to a pure-stdlib ANSI renderer otherwise — so it still runs on
# locked-down machines without pip packages or curl.
import sys, os, json, time, argparse, threading, itertools
import urllib.request, urllib.error

try:
    import rich  # noqa: F401
    RICH_OK = True
except Exception:
    RICH_OK = False

GPUTOOL_DIR = os.path.join(os.path.expanduser("~"), ".gputool")
CONFIG_PATH = os.path.join(GPUTOOL_DIR, "chat_config.json")

# ----------------------------------------------------------------------------- config
def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f) or {}
    except Exception:
        return {}

def save_config(cfg):
    try:
        os.makedirs(GPUTOOL_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        try:
            os.chmod(CONFIG_PATH, 0o600)  # the file can hold an API key
        except Exception:
            pass
    except Exception:
        pass

def save_history(messages, path=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    if not path:
        path = os.path.join(os.getcwd(), "gputool_chat_%s.md" % ts)
    path = os.path.expanduser(path)
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    else:
        titles = {"system": "System", "user": "You", "assistant": "Assistant"}
        out = ["# gputool chat — %s\n" % ts]
        for m in messages:
            out.append("## %s\n\n%s\n" % (titles.get(m["role"], m["role"]), m.get("content", "")))
        with open(path, "w") as f:
            f.write("\n".join(out))
    return path

# ----------------------------------------------------------------------------- helpers
def read_line(prompt):
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return ""

def read_secret(prompt):
    if sys.stdin.isatty():
        try:
            import getpass
            return getpass.getpass(prompt)
        except Exception:
            pass
    return read_line(prompt)

def build_base(host, port, url):
    if url:
        b = url.rstrip("/")
        return b if b.endswith("/v1") else b + "/v1"
    return "http://%s:%s/v1" % (host, port)

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
            return ("Authentication failed (401). The server requires an API key — "
                    "use /server to re-enter it, pass --api-key, or set GPUTOOL_LLAMA_API_KEY.")
        return "HTTP %s: %s" % (e.code, body[:300])
    if isinstance(e, urllib.error.URLError):
        return ("Cannot connect: %s. Is the server running and reachable at the given host/port?"
                % getattr(e, "reason", e))
    return str(e)

def iter_events(resp):
    """Yield ('reasoning'|'content'|'usage', value) tuples from an SSE stream."""
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
            yield ("usage", obj["usage"])
        for ch in obj.get("choices") or []:
            d = ch.get("delta", {})
            if d.get("reasoning_content"):
                yield ("reasoning", d["reasoning_content"])
            if d.get("content"):
                yield ("content", d["content"])

def stats_line(usage, dt):
    if not usage:
        return "(%.1fs)" % dt
    ct = usage.get("completion_tokens", 0)
    tps = (ct / dt) if dt > 0 else 0
    return "(%d prompt + %d completion tokens · %.1f tok/s · %.1fs)" % (
        usage.get("prompt_tokens", 0), ct, tps, dt)

HELP_ROWS = [
    ("/exit", "Quit the chat (also /quit, /q)"),
    ("/server", "Connect to a different server IP / API key"),
    ("/save [file]", "Save the conversation (.md default, or .json)"),
    ("/reset", "Clear conversation history (keeps system prompt)"),
    ("/system <text>", "Set (or clear) the system prompt"),
    ("/think on|off", "Toggle the model's reasoning output"),
    ("/help", "Show this help (also /?)"),
]

# ----------------------------------------------------------------------------- plain (stdlib) renderer
C = {}
def setup_colors(on):
    names = dict(reset="\033[0m", bold="\033[1m", dim="\033[2m",
                 red="\033[31m", green="\033[32m", yellow="\033[33m",
                 blue="\033[34m", magenta="\033[35m", cyan="\033[36m")
    for k, v in names.items():
        C[k] = v if on else ""

class _Spinner:
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

class PlainRenderer:
    def __init__(self, show_think, color):
        self.show_think = show_think
        setup_colors(color and sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb"))
        self._spin = None; self._first = True; self._in_think = False

    def banner(self, info):
        print("%s%s" % (C['cyan'], C['bold']))
        print("══════════════════════════════════════════════════")
        print(" 🦙 gputool chat — local OpenAI-compatible LLM")
        print("══════════════════════════════════════════════════%s" % C['reset'])
        print("%s  Endpoint : %s" % (C['dim'], info["endpoint"]))
        print("  Model    : %s" % info["model"])
        print("  Auth     : %s   Streaming: %s   Thinking: %s%s" % (
            info["auth"], info["streaming"], info["thinking"], C['reset']))
        print("%s  Type a message and press Enter.  /help for commands, /exit to quit.%s"
              % (C['dim'], C['reset']))
        print()

    def help(self):
        for cmd, desc in HELP_ROWS:
            print("%s  %-16s%s %s" % (C['bold'], cmd, C['reset'], desc))

    def info(self, msg):   print("%s%s%s" % (C['dim'], msg, C['reset']))
    def notice(self, msg): print("%s%s%s" % (C['yellow'], msg, C['reset']))
    def error(self, msg):  print("%s✖ %s%s" % (C['red'], msg, C['reset']))
    def stats(self, line): print("%s%s%s" % (C['dim'], line, C['reset']))

    def ask_user(self):
        try:
            return input("%s%sYou ▸ %s" % (C['green'], C['bold'], C['reset']))
        except (EOFError, KeyboardInterrupt):
            return None

    def begin(self):
        self._first = True; self._in_think = False
        self._spin = _Spinner(); self._spin.start()
    def _label(self):
        sys.stdout.write("%s%sAssistant ▸ %s" % (C['blue'], C['bold'], C['reset']))
    def reasoning(self, t):
        if not self.show_think:
            return
        if self._first:
            self._spin.end(); self._label(); self._first = False
        if not self._in_think:
            sys.stdout.write("%s💭 " % C['dim']); self._in_think = True
        sys.stdout.write(t); sys.stdout.flush()
    def content(self, t):
        if self._first:
            self._spin.end(); self._label(); self._first = False
        if self._in_think:
            sys.stdout.write("%s\n           " % C['reset']); self._in_think = False
        sys.stdout.write(t); sys.stdout.flush()
    def end(self):
        if self._first:
            self._spin.end()
        sys.stdout.write("\n")

# ----------------------------------------------------------------------------- rich renderer
class RichRenderer:
    def __init__(self, show_think, color):
        from rich.console import Console
        self.show_think = show_think
        self.console = Console(no_color=not color)
        self._live = None; self._cbuf = ""; self._rbuf = ""

    def banner(self, info):
        from rich.panel import Panel
        from rich.text import Text
        body = Text()
        body.append("Endpoint  ", style="bold"); body.append(info["endpoint"] + "\n", style="cyan")
        body.append("Model     ", style="bold"); body.append(info["model"] + "\n")
        body.append("Auth %s   Streaming %s   Thinking %s\n" % (
            info["auth"], info["streaming"], info["thinking"]), style="dim")
        body.append("\nType a message and press Enter.  ", style="dim")
        body.append("/help", style="bold cyan"); body.append(" for commands · ", style="dim")
        body.append("/exit", style="bold cyan"); body.append(" to quit.", style="dim")
        self.console.print(Panel(body, title="🦙 gputool chat — local LLM",
                                 border_style="cyan", padding=(1, 2)))

    def help(self):
        from rich.table import Table
        t = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        t.add_column(style="bold cyan", no_wrap=True)
        t.add_column(style="dim")
        for cmd, desc in HELP_ROWS:
            t.add_row(cmd, desc)
        self.console.print(t)

    def info(self, msg):   self.console.print(msg, style="dim")
    def notice(self, msg): self.console.print(msg, style="yellow")
    def error(self, msg):  self.console.print("✖ " + msg, style="bold red")
    def stats(self, line): self.console.print(line, style="dim")

    def ask_user(self):
        try:
            return self.console.input("[bold green]You ▸ [/]")
        except (EOFError, KeyboardInterrupt):
            return None

    def _render(self):
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.spinner import Spinner
        from rich.text import Text
        if not self._cbuf and not self._rbuf:
            return Panel(Spinner("dots", text=Text(" thinking…", style="dim")),
                         title="Assistant ▸", border_style="blue", padding=(0, 1))
        md = ""
        if self._rbuf and self.show_think:
            quoted = "\n".join("> " + ln for ln in self._rbuf.strip().splitlines())
            md += "> 💭 *thinking…*\n" + quoted + "\n\n"
        md += self._cbuf
        return Panel(Markdown(md) if md.strip() else Text("…"),
                     title="Assistant ▸", border_style="blue", padding=(0, 1))

    def begin(self):
        from rich.live import Live
        self._cbuf = ""; self._rbuf = ""
        self._live = Live(self._render(), console=self.console,
                          refresh_per_second=12, transient=False)
        self._live.start()
    def reasoning(self, t):
        self._rbuf += t
        if self._live:
            self._live.update(self._render())
    def content(self, t):
        self._cbuf += t
        if self._live:
            self._live.update(self._render())
    def end(self):
        if self._live:
            self._live.update(self._render())
            self._live.stop(); self._live = None

# ----------------------------------------------------------------------------- chat turn
def chat_turn(endpoint, headers, payload, renderer):
    """Run one request. Returns (assistant_text, usage) or (None, None) on error."""
    try:
        resp = request(endpoint, headers, payload)
    except Exception as e:
        renderer.error(err_for(e)); return None, None

    if not payload.get("stream"):
        try:
            obj = json.load(resp)
        except Exception as e:
            renderer.error(err_for(e)); return None, None
        msg = (obj.get("choices") or [{}])[0].get("message", {})
        renderer.begin()
        if renderer.show_think and msg.get("reasoning_content"):
            renderer.reasoning(msg["reasoning_content"])
        renderer.content(msg.get("content") or "")
        renderer.end()
        return msg.get("content") or "", obj.get("usage")

    renderer.begin()
    parts = []; usage = None
    try:
        for kind, val in iter_events(resp):
            if kind == "usage":
                usage = val
            elif kind == "reasoning":
                renderer.reasoning(val)
            elif kind == "content":
                parts.append(val); renderer.content(val)
    except KeyboardInterrupt:
        pass
    renderer.end()
    return "".join(parts), usage

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(prog="gputool chat", add_help=True,
        description="Chat with a local OpenAI-compatible LLM server (e.g. gputool's llama.cpp).")
    ap.add_argument("message", nargs="*", help="one-shot message; omit to start an interactive session")
    ap.add_argument("--host", default=None, help="server host/IP (default: saved or 127.0.0.1)")
    ap.add_argument("--port", default=None, help="server port (default: saved or 8080)")
    ap.add_argument("--url", default=None, help="full base URL override, e.g. http://10.31.96.155:8080/v1")
    ap.add_argument("--api-key", default=None, help="bearer token (default: saved or $GPUTOOL_LLAMA_API_KEY)")
    ap.add_argument("--model", default=None, help="model name (default: auto-detected from /v1/models)")
    ap.add_argument("--system", default=None, help="system prompt")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--think", action="store_true", help="enable model reasoning/thinking output")
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="disable token streaming")
    ap.add_argument("--no-color", dest="color", action="store_false", help="disable colored output")
    ap.add_argument("--plain", action="store_true", help="force the plain stdlib renderer (no rich)")
    ap.add_argument("--reset-config", action="store_true", help="ignore and overwrite the saved chat config")
    args = ap.parse_args()

    cfg = {} if args.reset_config else load_config()

    if args.plain or not RICH_OK:
        renderer = PlainRenderer(args.think, args.color)
    else:
        renderer = RichRenderer(args.think, args.color)

    env_key = os.environ.get("GPUTOOL_LLAMA_API_KEY")

    def prompt_server():
        dh = args.host or cfg.get("host") or "127.0.0.1"
        dp = args.port or cfg.get("port") or "8080"
        if args.url:
            return None, None, args.url
        raw = read_line("Server IP or URL [%s:%s]: " % (dh, dp)).strip()
        if not raw:
            return dh, dp, None
        if "://" in raw:
            return None, None, raw
        host, port = raw, dp
        if ":" in raw and not raw.startswith("["):
            h, _, p = raw.partition(":")
            host, port = (h or dh), (p or dp)
        return host, port, None

    def prompt_key():
        saved = args.api_key or env_key or cfg.get("api_key") or ""
        hint = " [Enter to keep saved]" if saved else " [blank if none]"
        val = read_secret("API key%s: " % hint).strip()
        return val if val else saved

    def persist(host, port, url, key):
        cfg["host"] = host or ""
        cfg["port"] = port or ""
        cfg["url"] = url or ""
        if key:
            cfg["api_key"] = key
        save_config(cfg)

    def connect(host, port, url, key):
        base = build_base(host, port, url)
        endpoint = base + "/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if key:
            headers["Authorization"] = "Bearer " + key
        model = args.model or fetch_model(base, headers) or "local-model"
        return base, endpoint, headers, model

    # Resolve the connection: prompt interactively for a bare `gputool chat`,
    # otherwise (one-shot message) use flags / saved config without prompting.
    if args.message:
        host = args.host or cfg.get("host") or "127.0.0.1"
        port = args.port or cfg.get("port") or "8080"
        url = args.url or (cfg.get("url") or None)
        key = args.api_key or env_key or cfg.get("api_key") or ""
    else:
        host, port, url = prompt_server()
        key = prompt_key()
        persist(host, port, url, key)

    base, endpoint, headers, model = connect(host, port, url, key)

    think = {"enabled": args.think}
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

    def ask(user_text):
        messages.append({"role": "user", "content": user_text})
        renderer.show_think = think["enabled"]
        t = time.time()
        text, usage = chat_turn(endpoint, headers, build_payload(), renderer)
        if text is None:
            messages.pop()
            return False
        messages.append({"role": "assistant", "content": text})
        renderer.stats(stats_line(usage, time.time() - t))
        return True

    def info_block():
        return {"endpoint": endpoint, "model": model,
                "auth": "on" if key else "off",
                "streaming": "on" if args.stream else "off",
                "thinking": "on" if think["enabled"] else "off"}

    # One-shot mode
    if args.message:
        sys.exit(0 if ask(" ".join(args.message)) else 1)

    # Interactive mode
    if not RICH_OK and not args.plain:
        renderer.info("Tip: `pip install rich` for a nicer UI (streaming Markdown & syntax highlighting).")
    renderer.banner(info_block())

    while True:
        user = renderer.ask_user()
        if user is None:
            renderer.notice("Bye!"); break
        user = user.strip()
        if not user:
            continue
        low = user.lower()
        if low in ("/exit", "/quit", "/q"):
            renderer.notice("Bye!"); break
        if low in ("/help", "/?"):
            renderer.help(); continue
        if low in ("/reset", "/clear"):
            messages[:] = [m for m in messages if m["role"] == "system"]
            renderer.notice("↺ Conversation reset."); continue
        if low.startswith("/save"):
            arg = user[len("/save"):].strip() or None
            try:
                p = save_history(messages, arg)
                renderer.notice("💾 Saved conversation to %s" % p)
            except Exception as e:
                renderer.error("Save failed: %s" % e)
            continue
        if low.startswith("/server"):
            host, port, url = prompt_server()
            key = prompt_key()
            persist(host, port, url, key)
            base, endpoint, headers, model = connect(host, port, url, key)
            renderer.notice("🔌 Connected to %s   (model: %s)" % (base, model))
            continue
        if low.startswith("/system"):
            sp = user[len("/system"):].strip()
            messages[:] = [m for m in messages if m["role"] != "system"]
            if sp:
                messages.insert(0, {"role": "system", "content": sp})
                renderer.notice("✎ System prompt set.")
            else:
                renderer.notice("✎ System prompt cleared.")
            continue
        if low.startswith("/think"):
            a = low[len("/think"):].strip()
            think["enabled"] = (a == "on") if a in ("on", "off") else (not think["enabled"])
            renderer.notice("🧠 Thinking %s." % ("enabled" if think["enabled"] else "disabled"))
            continue
        ask(user)

if __name__ == "__main__":
    main()
