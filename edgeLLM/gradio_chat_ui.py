#!/usr/bin/env python3
"""
gradio_chat_ui.py — a simple multi-backend chat UI (Gradio) for Edge AI labs.

It talks to the SAME OpenAI-compatible backends as `sjsujetsontool chat`:
  • Local Jetson llama.cpp  (http://localhost:8080/v1 — start with `sjsujetsontool llama`)
  • NVIDIA Build API · OpenAI · Anthropic Claude (OpenAI-compatible endpoint) · any custom URL

Features
  - Chat with history, choose backend + model, adjust max_tokens / thinking.
  - Attach a **text file** (its content is added to your prompt).
  - Attach an **image** (sent as OpenAI `image_url`) for vision models (Qwen3.5, Gemma-4, GPT-4o…).
  - Cloud API keys are read from ~/.env.local (the same file the CLI uses); you can also paste a key.

Run:
  pip install gradio requests
  python3 edgeLLM/gradio_chat_ui.py            # http://localhost:7860
"""
import os, time, base64, mimetypes, argparse
import requests
import gradio as gr

# Gradio 6.0 changed a few APIs (Chatbot is messages-only and dropped `type=`;
# `theme` moved from Blocks() to launch()). Detect the major version and adapt.
try:
    _GVER = int(gr.__version__.split(".")[0])
except Exception:
    _GVER = 5
try:
    _THEME = gr.themes.Soft(primary_hue="indigo")
except Exception:
    _THEME = None

# --------------------------------------------------------------------------- API keys (~/.env.local)
HOME_ENV = os.path.join(os.path.expanduser("~"), ".env.local")
_ENV_FALLBACKS = ["/Developer/edgeAI/edgeLLM/nextjs-nemotron-app/.env.local"]

def _parse_env(path):
    d = {}
    try:
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("="); d[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return d

def load_env_keys():
    merged = {}
    for p in _ENV_FALLBACKS:
        merged.update(_parse_env(p))
    merged.update(_parse_env(HOME_ENV))   # ~/.env.local wins
    return merged

ENV_KEYS = load_env_keys()

# --------------------------------------------------------------------------- backends (match sjsujetsontool chat)
BACKENDS = {
    "Local Jetson llama.cpp": dict(url="http://localhost:8080/v1", key_env=None, ntk=False, model=""),
    "NVIDIA Build API":       dict(url="https://integrate.api.nvidia.com/v1", key_env="NVIDIA_API_KEY", ntk=True,
                                   model="nvidia/llama-3.1-nemotron-nano-8b-v1"),
    "OpenAI":                 dict(url="https://api.openai.com/v1", key_env="OPENAI_API_KEY", ntk=True, model="gpt-4o-mini"),
    "Anthropic Claude":       dict(url="https://api.anthropic.com/v1", key_env="ANTHROPIC_API_KEY", ntk=True,
                                   model="claude-haiku-4-5"),
    "Custom (OpenAI-compatible)": dict(url="", key_env=None, ntk=False, model=""),
}

def backend_defaults(name):
    b = BACKENDS[name]
    key = ENV_KEYS.get(b["key_env"], "") if b["key_env"] else ""
    return b["model"], key, b["url"]

def detect_model(base, headers):
    try:
        r = requests.get(base.rstrip("/") + "/models", headers=headers, timeout=5)
        return (r.json().get("data") or [{}])[0].get("id") or "local-model"
    except Exception:
        return "local-model"

def _image_data_uri(path):
    data = open(path, "rb").read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())

# --------------------------------------------------------------------------- chat
def respond(message, image, textfile, history, backend, model, api_key, custom_url, max_tokens, thinking):
    history = history or []
    message = (message or "").strip()
    if not message and not image and not textfile:
        return history, history, "", None, None

    b = BACKENDS.get(backend, BACKENDS["Local Jetson llama.cpp"])
    base = (custom_url.strip() if (backend.startswith("Custom") and custom_url.strip()) else b["url"]).rstrip("/")
    key = (api_key or "").strip() or (ENV_KEYS.get(b["key_env"], "") if b["key_env"] else "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    mdl = (model or "").strip() or b["model"] or detect_model(base, headers)

    # Build the prompt text (+ optional attached text file)
    text = message
    note = ""
    if textfile:
        try:
            content = open(textfile, "r", errors="ignore").read()
            text += "\n\n--- attached file: %s ---\n%s" % (os.path.basename(textfile), content[:20000])
            note += "  📎 " + os.path.basename(textfile)
        except Exception as e:
            note += "  ⚠️ file read failed"
    if image:
        note += "  🖼️ image"

    # OpenAI message content: plain string, or array with an image for vision models
    if image:
        user_content = [{"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": _image_data_uri(image)}}]
    else:
        user_content = text

    # Prior turns (text only) + this turn
    api_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    api_messages.append({"role": "user", "content": user_content})

    payload = {"model": mdl, "messages": api_messages, "max_tokens": int(max_tokens), "stream": False}
    if not thinking and not b["ntk"]:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    # Show the user's turn immediately
    history = history + [{"role": "user", "content": message + note}]
    t0 = time.time()
    try:
        r = requests.post(base + "/chat/completions", headers=headers, json=payload, timeout=300)
        if not r.ok:
            reply = "⚠️ HTTP %s: %s" % (r.status_code, r.text[:300])
            usage = {}
        else:
            d = r.json()
            reply = d["choices"][0]["message"].get("content") or "(empty)"
            usage = d.get("usage", {})
    except Exception as e:
        reply, usage = "⚠️ Request failed: %s" % e, {}

    dt = time.time() - t0
    ct = usage.get("completion_tokens", 0)
    if ct:
        reply += "\n\n— *%d tok · %.1f tok/s · %.1fs*" % (ct, ct / dt if dt else 0, dt)
    history = history + [{"role": "assistant", "content": reply}]
    return history, history, "", None, None  # clears prompt, image, file

# --------------------------------------------------------------------------- UI
_blocks_kw = {"title": "Edge AI Chat UI"}
if _THEME is not None and _GVER < 6:        # Gradio <6: theme on Blocks
    _blocks_kw["theme"] = _THEME
with gr.Blocks(**_blocks_kw) as demo:
    gr.Markdown("## 🧠 Edge AI Chat UI — local & cloud backends (text + file + image)")
    with gr.Row():
        with gr.Column(scale=3):
            _chat_kw = {"label": "Chat", "height": 520}
            if _GVER < 6:                   # Gradio <6: opt into messages + copy button
                _chat_kw["type"] = "messages"
                _chat_kw["show_copy_button"] = True
            chatbot = gr.Chatbot(**_chat_kw)
            with gr.Row():
                prompt = gr.Textbox(placeholder="Ask something…  (attach a file/image on the right)",
                                    lines=2, scale=6, show_label=False)
                send = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                image_in = gr.Image(label="🖼️ Image (for vision models)", type="filepath", height=120)
                file_in = gr.File(label="📎 Text file", type="filepath", file_types=[".txt", ".md", ".py", ".json", ".csv", ".log"])
            clear_btn = gr.Button("🧹 Clear chat")
        with gr.Column(scale=1):
            backend = gr.Dropdown(list(BACKENDS.keys()), value="Local Jetson llama.cpp", label="Backend")
            model = gr.Textbox(value="", label="Model (blank = auto/default)")
            api_key = gr.Textbox(value="", label="API key (from ~/.env.local; paste to override)", type="password")
            custom_url = gr.Textbox(value="", label="Custom base URL (for Custom backend)", visible=False)
            max_tokens = gr.Slider(64, 2048, value=256, step=64, label="max_tokens")
            thinking = gr.Checkbox(value=False, label="Enable thinking (slower)")
            gr.Markdown("Keys are read from **~/.env.local** (private — don't share it). "
                        "Start a local model with `sjsujetsontool llama`.")

    state = gr.State([])

    def on_backend(name):
        mdl, key, url = backend_defaults(name)
        return mdl, key, gr.update(value=url, visible=name.startswith("Custom"))
    backend.change(on_backend, inputs=[backend], outputs=[model, api_key, custom_url])

    _inputs = [prompt, image_in, file_in, state, backend, model, api_key, custom_url, max_tokens, thinking]
    _outputs = [chatbot, state, prompt, image_in, file_in]
    send.click(respond, inputs=_inputs, outputs=_outputs)
    prompt.submit(respond, inputs=_inputs, outputs=_outputs)
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Edge AI multi-backend chat UI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    print("Starting Edge AI Chat UI on http://0.0.0.0:%d" % args.port)
    _launch_kw = {"server_name": "0.0.0.0", "server_port": args.port, "share": args.share, "show_error": True}
    if _THEME is not None and _GVER >= 6:   # Gradio >=6: theme on launch()
        _launch_kw["theme"] = _THEME
    demo.queue(max_size=16).launch(**_launch_kw)
