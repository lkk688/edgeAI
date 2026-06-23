---
marp: true
theme: sjsu
paginate: true
size: 16:9
title: Web AI App — Next.js + NVIDIA Nemotron
---

<!-- _class: lead -->
# 🌐 Build a Web AI App
### Next.js + NVIDIA Nemotron (Build API)

`SJSU · Edge AI`

<span class="tiny">A streaming chatbot web app that runs on your Jetson and serves to any browser.</span>

---

## <span class="step">1</span> What you'll build

<div class="cols">
<div>

- A **Next.js** web app with a **streaming chat** UI.
- Talks to **NVIDIA Nemotron** (and optionally **OpenAI** / **Anthropic**) via cloud APIs.
- **Runs on the Jetson**, opened from your laptop's browser.
- 🔒 Your API key stays **server‑side** — never sent to the browser.

</div>
<div>

![w:430](../curriculum/figures/chat_ui.png)

</div>
</div>

---

## <span class="step">2</span> The big picture

<div class="cols">
<div>

Your browser talks to a small **server on the Jetson**, which talks to the model API:

`Browser → /api/chat (on Jetson) → NVIDIA/OpenAI/Anthropic → stream back`

- The **page** runs in the browser (buttons, live text).
- The **API route** runs on the server — it holds the key and calls the model.
- The browser **never sees the key**.

</div>
<div>

![w:430](../curriculum/figures/security_architecture.png)

</div>
</div>

---

## <span class="step">3</span> Key web concepts (background)

- **Next.js** = a React framework. Each folder under `app/` is a page (the *App Router*).
- **Server Components / routes** run on the Jetson — they can hold **secrets** and call APIs.
- **Client Components** run in the **browser** — they're interactive (`"use client"`).
- **Streaming (SSE)**: the server forwards tokens one chunk at a time, so the answer appears live.

> Rule of thumb: **secrets + external API calls → server**; **buttons + live updates → client**.

---

## <span class="step">4</span> Where's the code (and what each part does)

```text
edgeLLM/nextjs-nemotron-app/
  app/page.js               # the chat UI  — Client Component (browser)
  app/components/ChatUI.js  # streaming chat box
  app/api/chat/route.js     # server route — holds the key, streams from the model
  app/api/models/route.js   # the model dropdown list
  lib/providers.js          # picks NVIDIA/OpenAI/Anthropic + reads keys from ~/.env.local
  .env.local                # optional local keys (git-ignored)
```

<span class="tiny">📦 Repo: [edgeLLM/nextjs-nemotron-app](https://github.com/lkk688/edgeAI/tree/main/edgeLLM/nextjs-nemotron-app)</span>

---

## <span class="step">5</span> Setup — API keys

**Keys** come from your **`~/.env.local`** (the same file `sjsujetsontool chat` saved). Add any of:

```bash
echo "NVIDIA_API_KEY=nvapi-…"     >> ~/.env.local   # build.nvidia.com (free)
echo "OPENAI_API_KEY=sk-…"        >> ~/.env.local   # platform.openai.com
echo "ANTHROPIC_API_KEY=sk-ant-…" >> ~/.env.local   # console.anthropic.com
```

The app picks the provider from the **model you choose** (`nvidia/…`, `gpt-…`, `claude-…`).

> 🛠️ **No Node on the host or in the container yet?** That's normal — the next slide installs
> everything with one command. *No `sudo` is needed anywhere.*

---

## <span class="step">6</span> Run it — `sjsujetsontool node`

One command installs Node in the container, runs `npm install`, and starts the dev server.
Run from **anywhere** on the host (your home is fine):

```bash
sjsujetsontool node             # interactive: prompts for path + mode
```

It asks the path with a sensible default — press *Enter* for this lesson:

```text
🟢 node v20.20.2 · npm 10.8.2  (inside container jetson-dev)
📁 Project path? [Enter = /Developer/edgeAI/edgeLLM/nextjs-nemotron-app]:
📦 Project: /Developer/edgeAI/edgeLLM/nextjs-nemotron-app
▶️  Start the frontend now? [f]oreground / [b]ackground / [n]o:  b
🚀 Starting in BACKGROUND on port 3000.   • URL: http://192.168.5.206:3000
```

**Shortcuts (skip the prompts):**

```bash
sjsujetsontool node bg                          # bg, default path
sjsujetsontool node fg /Developer/my-vite-app   # fg, explicit path (any order)
sjsujetsontool node /Developer/my-app bg        # path + mode, swapped
sjsujetsontool node stop                        # stop a background server
```

> Default path: `/Developer/edgeAI/edgeLLM/nextjs-nemotron-app` — override with
> `SJSUJETSONTOOL_NODE_DIR=/Developer/foo` in your shell rc.
> Path **must live under `/Developer/`** (that's the dir the container mounts 1:1 from the host).

---

## <span class="step">7</span> Manual install (what `sjsujetsontool node` does for you)

If you ever need to install Node by hand — or you want to see what the one-step command
runs — open a container shell and use NodeSource's apt repo (Ubuntu 24.04 aarch64, root inside,
*no sudo*):

```bash
sjsujetsontool shell                                # drops into root@jetson-dev:/workspace
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs                           # → node v20.20.2 · npm 10.8.2
```

Then build and run the app the classic way:

```bash
cd /Developer/edgeAI/edgeLLM/nextjs-nemotron-app
npm install                                         # first time only (~30-60 s)
npm run dev                                         # serves on 0.0.0.0:3000
```

> The Node install lives in the **container's writable layer** — it persists across
> `sjsujetsontool shell` invocations until the image is rebuilt. `node_modules/` lives
> on the host SSD because the parent dir is a host mount.

---

## <span class="step">8</span> Open it from your laptop

Find the Jetson's IP and open it in any browser:

```bash
hostname -I | awk '{print $1}'        # e.g. 192.168.5.206  → http://192.168.5.206:3000
```

Each message streams **Jetson → model API → Jetson → your browser**, with a live TTFT and
tokens-per-second line under the chat. To stop a backgrounded server:

```bash
sjsujetsontool node stop
```

---

## <span class="step">9</span> Extend it — same pattern every time

Every feature = **one page** (UI) + **one API route** (server logic). Copy the pattern:

```text
app/<feature>/page.js        # the page/UI
app/api/<feature>/route.js   # server: read key, call a model, return/stream
```

The bonus labs are exactly this — add a route + page and you've extended the app:

<div class="cols">
<div>

🔎 **Retrieval** · 🖼️ **Omni** (vision) · 🎙️ **ASR** · 🔊 **TTS**

</div>
<div>

![w:280](../curriculum/figures/retrieval_ui.png) ![w:280](../curriculum/figures/omni_ui.png)

</div>
</div>

---

## <span class="step">10</span> Make it your own (push to GitHub)

Copy the app into your own folder, then create your repo:

```bash
cp -r /Developer/edgeAI/edgeLLM/nextjs-nemotron-app ~/my-ai-app
cd ~/my-ai-app && rm -rf node_modules .next
git init && git add -A && git commit -m "My Edge AI web app"
```

Create an empty repo on **github.com** (the **＋ → New repository**), then:

```bash
git remote add origin https://github.com/<your-username>/my-ai-app.git
git branch -M main && git push -u origin main
```

<span class="tiny">With the GitHub CLI it's one line: <code>gh repo create my-ai-app --public --source=. --push</code></span>

---

<!-- _class: lead -->
## 📚 Full walkthrough

Step‑by‑step build, code, and the four bonus labs:

[lkk688.github.io/edgeAI/curriculum/11_nextjs_nemotron_app](https://lkk688.github.io/edgeAI/curriculum/11_nextjs_nemotron_app/)

<span class="tiny">← Back to the [Get‑Started slides](get-started.html)</span>
