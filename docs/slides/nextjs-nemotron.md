---
marp: true
paginate: true
size: 16:9
title: Web AI App — Next.js + NVIDIA Nemotron
---

<style>
:root { --blue:#0055A2; --gold:#E5A823; --ink:#202a3c; }
section { background:#fff; color:var(--ink); font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:21px; line-height:1.45; padding:46px 62px 54px; border-top:7px solid var(--blue); }
section::before { content:""; position:absolute; left:0; right:0; top:7px; height:3px; background:var(--gold); }
h1 { color:var(--blue); font-size:1.85em; margin:0 0 .3em; }
h2 { color:var(--blue); font-size:1.3em; border-bottom:2px solid var(--gold); padding-bottom:6px; margin:0 0 .5em; }
h3 { color:#0a3d7a; }
strong { color:var(--blue); }
a { color:var(--blue); text-decoration:none; border-bottom:1px solid var(--gold); }
code { background:#eef2f8; color:#0a3d7a; border-radius:5px; padding:.05em .35em; font-size:.92em; }
pre { background:#0f1830; border-radius:10px; box-shadow:0 6px 18px rgba(8,20,50,.12); }
pre code { background:transparent; color:#e8eefc; font-size:.78em; line-height:1.5; }
blockquote { border-left:4px solid var(--gold); background:#fbf6e9; color:#5b4a22; padding:.4em .9em; border-radius:6px; }
img { border-radius:10px; box-shadow:0 6px 18px rgba(8,20,50,.18); }
section::after { color:#9aa7bd; }
.step { background:var(--blue); color:#fff; border-radius:999px; padding:.03em .6em; font-weight:700; font-size:.85em; }
.tiny { font-size:.78em; color:#5d6b82; }
.cols { display:flex; gap:26px; align-items:center; } .cols > * { flex:1; }
section.lead { text-align:center; border-top-width:10px; }
section.lead h1 { font-size:2.3em; }
</style>

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

## <span class="step">5</span> Setup — keys + Node

**Keys** come from your **`~/.env.local`** (the same file `sjsujetsontool chat` saved). Add any of:

```bash
echo "NVIDIA_API_KEY=nvapi-…"     >> ~/.env.local   # build.nvidia.com (free)
echo "OPENAI_API_KEY=sk-…"        >> ~/.env.local   # platform.openai.com
echo "ANTHROPIC_API_KEY=sk-ant-…" >> ~/.env.local   # console.anthropic.com
```

The app picks the provider from the **model you choose** (`nvidia/…`, `gpt-…`, `claude-…`).

> 🟢 **Node/npm live in the container** — there's no `npm` on the host. Run everything inside
> `sjsujetsontool shell` (it also passes your `~/.env.local` keys into the container).

---

## <span class="step">6</span> Run it

```bash
sjsujetsontool shell                                  # Node 20 + npm + your keys
cd /Developer/edgeAI/edgeLLM/nextjs-nemotron-app
npm install                                           # first time (~30-60 s)
npm run dev                                           # serves on 0.0.0.0:3000
```

Open it **from your laptop** using the Jetson's IP:

```bash
hostname -I | awk '{print $1}'        # e.g. 192.168.5.206  → http://192.168.5.206:3000
```

> Each message streams: Jetson → model API → Jetson → your browser, with a TTFT / tokens-per-second line.

---

## <span class="step">7</span> Extend it — same pattern every time

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

## <span class="step">8</span> Make it your own (push to GitHub)

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
