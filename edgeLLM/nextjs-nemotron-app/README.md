# Next.js × NVIDIA Nemotron — Edge AI Tutorial App

A minimal Next.js (App Router) chat client that streams responses from the
**NVIDIA Build** OpenAI-compatible API, using **Llama-Nemotron** reasoning
models hosted on `https://integrate.api.nvidia.com/v1`.

The full step-by-step write-up lives at
[`docs/curriculum/11_nextjs_nemotron_app.md`](../../docs/curriculum/11_nextjs_nemotron_app.md).

## Quick start

```bash
cp .env.local.example .env.local
# edit .env.local and paste your NVIDIA Build key (https://build.nvidia.com)

npm install
npm run dev      # http://localhost:3000  (and http://<jetson-ip>:3000 on LAN)
```

## Layout

```
app/
├── layout.js               # root layout — mounts <NavBar/> + global CSS
├── page.js                 # /          → <ChatUI/>
├── retrieval/page.js       # /retrieval → <RetrievalLab/>
├── omni/page.js            # /omni      → <OmniLab/>
├── asr/page.js             # /asr       → <AsrLab/>
├── tts/page.js             # /tts       → <TtsLab/>
├── globals.css             # NVIDIA-green dark theme + nav styles
├── components/
│   ├── NavBar.js           # client: shared top nav, active-link highlight
│   ├── ChatUI.js           # client: streaming chat UI
│   ├── RetrievalLab.js     # client: embed → rerank lab
│   ├── OmniLab.js          # client: image + audio multimodal lab
│   ├── AsrLab.js           # client: file/mic ASR with browser PCM encoder
│   └── TtsLab.js           # client: text + voice-ref zero-shot TTS UI
└── api/
    ├── chat/route.js       # POST /api/chat    → SSE chat proxy
    ├── embed/route.js      # POST /api/embed   → batch embeddings proxy
    ├── rerank/route.js     # POST /api/rerank  → cross-encoder rerank proxy
    ├── omni/route.js       # POST /api/omni    → multimodal SSE proxy
    ├── asr/route.js        # POST /api/asr     → forwards SSE from sidecar
    ├── tts/route.js        # POST /api/tts     → forwards multipart to sidecar
    └── models/route.js     # GET  /api/models  → returns model picker list

asr_sidecar/                # Python service — used by ASR + TTS labs
├── asr_sidecar.py          # FastAPI: /transcribe (SSE) + /synthesize (WAV)
└── requirements.txt        # fastapi + uvicorn + python-multipart + nvidia-riva-client
```

Five pages, six POST routes, one GET route, one shared NavBar, and a small
Python sidecar that bridges Riva gRPC for both the ASR and TTS labs. The
browser only ever talks to this server — every `nvapi-…` key stays in
`process.env` on the Jetson.

## Why a server route?

The browser never sees `NVIDIA_API_KEY`. Requests go: **browser → /api/chat → NVIDIA Build**.
The server reads `process.env.NVIDIA_API_KEY` and forwards the SSE stream byte-for-byte.

## Running on Jetson Orin Nano

```bash
# from your laptop (one-time):
rsync -av --exclude node_modules --exclude .next \
  edgeLLM/nextjs-nemotron-app/ jetsonorin:~/nextjs-nemotron-app/

# on the Jetson:
ssh jetsonorin
cd ~/nextjs-nemotron-app
npm install
cp .env.local.example .env.local && nano .env.local   # paste key
npm run dev
```

Then on your laptop, open `http://<jetson-ip>:3000`.
