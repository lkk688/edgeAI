# ASR sidecar (FastAPI + Uvicorn)

A ~120-line FastAPI service that lets the Next.js app reach
`nvidia/nemotron-asr-streaming` on NVIDIA Build. The model is exposed as an
NVCF **gRPC** function — there is no production-quality pure-Node Riva
client — so we run a tiny Python process that:

1. Accepts raw 16-bit LINEAR_PCM mono audio on `POST /transcribe`.
2. Streams it to NVCF gRPC using the official `nvidia-riva-client`.
3. Emits one **Server-Sent Event** per Riva result (`partial`, `final`,
   `done`, `error`) using FastAPI's `StreamingResponse`.

Next.js's `/api/asr` route just proxies that SSE stream to the browser.

## Why FastAPI?

The sidecar started life as a Flask app. We rewrote it in FastAPI for three
reasons that matter in this codebase:

| Capability | Flask | FastAPI |
|---|---|---|
| Native async / ASGI streaming | needs Quart or `flask[async]` | first-class (`StreamingResponse`) |
| Type-checked query params | manual `request.args.get` | function signature: `sr: int = 16000` |
| Interactive API docs | none built-in | free Swagger UI at `/docs`, ReDoc at `/redoc` |
| Threadpool for blocking calls | manual | automatic for sync generators |

Riva's Python client is **blocking** (the gRPC iterator yields synchronously
in a loop). FastAPI's `StreamingResponse` accepts a sync generator and runs
its iteration in a threadpool, so the event loop is never blocked.

## Layout

```
asr_sidecar/
├── asr_sidecar.py      # FastAPI app + Riva bridge
├── requirements.txt    # fastapi + uvicorn + nvidia-riva-client
└── README.md           # this file
```

## Run on the Jetson

```bash
ssh jetsonorin
source ~/.venv/bin/activate

# install deps once into the venv (the curriculum venv has no own pip,
# so we use the system pip and point it at the venv site-packages):
/usr/bin/python3 -m pip install \
    --target ~/.venv/lib/python3.10/site-packages \
    -r ~/nextjs-nemotron-app/asr_sidecar/requirements.txt

export NVIDIA_API_KEY=nvapi-...
cd ~/nextjs-nemotron-app/asr_sidecar
python asr_sidecar.py
# → INFO:     starting ASR sidecar on 0.0.0.0:8001 — docs at /docs
# → INFO:     Uvicorn running on http://0.0.0.0:8001
```

Keep this terminal open while Next.js runs in a second one.

Open <http://localhost:8001/docs> on the Jetson for FastAPI's interactive
Swagger UI — you can click "Try it out" on `/transcribe` and paste a base64
audio file straight into the browser to test without going through Next.js.

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `NVIDIA_API_KEY` | *(required)* | NVIDIA Build API key (nvapi-…) |
| `ASR_SIDECAR_PORT` | `8001` | Port Uvicorn binds to |
| `NEMOTRON_ASR_FUNCTION_ID` | `bb0837de-…` | NVCF function ID for `nemotron-asr-streaming` |
| `NEMOTRON_ASR_URI` | `grpc.nvcf.nvidia.com:443` | gRPC endpoint |
| `ASR_CHUNK_MS` | `320` | gRPC chunk size — smaller = more partials, more network |

## Quick test (no UI required)

```bash
# any 16 kHz mono 16-bit WAV; the JFK sample is convenient:
curl -sL https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav -o /tmp/jfk.wav

# strip the 44-byte WAV header so we send raw PCM:
dd if=/tmp/jfk.wav of=/tmp/jfk.pcm bs=1 skip=44 status=none

curl -N -X POST 'http://localhost:8001/transcribe?sr=16000&lang=en-US' \
  --data-binary @/tmp/jfk.pcm
```

Expected output:

```
data: {"type":"partial","text":"And","stability":0.0,"elapsed_ms":420}
data: {"type":"partial","text":"And so, my fellow","stability":0.0,...}
data: {"type":"final","text":"And so, my fellow Americans ",...}
...
data: {"type":"done","elapsed_ms":2110}
data: [DONE]
```
