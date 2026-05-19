# Riva speech sidecar (FastAPI + Uvicorn)

A small FastAPI service that lets the Next.js app reach two NVIDIA Build
models that are only available over **gRPC**:

| Endpoint            | NVIDIA Build model                | Lab     | Wire format            |
|---------------------|-----------------------------------|---------|------------------------|
| `POST /transcribe`  | `nvidia/nemotron-asr-streaming`   | ASR (§9)| raw PCM in / SSE out   |
| `POST /synthesize`  | `nvidia/magpie-tts-zeroshot`      | TTS (§10)| multipart in / WAV out |

Both share the same gRPC connection setup (`grpc.nvcf.nvidia.com:443` +
function-id metadata) and the same authentication path — adding a new
Riva-backed model is mostly a copy-paste away.

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
| `MAGPIE_TTS_FUNCTION_ID`   | `55cf67bf-…` | NVCF function ID for `magpie-tts-zeroshot` |
| `NVCF_GRPC_URI` | `grpc.nvcf.nvidia.com:443` | gRPC endpoint (shared by both models) |
| `ASR_CHUNK_MS` | `320` | gRPC chunk size — smaller = more partials, more network |

## Quick tests (no UI required)

### ASR — `/transcribe`

```bash
# any 16 kHz mono 16-bit WAV; the JFK sample is convenient:
curl -sL https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav -o /tmp/jfk.wav

# strip the 44-byte WAV header so we send raw PCM:
dd if=/tmp/jfk.wav of=/tmp/jfk.pcm bs=1 skip=44 status=none

curl -N -X POST 'http://localhost:8001/transcribe?sr=16000&lang=en-US' \
  --data-binary @/tmp/jfk.pcm
```

Expected SSE output:

```
data: {"type":"partial","text":"And","stability":0.0,"elapsed_ms":420}
data: {"type":"partial","text":"And so, my fellow","stability":0.0,...}
data: {"type":"final","text":"And so, my fellow Americans ",...}
data: {"type":"done","elapsed_ms":2110}
data: [DONE]
```

### TTS — `/synthesize`

```bash
# Trim the JFK sample to 8 s (3-10 s reference is required) using Python's
# stdlib wave module:
python3 - <<'PY'
import wave
with wave.open('/tmp/jfk.wav','rb') as w, wave.open('/tmp/jfk_8s.wav','wb') as o:
    o.setnchannels(w.getnchannels()); o.setsampwidth(w.getsampwidth())
    o.setframerate(w.getframerate())
    o.writeframes(w.readframes(min(w.getnframes(), w.getframerate()*8)))
PY

curl -s -o /tmp/synth.wav -D - --max-time 60 \
  -X POST 'http://localhost:8001/synthesize' \
  -F 'text=Hello from the Jetson Orin Nano.' \
  -F 'language_code=en-US' \
  -F 'sample_rate_hz=22050' \
  -F 'quality=20' \
  -F 'voice_ref=@/tmp/jfk_8s.wav;type=audio/wav' \
  | head -20
file /tmp/synth.wav        # → WAVE audio, Microsoft PCM, 16 bit, mono 22050 Hz
```

You should see `HTTP/1.1 200 OK`, `Content-Type: audio/wav`, and
`X-Synth-Elapsed-Ms: <number>` in the response headers, plus a playable
WAV at `/tmp/synth.wav`.
