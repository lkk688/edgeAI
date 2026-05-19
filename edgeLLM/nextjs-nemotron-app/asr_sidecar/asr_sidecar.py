"""ASR sidecar for the Next.js ASR Lab — FastAPI edition.

Why this file exists
--------------------
NVIDIA Build hosts `nvidia/nemotron-asr-streaming` as an NVCF gRPC function
(Riva ASR over gRPC, port 50051). gRPC + HTTP/2 streaming is awkward to call
from a Next.js Node.js route handler — there is no maintained pure-Node
client for Riva that ships with proto files. Instead, this short FastAPI
service speaks Riva gRPC on one side and exposes a tiny **SSE-over-HTTP**
interface on the other. The Next.js `/api/asr` route simply proxies the SSE
stream to the browser.

Why FastAPI (instead of Flask)
------------------------------
- **ASGI streaming.** `fastapi.responses.StreamingResponse` is the same
  abstraction the rest of the modern Python async ecosystem uses, and it
  hands a sync generator off to a threadpool automatically — which is
  exactly what we need to bridge Riva's blocking gRPC client.
- **Free OpenAPI docs.** Visit http://localhost:8001/docs in the browser to
  see an interactive Swagger UI for `/health` and `/transcribe`.
- **Type-checked query parameters.** `sr: int = 16000` is parsed and
  validated for free.

API
---
POST /transcribe?sr=16000&lang=en-US&interim=true
  body  : raw 16-bit little-endian PCM mono audio (no WAV header)
  stream: text/event-stream; one JSON event per Riva result
            data: {"type":"partial","text":"...","elapsed_ms":...}
            data: {"type":"final",  "text":"...","elapsed_ms":...}
            data: {"type":"done",                "elapsed_ms":...}
            data: {"type":"error","message":"..."}
            data: [DONE]

GET /health -> {"ok": true, "function_id": "..."}
GET /docs   -> interactive Swagger UI (FastAPI default)

Dependencies (see requirements.txt):
  fastapi, uvicorn[standard], nvidia-riva-client
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncIterator, Iterator

import riva.client
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

FUNCTION_ID = os.environ.get(
    "NEMOTRON_ASR_FUNCTION_ID", "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
)
NVCF_URI = os.environ.get("NEMOTRON_ASR_URI", "grpc.nvcf.nvidia.com:443")
PORT = int(os.environ.get("ASR_SIDECAR_PORT", "8001"))

# Audio is chunked into ~320 ms frames before being sent over gRPC. Smaller
# chunks = more frequent partial transcripts, more network overhead.
CHUNK_MS = int(os.environ.get("ASR_CHUNK_MS", "320"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("asr-sidecar")

app = FastAPI(
    title="Nemotron ASR Sidecar",
    description=(
        "Bridges NVIDIA Build's `nemotron-asr-streaming` (Riva gRPC) "
        "to a simple SSE HTTP interface for the Next.js ASR Lab."
    ),
    version="1.0.0",
)


def make_asr_service() -> riva.client.ASRService:
    """Create an authenticated Riva ASR client targeting NVCF."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set in the sidecar environment.")
    auth = riva.client.Auth(
        None,            # ssl_root_cert (use system CAs)
        True,            # use_ssl
        NVCF_URI,        # uri
        [                # metadata
            ("function-id", FUNCTION_ID),
            ("authorization", f"Bearer {api_key}"),
        ],
    )
    return riva.client.ASRService(auth)


@app.get("/health")
def health() -> dict:
    """Liveness check — also confirms which NVCF function this sidecar targets."""
    return {"ok": True, "function_id": FUNCTION_ID}


@app.post("/transcribe")
async def transcribe(
    request: Request,
    sr: int = 16000,
    lang: str = "en-US",
    interim: bool = True,
) -> StreamingResponse:
    """Stream a transcript for raw 16-bit LE PCM audio in the request body."""
    pcm = await request.body()
    if not pcm:
        # Even error responses go out as SSE so the client's parser
        # never has to special-case content types.
        return StreamingResponse(
            iter([_sse({"type": "error", "message": "empty body"}), _sse_done()]),
            media_type="text/event-stream",
            status_code=400,
        )

    bytes_per_chunk = max(2, int(sr * (CHUNK_MS / 1000.0)) * 2)
    log.info(
        "transcribe sr=%s lang=%s bytes=%s chunk=%sms (%s B)",
        sr, lang, len(pcm), CHUNK_MS, bytes_per_chunk,
    )

    def chunk_audio() -> Iterator[bytes]:
        for i in range(0, len(pcm), bytes_per_chunk):
            yield pcm[i : i + bytes_per_chunk]

    def event_stream() -> Iterator[str]:
        """Sync generator — Starlette will iterate it in a threadpool, which
        is exactly right for Riva's blocking gRPC client."""
        try:
            asr = make_asr_service()
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse_done()
            return

        cfg = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=sr,
                language_code=lang,
                max_alternatives=1,
                enable_automatic_punctuation=True,
            ),
            interim_results=interim,
        )

        t0 = time.time()
        try:
            stream = asr.streaming_response_generator(
                audio_chunks=chunk_audio(),
                streaming_config=cfg,
            )
            for resp in stream:
                for result in resp.results:
                    for alt in result.alternatives:
                        yield _sse({
                            "type": "final" if result.is_final else "partial",
                            "text": alt.transcript,
                            "stability": float(getattr(result, "stability", 0.0) or 0.0),
                            "elapsed_ms": int((time.time() - t0) * 1000),
                        })
            yield _sse({"type": "done", "elapsed_ms": int((time.time() - t0) * 1000)})
        except Exception as exc:
            log.exception("ASR stream failed")
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            yield _sse_done()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


if __name__ == "__main__":
    log.info(
        "starting ASR sidecar on 0.0.0.0:%s (function=%s) — docs at /docs",
        PORT,
        FUNCTION_ID,
    )
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
