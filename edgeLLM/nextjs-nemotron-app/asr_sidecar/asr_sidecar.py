"""Speech sidecar for the Next.js ASR + TTS Labs — FastAPI edition.

Why this file exists
--------------------
Two NVIDIA Build models we use are exposed as NVCF **gRPC** functions —
there is no JSON REST shim and no maintained pure-Node Riva client. This
short FastAPI service speaks Riva gRPC on one side and exposes plain HTTP
on the other. Both Next.js routes (`/api/asr`, `/api/tts`) are thin
proxies onto this sidecar.

Endpoints
---------
POST /transcribe?sr=16000&lang=en-US&interim=true
  Streaming ASR. Body is raw 16-bit LE PCM mono audio. Response is SSE:
      data: {"type":"partial","text":"...","elapsed_ms":...}
      data: {"type":"final",  "text":"...","elapsed_ms":...}
      data: {"type":"done",                "elapsed_ms":...}
      data: {"type":"error","message":"..."}
      data: [DONE]
  → uses Riva ASRService against `nvidia/nemotron-asr-streaming`.

POST /synthesize        (multipart/form-data)
  Zero-shot TTS. Form fields:
      text:           the sentence to synthesize
      voice_ref:      a 3–10 s WAV file with the reference voice
      language_code:  default "en-US"
      sample_rate_hz: default 22050
      quality:        zero-shot quality 1–40, default 20
  Returns a single 16-bit mono WAV file (audio/wav).
  → uses Riva SpeechSynthesisService against `nvidia/magpie-tts-zeroshot`.

GET /health -> {"ok": true, "asr_function_id": "...", "tts_function_id": "..."}
GET /docs   -> interactive Swagger UI for both endpoints

Why FastAPI
-----------
- **First-class streaming** for ASR (`StreamingResponse(generator)`).
- **Native multipart** for TTS — declare `voice_ref: UploadFile = File(...)`
  and `text: str = Form(...)` and FastAPI parses the boundary for you.
- **Free OpenAPI docs** at `/docs` — students can click "Try it out" on
  either endpoint without writing any client code.

Dependencies (see requirements.txt):
  fastapi, uvicorn[standard], nvidia-riva-client, python-multipart
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import tempfile
import time
from pathlib import Path
from typing import Iterator

import riva.client
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

ASR_FUNCTION_ID = os.environ.get(
    "NEMOTRON_ASR_FUNCTION_ID", "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
)
TTS_FUNCTION_ID = os.environ.get(
    "MAGPIE_TTS_FUNCTION_ID", "55cf67bf-600f-4b04-8eac-12ed39537a08"
)
NVCF_URI = os.environ.get("NVCF_GRPC_URI", "grpc.nvcf.nvidia.com:443")
PORT = int(os.environ.get("ASR_SIDECAR_PORT", "8001"))

# Audio is chunked into ~320 ms frames before being sent over gRPC. Smaller
# chunks = more frequent partial transcripts, more network overhead.
CHUNK_MS = int(os.environ.get("ASR_CHUNK_MS", "320"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("asr-sidecar")

app = FastAPI(
    title="Riva Speech Sidecar",
    description=(
        "Bridges NVIDIA Build's `nemotron-asr-streaming` and "
        "`magpie-tts-zeroshot` (both Riva gRPC) onto plain HTTP for the "
        "Next.js ASR + TTS Labs."
    ),
    version="1.1.0",
)


def _make_auth(function_id: str) -> riva.client.Auth:
    """Build a Riva Auth object pointed at one specific NVCF function."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set in the sidecar environment.")
    return riva.client.Auth(
        None,            # ssl_root_cert (use system CAs)
        True,            # use_ssl
        NVCF_URI,        # uri
        [                # gRPC metadata: function-id picks the model
            ("function-id", function_id),
            ("authorization", f"Bearer {api_key}"),
        ],
    )


def make_asr_service() -> riva.client.ASRService:
    return riva.client.ASRService(_make_auth(ASR_FUNCTION_ID))


def make_tts_service() -> riva.client.SpeechSynthesisService:
    return riva.client.SpeechSynthesisService(_make_auth(TTS_FUNCTION_ID))


@app.get("/health")
def health() -> dict:
    """Liveness check — also confirms which NVCF functions this sidecar targets."""
    return {
        "ok": True,
        "asr_function_id": ASR_FUNCTION_ID,
        "tts_function_id": TTS_FUNCTION_ID,
    }


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


# ---------------------------------------------------------------------------
# TTS endpoint — wraps Riva SpeechSynthesisService for magpie-tts-zeroshot
# ---------------------------------------------------------------------------

def _pcm_to_wav(pcm: bytes, sr: int, nch: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM bytes in a minimal RIFF/WAVE container."""
    n = len(pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + n))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack(
        "<IHHIIHH",
        16,                            # fmt chunk size
        1,                             # PCM format
        nch,                           # channels
        sr,                            # sample rate
        sr * nch * sample_width,       # byte rate
        nch * sample_width,            # block align
        sample_width * 8,              # bits per sample
    ))
    buf.write(b"data")
    buf.write(struct.pack("<I", n))
    buf.write(pcm)
    return buf.getvalue()


@app.post(
    "/synthesize",
    responses={200: {"content": {"audio/wav": {}}}},
    response_class=Response,
)
async def synthesize(
    voice_ref: UploadFile = File(..., description="3-10 s WAV reference voice"),
    text: str = Form(..., description="Sentence to synthesize"),
    language_code: str = Form("en-US"),
    sample_rate_hz: int = Form(22050),
    quality: int = Form(20, ge=1, le=40),
):
    """Zero-shot TTS — returns a single WAV file synthesized in `voice_ref`'s voice."""
    if not text.strip():
        return Response(content=b"text is empty", status_code=400, media_type="text/plain")

    # Riva's Python client wants a filesystem Path, not bytes — save the
    # upload to a temp file and clean up afterwards. The file is small
    # (≤10 s of audio), so this is cheap.
    ref_bytes = await voice_ref.read()
    if not ref_bytes:
        return Response(content=b"voice_ref is empty", status_code=400, media_type="text/plain")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(ref_bytes)
        prompt_path = Path(tmp.name)

    log.info(
        "synthesize text=%r voice_ref=%s bytes lang=%s sr=%s quality=%s",
        text[:60] + ("…" if len(text) > 60 else ""),
        len(ref_bytes), language_code, sample_rate_hz, quality,
    )

    try:
        tts = make_tts_service()
        t0 = time.time()
        resp = tts.synthesize(
            text=text,
            language_code=language_code,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=sample_rate_hz,
            zero_shot_audio_prompt_file=prompt_path,
            # Let the server detect the container from the file bytes.
            # Hard-coding LINEAR_PCM fails with "config format doesn't match
            # with header format" when a WAV header is present.
            audio_prompt_encoding=riva.client.AudioEncoding.ENCODING_UNSPECIFIED,
            zero_shot_quality=quality,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        wav = _pcm_to_wav(resp.audio, sample_rate_hz)
        audio_sec = len(resp.audio) / 2 / sample_rate_hz
        log.info(
            "synthesize ok in %sms — %s bytes WAV (%.2fs audio)",
            elapsed_ms, len(wav), audio_sec,
        )
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "X-Synth-Elapsed-Ms":     str(elapsed_ms),
                "X-Synth-Audio-Seconds":  f"{audio_sec:.3f}",
                "X-Synth-Sample-Rate":    str(sample_rate_hz),
            },
        )
    except Exception as exc:
        log.exception("synthesize failed")
        return Response(
            content=f"{type(exc).__name__}: {exc}".encode(),
            status_code=500,
            media_type="text/plain",
        )
    finally:
        try:
            prompt_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    log.info(
        "starting Riva speech sidecar on 0.0.0.0:%s — docs at /docs "
        "(asr=%s, tts=%s)",
        PORT, ASR_FUNCTION_ID, TTS_FUNCTION_ID,
    )
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
