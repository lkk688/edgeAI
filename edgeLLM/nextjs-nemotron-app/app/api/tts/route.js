// POST /api/tts — proxies a TTS multipart upload to the FastAPI sidecar.
//
// Request: multipart/form-data with at minimum:
//   - text       (string)
//   - voice_ref  (file, 3-10 s WAV reference voice)
// Optional form fields: language_code, sample_rate_hz, quality.
//
// Response: audio/wav binary (the synthesized speech), with informational
// X-Synth-* headers passed through from the sidecar.
//
// The sidecar talks Riva gRPC to NVCF for `nvidia/magpie-tts-zeroshot`.
// See asr_sidecar/asr_sidecar.py for that side of the wire.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SIDECAR_URL =
  process.env.ASR_SIDECAR_URL || "http://localhost:8001";

export async function POST(req) {
  const contentType = req.headers.get("content-type") || "";
  if (!contentType.includes("multipart/form-data")) {
    return Response.json(
      {
        error:
          "Expected multipart/form-data with `text` and `voice_ref` fields.",
      },
      { status: 400 }
    );
  }

  let upstream;
  try {
    upstream = await fetch(`${SIDECAR_URL}/synthesize`, {
      method: "POST",
      // Forward the multipart body unchanged — the boundary is in the
      // Content-Type header, so the sidecar parses it correctly.
      body: req.body,
      duplex: "half",
      headers: { "Content-Type": contentType },
    });
  } catch (err) {
    return Response.json(
      {
        error:
          `Cannot reach Riva sidecar at ${SIDECAR_URL}. ` +
          `Start it with: cd asr_sidecar && python asr_sidecar.py. ` +
          `(${err.message})`,
      },
      { status: 502 }
    );
  }

  if (!upstream.ok) {
    const text = await upstream.text().catch(() => "");
    return Response.json(
      { error: `Sidecar ${upstream.status}: ${text.slice(0, 500)}` },
      { status: upstream.status || 502 }
    );
  }

  // Pass the WAV bytes back to the browser, preserving the helpful
  // X-Synth-* timing headers so the UI can display them.
  const headers = {
    "Content-Type":  upstream.headers.get("content-type") || "audio/wav",
    "Cache-Control": "no-store",
  };
  for (const k of ["x-synth-elapsed-ms", "x-synth-audio-seconds", "x-synth-sample-rate"]) {
    const v = upstream.headers.get(k);
    if (v) headers[k] = v;
  }
  return new Response(upstream.body, { status: 200, headers });
}
