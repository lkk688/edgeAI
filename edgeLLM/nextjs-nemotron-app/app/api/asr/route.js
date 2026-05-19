// POST /api/asr — proxies raw PCM audio to the Python ASR sidecar.
//
// Request:
//   body         : raw 16-bit little-endian PCM mono audio
//   query string : ?sr=16000&lang=en-US&interim=1
//
// Response: text/event-stream, forwarded verbatim from the sidecar.
//   Each event is one of:
//     {"type":"partial","text":"...","elapsed_ms":N}
//     {"type":"final","text":"...","elapsed_ms":N}
//     {"type":"done","elapsed_ms":N}
//     {"type":"error","message":"..."}
//
// The sidecar talks Riva gRPC to NVCF (the actual NVIDIA Build endpoint).
// See asr_sidecar/asr_sidecar.py for that side of the wire.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SIDECAR_URL =
  process.env.ASR_SIDECAR_URL || "http://localhost:8001";

export async function POST(req) {
  const url = new URL(req.url);
  const qs = url.searchParams.toString();
  const upstreamUrl =
    `${SIDECAR_URL}/transcribe` + (qs ? `?${qs}` : "");

  let upstream;
  try {
    upstream = await fetch(upstreamUrl, {
      method: "POST",
      // Forward the raw body (Web ReadableStream → fetch will stream it).
      body: req.body,
      headers: {
        "Content-Type": "application/octet-stream",
        Accept: "text/event-stream",
      },
      // Required by Node fetch when forwarding a streaming request body.
      duplex: "half",
    });
  } catch (err) {
    return Response.json(
      {
        error:
          `Cannot reach ASR sidecar at ${SIDECAR_URL}. ` +
          `Start it with: cd asr_sidecar && python asr_sidecar.py. ` +
          `(${err.message})`,
      },
      { status: 502 }
    );
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    return Response.json(
      { error: `Sidecar ${upstream.status}: ${text.slice(0, 400)}` },
      { status: upstream.status || 502 }
    );
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
