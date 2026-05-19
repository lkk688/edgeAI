// POST /api/embed — proxies a batch embeddings request to NVIDIA Build.
//
// Request body:
//   {
//     "inputs": ["text 1", "text 2", ...],   // required, non-empty
//     "input_type": "query" | "passage",     // required for embedqa models
//     "model": "nvidia/nv-embedqa-e5-v5"     // optional
//   }
//
// Response (forwarded JSON):
//   { "vectors": [[...1024 floats...], ...], "dim": 1024, "model": "...", "usage": {...} }
//
// As with /api/chat, the NVIDIA_API_KEY is read on the server only.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const NVIDIA_BASE_URL =
  process.env.NVIDIA_BASE_URL || "https://integrate.api.nvidia.com/v1";

const DEFAULT_EMBED_MODEL =
  process.env.NVIDIA_EMBED_MODEL || "nvidia/nv-embedqa-e5-v5";

export async function POST(req) {
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    return Response.json({ error: "NVIDIA_API_KEY is not set." }, { status: 500 });
  }

  let body;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Body must be valid JSON." }, { status: 400 });
  }

  const {
    inputs,
    input_type = "query",
    model = DEFAULT_EMBED_MODEL,
  } = body || {};

  if (!Array.isArray(inputs) || inputs.length === 0) {
    return Response.json(
      { error: "`inputs` must be a non-empty array of strings." },
      { status: 400 }
    );
  }
  if (input_type !== "query" && input_type !== "passage") {
    return Response.json(
      { error: '`input_type` must be "query" or "passage".' },
      { status: 400 }
    );
  }

  const upstream = await fetch(`${NVIDIA_BASE_URL}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model, input: inputs, input_type }),
  });

  const text = await upstream.text();
  if (!upstream.ok) {
    return Response.json(
      { error: `Upstream ${upstream.status}: ${text.slice(0, 400)}` },
      { status: upstream.status || 502 }
    );
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return Response.json({ error: "Upstream returned non-JSON." }, { status: 502 });
  }

  const vectors = (data.data || []).map((d) => d.embedding);
  return Response.json({
    vectors,
    dim: vectors[0]?.length || 0,
    model,
    usage: data.usage || null,
  });
}
