// POST /api/rerank — proxies a reranking request to NVIDIA Build.
//
// NVIDIA's rerank endpoint lives at a different host/path than the chat and
// embeddings APIs:
//
//   POST https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking
//   {
//     "model":    "nvidia/rerank-qa-mistral-4b",
//     "query":    {"text": "..."},
//     "passages": [{"text": "..."}, ...]
//   }
//   → { "rankings": [{"index": <int>, "logit": <float>}, ...] }   (sorted)
//
// Request body to /api/rerank:
//   { "query": "...", "passages": ["...", "...", ...], "model": "..." (optional) }
//
// Response:
//   { "rankings": [{ "index": 0, "logit": 9.6, "passage": "..."}, ...], "model": "..." }

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const NVIDIA_RERANK_URL =
  process.env.NVIDIA_RERANK_URL ||
  "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking";

const DEFAULT_RERANK_MODEL =
  process.env.NVIDIA_RERANK_MODEL || "nvidia/rerank-qa-mistral-4b";

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

  const { query, passages, model = DEFAULT_RERANK_MODEL } = body || {};
  if (typeof query !== "string" || !query.trim()) {
    return Response.json({ error: "`query` must be a non-empty string." }, { status: 400 });
  }
  if (!Array.isArray(passages) || passages.length === 0) {
    return Response.json(
      { error: "`passages` must be a non-empty array of strings." },
      { status: 400 }
    );
  }

  const upstream = await fetch(NVIDIA_RERANK_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      query: { text: query },
      passages: passages.map((p) => ({ text: String(p) })),
    }),
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

  const rankings = (data.rankings || []).map((r) => ({
    index: r.index,
    logit: r.logit,
    passage: passages[r.index],
  }));

  return Response.json({ rankings, model });
}
