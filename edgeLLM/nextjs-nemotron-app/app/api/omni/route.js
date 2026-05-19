// POST /api/omni — proxies a multimodal streaming chat completion to NVIDIA
// Build's omni reasoning model (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`).
//
// Request body (JSON):
//   {
//     "prompt": "What's in this image?",
//     "history": [{role,content}...]               // optional prior turns (text only)
//     "image":  { "data_url": "data:image/png;base64,..." }   // optional
//     "audio":  { "data_url": "data:audio/wav;base64,...", "format": "wav" }   // optional
//     "thinking": true,                             // default true (it's a reasoning model)
//     "reasoning_budget": 4096,
//     "max_tokens":        8192,
//     "temperature":       0.6,
//     "top_p":             0.95,
//     "model":             "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
//   }
//
// Response: text/event-stream of OpenAI-compatible chunks; the client reads
// `delta.content` and `delta.reasoning_content` exactly like /api/chat.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const NVIDIA_BASE_URL =
  process.env.NVIDIA_BASE_URL || "https://integrate.api.nvidia.com/v1";

const DEFAULT_OMNI_MODEL =
  process.env.NVIDIA_OMNI_MODEL ||
  "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning";

function jsonError(status, message) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// Build the multimodal `content` array for the user turn.
function buildUserContent({ prompt, image, audio }) {
  const parts = [];
  if (prompt && prompt.trim()) {
    parts.push({ type: "text", text: prompt });
  }
  if (image && image.data_url) {
    parts.push({ type: "image_url", image_url: { url: image.data_url } });
  }
  if (audio && audio.data_url) {
    // Strip the "data:audio/wav;base64," prefix — NVIDIA accepts both, but
    // the OpenAI-standard `input_audio` block wants raw base64.
    const m = /^data:audio\/([a-z0-9]+);base64,(.+)$/i.exec(audio.data_url);
    const format = (audio.format || (m && m[1]) || "wav").toLowerCase();
    const data = m ? m[2] : audio.data_url;
    parts.push({ type: "input_audio", input_audio: { data, format } });
  }
  // No attachments → fall back to a plain string content (the model accepts both).
  if (parts.length === 0) return prompt || "";
  if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
  return parts;
}

export async function POST(req) {
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) return jsonError(500, "NVIDIA_API_KEY is not set.");

  let body;
  try {
    body = await req.json();
  } catch {
    return jsonError(400, "Body must be valid JSON.");
  }

  const {
    prompt = "",
    history = [],
    image = null,
    audio = null,
    thinking = true,
    reasoning_budget = 4096,
    max_tokens = 8192,
    temperature = 0.6,
    top_p = 0.95,
    model = DEFAULT_OMNI_MODEL,
  } = body || {};

  if (!prompt.trim() && !image && !audio) {
    return jsonError(
      400,
      "Provide at least one of: `prompt`, `image`, or `audio`."
    );
  }

  const userContent = buildUserContent({ prompt, image, audio });
  const messages = [...history, { role: "user", content: userContent }];

  const payload = {
    model,
    messages,
    temperature,
    top_p,
    max_tokens,
    stream: true,
    stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: !!thinking },
  };
  if (thinking && reasoning_budget) {
    payload.reasoning_budget = reasoning_budget;
  }

  const upstream = await fetch(`${NVIDIA_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    return jsonError(
      upstream.status || 502,
      `Upstream ${upstream.status}: ${text.slice(0, 400)}`
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
