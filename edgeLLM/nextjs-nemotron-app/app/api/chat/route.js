// Server-side route that proxies chat completions to NVIDIA Build.
// The NVIDIA_API_KEY stays on the server — the browser never sees it.
//
// Request body (JSON):
//   {
//     "messages": [{role: "user"|"assistant"|"system", content: string}, ...],
//     "model":   "nvidia/llama-3.3-nemotron-super-49b-v1"  // optional
//     "thinking": true | false                              // optional
//     "temperature": 0.6                                    // optional
//     "max_tokens":  2048                                   // optional
//   }
//
// Response: text/event-stream (SSE) of OpenAI-compatible chunks, forwarded
// verbatim from NVIDIA so the client can parse `delta.content`,
// `delta.reasoning_content`, and `usage`.

export const runtime = "nodejs";          // streaming works on Node runtime
export const dynamic = "force-dynamic";   // never cache

import { resolveProvider } from "@/lib/providers";

const DEFAULT_MODEL =
  process.env.NVIDIA_MODEL || "nvidia/llama-3.3-nemotron-super-49b-v1";

export async function POST(req) {
  let body;
  try {
    body = await req.json();
  } catch {
    return jsonError(400, "Request body must be valid JSON.");
  }

  const {
    messages,
    model = DEFAULT_MODEL,
    thinking = false,
    temperature = 0.6,
    max_tokens = 2048,
  } = body || {};

  if (!Array.isArray(messages) || messages.length === 0) {
    return jsonError(400, "`messages` must be a non-empty array.");
  }

  // Pick the provider (NVIDIA / OpenAI / Anthropic) from the model id and read
  // its key from ~/.env.local or this app's .env.local.
  const provider = resolveProvider(model);
  if (!provider.apiKey) {
    return jsonError(
      500,
      `${provider.keyEnv} is not set. Add it to ~/.env.local (e.g. via 'sjsujetsontool chat' / 'setup-nvapi') or this app's .env.local.`
    );
  }

  const payload = {
    model,
    messages,
    temperature,
    max_tokens,
    stream: true,
    stream_options: { include_usage: true },
  };

  // Visible "thinking" is a Nemotron/llama.cpp feature; other providers reject it.
  if (thinking && provider.thinking) {
    payload.chat_template_kwargs = { enable_thinking: true };
  }

  let upstream;
  try {
    upstream = await fetch(`${provider.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${provider.apiKey}`,
        Accept: "text/event-stream",
      },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    return jsonError(502, `Failed to reach ${provider.name}: ${err.message}`);
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    return jsonError(
      upstream.status || 502,
      `Upstream error (${upstream.status}): ${text.slice(0, 500)}`
    );
  }

  // Forward the SSE stream straight through to the browser.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}

function jsonError(status, message) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
