// POST /api/agent — forwards a task to the Python edge_agent sidecar and
// streams the ReAct trace (SSE) back to the browser.
//
// Request body:
//   {
//     "task":        "…",                       // required
//     "root":        "/abs/path",               // optional override
//     "model":       "qwen/qwen3-coder-…",      // optional, default qwen3-coder
//     "temperature": 0.1,
//     "max_steps":   8
//   }
//
// Response: text/event-stream forwarded byte-for-byte from the sidecar.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

import { resolveProvider, envFromHome } from "@/lib/providers";

const SIDECAR_URL = process.env.AGENT_SIDECAR_URL || "http://localhost:8002";

const DEFAULT_AGENT_MODEL =
  process.env.AGENT_MODEL || "minimaxai/minimax-m2.7";

export async function POST(req) {
  let body;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "body must be JSON" }, { status: 400 });
  }

  const task = (body.task || "").trim();
  if (!task) {
    return Response.json({ error: "`task` is required" }, { status: 400 });
  }

  const model = body.model || DEFAULT_AGENT_MODEL;
  // Reuse the chat lab's provider resolver — it understands NVIDIA / OpenAI /
  // Anthropic model ids and reads ~/.env.local for keys.
  const provider = resolveProvider(model);
  if (!provider.apiKey) {
    return Response.json(
      {
        error:
          `${provider.keyEnv} is not set. Add it to ~/.env.local or this app's .env.local.`,
      },
      { status: 500 }
    );
  }

  // The sidecar forwards `SERPAPI_API_KEY` from its own env into the
  // edge_agent process, so the web_search tool needs the key on the sidecar
  // host. The /api/agent route does not need to thread it through.
  envFromHome();

  const sidecarPayload = {
    task,
    root: body.root || undefined,
    model,
    base_url: provider.baseUrl,
    api_key: provider.apiKey,
    temperature: body.temperature ?? 0.1,
    max_steps: body.max_steps || 8,
  };

  let upstream;
  try {
    upstream = await fetch(`${SIDECAR_URL}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(sidecarPayload),
    });
  } catch (err) {
    return Response.json(
      {
        error:
          `Cannot reach agent sidecar at ${SIDECAR_URL}. ` +
          `Start it with: cd agent_sidecar && python agent_sidecar.py. ` +
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

// GET /api/agent — proxies the sidecar's /health for the UI to call once
// on mount (it can show available tools and whether web_search is enabled).
export async function GET() {
  try {
    const upstream = await fetch(`${SIDECAR_URL}/health`, { cache: "no-store" });
    if (!upstream.ok) {
      return Response.json({ ok: false, error: `sidecar ${upstream.status}` });
    }
    const data = await upstream.json();
    return Response.json(data);
  } catch (err) {
    return Response.json({ ok: false, error: String(err.message || err) });
  }
}
