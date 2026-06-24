// POST /api/agent — forwards a task to the Python edge_agent sidecar and
// streams the ReAct trace (SSE) back to the browser.
//
// Request body:
//   {
//     "task":        "…",                          // required
//     "root":        "/abs/path",                  // optional override
//
//     // Backend selection (mirrors `sjsujetsontool chat`):
//     "backend":     "nvidia" | "llama" | "openai" | "anthropic" | "custom",
//     "model":       "minimaxai/minimax-m2.7",
//     "base_url":    "http://localhost:8080/v1",   // required for "custom",
//                                                  // overrides default for others
//     "api_key":     "sk-…",                       // optional; provided for "custom"
//                                                  // or to override an env-supplied key
//
//     // Loop knobs:
//     "temperature": 0.1,
//     "max_steps":   8
//   }
//
// Response: text/event-stream forwarded byte-for-byte from the sidecar.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

import { resolveBackend, envFromHome } from "@/lib/providers";

const SIDECAR_URL = process.env.AGENT_SIDECAR_URL || "http://localhost:8002";

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

  // Pick the backend (default = NVIDIA Build, current behaviour). The
  // resolver returns a normalized {baseUrl, apiKey, model} regardless of
  // which backend the user chose.
  const backendId = body.backend || "nvidia";
  const backend = resolveBackend(backendId, {
    baseUrl: body.base_url,
    apiKey:  body.api_key,
    model:   body.model,
  });

  // A baseUrl is mandatory. Custom-backend users may forget it.
  if (!backend.baseUrl) {
    return Response.json(
      { error: `No base URL configured for backend "${backendId}". For "custom", set base_url on the request or CUSTOM_BASE_URL in .env.local.` },
      { status: 400 }
    );
  }

  // Cloud backends need a real key. Local llama.cpp and custom-without-auth
  // are fine with the "EMPTY" placeholder the resolver returns.
  if (backend.keyEnv && backend.apiKey === "EMPTY") {
    return Response.json(
      { error: `${backend.keyEnv} is not set. Add it to ~/.env.local or this app's .env.local.` },
      { status: 500 }
    );
  }

  envFromHome();          // also makes SERPAPI_API_KEY visible to the sidecar

  const sidecarPayload = {
    task,
    root:        body.root || undefined,
    model:       backend.model,
    base_url:    backend.baseUrl,
    api_key:     backend.apiKey,
    temperature: body.temperature ?? 0.1,
    max_steps:   body.max_steps || 8,
  };

  let upstream;
  try {
    upstream = await fetch(`${SIDECAR_URL}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(sidecarPayload),
    });
  } catch (err) {
    return Response.json(
      {
        error:
          `Cannot reach the FastAPI agent backend at ${SIDECAR_URL}. ` +
          `Start it on the Jetson with:  sjsujetsontool agent bg  ` +
          `(or run it by hand: cd agent_sidecar/ && python agent_sidecar.py). ` +
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

// GET /api/agent — proxies the sidecar's /health AND advertises the
// available backend menu so the UI doesn't have to hard-code it.
export async function GET() {
  // Lazy-import the menu so this route still loads if providers.js breaks.
  const { BACKEND_MENU } = await import("@/lib/providers");
  let sidecar = null;
  try {
    const r = await fetch(`${SIDECAR_URL}/health`, { cache: "no-store" });
    if (r.ok) sidecar = await r.json();
  } catch (err) {
    sidecar = { ok: false, error: String(err.message || err) };
  }
  return Response.json({
    sidecar,
    backends: BACKEND_MENU,
  });
}
