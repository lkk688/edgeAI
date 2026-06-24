"use client";

import { useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Backend menu — same options as `sjsujetsontool chat` (local llama.cpp /
// NVIDIA / OpenAI / Anthropic / custom). The route's GET /api/agent
// returns the same list, but we hard-code it here too so the UI works
// before /health responds.
// ---------------------------------------------------------------------------

const BACKEND_MENU = [
  {
    id: "nvidia",
    name: "NVIDIA Build (cloud)",
    requiresKey: true,
    defaultModel: "minimaxai/minimax-m2.7",
    models: [
      { id: "minimaxai/minimax-m2.7",                       label: "MiniMax M2.7 (default)" },
      { id: "minimaxai/minimax-m3",                         label: "MiniMax M3" },
      { id: "z-ai/glm-5.1",                                 label: "Z-AI GLM 5.1" },
      { id: "nvidia/llama-3.3-nemotron-super-49b-v1.5",     label: "Nemotron Super 49B v1.5" },
      { id: "nvidia/llama-3.1-nemotron-70b-instruct",       label: "Nemotron Llama 3.1 70B" },
      { id: "deepseek-ai/deepseek-v4-pro",                  label: "DeepSeek v4 Pro" },
      { id: "mistralai/mistral-large-3-675b-instruct-2512", label: "Mistral Large 3 675B" },
    ],
  },
  {
    id: "llama",
    name: "Local llama.cpp (Jetson :8080)",
    requiresKey: false,
    defaultModel: "local",
    models: [{ id: "local", label: "whatever llama-server is running" }],
    helpText: "Start it first on the Jetson with: sjsujetsontool llama bg",
  },
  {
    id: "node05",
    name: "Shared SJSU llama.cpp (node05)",
    requiresKey: false,
    defaultModel: "Qwen3.5-9B-UD-Q6_K_XL.gguf",
    models: [
      { id: "Qwen3.5-9B-UD-Q6_K_XL.gguf", label: "Qwen3.5-9B (9 B, Q6_K_XL — same as sjsujetsontool chat)" },
    ],
    helpText:
      "Our shared LLM server at https://llm.forgengi.org/node05/v1 — no key required. Same backend " +
      "the sjsujetsontool chat menu uses for its 'Our shared LLM server'.",
  },
  {
    id: "openai",
    name: "OpenAI",
    requiresKey: true,
    defaultModel: "gpt-4o-mini",
    models: [
      { id: "gpt-4o-mini", label: "GPT-4o mini" },
      { id: "gpt-4o",      label: "GPT-4o" },
      { id: "gpt-4.1",     label: "GPT-4.1" },
    ],
  },
  {
    id: "anthropic",
    name: "Anthropic",
    requiresKey: true,
    defaultModel: "claude-sonnet-4-6",
    models: [
      { id: "claude-haiku-4-5",  label: "Claude Haiku 4.5" },
      { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6" },
    ],
  },
  {
    id: "custom",
    name: "Custom (OpenAI-compatible)",
    requiresKey: false,
    defaultModel: "",
    models: [],
    helpText:
      "Any /v1/chat/completions endpoint (vLLM, Ollama, Together, an enterprise gateway, …).",
  },
];

const DEFAULT_BACKEND_ID = "nvidia";

const SAMPLE_TASKS = [
  "Read calculator.py and summarize what it does in two sentences.",
  "Find every TODO in the project and list them.",
  "Fix the typo in calculator.py — `doubel` should be `double` — and write the corrected file.",
  "Add a new function `power(base, exp)` to calculator.py.",
];

const SAMPLE_TASKS_WITH_WEB = [
  "Search the web for the latest LangChain version and write a one-paragraph summary into webnote.md.",
];

// Parse one SSE chunk of `data: {…}\n\n`-style events.
async function readSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        onEvent(JSON.parse(payload));
      } catch {
        /* ignore malformed lines */
      }
    }
  }
}

const ICON = {
  read_file:    "📄",
  grep:         "🔎",
  search_files: "📁",
  write_file:   "📝",
  edit_file:    "✏️ ",
  web_search:   "🌐",
};

const inputStyle = {
  background: "var(--panel)",
  color: "var(--text)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  padding: "8px 10px",
  font: "inherit",
};

export default function AgentLab() {
  const [task, setTask] = useState(SAMPLE_TASKS[0]);
  const [backendId, setBackendId] = useState(DEFAULT_BACKEND_ID);
  const backendDef = BACKEND_MENU.find((b) => b.id === backendId) || BACKEND_MENU[0];

  // Model: per-backend, with a per-backend default. Custom backend uses a free text input.
  const [modelByBackend, setModelByBackend] = useState(() =>
    Object.fromEntries(BACKEND_MENU.map((b) => [b.id, b.defaultModel]))
  );
  const model = modelByBackend[backendId] || backendDef.defaultModel || "";

  // Custom backend extras (URL + optional key). Kept separate from `model` so
  // switching backends preserves the picked model per backend.
  const [customBaseUrl, setCustomBaseUrl] = useState("");
  const [customApiKey, setCustomApiKey] = useState("");

  // For llama backend, allow base_url override (rare — e.g. a remote llama-server).
  const [llamaBaseUrl, setLlamaBaseUrl] = useState("");

  const [maxSteps, setMaxSteps] = useState(8);
  const [root, setRoot] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [events, setEvents] = useState([]);
  const [final, setFinal] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState([]);
  const traceRef = useRef(null);
  const abortRef = useRef(null);

  function setModel(value) {
    setModelByBackend((prev) => ({ ...prev, [backendId]: value }));
  }

  // Load /api/agent (GET) for tool availability + workspace path.
  useEffect(() => {
    fetch("/api/agent", { cache: "no-store" })
      .then((r) => r.json())
      .then((j) => setHealth(j))
      .catch(() => setHealth(null));
  }, []);

  // Auto-scroll trace as new events arrive.
  useEffect(() => {
    if (traceRef.current) traceRef.current.scrollTop = traceRef.current.scrollHeight;
  }, [events, final]);

  function buildRequestBody() {
    const body = {
      task,
      backend: backendId,
      model,
      max_steps: maxSteps,
      root: root.trim() || undefined,
    };
    if (backendId === "custom") {
      body.base_url = customBaseUrl.trim();
      if (customApiKey.trim()) body.api_key = customApiKey.trim();
    } else if (backendId === "llama" && llamaBaseUrl.trim()) {
      body.base_url = llamaBaseUrl.trim();
    }
    return body;
  }

  async function runAgent() {
    if (!task.trim() || busy) return;
    if (backendId === "custom" && !customBaseUrl.trim()) {
      setError("Custom backend needs a base URL (e.g. http://192.168.1.10:8000/v1).");
      return;
    }
    if (backendId === "custom" && !model.trim()) {
      setError("Custom backend needs a model id.");
      return;
    }
    setError("");
    setEvents([]);
    setFinal(null);
    setMetrics(null);
    setBusy(true);
    const startedAt = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildRequestBody()),
        signal: controller.signal,
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${res.status}`);
      }
      await readSSE(res, (evt) => {
        if (evt.type === "error") {
          setError(evt.message || "agent error");
          return;
        }
        if (evt.type === "final") {
          setFinal(evt);
          setMetrics({
            wall_ms: performance.now() - startedAt,
            elapsed_ms: evt.elapsed_ms,
            steps: evt.n,
            exhausted: !!evt.exhausted,
          });
          setHistory((prev) => [
            ...prev,
            { task, answer: evt.answer, n: evt.n, backend: backendId, model },
          ]);
          return;
        }
        setEvents((prev) => [...prev, evt]);
      });
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message || String(e));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function cancel() { if (abortRef.current) abortRef.current.abort(); }
  function resetTrace() {
    setEvents([]);
    setFinal(null);
    setMetrics(null);
    setError("");
  }

  const webEnabled = !!health?.sidecar?.web_search;
  const sidecarOk = !!health?.sidecar?.ok;
  const tools = health?.sidecar?.tools || [];
  const sampleTasks = webEnabled
    ? [...SAMPLE_TASKS, ...SAMPLE_TASKS_WITH_WEB]
    : SAMPLE_TASKS;

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <div className="brand-title">Agent Lab — multi-round file & web agent</div>
            <div className="brand-sub">
              edge_agent (ReAct) ·{" "}
              {sidecarOk ? `${tools.length} tools available · ${tools.join(" · ")}` : "sidecar offline"}
              {sidecarOk && !webEnabled ? " · web_search disabled (no SERPAPI_API_KEY)" : ""}
            </div>
          </div>
        </div>
      </header>

      <div className="lab-grid">
        <section className="lab-col">
          <label className="lab-label">Backend</label>
          <select
            value={backendId}
            onChange={(e) => { setBackendId(e.target.value); setError(""); }}
            disabled={busy}
            style={inputStyle}
          >
            {BACKEND_MENU.map((b) => (
              <option key={b.id} value={b.id}>{b.name}</option>
            ))}
          </select>
          {backendDef.helpText && (
            <div className="attachment-meta">💡 {backendDef.helpText}</div>
          )}

          {/* Per-backend model selector OR free text */}
          <label className="lab-label" style={{ marginTop: 8 }}>Model</label>
          {backendDef.models && backendDef.models.length > 0 ? (
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={busy}
              style={inputStyle}
            >
              {backendDef.models.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={busy}
              placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct"
              style={inputStyle}
            />
          )}

          {/* Custom backend: URL + key */}
          {backendId === "custom" && (
            <>
              <label className="lab-label" style={{ marginTop: 8 }}>
                Base URL (must end with /v1)
              </label>
              <input
                type="text"
                value={customBaseUrl}
                onChange={(e) => setCustomBaseUrl(e.target.value)}
                disabled={busy}
                placeholder="http://192.168.1.10:8000/v1"
                style={inputStyle}
              />
              <label className="lab-label" style={{ marginTop: 8 }}>
                API key (optional — leave blank if the server doesn't require one)
              </label>
              <input
                type="password"
                value={customApiKey}
                onChange={(e) => setCustomApiKey(e.target.value)}
                disabled={busy}
                placeholder="sk-…"
                style={inputStyle}
              />
            </>
          )}

          {/* llama backend: optional base_url override */}
          {backendId === "llama" && (
            <>
              <label className="lab-label" style={{ marginTop: 8 }}>
                Base URL override (default: <code>http://localhost:8080/v1</code>)
              </label>
              <input
                type="text"
                value={llamaBaseUrl}
                onChange={(e) => setLlamaBaseUrl(e.target.value)}
                disabled={busy}
                placeholder="http://localhost:8080/v1"
                style={inputStyle}
              />
            </>
          )}

          <label className="lab-label" style={{ marginTop: 12 }}>
            Task ({task.length} chars)
          </label>
          <textarea
            className="lab-textarea"
            rows={5}
            value={task}
            onChange={(e) => setTask(e.target.value)}
            disabled={busy}
            placeholder="What do you want the agent to do?"
          />
          <div className="file-row" style={{ flexWrap: "wrap" }}>
            {sampleTasks.map((p, i) => (
              <button
                key={i}
                className="btn btn-ghost btn-sm"
                onClick={() => setTask(p)}
                disabled={busy}
                title={p}
              >
                sample {i + 1}
              </button>
            ))}
          </div>

          <label className="lab-label" style={{ marginTop: 10 }}>Workspace</label>
          <input
            type="text"
            value={root}
            onChange={(e) => setRoot(e.target.value)}
            disabled={busy}
            placeholder={health?.sidecar?.default_root || "/path/to/project (defaults to sidecar workspace)"}
            style={inputStyle}
          />

          <div className="lab-controls">
            <label className="toggle">
              max steps:
              <input
                type="number"
                min="2"
                max={health?.sidecar?.max_steps_hard || 12}
                value={maxSteps}
                onChange={(e) => setMaxSteps(parseInt(e.target.value, 10) || 8)}
                disabled={busy}
                style={{ width: 56, marginLeft: 4 }}
              />
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-ghost btn-sm" onClick={resetTrace} disabled={busy}>
                Clear
              </button>
              {busy ? (
                <button className="btn" onClick={cancel}>Stop</button>
              ) : (
                <button className="btn" onClick={runAgent} disabled={!task.trim()}>
                  Run agent
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="bubble error" style={{ alignSelf: "stretch" }}>
              <span className="bubble-role">Error</span>
              {error}
            </div>
          )}
          {metrics && (
            <div className="metrics">
              backend {backendDef.name} · steps {metrics.steps}
              {metrics.exhausted ? " (forced)" : ""} · sidecar{" "}
              {(metrics.elapsed_ms / 1000).toFixed(2)} s · wall{" "}
              {(metrics.wall_ms / 1000).toFixed(2)} s
            </div>
          )}

          {history.length > 0 && (
            <>
              <div className="lab-label" style={{ marginTop: 14 }}>Previous runs</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {history.slice(-5).reverse().map((h, i) => (
                  <div key={i} className="result-card">
                    <div className="result-meta">
                      #{history.length - i} · {h.n} steps · {h.backend} · {h.model}
                    </div>
                    <div style={{ fontSize: 12, color: "var(--muted)" }}>{h.task}</div>
                    <div>{h.answer.length > 200 ? h.answer.slice(0, 200) + "…" : h.answer}</div>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>

        <section className="lab-col" ref={traceRef} style={{ overflow: "auto" }}>
          <div className="lab-label">Live agent trace</div>
          {events.length === 0 && !final && (
            <div className="empty-hint" style={{ padding: 16 }}>
              {busy
                ? "Agent is thinking…"
                : "Pick a backend, type a task, and press Run agent to watch each Thought / Action / Observation arrive in real time."}
            </div>
          )}
          {events.map((evt, i) => {
            if (evt.type === "start") {
              return (
                <div key={i} className="agent-card agent-card-start">
                  <div className="agent-meta">workspace · {evt.tools?.length} tools</div>
                  <div>
                    <code>{evt.root}</code>
                    <br />
                    <span style={{ fontSize: 12 }}>model: {evt.model} · max_steps: {evt.max_steps}</span>
                  </div>
                </div>
              );
            }
            if (evt.type === "step") {
              return (
                <div key={i} className="agent-card agent-card-step">
                  <div className="agent-meta">
                    step {evt.n} · {ICON[evt.action] || "🛠"} {evt.action}
                  </div>
                  {evt.thought && (
                    <div className="agent-thought">
                      <em>Thought:</em> {evt.thought}
                    </div>
                  )}
                  <div className="agent-action">
                    <code>
                      {evt.action}({JSON.stringify(evt.input)})
                    </code>
                  </div>
                </div>
              );
            }
            if (evt.type === "observation") {
              const text = evt.text || "";
              const isError = text.startsWith("ERROR");
              return (
                <div
                  key={i}
                  className={`agent-card agent-card-obs ${isError ? "agent-card-obs-err" : ""}`}
                >
                  <div className="agent-meta">step {evt.n} · observation</div>
                  <pre className="agent-obs">
                    {text.length > 1200 ? text.slice(0, 1200) + "\n… [truncated]" : text}
                  </pre>
                </div>
              );
            }
            if (evt.type === "nudge") {
              return (
                <div key={i} className="agent-card agent-card-nudge">
                  <div className="agent-meta">step {evt.n} · nudge</div>
                  <div>Model produced no Action. Nudging it to follow the protocol…</div>
                </div>
              );
            }
            return null;
          })}
          {final && (
            <div className="agent-card agent-card-final">
              <div className="agent-meta">final answer · step {final.n}</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{final.answer}</div>
            </div>
          )}
        </section>
      </div>
    </>
  );
}
