"use client";

import { useEffect, useRef, useState } from "react";

// Coding-capable models from build.nvidia.com that we know support tool use.
// (Anthropic / OpenAI models also work — they'll be auto-resolved by the route.)
const CODING_MODELS = [
  { id: "minimaxai/minimax-m2.7",                  label: "MiniMax M2.7 (default)" },
  { id: "minimaxai/minimax-m3",                    label: "MiniMax M3" },
  { id: "nvidia/llama-3.3-nemotron-super-49b-v1.5", label: "Nemotron Super 49B v1.5" },
  { id: "nvidia/llama-3.1-nemotron-70b-instruct",  label: "Nemotron Llama 3.1 70B" },
  { id: "deepseek-ai/deepseek-v4-pro",             label: "DeepSeek v4 Pro" },
  { id: "mistralai/mistral-large-3-675b-instruct-2512", label: "Mistral Large 3 675B" },
  { id: "claude-sonnet-4-6",                       label: "Anthropic Claude Sonnet 4.6" },
  { id: "gpt-4o-mini",                             label: "OpenAI GPT-4o mini" },
];

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

export default function AgentLab() {
  const [task, setTask] = useState(SAMPLE_TASKS[0]);
  const [model, setModel] = useState(CODING_MODELS[0].id);
  const [maxSteps, setMaxSteps] = useState(8);
  const [root, setRoot] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [events, setEvents] = useState([]);          // {type, n, …}
  const [final, setFinal] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState([]);        // previous runs (task, final)
  const traceRef = useRef(null);
  const abortRef = useRef(null);

  // Load /api/agent (GET) for tool availability + workspace path.
  useEffect(() => {
    fetch("/api/agent", { cache: "no-store" })
      .then((r) => r.json())
      .then((j) => setHealth(j))
      .catch(() => setHealth({ ok: false }));
  }, []);

  // Auto-scroll trace as new events arrive.
  useEffect(() => {
    if (traceRef.current) traceRef.current.scrollTop = traceRef.current.scrollHeight;
  }, [events, final]);

  async function runAgent() {
    if (!task.trim() || busy) return;
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
        body: JSON.stringify({
          task,
          model,
          max_steps: maxSteps,
          root: root.trim() || undefined,
        }),
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
            { task, answer: evt.answer, n: evt.n, model },
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

  function cancel() {
    if (abortRef.current) abortRef.current.abort();
  }

  function resetTrace() {
    setEvents([]);
    setFinal(null);
    setMetrics(null);
    setError("");
  }

  const webEnabled = !!health?.web_search;
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
              edge_agent (ReAct) · {health?.tools?.join(" · ") || "loading tools…"}
              {webEnabled ? "" : " · web_search disabled (no SERPAPI_API_KEY)"}
            </div>
          </div>
        </div>
        <div className="model-row">
          <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy}>
            {CODING_MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>
      </header>

      <div className="lab-grid">
        <section className="lab-col">
          <label className="lab-label">Task ({task.length} chars)</label>
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
            placeholder={health?.default_root || "/path/to/project (defaults to sidecar workspace)"}
            style={{
              background: "var(--panel)", color: "var(--text)",
              border: "1px solid var(--border)", borderRadius: 8,
              padding: "8px 10px", font: "inherit",
            }}
          />

          <div className="lab-controls">
            <label className="toggle">
              max steps:
              <input
                type="number"
                min="2"
                max={health?.max_steps_hard || 12}
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
              steps {metrics.steps}
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
                      #{history.length - i} · {h.n} steps · {h.model}
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
              {busy ? "Agent is thinking…" : "Press Run agent and watch each Thought / Action / Observation arrive in real time."}
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
                  <pre className="agent-obs">{text.length > 1200 ? text.slice(0, 1200) + "\n… [truncated]" : text}</pre>
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
