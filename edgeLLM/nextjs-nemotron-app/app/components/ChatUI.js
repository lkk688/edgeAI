"use client";

import { useEffect, useRef, useState } from "react";

// Parses a streamed OpenAI/NVIDIA-compatible SSE response line-by-line.
// Calls onDelta({ content, reasoning, usage }) for every chunk it understands.
async function readSSE(response, onDelta) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by blank lines; each line begins with "data: ".
    let nl;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line.startsWith("data:")) continue;

      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;

      try {
        const chunk = JSON.parse(payload);
        const choice = chunk.choices && chunk.choices[0];
        const delta = choice && choice.delta;
        onDelta({
          content: delta && delta.content,
          reasoning: delta && delta.reasoning_content,
          usage: chunk.usage,
        });
      } catch {
        // ignore malformed lines
      }
    }
  }
}

export default function ChatUI() {
  const [messages, setMessages] = useState([]);          // {role, content, reasoning?}
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [thinking, setThinking] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const chatRef = useRef(null);
  const abortRef = useRef(null);

  // Load model list once.
  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((j) => {
        setModels(j.models || []);
        setModel(j.default || (j.models && j.models[0] && j.models[0].id) || "");
      })
      .catch(() => {});
  }, []);

  // Auto-scroll to bottom on new content.
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;
    setError("");
    setMetrics(null);

    const newUser = { role: "user", content: text };
    const newAssistant = { role: "assistant", content: "", reasoning: "" };
    setMessages((prev) => [...prev, newUser, newAssistant]);
    setInput("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const startedAt = performance.now();
    let firstTokenAt = null;
    let finalUsage = null;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          thinking,
          messages: [
            { role: "system", content: "You are a helpful, concise assistant." },
            ...messages,
            newUser,
          ],
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${res.status}`);
      }

      await readSSE(res, ({ content, reasoning, usage }) => {
        if ((content || reasoning) && firstTokenAt == null) {
          firstTokenAt = performance.now();
        }
        if (usage) finalUsage = usage;

        if (content || reasoning) {
          setMessages((prev) => {
            const next = prev.slice();
            const last = { ...next[next.length - 1] };
            if (reasoning) last.reasoning = (last.reasoning || "") + reasoning;
            if (content) last.content = (last.content || "") + content;
            next[next.length - 1] = last;
            return next;
          });
        }
      });

      const total = (performance.now() - startedAt) / 1000;
      const ttft = firstTokenAt ? (firstTokenAt - startedAt) / 1000 : null;
      setMetrics({ total, ttft, usage: finalUsage });
    } catch (e) {
      if (e.name !== "AbortError") {
        setError(e.message || String(e));
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function cancel() {
    if (abortRef.current) abortRef.current.abort();
  }

  function reset() {
    setMessages([]);
    setError("");
    setMetrics(null);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  const selectedModel = models.find((m) => m.id === model);

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <div className="brand-title">Next.js × NVIDIA Nemotron</div>
            <div className="brand-sub">
              A minimal chat client that streams from NVIDIA Build.
            </div>
          </div>
        </div>
        <div className="model-row">
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={busy}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          {selectedModel && selectedModel.supportsThinking && (
            <label className="toggle">
              <input
                type="checkbox"
                checked={thinking}
                onChange={(e) => setThinking(e.target.checked)}
                disabled={busy}
              />
              Show thinking
            </label>
          )}
          <button className="btn btn-ghost" onClick={reset} disabled={busy}>
            Reset
          </button>
        </div>
      </header>

      <div className="chat-area" ref={chatRef}>
        {messages.length === 0 && (
          <div className="empty-hint">
            Ask anything — try <em>“Write a haiku about edge GPUs.”</em>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ display: "contents" }}>
            {m.role === "assistant" && m.reasoning && (
              <div className="bubble thinking">
                <span className="bubble-role">Thinking</span>
                {m.reasoning}
              </div>
            )}
            <div className={`bubble ${m.role}`}>
              <span className="bubble-role">{m.role}</span>
              {m.content || (m.role === "assistant" && busy ? "…" : "")}
            </div>
          </div>
        ))}
        {error && (
          <div className="bubble error">
            <span className="bubble-role">Error</span>
            {error}
          </div>
        )}
        {metrics && metrics.usage && (
          <div className="metrics">
            tokens · prompt {metrics.usage.prompt_tokens} · completion{" "}
            {metrics.usage.completion_tokens} · total {metrics.usage.total_tokens}
            {metrics.ttft != null && ` · TTFT ${metrics.ttft.toFixed(2)}s`} ·
            wall {metrics.total.toFixed(2)}s
          </div>
        )}
      </div>

      <div className="input-row">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message and press Enter (Shift+Enter for newline)…"
          disabled={busy}
        />
        {busy ? (
          <button className="btn" onClick={cancel}>
            Stop
          </button>
        ) : (
          <button className="btn" onClick={sendMessage} disabled={!input.trim()}>
            Send
          </button>
        )}
      </div>
    </>
  );
}
