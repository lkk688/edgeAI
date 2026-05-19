"use client";

import { useEffect, useRef, useState } from "react";

// Same minimal SSE parser used by ChatUI — kept local so the file is self-
// contained for the tutorial.
async function readSSE(response, onDelta) {
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
        const chunk = JSON.parse(payload);
        const delta = chunk.choices?.[0]?.delta;
        onDelta({
          content: delta?.content,
          reasoning: delta?.reasoning_content,
          usage: chunk.usage,
        });
      } catch {
        /* ignore */
      }
    }
  }
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function humanSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

const MAX_FILE_MB = 8;

export default function OmniLab() {
  const [prompt, setPrompt] = useState("Describe what you see and/or hear.");
  const [image, setImage] = useState(null); // { name, size, type, data_url }
  const [audio, setAudio] = useState(null); // { name, size, type, data_url, format }
  const [thinking, setThinking] = useState(true);
  const [reasoningBudget, setReasoningBudget] = useState(4096);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [reasoningOut, setReasoningOut] = useState("");
  const [answerOut, setAnswerOut] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [showReasoning, setShowReasoning] = useState(true);
  const imageInputRef = useRef(null);
  const audioInputRef = useRef(null);
  const abortRef = useRef(null);
  const outRef = useRef(null);

  useEffect(() => {
    if (outRef.current) outRef.current.scrollTop = outRef.current.scrollHeight;
  }, [reasoningOut, answerOut]);

  async function onImagePick(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`Image is too large (${humanSize(file.size)}). Limit is ${MAX_FILE_MB} MB.`);
      e.target.value = "";
      return;
    }
    setError("");
    const data_url = await fileToDataURL(file);
    setImage({ name: file.name, size: file.size, type: file.type, data_url });
  }

  async function onAudioPick(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`Audio is too large (${humanSize(file.size)}). Limit is ${MAX_FILE_MB} MB.`);
      e.target.value = "";
      return;
    }
    setError("");
    const data_url = await fileToDataURL(file);
    const m = /^data:audio\/([a-z0-9]+);/i.exec(data_url);
    const format = m ? m[1].toLowerCase() : "wav";
    setAudio({ name: file.name, size: file.size, type: file.type, data_url, format });
  }

  function clearImage() {
    setImage(null);
    if (imageInputRef.current) imageInputRef.current.value = "";
  }
  function clearAudio() {
    setAudio(null);
    if (audioInputRef.current) audioInputRef.current.value = "";
  }

  async function run() {
    if (busy) return;
    if (!prompt.trim() && !image && !audio) {
      setError("Type a prompt, attach an image, or attach an audio file.");
      return;
    }
    setError("");
    setReasoningOut("");
    setAnswerOut("");
    setMetrics(null);
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const startedAt = performance.now();
    let firstTokenAt = null;
    let finalUsage = null;

    try {
      const res = await fetch("/api/omni", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          image: image ? { data_url: image.data_url } : null,
          audio: audio
            ? { data_url: audio.data_url, format: audio.format }
            : null,
          thinking,
          reasoning_budget: Math.max(0, parseInt(reasoningBudget, 10) || 0),
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${res.status}`);
      }

      await readSSE(res, ({ content, reasoning, usage }) => {
        if ((content || reasoning) && firstTokenAt == null) {
          firstTokenAt = performance.now();
        }
        if (usage) finalUsage = usage;
        if (reasoning) setReasoningOut((s) => s + reasoning);
        if (content)   setAnswerOut((s) => s + content);
      });

      const total = (performance.now() - startedAt) / 1000;
      const ttft = firstTokenAt ? (firstTokenAt - startedAt) / 1000 : null;
      setMetrics({ total, ttft, usage: finalUsage });
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

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <div className="brand-title">Omni Lab — image + audio + text</div>
            <div className="brand-sub">
              nvidia/nemotron-3-nano-omni-30b-a3b-reasoning · streaming with visible thinking.
            </div>
          </div>
        </div>
      </header>

      <div className="lab-grid">
        <section className="lab-col">
          <label className="lab-label">Prompt</label>
          <textarea
            className="lab-textarea"
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={busy}
            placeholder="Ask something about the attached image/audio, or just chat."
          />

          <label className="lab-label">Image (optional · png / jpg / webp)</label>
          <div className="file-row">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              onChange={onImagePick}
              disabled={busy}
            />
            {image && (
              <button className="btn btn-ghost btn-sm" onClick={clearImage} disabled={busy}>
                clear
              </button>
            )}
          </div>
          {image && (
            <div className="attachment">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={image.data_url} alt="preview" className="attachment-img" />
              <div className="attachment-meta">
                {image.name} · {humanSize(image.size)}
              </div>
            </div>
          )}

          <label className="lab-label">Audio (optional · wav / mp3)</label>
          <div className="file-row">
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*"
              onChange={onAudioPick}
              disabled={busy}
            />
            {audio && (
              <button className="btn btn-ghost btn-sm" onClick={clearAudio} disabled={busy}>
                clear
              </button>
            )}
          </div>
          {audio && (
            <div className="attachment">
              <audio controls src={audio.data_url} className="attachment-audio" />
              <div className="attachment-meta">
                {audio.name} · {humanSize(audio.size)} · format <code>{audio.format}</code>
              </div>
            </div>
          )}

          <div className="lab-controls">
            <label className="toggle">
              <input
                type="checkbox"
                checked={thinking}
                onChange={(e) => setThinking(e.target.checked)}
                disabled={busy}
              />
              Enable thinking
            </label>
            <label className="toggle">
              budget:
              <input
                type="number"
                min="0"
                max="32768"
                step="512"
                value={reasoningBudget}
                onChange={(e) => setReasoningBudget(e.target.value)}
                disabled={busy || !thinking}
                style={{ width: 80, marginLeft: 4 }}
              />
            </label>
            {busy ? (
              <button className="btn" onClick={cancel}>Stop</button>
            ) : (
              <button className="btn" onClick={run}>Run</button>
            )}
          </div>

          {error && (
            <div className="bubble error" style={{ alignSelf: "stretch" }}>
              <span className="bubble-role">Error</span>
              {error}
            </div>
          )}
          {metrics && metrics.usage && (
            <div className="metrics">
              tokens · prompt {metrics.usage.prompt_tokens} · completion{" "}
              {metrics.usage.completion_tokens} · total{" "}
              {metrics.usage.total_tokens}
              {metrics.ttft != null && ` · TTFT ${metrics.ttft.toFixed(2)}s`} ·
              wall {metrics.total.toFixed(2)}s
            </div>
          )}
        </section>

        <section className="lab-col" ref={outRef} style={{ overflow: "auto" }}>
          {thinking && (
            <>
              <div className="lab-label" style={{ display: "flex", justifyContent: "space-between" }}>
                <span>Reasoning ({reasoningOut.length} chars)</span>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={showReasoning}
                    onChange={(e) => setShowReasoning(e.target.checked)}
                  />
                  show
                </label>
              </div>
              {showReasoning && (
                <div className="bubble thinking" style={{ alignSelf: "stretch", maxWidth: "100%" }}>
                  {reasoningOut || (busy ? "…" : "(empty — model has not started thinking yet)")}
                </div>
              )}
            </>
          )}

          <div className="lab-label" style={{ marginTop: 12 }}>Answer</div>
          <div className="bubble assistant" style={{ alignSelf: "stretch", maxWidth: "100%" }}>
            {answerOut || (busy ? "…" : "(empty — run the model)")}
          </div>
        </section>
      </div>
    </>
  );
}
