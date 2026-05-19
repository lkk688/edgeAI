"use client";

import { useEffect, useRef, useState } from "react";

const LANGS = [
  { code: "en-US", label: "English (US)" },
  { code: "en-GB", label: "English (UK)" },
  { code: "es-US", label: "Spanish (US)" },
  { code: "fr-FR", label: "French" },
  { code: "de-DE", label: "German" },
];

const TARGET_SR = 16000;        // Riva expects LINEAR_PCM at this rate
const MAX_FILE_MB = 12;         // cap upload size (pre-decode)
const MAX_RECORD_SEC = 60;      // safety cap on mic recording

// ---------- audio helpers ------------------------------------------------

// Decode any browser-supported audio into a Float32 mono 16 kHz PCM buffer,
// then convert to little-endian 16-bit PCM bytes ready for /api/asr.
async function audioBufferToInt16PCM(buf) {
  // 1) Down-mix to mono.
  const mono = new Float32Array(buf.length);
  for (let ch = 0; ch < buf.numberOfChannels; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < buf.length; i++) {
      mono[i] += data[i] / buf.numberOfChannels;
    }
  }

  // 2) Resample to 16 kHz with an OfflineAudioContext. The browser's built-in
  //    resampler is good enough for ASR.
  let samples = mono;
  if (buf.sampleRate !== TARGET_SR) {
    const targetLen = Math.round((mono.length * TARGET_SR) / buf.sampleRate);
    const offline = new OfflineAudioContext(1, targetLen, TARGET_SR);
    const monoBuf = offline.createBuffer(1, mono.length, buf.sampleRate);
    monoBuf.getChannelData(0).set(mono);
    const src = offline.createBufferSource();
    src.buffer = monoBuf;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    samples = rendered.getChannelData(0);
  }

  // 3) Float32 [-1,1] → Int16 LE.
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm.buffer;
}

async function decodeToAudioBuffer(arrayBuffer) {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctor();
  try {
    return await ctx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    // ctx.close() is async; we don't await because we don't need it.
    ctx.close?.();
  }
}

function humanSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

// ---------- SSE reader ---------------------------------------------------

async function readASRStream(response, onEvent) {
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

// ---------- component ----------------------------------------------------

export default function AsrLab() {
  const [mode, setMode] = useState("file"); // "file" | "mic"
  const [lang, setLang] = useState("en-US");

  // file mode
  const [file, setFile] = useState(null);
  const [audioBuf, setAudioBuf] = useState(null); // AudioBuffer
  const fileInputRef = useRef(null);

  // mic mode
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const recorderRef = useRef(null);
  const recordTimerRef = useRef(null);
  const [recordSec, setRecordSec] = useState(0);

  // transcription state
  const [busy, setBusy] = useState(false);
  const [finals, setFinals] = useState([]);     // [{text, elapsed_ms}]
  const [partial, setPartial] = useState("");
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState(null); // {durationSec, pcmBytes, totalMs}
  const abortRef = useRef(null);

  // ---------- file mode ----------

  async function onFilePick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`File is too large (${humanSize(f.size)}). Limit is ${MAX_FILE_MB} MB.`);
      e.target.value = "";
      return;
    }
    setError("");
    setFile(f);
    setAudioBuf(null);
    try {
      const buf = await f.arrayBuffer();
      const decoded = await decodeToAudioBuffer(buf);
      setAudioBuf(decoded);
    } catch (err) {
      setError(`Could not decode audio: ${err.message || err}`);
    }
  }

  function clearFile() {
    setFile(null);
    setAudioBuf(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  // ---------- mic mode ----------

  async function startRecording() {
    setError("");
    setRecordedBlob(null);
    setRecordSec(0);
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setError(
        "Microphone permission denied. Browsers only expose getUserMedia on " +
          "https:// or localhost — open this page on the Jetson itself, or via " +
          "an SSH tunnel, not over plain HTTP from another host."
      );
      return;
    }

    const mime =
      MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "";
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
      const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      setRecordedBlob(blob);
      try {
        const buf = await blob.arrayBuffer();
        const decoded = await decodeToAudioBuffer(buf);
        setAudioBuf(decoded);
      } catch (err) {
        setError(`Could not decode recording: ${err.message || err}`);
      }
    };
    recorderRef.current = rec;
    rec.start();
    setRecording(true);
    const started = performance.now();
    recordTimerRef.current = setInterval(() => {
      const elapsed = (performance.now() - started) / 1000;
      setRecordSec(elapsed);
      if (elapsed >= MAX_RECORD_SEC) {
        stopRecording();
      }
    }, 100);
  }

  function stopRecording() {
    if (!recorderRef.current) return;
    if (recorderRef.current.state !== "inactive") recorderRef.current.stop();
    recorderRef.current = null;
    setRecording(false);
  }

  function clearMic() {
    setRecordedBlob(null);
    setAudioBuf(null);
    setRecordSec(0);
  }

  useEffect(() => () => {
    // cleanup if user navigates away mid-recording
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
  }, []);

  // ---------- transcribe ----------

  async function transcribe() {
    if (!audioBuf) {
      setError("No audio loaded. Pick a file or record from the mic first.");
      return;
    }
    setError("");
    setFinals([]);
    setPartial("");
    setMetrics(null);
    setBusy(true);

    const startedAt = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const pcm = await audioBufferToInt16PCM(audioBuf);
      const res = await fetch(
        `/api/asr?sr=${TARGET_SR}&lang=${encodeURIComponent(lang)}&interim=1`,
        {
          method: "POST",
          body: pcm,
          headers: { "Content-Type": "application/octet-stream" },
          signal: controller.signal,
        }
      );
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${res.status}`);
      }
      let lastElapsed = 0;
      await readASRStream(res, (evt) => {
        if (evt.type === "partial") {
          setPartial(evt.text || "");
          lastElapsed = evt.elapsed_ms || lastElapsed;
        } else if (evt.type === "final") {
          setFinals((prev) => [
            ...prev,
            { text: evt.text || "", elapsed_ms: evt.elapsed_ms || 0 },
          ]);
          setPartial("");
          lastElapsed = evt.elapsed_ms || lastElapsed;
        } else if (evt.type === "error") {
          setError(evt.message || "ASR sidecar error");
        } else if (evt.type === "done") {
          lastElapsed = evt.elapsed_ms || lastElapsed;
        }
      });
      setMetrics({
        durationSec: audioBuf.duration,
        pcmBytes: pcm.byteLength,
        totalMs: performance.now() - startedAt,
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

  // ---------- render ----------

  const ready = !!audioBuf && !busy;
  const fullTranscript =
    finals.map((f) => f.text).join("").trim() +
    (partial ? (finals.length ? " " : "") + partial : "");

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <div className="brand-title">ASR Lab — streaming speech-to-text</div>
            <div className="brand-sub">
              nvidia/nemotron-asr-streaming · via Riva gRPC sidecar · streamed
              partials &amp; finals.
            </div>
          </div>
        </div>
        <div className="model-row">
          <select value={lang} onChange={(e) => setLang(e.target.value)} disabled={busy}>
            {LANGS.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className="asr-tabs">
        <button
          className={`asr-tab ${mode === "file" ? "is-active" : ""}`}
          onClick={() => setMode("file")}
          disabled={busy}
        >
          Upload file
        </button>
        <button
          className={`asr-tab ${mode === "mic" ? "is-active" : ""}`}
          onClick={() => setMode("mic")}
          disabled={busy}
        >
          Record microphone
        </button>
      </div>

      <div className="lab-grid">
        <section className="lab-col">
          {mode === "file" ? (
            <>
              <label className="lab-label">Audio file (wav / mp3 / m4a / webm · ≤{MAX_FILE_MB} MB)</label>
              <div className="file-row">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={onFilePick}
                  disabled={busy}
                />
                {file && (
                  <button className="btn btn-ghost btn-sm" onClick={clearFile} disabled={busy}>
                    clear
                  </button>
                )}
              </div>
              {file && (
                <div className="attachment">
                  {recordedBlob ? (
                    <audio controls src={URL.createObjectURL(recordedBlob)} className="attachment-audio" />
                  ) : (
                    <audio controls src={URL.createObjectURL(file)} className="attachment-audio" />
                  )}
                  <div className="attachment-meta">
                    {file.name} · {humanSize(file.size)}
                    {audioBuf && (
                      <>
                        {" · "}
                        {audioBuf.duration.toFixed(2)} s · {audioBuf.sampleRate} Hz ·{" "}
                        {audioBuf.numberOfChannels} ch
                      </>
                    )}
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <label className="lab-label">Microphone (≤{MAX_RECORD_SEC} s)</label>
              <div className="mic-row">
                {!recording ? (
                  <button className="btn" onClick={startRecording} disabled={busy}>
                    ● Record
                  </button>
                ) : (
                  <button className="btn btn-rec" onClick={stopRecording}>
                    ■ Stop ({recordSec.toFixed(1)} s)
                  </button>
                )}
                {recordedBlob && !recording && (
                  <button className="btn btn-ghost btn-sm" onClick={clearMic} disabled={busy}>
                    clear
                  </button>
                )}
              </div>
              <div className="attachment-meta">
                Browser sends mic audio in default WebM/Opus, then re-encodes
                to 16 kHz mono LINEAR_PCM in JavaScript before posting.
                getUserMedia requires HTTPS or localhost — open this page on
                the Jetson itself, not from another host over plain HTTP.
              </div>
              {recordedBlob && (
                <div className="attachment">
                  <audio controls src={URL.createObjectURL(recordedBlob)} className="attachment-audio" />
                  <div className="attachment-meta">
                    {humanSize(recordedBlob.size)} ({recordedBlob.type})
                    {audioBuf && (
                      <>
                        {" · "}
                        {audioBuf.duration.toFixed(2)} s · {audioBuf.sampleRate} Hz ·{" "}
                        {audioBuf.numberOfChannels} ch
                      </>
                    )}
                  </div>
                </div>
              )}
            </>
          )}

          <div className="lab-controls">
            <div className="attachment-meta">
              target: 16 kHz mono 16-bit PCM (Riva LINEAR_PCM)
            </div>
            {busy ? (
              <button className="btn" onClick={cancel}>Stop</button>
            ) : (
              <button className="btn" onClick={transcribe} disabled={!ready}>
                Transcribe
              </button>
            )}
          </div>

          {error && (
            <div className="bubble error" style={{ alignSelf: "stretch" }}>
              <span className="bubble-role">Error</span>
              {error}
            </div>
          )}
          {metrics && (
            <div className="metrics">
              audio {metrics.durationSec.toFixed(2)} s · pcm{" "}
              {humanSize(metrics.pcmBytes)} · wall{" "}
              {(metrics.totalMs / 1000).toFixed(2)} s
            </div>
          )}
        </section>

        <section className="lab-col" style={{ overflow: "hidden" }}>
          <div className="lab-label">Live transcript</div>
          <div className="asr-transcript">
            {finals.length === 0 && !partial && (
              <span style={{ color: "var(--muted)" }}>
                {busy
                  ? "Listening…"
                  : "Load audio or record from the mic, then click Transcribe."}
              </span>
            )}
            {finals.map((f, i) => (
              <span key={i} className="asr-final">
                {f.text}
              </span>
            ))}
            {partial && <span className="asr-partial">{partial}</span>}
          </div>

          {finals.length > 0 && (
            <>
              <div className="lab-label" style={{ marginTop: 12 }}>
                Final segments
              </div>
              {finals.map((f, i) => (
                <div key={i} className="result-card">
                  <div className="result-meta">
                    #{i + 1} · t+{(f.elapsed_ms / 1000).toFixed(2)} s
                  </div>
                  <div>{f.text}</div>
                </div>
              ))}
            </>
          )}

          {fullTranscript && !busy && (
            <button
              className="btn btn-ghost btn-sm"
              style={{ marginTop: 8, alignSelf: "flex-start" }}
              onClick={() => navigator.clipboard.writeText(fullTranscript)}
            >
              Copy full transcript
            </button>
          )}
        </section>
      </div>
    </>
  );
}
