"use client";

import { useEffect, useRef, useState } from "react";

const LANGS = [
  { code: "en-US", label: "English (US)" },
  { code: "en-GB", label: "English (UK)" },
  { code: "es-US", label: "Spanish (US)" },
  { code: "fr-FR", label: "French" },
  { code: "de-DE", label: "German" },
];

const REF_SR        = 16000;   // sample rate we send the voice reference at
const OUTPUT_SR     = 22050;   // sample rate of synthesized output
const MIN_REF_SEC   = 3;
const MAX_REF_SEC   = 10;
const MAX_REF_MB    = 12;
const MAX_TEXT_LEN  = 800;
const MAX_RECORD_SEC = 10;

// ---------- audio helpers ----------

async function decodeToAudioBuffer(arrayBuffer) {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctor();
  try {
    return await ctx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    ctx.close?.();
  }
}

// Down-mix + resample → mono Int16 LE PCM at the requested sample rate.
async function audioBufferToMonoInt16(buf, targetSR) {
  const mono = new Float32Array(buf.length);
  for (let ch = 0; ch < buf.numberOfChannels; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < buf.length; i++) mono[i] += data[i] / buf.numberOfChannels;
  }

  let samples = mono;
  if (buf.sampleRate !== targetSR) {
    const targetLen = Math.round((mono.length * targetSR) / buf.sampleRate);
    const offline = new OfflineAudioContext(1, targetLen, targetSR);
    const inBuf = offline.createBuffer(1, mono.length, buf.sampleRate);
    inBuf.getChannelData(0).set(mono);
    const src = offline.createBufferSource();
    src.buffer = inBuf;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    samples = rendered.getChannelData(0);
  }

  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
}

// Wrap a mono Int16 buffer in a minimal RIFF/WAVE container → Blob.
function pcmToWavBlob(int16, sampleRate) {
  const n = int16.byteLength;
  const buf = new ArrayBuffer(44 + n);
  const v = new DataView(buf);
  let p = 0;
  const w8 = (s) => { for (const c of s) v.setUint8(p++, c.charCodeAt(0)); };
  const w32 = (x) => { v.setUint32(p, x, true); p += 4; };
  const w16 = (x) => { v.setUint16(p, x, true); p += 2; };
  w8("RIFF"); w32(36 + n); w8("WAVE"); w8("fmt ");
  w32(16); w16(1); w16(1);
  w32(sampleRate); w32(sampleRate * 2); w16(2); w16(16);
  w8("data"); w32(n);
  new Int16Array(buf, 44).set(int16);
  return new Blob([buf], { type: "audio/wav" });
}

function humanSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

// ---------- component ----------

const SAMPLE_PROMPTS = [
  "Hello from the Jetson Orin Nano. Welcome to the zero-shot voice cloning lab.",
  "The quick brown fox jumps over the lazy dog.",
  "Edge AI is the practice of running machine learning models directly on devices, " +
    "without sending data to the cloud.",
];

export default function TtsLab() {
  const [text, setText] = useState(SAMPLE_PROMPTS[0]);
  const [mode, setMode] = useState("file"); // "file" | "mic"
  const [lang, setLang] = useState("en-US");
  const [quality, setQuality] = useState(20);

  // voice reference state
  const [refFile, setRefFile] = useState(null);
  const [refBlob, setRefBlob] = useState(null);     // either uploaded File or recorded Blob
  const [refBuf, setRefBuf]   = useState(null);     // decoded AudioBuffer
  const fileInputRef = useRef(null);

  // mic recording state
  const [recording, setRecording] = useState(false);
  const [recordSec, setRecordSec] = useState(0);
  const recorderRef = useRef(null);
  const recordTimerRef = useRef(null);

  // synthesis state
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [outputUrl, setOutputUrl] = useState(null);
  const [outputBlob, setOutputBlob] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const abortRef = useRef(null);

  // ---------- ref voice: file mode ----------

  async function onFilePick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > MAX_REF_MB * 1024 * 1024) {
      setError(`Reference is too large (${humanSize(f.size)}). Limit is ${MAX_REF_MB} MB.`);
      e.target.value = "";
      return;
    }
    setError("");
    setRefFile(f);
    setRefBlob(f);
    setRefBuf(null);
    try {
      const buf = await f.arrayBuffer();
      const decoded = await decodeToAudioBuffer(buf);
      setRefBuf(decoded);
    } catch (err) {
      setError(`Could not decode audio: ${err.message || err}`);
    }
  }

  function clearRef() {
    setRefFile(null);
    setRefBlob(null);
    setRefBuf(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  // ---------- ref voice: mic mode ----------

  async function startRecording() {
    setError("");
    setRefBlob(null);
    setRefBuf(null);
    setRecordSec(0);
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setError(
        "Microphone permission denied. Browsers only expose getUserMedia on " +
          "https:// or localhost — open this page on the Jetson itself or " +
          "via an SSH tunnel, not over plain HTTP from another host."
      );
      return;
    }
    const mime =
      MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" :
      MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
      const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      setRefBlob(blob);
      setRefFile(null);
      try {
        const decoded = await decodeToAudioBuffer(await blob.arrayBuffer());
        setRefBuf(decoded);
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
      if (elapsed >= MAX_RECORD_SEC) stopRecording();
    }, 100);
  }

  function stopRecording() {
    if (!recorderRef.current) return;
    if (recorderRef.current.state !== "inactive") recorderRef.current.stop();
    recorderRef.current = null;
    setRecording(false);
  }

  useEffect(() => () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    if (outputUrl) URL.revokeObjectURL(outputUrl);
  }, []); // eslint-disable-line

  // ---------- synthesize ----------

  const refSeconds = refBuf ? refBuf.duration : 0;
  const refValid =
    refBuf && refSeconds >= MIN_REF_SEC && refSeconds <= MAX_REF_SEC + 0.5;

  async function synthesize() {
    if (!text.trim()) {
      setError("Type the text you want spoken.");
      return;
    }
    if (text.length > MAX_TEXT_LEN) {
      setError(`Text is over ${MAX_TEXT_LEN} characters.`);
      return;
    }
    if (!refBuf) {
      setError(`Provide a ${MIN_REF_SEC}-${MAX_REF_SEC} s reference voice (upload or record).`);
      return;
    }
    if (!refValid) {
      setError(
        `Reference must be between ${MIN_REF_SEC} and ${MAX_REF_SEC} seconds ` +
        `(got ${refSeconds.toFixed(2)} s).`
      );
      return;
    }
    setError("");
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    setOutputUrl(null);
    setOutputBlob(null);
    setMetrics(null);
    setBusy(true);
    const startedAt = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // Always re-encode the reference to a clean 16 kHz mono WAV so the
      // sidecar's Riva call gets a format the server is happy with —
      // regardless of whether the user uploaded an MP3 or recorded WebM/Opus.
      const pcm = await audioBufferToMonoInt16(refBuf, REF_SR);
      const wavBlob = pcmToWavBlob(pcm, REF_SR);

      const form = new FormData();
      form.append("text", text);
      form.append("voice_ref", wavBlob, "voice_ref.wav");
      form.append("language_code", lang);
      form.append("sample_rate_hz", String(OUTPUT_SR));
      form.append("quality", String(quality));

      const res = await fetch("/api/tts", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setOutputBlob(blob);
      setOutputUrl(url);

      setMetrics({
        wall_ms: performance.now() - startedAt,
        synth_ms: parseInt(res.headers.get("x-synth-elapsed-ms") || "0", 10),
        audio_sec: parseFloat(res.headers.get("x-synth-audio-seconds") || "0"),
        sample_rate: parseInt(res.headers.get("x-synth-sample-rate") || OUTPUT_SR, 10),
        out_bytes: blob.size,
      });
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message || String(e));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function cancel() { if (abortRef.current) abortRef.current.abort(); }

  function downloadOutput() {
    if (!outputBlob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(outputBlob);
    a.download = "synth.wav";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  const refPreviewUrl = refBlob ? URL.createObjectURL(refBlob) : null;

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <div className="brand-title">TTS Lab — zero-shot voice cloning</div>
            <div className="brand-sub">
              nvidia/magpie-tts-zeroshot · 3–10 s reference voice + text → synthesized speech.
            </div>
          </div>
        </div>
        <div className="model-row">
          <select value={lang} onChange={(e) => setLang(e.target.value)} disabled={busy}>
            {LANGS.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </div>
      </header>

      <div className="lab-grid">
        <section className="lab-col">
          <label className="lab-label">Text to synthesize ({text.length}/{MAX_TEXT_LEN})</label>
          <textarea
            className="lab-textarea"
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={busy}
            maxLength={MAX_TEXT_LEN}
          />
          <div className="file-row">
            {SAMPLE_PROMPTS.map((p, i) => (
              <button
                key={i}
                className="btn btn-ghost btn-sm"
                onClick={() => setText(p)}
                disabled={busy}
                title={p}
              >
                sample {i + 1}
              </button>
            ))}
          </div>

          <div className="asr-tabs" style={{ marginTop: 8 }}>
            <button
              className={`asr-tab ${mode === "file" ? "is-active" : ""}`}
              onClick={() => setMode("file")}
              disabled={busy}
            >
              Upload reference
            </button>
            <button
              className={`asr-tab ${mode === "mic" ? "is-active" : ""}`}
              onClick={() => setMode("mic")}
              disabled={busy}
            >
              Record reference
            </button>
          </div>

          {mode === "file" ? (
            <>
              <label className="lab-label">
                Reference voice ({MIN_REF_SEC}–{MAX_REF_SEC} s · wav / mp3 / m4a / webm)
              </label>
              <div className="file-row">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={onFilePick}
                  disabled={busy}
                />
                {refFile && (
                  <button className="btn btn-ghost btn-sm" onClick={clearRef} disabled={busy}>
                    clear
                  </button>
                )}
              </div>
            </>
          ) : (
            <>
              <label className="lab-label">
                Reference voice (record {MIN_REF_SEC}–{MAX_REF_SEC} s of your own speech)
              </label>
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
                {refBlob && !recording && (
                  <button className="btn btn-ghost btn-sm" onClick={clearRef} disabled={busy}>
                    clear
                  </button>
                )}
              </div>
              <div className="attachment-meta">
                Tip: read a normal sentence at your usual pace for {MIN_REF_SEC}–{MAX_REF_SEC} s.
                Avoid background noise — the model will copy the timbre of whatever it hears.
              </div>
            </>
          )}

          {refBlob && refPreviewUrl && (
            <div className="attachment">
              <audio controls src={refPreviewUrl} className="attachment-audio" />
              <div className="attachment-meta">
                {humanSize(refBlob.size)}
                {refBuf && (
                  <>
                    {" · "}
                    {refBuf.duration.toFixed(2)} s · {refBuf.sampleRate} Hz ·{" "}
                    {refBuf.numberOfChannels} ch
                    {!refValid && (
                      <span style={{ color: "var(--error)" }}>
                        {" · outside "}
                        {MIN_REF_SEC}–{MAX_REF_SEC} s window
                      </span>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          <div className="lab-controls">
            <label className="toggle">
              quality (1–40):
              <input
                type="number"
                min="1"
                max="40"
                value={quality}
                onChange={(e) => setQuality(parseInt(e.target.value, 10) || 20)}
                disabled={busy}
                style={{ width: 56, marginLeft: 4 }}
              />
            </label>
            {busy ? (
              <button className="btn" onClick={cancel}>Stop</button>
            ) : (
              <button className="btn" onClick={synthesize} disabled={!refValid || !text.trim()}>
                Synthesize
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
              synth {(metrics.synth_ms / 1000).toFixed(2)} s · wall{" "}
              {(metrics.wall_ms / 1000).toFixed(2)} s · audio{" "}
              {metrics.audio_sec.toFixed(2)} s @ {metrics.sample_rate} Hz ·{" "}
              {humanSize(metrics.out_bytes)}
            </div>
          )}
        </section>

        <section className="lab-col">
          <div className="lab-label">Synthesized output</div>
          {outputUrl ? (
            <div className="attachment">
              <audio controls src={outputUrl} autoPlay className="attachment-audio" />
              <div className="attachment-meta">
                {outputBlob && `${humanSize(outputBlob.size)} · audio/wav`}
              </div>
              <button
                className="btn btn-ghost btn-sm"
                style={{ alignSelf: "flex-start" }}
                onClick={downloadOutput}
              >
                Download .wav
              </button>
            </div>
          ) : (
            <div className="empty-hint" style={{ padding: 12 }}>
              {busy
                ? "Synthesizing — magpie-tts-zeroshot typically returns in 5–15 s for short prompts."
                : "Pick or record a reference voice, type some text, then press Synthesize."}
            </div>
          )}

          <div className="lab-label" style={{ marginTop: 12 }}>How it works</div>
          <div className="attachment">
            <ol style={{ margin: "6px 0 0 18px", padding: 0, color: "var(--muted)", fontSize: 13 }}>
              <li>Browser decodes your reference → 16 kHz mono 16-bit PCM → wraps in a WAV.</li>
              <li>Multipart POST → <code>/api/tts</code> → FastAPI sidecar.</li>
              <li>Sidecar calls Riva <code>SpeechSynthesisService.synthesize()</code> against the
                  <code>magpie-tts-zeroshot</code> NVCF gRPC function with your text + the WAV
                  as <code>zero_shot_audio_prompt_file</code>.</li>
              <li>Sidecar wraps the returned PCM in a WAV header and returns it as
                  <code>audio/wav</code>.</li>
              <li>Browser plays the blob from a generated object URL.</li>
            </ol>
          </div>
        </section>
      </div>
    </>
  );
}
