"use client";

import { useState } from "react";

// Cosine similarity between two equal-length vectors.
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

const DEFAULT_CORPUS = [
  "The Jetson Orin Nano is an NVIDIA edge AI computer that delivers up to 40 TOPS of compute.",
  "Bananas are yellow tropical fruit that grow on plants in clusters called hands.",
  "Llama-Nemotron models are NVIDIA-tuned variants of Meta's Llama 3 series, specialized for reasoning and tool use.",
  "Next.js is a React framework with file-based routing, server components, and built-in API routes.",
  "CUDA is NVIDIA's parallel computing platform that lets developers run general-purpose code on GPUs.",
  "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
  "Retrieval-Augmented Generation (RAG) combines a vector search step with an LLM to ground answers in your own documents.",
  "Mount Everest sits on the border between Nepal and Tibet and is the tallest mountain above sea level.",
];

const DEFAULT_QUERY = "What hardware does NVIDIA make for edge AI?";

export default function RetrievalLab() {
  const [corpusText, setCorpusText] = useState(DEFAULT_CORPUS.join("\n"));
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [topK, setTopK] = useState(3);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // Vector-search results: [{ doc, score, index }]
  const [embedResults, setEmbedResults] = useState([]);
  // Rerank results over the top-k embedding picks: [{ doc, logit, originalIndex }]
  const [rerankResults, setRerankResults] = useState([]);
  const [metrics, setMetrics] = useState(null);

  async function run() {
    setError("");
    setEmbedResults([]);
    setRerankResults([]);
    setMetrics(null);

    const docs = corpusText
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (docs.length === 0) {
      setError("Paste at least one passage into the corpus.");
      return;
    }
    if (!query.trim()) {
      setError("Enter a query.");
      return;
    }

    setBusy(true);
    const startedAt = performance.now();
    try {
      // 1) Embed query and passages in two separate calls — NVIDIA embedqa
      //    requires different `input_type` values for queries vs. passages.
      const [qRes, pRes] = await Promise.all([
        fetch("/api/embed", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ inputs: [query], input_type: "query" }),
        }).then((r) => r.json()),
        fetch("/api/embed", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ inputs: docs, input_type: "passage" }),
        }).then((r) => r.json()),
      ]);

      if (qRes.error) throw new Error(qRes.error);
      if (pRes.error) throw new Error(pRes.error);

      const qVec  = qRes.vectors[0];
      const pVecs = pRes.vectors;
      const embedAt = performance.now();

      // 2) Score everything by cosine similarity in the browser.
      const scored = pVecs
        .map((v, i) => ({ index: i, doc: docs[i], score: cosine(qVec, v) }))
        .sort((a, b) => b.score - a.score);
      const k = Math.min(Math.max(parseInt(topK, 10) || 3, 1), scored.length);
      const topByEmbed = scored.slice(0, k);
      setEmbedResults(topByEmbed);

      // 3) Rerank the top-k embedding picks with the cross-encoder.
      const rerankReq = await fetch("/api/rerank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          passages: topByEmbed.map((r) => r.doc),
        }),
      });
      const rerank = await rerankReq.json();
      if (rerank.error) throw new Error(rerank.error);
      const rerankAt = performance.now();

      setRerankResults(
        (rerank.rankings || []).map((r) => ({
          doc: r.passage,
          logit: r.logit,
          originalIndex: topByEmbed[r.index].index,
        }))
      );

      setMetrics({
        embed_ms: embedAt - startedAt,
        rerank_ms: rerankAt - embedAt,
        total_ms: rerankAt - startedAt,
        embed_dim: qRes.dim,
        docs: docs.length,
      });
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <div className="brand-title">Retrieval Lab — Embed → Rerank</div>
            <div className="brand-sub">
              nv-embedqa-e5-v5 (1024-d) · rerank-qa-mistral-4b · running on NVIDIA Build.
            </div>
          </div>
        </div>
      </header>

      <div className="lab-grid">
        <section className="lab-col">
          <label className="lab-label">Corpus — one passage per line</label>
          <textarea
            className="lab-textarea"
            value={corpusText}
            onChange={(e) => setCorpusText(e.target.value)}
            disabled={busy}
            rows={10}
          />

          <label className="lab-label">Query</label>
          <textarea
            className="lab-textarea"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={busy}
            rows={2}
          />

          <div className="lab-controls">
            <label className="toggle">
              top-k from embeddings:
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                disabled={busy}
                style={{ width: 56, marginLeft: 4 }}
              />
            </label>
            <button className="btn" onClick={run} disabled={busy}>
              {busy ? "Searching…" : "Search + rerank"}
            </button>
          </div>

          {error && <div className="bubble error" style={{ alignSelf: "stretch" }}>{error}</div>}
          {metrics && (
            <div className="metrics">
              {metrics.docs} docs · dim {metrics.embed_dim} · embed{" "}
              {metrics.embed_ms.toFixed(0)} ms · rerank{" "}
              {metrics.rerank_ms.toFixed(0)} ms · total{" "}
              {metrics.total_ms.toFixed(0)} ms
            </div>
          )}
        </section>

        <section className="lab-col">
          <div className="lab-label">
            1) Vector search · cosine similarity (top {embedResults.length || topK})
          </div>
          {embedResults.length === 0 && (
            <div className="empty-hint" style={{ padding: 12 }}>Results appear here.</div>
          )}
          {embedResults.map((r, i) => (
            <div key={i} className="result-card">
              <div className="result-meta">
                #{i + 1} · doc {r.index} · cosine {r.score.toFixed(3)}
              </div>
              <div>{r.doc}</div>
            </div>
          ))}

          <div className="lab-label" style={{ marginTop: 12 }}>
            2) Cross-encoder rerank (highest logit = most relevant)
          </div>
          {rerankResults.length === 0 && (
            <div className="empty-hint" style={{ padding: 12 }}>
              Run a search to see the rerank order.
            </div>
          )}
          {rerankResults.map((r, i) => (
            <div key={i} className="result-card result-card-rerank">
              <div className="result-meta">
                #{i + 1} · doc {r.originalIndex} · logit {r.logit.toFixed(2)}
              </div>
              <div>{r.doc}</div>
            </div>
          ))}
        </section>
      </div>
    </>
  );
}
