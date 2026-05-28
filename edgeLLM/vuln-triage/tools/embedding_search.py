"""Tool: NVIDIA-Build embedding search over a small CVE corpus.

Used by §12d. Builds an in-memory cosine-similarity index once per
process, then exposes `similar_cves(query, k)` as a tool the LLM can
call to retrieve "have we seen something like this before?" context.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import httpx

NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
EMBED_MODEL = os.environ.get("TRIAGE_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

# Lazy module-level cache so we only embed the corpus once per process.
_INDEX: list[dict[str, Any]] | None = None


def _embed_batch(texts: list[str], *, input_type: str) -> list[list[float]]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    with httpx.Client(timeout=60.0, trust_env=False) as client:
        resp = client.post(
            f"{NVIDIA_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "input": texts,
                "input_type": input_type,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return [d["embedding"] for d in data["data"]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _default_corpus_path() -> Path:
    return Path(__file__).resolve().parent.parent / "examples" / "cve_corpus.jsonl"


def _load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    if not corpus_path.is_file():
        raise FileNotFoundError(f"CVE corpus missing: {corpus_path}")
    rows: list[dict[str, Any]] = []
    for line in corpus_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _build_index(corpus_path: Path) -> list[dict[str, Any]]:
    rows = _load_corpus(corpus_path)
    texts = [
        f"{r['package']}: {r['summary']}. Patterns: {', '.join(r.get('patterns', []))}"
        for r in rows
    ]
    # NVIDIA embedqa is asymmetric — corpus uses "passage", queries use "query".
    vecs = _embed_batch(texts, input_type="passage")
    indexed: list[dict[str, Any]] = []
    for r, v, t in zip(rows, vecs, texts):
        r = dict(r)
        r["_vector"] = v
        r["_text"] = t
        indexed.append(r)
    return indexed


def similar_cves(
    query: str,
    *,
    k: int = 3,
    corpus_path: str | None = None,
) -> dict[str, Any]:
    """Return the top-k most semantically similar CVE notes for `query`.

    The index is built lazily on first call and cached in this module.
    """
    global _INDEX  # pylint: disable=global-statement
    path = Path(corpus_path) if corpus_path else _default_corpus_path()
    if _INDEX is None:
        _INDEX = _build_index(path)

    q_vec = _embed_batch([query], input_type="query")[0]
    scored = [
        (_cosine(q_vec, row["_vector"]), row) for row in _INDEX
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, row in scored[:k]:
        out.append(
            {
                "cve_id": row.get("cve_id"),
                "package": row.get("package"),
                "summary": row.get("summary"),
                "patterns": row.get("patterns", []),
                "guidance": row.get("guidance", ""),
                "score": round(score, 3),
            }
        )
    return {"query": query, "k": k, "matches": out}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "HTTP proxy authorization header leak"
    print(json.dumps(similar_cves(q), indent=2))
