#!/usr/bin/env python3
"""
benchmark_llm.py — benchmark a served OpenAI-compatible LLM (tokens/sec, TTFT).

Sends a set of prompts to a running server (e.g. `sjsujetsontool llama` on
:8080) and reports per-prompt and average generation speed. Uses llama.cpp's
`timings` when present (separate prefill vs generation rates), else measures
wall-clock. Pure stdlib.

Examples:
  python3 benchmark_llm.py                                  # local :8080
  python3 benchmark_llm.py --url https://llm.forgengi.org/node05/v1 --api-key sjsugputool
  python3 benchmark_llm.py --runs 3 --max-tokens 128
"""
import argparse, json, time, statistics, urllib.request, urllib.error

PROMPTS = [
    "Explain the importance of edge AI in smart cities.",
    "Write a haiku about an NVIDIA Jetson.",
    "List three uses of on-device language models.",
]


def one_call(endpoint, headers, model, prompt, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    t0 = time.time()
    ttft = None
    usage = timings = None
    resp = urllib.request.urlopen(req, timeout=120)
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("timings"):
            timings = obj["timings"]
        for ch in obj.get("choices") or []:
            if ch.get("delta", {}).get("content") and ttft is None:
                ttft = time.time() - t0
    dt = time.time() - t0
    gen = (usage or {}).get("completion_tokens", 0)
    gen_tps = (timings or {}).get("predicted_per_second") or (gen / dt if dt else 0)
    pre_tps = (timings or {}).get("prompt_per_second")
    return {"dt": dt, "ttft": ttft or 0, "gen": gen, "gen_tps": gen_tps, "pre_tps": pre_tps}


def main():
    ap = argparse.ArgumentParser(description="Benchmark a served OpenAI-compatible LLM.")
    ap.add_argument("--url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="sk-no-key-required")
    ap.add_argument("--model", default="local")
    ap.add_argument("--runs", type=int, default=2, help="runs per prompt")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + args.api_key}

    print("Benchmarking %s  (%d prompts x %d runs)\n" % (endpoint, len(PROMPTS), args.runs))
    gen_rates, ttfts = [], []
    for i, prompt in enumerate(PROMPTS, 1):
        for r in range(args.runs):
            try:
                m = one_call(endpoint, headers, args.model, prompt, args.max_tokens)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print("  prompt %d run %d: ERROR %s" % (i, r + 1, e))
                continue
            gen_rates.append(m["gen_tps"]); ttfts.append(m["ttft"])
            extra = (" · prefill %.0f tok/s" % m["pre_tps"]) if m["pre_tps"] else ""
            print("  prompt %d run %d: %d tok @ %.1f tok/s · TTFT %.2fs%s"
                  % (i, r + 1, m["gen"], m["gen_tps"], m["ttft"], extra))

    if gen_rates:
        print("\n📊 Average generation: %.1f tok/s   (min %.1f, max %.1f)"
              % (statistics.mean(gen_rates), min(gen_rates), max(gen_rates)))
        print("📊 Average TTFT      : %.2fs" % statistics.mean(ttfts))


if __name__ == "__main__":
    main()
