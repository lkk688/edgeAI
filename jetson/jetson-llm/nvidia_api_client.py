#!/usr/bin/env python3
"""
nvidia_api_client.py — query the NVIDIA Build API (cloud) from a Jetson.

The NVIDIA Build API is OpenAI-compatible, so this is a thin wrapper that
streams a chat completion and reports speed. The API key is read from, in order:
  1) --api-key
  2) $NVIDIA_API_KEY
  3) the NVIDIA_API_KEY= line in ./.env.local or edgeLLM/nextjs-nemotron-app/.env.local
Set it once with:  sjsujetsontool setup-nvapi

Examples:
  python3 nvidia_api_client.py -p "Explain speculative decoding."
  python3 nvidia_api_client.py -m nvidia/llama-3.3-nemotron-super-49b-v1 -p "Hi"
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
ENV_PATHS = [".env.local", "edgeLLM/nextjs-nemotron-app/.env.local",
             "/Developer/edgeAI/edgeLLM/nextjs-nemotron-app/.env.local"]


def find_key(cli_key):
    if cli_key:
        return cli_key
    if os.environ.get("NVIDIA_API_KEY"):
        return os.environ["NVIDIA_API_KEY"]
    for p in ENV_PATHS:
        try:
            for line in open(p):
                if line.startswith("NVIDIA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description="Query the NVIDIA Build API (OpenAI-compatible).")
    ap.add_argument("-p", "--prompt", default="Explain what edge AI is, in 3 sentences.")
    ap.add_argument("-m", "--model", default="nvidia/llama-3.1-nemotron-nano-8b-v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    key = find_key(args.api_key)
    if not key:
        print("❌ No NVIDIA API key. Run: sjsujetsontool setup-nvapi  (or pass --api-key)")
        sys.exit(1)

    payload = {"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
               "max_tokens": args.max_tokens, "stream": True}
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
               "Accept": "text/event-stream"}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=headers)

    print("→ NVIDIA Build API  (model=%s)\n" % args.model)
    t0 = time.time()
    ttft = None
    n = 0
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        print("HTTP %s: %s" % (e.code, e.read().decode("utf-8", "ignore")[:300]))
        sys.exit(1)
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
        for ch in obj.get("choices") or []:
            piece = ch.get("delta", {}).get("content")
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
                sys.stdout.write(piece)
                sys.stdout.flush()
    dt = time.time() - t0
    print("\n\n⚡ ~%d chunks · TTFT %.2fs · total %.1fs" % (n, ttft or 0, dt))


if __name__ == "__main__":
    main()
