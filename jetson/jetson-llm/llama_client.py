#!/usr/bin/env python3
"""
llama_client.py — minimal OpenAI-compatible client for a served LLM.

Works against ANY OpenAI-compatible endpoint:
  * local Jetson llama.cpp   (sjsujetsontool llama  -> http://localhost:8080/v1)
  * our shared LLM gateway   (https://llm.forgengi.org/node05/v1)
  * NVIDIA Build API         (https://integrate.api.nvidia.com/v1)

Pure stdlib (urllib) — no pip packages required. Prints the reply and tokens/sec.

Examples:
  python3 llama_client.py -p "Explain what a Jetson Orin Nano is."
  python3 llama_client.py --url https://llm.forgengi.org/node05/v1 --api-key sjsugputool -p "Hi"
  python3 llama_client.py --no-stream -p "One sentence on edge AI."
"""
import argparse, json, time, sys, urllib.request, urllib.error


def main():
    ap = argparse.ArgumentParser(description="Query an OpenAI-compatible LLM server.")
    ap.add_argument("-p", "--prompt", default="Explain what an NVIDIA Jetson Orin Nano is, in 3 sentences.")
    ap.add_argument("--url", default="http://localhost:8080/v1", help="API base URL (…/v1)")
    ap.add_argument("--api-key", default="sk-no-key-required", help="bearer token if the server requires one")
    ap.add_argument("--model", default="local", help="model name (llama.cpp ignores it; required for NVIDIA)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="disable token streaming")
    ap.add_argument("--no-think", action="store_true",
                    help="tell llama.cpp reasoning models (Qwen3, etc.) to skip thinking")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }
    if args.no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + args.api_key}
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)

    print("→ %s  (model=%s)\n" % (endpoint, args.model))
    t0 = time.time()
    completion_tokens = 0
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        print("HTTP %s: %s" % (e.code, e.read().decode("utf-8", "ignore")[:300]))
        sys.exit(1)
    except urllib.error.URLError as e:
        print("Cannot connect: %s (is the server running?)" % e.reason)
        sys.exit(1)

    if not args.stream:
        obj = json.load(resp)
        print(obj["choices"][0]["message"]["content"].strip())
        completion_tokens = (obj.get("usage") or {}).get("completion_tokens", 0)
    else:
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
                completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
            for ch in obj.get("choices") or []:
                piece = ch.get("delta", {}).get("content")
                if piece:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
        print()

    dt = time.time() - t0
    if completion_tokens:
        print("\n⚡ %d tokens in %.1fs = %.1f tok/s" % (completion_tokens, dt, completion_tokens / dt))
    else:
        print("\n⏱️  %.1fs" % dt)


if __name__ == "__main__":
    main()
