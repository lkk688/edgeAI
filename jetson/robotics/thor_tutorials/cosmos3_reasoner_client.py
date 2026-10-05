"""Query a Cosmos3-Edge reasoner served by vLLM (OpenAI API) and time it.

Uses the model card's own example (assets/example_reasoning_input.png + prompt).
Streams the answer so time-to-first-token and decode speed can be separated.
Standard library only, so it runs on the host or inside any container.

    python cosmos3_reasoner_client.py --assets ~/Developer/models/Cosmos3-Edge-assets --runs 3
"""
import argparse, base64, json, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/v1")
ap.add_argument("--assets", required=True, help="dir with example_reasoning_input.png / example_reasoning_prompt.json")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--runs", type=int, default=3)
a = ap.parse_args()

assets = Path(a.assets).expanduser()
prompt_file = assets / "example_reasoning_prompt.json"
prompt = json.loads(prompt_file.read_text()) if prompt_file.exists() else {}
text = prompt.get("prompt") or prompt.get("text") or (
    "The task is to put flower into the red bottle. Generate a plan consisting of subtasks for accomplish the task.")
img = "data:image/png;base64," + base64.b64encode((assets / "example_reasoning_input.png").read_bytes()).decode()

model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
print(f"model {model}\nprompt {text[:100]}")
for r in range(a.runs):
    body = {"model": model, "stream": True, "max_tokens": a.max_tokens, "temperature": 0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                                      {"type": "text", "text": text}]}]}
    req = urllib.request.Request(f"{a.url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = []; usage = {}
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                out.append(piece)
    total = time.time() - t0
    n = usage.get("completion_tokens", 0)
    rate = (n - 1) / (total - ttft) if n > 1 and ttft else 0
    print(f"run {r}: prompt {usage.get('prompt_tokens')} tok, TTFT {ttft*1000:.0f} ms, "
          f"{n} tok in {total:.2f} s, decode {rate:.1f} tok/s")
print("\n--- answer (last run) ---\n" + "".join(out)[:1500])
