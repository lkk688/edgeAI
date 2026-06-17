#!/usr/bin/env python3
"""
vision_test.py — send an image to a multimodal llama.cpp server (OpenAI vision format).

Works with any vision-capable OpenAI-compatible endpoint:
  * Jetson Gemma 4 E2B   (sjsujetsontool llama -> http://localhost:8080/v1, auto mmproj)
  * Qwen3.5 on our server (gputool serve-llamacpp --mmproj … -> llm.forgengi.org/node05/v1)

Pass --image <file> to use your own picture; otherwise a small test image
(a yellow circle on a blue square) is generated with Pillow.

Examples:
  python3 vision_test.py --image cat.jpg -p "What is in this photo?"
  python3 vision_test.py --url https://llm.forgengi.org/node05/v1 --api-key sjsugputool
"""
import argparse, base64, io, json, sys, urllib.request, urllib.error


def load_image_b64(path):
    if path:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/" + ext
        return mime, base64.b64encode(data).decode()
    # No image given: synthesize one (needs Pillow).
    try:
        from PIL import Image, ImageDraw
    except Exception:
        sys.exit("No --image given and Pillow not installed. Pass --image <file>.")
    img = Image.new("RGB", (200, 200), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 180, 180], fill="blue")
    d.ellipse([60, 60, 140, 140], fill="yellow")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "image/png", base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(description="Send an image to a vision LLM server.")
    ap.add_argument("--image", default=None, help="path to an image; omit to generate a test image")
    ap.add_argument("-p", "--prompt", default="Describe this image in one short sentence.")
    ap.add_argument("--url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="sk-no-key-required")
    ap.add_argument("--model", default="local")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    mime, b64 = load_image_b64(args.image)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}},
        ]}],
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        args.url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + args.api_key})
    print("→ %s  (image: %s)\n" % (args.url, args.image or "generated test image"))
    try:
        d = json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        if "mmproj" in body.lower() or e.code == 500:
            print("Server has no vision model loaded. Start it with an mmproj:")
            print("  gputool serve-llamacpp start <model.gguf> 8080 --mmproj mmproj-F16.gguf")
        sys.exit("HTTP %s: %s" % (e.code, body[:300]))
    print("VISION REPLY:", d["choices"][0]["message"]["content"].strip())


if __name__ == "__main__":
    main()
