"""Cosmos3-Edge-Policy-DROID through vLLM-Omni's /v1/videos API, timed.

Mirrors NVIDIA/cosmos cookbooks/cosmos3/generator/action/run_policy_with_vllm_omni.ipynb:
compose the DROID multiview frame (640x360 wrist on top, two 320x180 exterior views
below), send it with the Edge checkpoint's structured JSON prompt, poll the async job,
and read back the [chunk, 8] joint-position action chunk. Needs requests + Pillow + av
(the LeRobot venv has them).

    python cosmos3_policy_omni_client.py --assets <cosmos>/cookbooks/cosmos3/generator/action/assets/droid_lerobot_example \
        --steps 30 4 --runs 3
"""
import argparse, io, json, time
from pathlib import Path
import av, numpy as np, requests
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8001")
ap.add_argument("--assets", required=True)
ap.add_argument("--prompt", default="Pick up the object and place it in the target container.")
ap.add_argument("--steps", type=int, nargs="+", default=[30])
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--out", default="cosmos3_policy_out")
ap.add_argument("--size", default=None, help="override the generation size WxH, e.g. 320x192 (model card real-time row)")
ap.add_argument("--tier", type=int, default=480, help="extra_params image_size (480 or 256)")
a = ap.parse_args()

def first_frame(p):
    with av.open(str(p)) as c:
        return next(c.decode(video=0)).to_image().convert("RGB")
root = Path(a.assets).expanduser() / "videos"
wrist = first_frame(root / "observation.image.wrist_image_left/chunk-000/file-000.mp4")
ext1 = first_frame(root / "observation.image.exterior_image_1_left/chunk-000/file-000.mp4")
ext2 = first_frame(root / "observation.image.exterior_image_2_left/chunk-000/file-000.mp4")
W, H = wrist.size; bh, hw = H // 2, W // 2
img = Image.new("RGB", (W, H + bh)); img.paste(wrist, (0, 0))
img.paste(ext1.resize((hw, bh), Image.Resampling.BILINEAR), (0, H))
img.paste(ext2.resize((hw, bh), Image.Resampling.BILINEAR), (hw, H))
out = Path(a.out); out.mkdir(exist_ok=True); img.save(out / "policy_input.png")

SIZES = {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)}
ratio, (tw, th) = min(SIZES.items(), key=lambda kv: abs(img.height / img.width - kv[1][1] / kv[1][0]))
if a.size:
    tw, th = map(int, a.size.split("x"))
frames = a.chunk + 1
prompt = json.dumps({
    "cinematography": {"framing": "This video contains concatenated views from multiple camera perspectives. "
                                  "The top row is the wrist camera and the bottom row contains two external cameras."},
    "actions": [{"time": f"0:00-0:{round(frames / a.fps):02d}", "description": a.prompt.rstrip(".!?") + "."}],
    "duration": f"{int(frames / a.fps)}s", "fps": float(a.fps), "resolution": {"H": th, "W": tw}, "aspect_ratio": ratio,
}, separators=(",", ":"))
model = requests.get(f"{a.url}/v1/models", timeout=10).json()["data"][0]["id"]
print(f"model {model}  input {img.size} -> {tw}x{th}  chunk {a.chunk}  frames {frames}")

buf = io.BytesIO(); img.save(buf, format="PNG")
for steps in a.steps:
    lat = []
    for r in range(a.runs):
        form = {"prompt": prompt, "num_frames": frames, "fps": a.fps, "size": f"{tw}x{th}",
                "num_inference_steps": steps, "guidance_scale": 1.0, "flow_shift": 5.0, "seed": r,
                "extra_params": json.dumps({"action_mode": "policy", "domain_name": "droid_lerobot",
                                            "raw_action_dim": 8, "action_chunk_size": a.chunk,
                                            "image_size": a.tier, "guardrails": False})}
        t0 = time.time()
        job = requests.post(f"{a.url}/v1/videos", data={k: str(v) for k, v in form.items()},
                            files={"input_reference": ("policy_input.png", buf.getvalue(), "image/png")}, timeout=120)
        job.raise_for_status(); jid = job.json()["id"]
        while True:
            st = requests.get(f"{a.url}/v1/videos/{jid}", timeout=30).json()
            if st.get("status") in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)
        dt = time.time() - t0
        if st["status"] != "completed":
            raise SystemExit(json.dumps(st, indent=2)[:2000])
        act = np.asarray(st["action"]["data"], dtype=np.float32)
        lat.append(dt)
        print(f"  steps {steps:2d} run {r}: {dt*1000:8.0f} ms  action {act.shape}  finite {np.isfinite(act).all()}")
    (out / f"action_steps{steps}.json").write_text(json.dumps(st["action"], indent=1))
    steady = np.mean(lat[1:] or lat)
    print(f"steps {steps}: steady {steady*1000:.0f} ms per [{a.chunk},8] chunk "
          f"-> budget at {a.fps} Hz {a.chunk/a.fps*1000:.0f} ms, RTF {a.chunk/a.fps/steady:.2f}")
video = requests.get(f"{a.url}/v1/videos/{jid}/content", timeout=300)
if video.content:
    (out / "policy_rollout.mp4").write_bytes(video.content); print("saved", out / "policy_rollout.mp4")
