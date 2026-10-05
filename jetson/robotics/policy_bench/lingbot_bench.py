"""Chunk latency of LingBot-VLA (Robbyant, own codebase on LeRobot 0.4.2).

Uses the repo's own deploy path (deploy.lingbot_vla_policy.LingbotVLAServer):
FeatureTransform -> Qwen2.5-VL processor -> sample_actions -> unnormalize, i.e.
exactly what its websocket server runs per request. Must run from the
lingbot-vla checkout (configs/ and assets/ are relative paths).

Setup (Thor): venv with torch 2.11+cu130, lerobot 0.4.2 (torch overridden),
transformers 4.51.3, the Jetson AI Lab flash-attn wheel, `pip install -e . --no-deps`,
QWEN25_PATH -> Qwen/Qwen2.5-VL-3B-Instruct.

--v2 runs LingBot-VLA 2.0 (github.com/robbyant/lingbot-vla-v2, Qwen3-VL-4B + MoE
action expert, ~6B) from that checkout instead; same env recipe with
transformers 4.57.3 and QWEN3VL_PATH -> Qwen/Qwen3-VL-4B-Instruct. Its README
notes the release was validated in FP32 and BF16 can change success rates.

The released post-trained checkpoint is RoboTwin (dual arm, 14-D joints,
3 cameras); an SO-101 run needs a fine-tune, but the compute is the same.
"""
import argparse, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--v2", action="store_true", help="LingBot-VLA 2.0 (run from the lingbot-vla-v2 checkout)")
ap.add_argument("--model", default=None)
ap.add_argument("--fp32", action="store_true", help="fp32 instead of bf16")
ap.add_argument("--robot", default="robotwin")
ap.add_argument("--denoise", type=int, default=10)
ap.add_argument("--use-length", type=int, default=25, help="actions executed per chunk (repo's real-robot default)")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--runs", type=int, default=6)
ap.add_argument("--hz", type=float, default=30.0)
a = ap.parse_args()

sys.path.insert(0, os.getcwd())
from huggingface_hub import snapshot_download
gb = lambda x: x / 1024 ** 3
t0 = time.time()
if a.v2:
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    a.model = a.model or "robbyant/lingbot-vla-v2-6b-robotwin"
    path = os.path.join(snapshot_download(a.model), "checkpoints/global_step_50000/hf_ckpt")
    os.environ.setdefault("QWEN3VL_PATH", snapshot_download("Qwen/Qwen3-VL-4B-Instruct"))
    srv = LingbotVLAv2Server(path, use_length=a.use_length, chunk_ret=True, use_bf16=not a.fp32,
                             use_fp32=a.fp32, use_compile=a.compile)
else:
    from deploy.lingbot_vla_policy import LingbotVLAServer
    a.model = a.model or "robbyant/lingbot-vla-4b-posttrain-robotwin"
    path = snapshot_download(a.model)
    os.environ.setdefault("QWEN25_PATH", snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct"))
    srv = LingbotVLAServer(path, use_length=a.use_length, num_denoising_step=a.denoise,
                           use_compile=a.compile, use_bf16=not a.fp32, use_fp32=a.fp32)
srv.reset(robo_name=a.robot)
torch.cuda.synchronize()
print(f"{a.model}: loaded in {time.time()-t0:.1f}s, resident {gb(torch.cuda.memory_allocated()):.2f} GB")

cams = [list(d.values())[0]["origin_keys"] for d in srv.vla.feature_transform.robot_config["images"]] \
    if hasattr(srv.vla.feature_transform, "robot_config") else \
    ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
torch.cuda.reset_peak_memory_stats()
lat = []
for i in range(a.runs):
    obs = {c: np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for c in cams}
    obs["observation.state"] = np.zeros(14, dtype=np.float32)
    obs["task"] = obs["prompt"] = "pick up the cube and place it in the box"
    torch.cuda.synchronize(); t = time.time()
    out = srv.infer(obs)
    torch.cuda.synchronize(); dt = time.time() - t
    lat.append(dt)
    print(f"  run {i}: {dt*1000:8.1f} ms   " + ", ".join(f"{k} {getattr(v, 'shape', type(v).__name__)}"
                                                    for k, v in out.items()))

steady = np.mean(lat[1:] or lat)
budget = a.use_length / a.hz
print(f"{'v2' if a.v2 else 'v1'} {'fp32' if a.fp32 else 'bf16'}, denoise {a.denoise if not a.v2 else 'cfg'}, compile {a.compile}, chunk {srv.config.chunk_size}, executed {a.use_length}")
print(f"steady {steady*1000:.1f} ms   peak {gb(torch.cuda.max_memory_allocated()):.2f} GB")
print(f"budget {a.use_length} actions @ {a.hz:g} Hz = {budget*1000:.0f} ms -> RTF {budget/steady:.2f} "
      f"({'real-time' if budget >= steady else 'NOT real-time'})")
