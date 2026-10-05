"""Chunk latency of Tencent Hy-Embodied-0.5-VLA (own codebase, not LeRobot).

Setup (Thor, JetPack 7): clone github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA,
venv with torch 2.11+cu130, the Jetson AI Lab aarch64 flash-attn 2.8.4 wheel
(pypi.jetson-ai-lab.io/sbsa/cu130), transformers<5, timm==1.0.21, then
`pip install -e . --no-deps`. Mirrors scripts/quick_start.py with timing.

Hy-VLA predicts end-effector deltas ([xyz + rot6d + gripper] per arm), not joint
angles, so an SO-101 would also need IK; no SO-101 checkpoint exists.
"""
import argparse, time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from hy_vla import HyVLA, HyVLAConfig

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="tencent/Hy-Embodied-0.5-VLA-UMI")
ap.add_argument("--history", type=int, default=6, help="K frames per camera (slot K-1 = now)")
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--runs", type=int, default=6)
ap.add_argument("--hz", type=float, default=30.0)
a = ap.parse_args()

gb = lambda x: x / 1024 ** 3
ckpt = snapshot_download(a.repo)
config = HyVLAConfig.from_pretrained(ckpt)
t0 = time.time()
policy = HyVLA.from_pretrained(ckpt, config=config)
policy.enable_video_encoder_if_needed()
policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()
torch.cuda.synchronize()
print(f"{a.repo}: loaded in {time.time()-t0:.1f}s, resident {gb(torch.cuda.memory_allocated()):.2f} GB")
cams = list(config.image_features)
print(f"cameras {cams}, chunk {config.chunk_size}, n_action_steps {config.n_action_steps}, "
      f"flow steps {config.num_steps}")

torch.cuda.reset_peak_memory_stats()
lat = []
for i in range(a.runs):
    img = torch.rand(1, a.history, 3, a.res, a.res, device="cuda", dtype=torch.bfloat16)
    batch = {k: img for k in cams}
    batch["observation.state"] = torch.zeros((1, config.max_state_dim), device="cuda", dtype=torch.bfloat16)
    batch["task"] = ["pick up the cube and place it in the box"]
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        act = policy.forward_evaluate(batch)["pred"]
    torch.cuda.synchronize(); dt = time.time() - t
    lat.append(dt)
    print(f"  run {i}: {dt*1000:8.1f} ms   pred {tuple(act.shape)}")

steady = np.mean(lat[1:] or lat)
budget = config.n_action_steps / a.hz
print(f"steady {steady*1000:.1f} ms   peak {gb(torch.cuda.max_memory_allocated()):.2f} GB")
print(f"budget {config.n_action_steps} actions @ {a.hz:g} Hz = {budget*1000:.0f} ms -> RTF {budget/steady:.2f} "
      f"({'real-time' if budget >= steady else 'NOT real-time'})")
