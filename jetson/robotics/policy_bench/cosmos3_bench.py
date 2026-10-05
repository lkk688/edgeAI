"""Latency of a Cosmos3 policy (world action model) through the official diffusers pipeline.

Cosmos3 is not in LeRobot yet (PR #3745 is a design draft), and the community
SO-101 conversion (geonmin-kim/Cosmos3-Edge-Policy-SO101-init) needs a private
LeRobot fork. That checkpoint's weights are bit-identical to
nvidia/Cosmos3-Edge-Policy-DROID, so the DROID release measures the same compute.

Each call generates chunk_size actions AND chunk_size+1 future video latents
(the model always co-generates video). --latent skips the VAE decode of that
video, which a robot does not need. Needs diffusers from git main.

    python cosmos3_bench.py --steps 4 --guidance 3.0 --fps 15   # NVIDIA's Thor setting
    python cosmos3_bench.py --steps 4 --guidance 1.0 --latent    # deployment-style
"""
import argparse, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/Cosmos3-Edge-Policy-DROID")
ap.add_argument("--domain", default="droid_lerobot")
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--steps", type=int, default=4, help="UniPC denoising steps")
ap.add_argument("--guidance", type=float, default=1.0, help=">1 runs CFG (2 passes per step)")
ap.add_argument("--fps", type=float, default=15)
ap.add_argument("--tier", type=int, default=480)
ap.add_argument("--size", default="640x540", help="conditioning canvas WxH")
ap.add_argument("--latent", action="store_true", help="skip decoding the co-generated video")
ap.add_argument("--runs", type=int, default=4)
ap.add_argument("--hz", type=float, default=30.0, help="robot control rate for the RTF column")
ap.add_argument("--profile", action="store_true", help="time each pipeline component")
ap.add_argument("--cudnn-benchmark", action="store_true",
                help="let cuDNN autotune the VAE's 3D convolutions (same math, faster kernels)")
a = ap.parse_args()

from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

torch.backends.cudnn.benchmark = a.cudnn_benchmark
gb = lambda x: x / 1024 ** 3
t0 = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                           safety_checker=None, enable_safety_checker=False)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0,
                                                     use_karras_sigmas=False)
torch.cuda.synchronize()
print(f"{a.model}: loaded in {time.time()-t0:.1f}s, resident {gb(torch.cuda.memory_allocated()):.2f} GB")

# --profile: wrap every top-level component (and the VAE's encode/decode) with a
# CUDA-synchronised timer so the per-call total can be split by component.
from collections import defaultdict
timers = defaultdict(float)
def timed(name, fn):
    def wrap(*args, **kw):
        torch.cuda.synchronize(); t = time.time()
        r = fn(*args, **kw)
        torch.cuda.synchronize(); timers[name] += time.time() - t
        return r
    return wrap
if a.profile:
    for name, comp in pipe.components.items():
        if isinstance(comp, torch.nn.Module):
            if hasattr(comp, "encode") and hasattr(comp, "decode"):
                comp.encode = timed(f"{name}.encode", comp.encode)
                comp.decode = timed(f"{name}.decode", comp.decode)
            else:
                comp.forward = timed(name, comp.forward)
    print("components:", [n for n, c in pipe.components.items() if isinstance(c, torch.nn.Module)])

w, h = map(int, a.size.split("x"))
img = Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
torch.cuda.reset_peak_memory_stats()
lat = []
for i in range(a.runs):
    g = torch.Generator(device="cuda").manual_seed(i)
    torch.cuda.synchronize(); t = time.time()
    out = pipe(prompt="pick up the cube and place it in the box",
               action=CosmosActionCondition(mode="policy", chunk_size=a.chunk, domain_name=a.domain,
                                            resolution_tier=a.tier, image=img, view_point="concat_view"),
               fps=a.fps, num_inference_steps=a.steps, guidance_scale=a.guidance,
               use_system_prompt=False, generator=g, enable_safety_check=False,
               output_type="latent" if a.latent else "pil")
    torch.cuda.synchronize(); dt = time.time() - t
    lat.append(dt)
    if a.profile:
        parts = "  ".join(f"{k} {v*1000:.0f}" for k, v in sorted(timers.items(), key=lambda x: -x[1]))
        print(f"         ms by component: {parts}  other {(dt-sum(timers.values()))*1000:.0f}")
        timers.clear()
    shp = tuple(out.action[0].shape) if out.action is not None else None
    print(f"  run {i}: {dt*1000:8.1f} ms   action {shp}")

steady = np.mean(lat[1:] or lat)
budget = a.chunk / a.hz
print(f"config: chunk {a.chunk}, {a.steps} steps, guidance {a.guidance}, fps {a.fps:g}, "
      f"{a.size} tier {a.tier}, {'latent (no video decode)' if a.latent else 'decoded video'}")
print(f"steady {steady*1000:.1f} ms   peak {gb(torch.cuda.max_memory_allocated()):.2f} GB")
print(f"budget {a.chunk} actions @ {a.hz:g} Hz = {budget*1000:.0f} ms -> RTF {budget/steady:.2f} "
      f"({'real-time' if budget >= steady else 'NOT real-time'});  @ 15 Hz RTF {a.chunk/15/steady:.2f}")
