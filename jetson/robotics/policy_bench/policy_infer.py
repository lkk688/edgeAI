"""Load a pretrained LeRobot checkpoint and run real inference steps.

The observation is synthesised from the checkpoint's OWN input_features, so the
same script works for any policy (SmolVLA, FLUX 3, pi05, ...) without guessing
camera keys. Runs the exact path a robot rollout uses:

    prepare_observation_for_inference -> preprocessor -> select_action -> postprocessor

Reports weights-resident memory, peak memory and chunk-inference latency.
Chunked policies compute a whole chunk on the first select_action and then pop
from a queue, so each timed step calls policy.reset() to force a real forward.
"""
import argparse, time, sys
import numpy as np
import torch

def gb(x): return x / 1024 ** 3

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--task", default="pick up the cube and place it in the box")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--robot-type", default="so101_follower")
ap.add_argument("--compat", action="store_true",
                help="rewrite a stale hub config.json to the installed LeRobot's field names")
ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                help="override a config field, e.g. --set dtype='\"bfloat16\"'")
ap.add_argument("--dump", metavar="NPY",
                help="also save one full action chunk (fixed obs seed 0, noise seed --noise-seed) in joint units")
ap.add_argument("--noise-seed", type=int, default=0)
ap.add_argument("--hz", type=float, default=30.0, help="control rate the data was recorded at")
a = ap.parse_args()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

dev = torch.device("cuda")
print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
print(f"checkpoint {a.repo}\n")

# Checkpoints on the hub lag behind LeRobot main (fields get renamed). --compat
# builds a local snapshot whose config.json uses the installed field names;
# weights and processor files are symlinked, nothing on the hub is modified.
RENAMES = {"model_dtype": "dtype"}
def compat_snapshot(repo, overrides):
    import dataclasses, json, os
    from pathlib import Path
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(repo)) if not os.path.isdir(repo) else Path(repo)
    raw = json.loads((src / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(PreTrainedConfig.get_choice_class(raw["type"]))}
    fixed, dropped = {"type": raw["type"]}, []
    for k, v in raw.items():
        k2 = RENAMES.get(k, k)
        if k2 in known: fixed[k2] = v
        elif k != "type": dropped.append(k)
    # MolmoAct2: enable_lora_vlm (bool) became train_mode_vlm ('fft'|'lora'|'freeze').
    # Leaving it unset picks the new default 'lora', which wraps the VLM in PEFT and
    # no longer matches the saved state_dict keys.
    if "enable_lora_vlm" in raw and "train_mode_vlm" in known and "train_mode_vlm" not in raw:
        fixed["train_mode_vlm"] = "lora" if raw["enable_lora_vlm"] else "fft"
    fixed.update(overrides)
    dst = Path.home() / ".cache" / "policy_bench" / repo.replace("/", "--")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "config.json" and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    (dst / "config.json").write_text(json.dumps(fixed, indent=2))
    print(f"compat snapshot {dst}\n  renamed {[k for k in raw if k in RENAMES]}  dropped {dropped}"
          f"  overrides {overrides}")
    return str(dst)

import json as _json
overrides = {k: _json.loads(v) for k, v in (x.split("=", 1) for x in a.set)}
if a.compat or overrides:
    a.repo = compat_snapshot(a.repo, overrides)
cfg = PreTrainedConfig.from_pretrained(a.repo)
cls = get_policy_class(cfg.type)
print(f"policy type: {cfg.type}  ->  {cls.__name__}")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
policy = cls.from_pretrained(a.repo)
policy.to(dev).eval()
torch.cuda.synchronize()
print(f"loaded in {time.time()-t0:6.1f}s   weights resident {gb(torch.cuda.memory_allocated()):6.2f} GB")

# Some checkpoints were saved with device="cpu"; force the saved device step to CUDA.
policy.config.device = "cuda"
pre, post = make_pre_post_processors(policy.config, pretrained_path=a.repo,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})

feats = policy.config.input_features
print("input contract:")
for k, f in feats.items():
    print(f"   {k:<34} {tuple(f.shape)}")
print(f"output: {[(k, tuple(f.shape)) for k, f in policy.config.output_features.items()]}\n")

def make_obs(rng=np.random):
    obs = {}
    for k, f in feats.items():
        shp = tuple(f.shape)
        if "image" in k:
            c, h, w = shp
            obs[k] = rng.randint(0, 256, (h, w, c), dtype=np.uint8)
        else:
            obs[k] = np.zeros(shp, dtype=np.float32)
    return obs

torch.cuda.reset_peak_memory_stats()
lat = []
for i in range(a.steps):
    policy.reset()
    batch = prepare_observation_for_inference(make_obs(), dev, a.task, a.robot_type)
    batch = pre(batch)
    torch.cuda.synchronize(); t = time.time()
    with torch.inference_mode():
        act = policy.select_action(batch)
    torch.cuda.synchronize(); dt = time.time() - t
    act = post(act)
    lat.append(dt)
    shape = tuple(act.shape) if torch.is_tensor(act) else type(act).__name__
    print(f"  step {i}: chunk inference {dt*1000:8.1f} ms   action {shape}")

steady = lat[1:] or lat
print(f"\nfirst call (warm-up)   {lat[0]*1000:8.1f} ms")
print(f"steady chunk inference {np.mean(steady)*1000:8.1f} ms  (mean of {len(steady)})")
print(f"peak GPU memory        {gb(torch.cuda.max_memory_allocated()):8.2f} GB")

# --dump: one whole chunk for numerical comparisons (e.g. bf16 vs fp32). Same
# observation every time; the flow-matching noise is fixed by --noise-seed.
if a.dump:
    policy.reset()
    batch = pre(prepare_observation_for_inference(make_obs(np.random.RandomState(0)), dev, a.task, a.robot_type))
    torch.manual_seed(a.noise_seed); torch.cuda.manual_seed_all(a.noise_seed)
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    chunk = post(chunk).float().cpu().numpy()
    np.save(a.dump, chunk)
    print(f"dumped chunk {chunk.shape} -> {a.dump}")

# Real-time check: a chunk must be computed before the previous one finishes
# executing. With n_action_steps executed per chunk at --hz, the budget is
# n_action_steps / hz seconds (async/RTC execution can overlap, sync cannot).
cfg_p = policy.config
n_exec = getattr(cfg_p, "n_action_steps", None) or getattr(cfg_p, "chunk_size", None)
if n_exec:
    budget = n_exec / a.hz
    rtf = budget / np.mean(steady)
    print(f"chunk_size {getattr(cfg_p, 'chunk_size', '?')}, n_action_steps {n_exec}, "
          f"denoise steps {getattr(cfg_p, 'num_inference_steps', getattr(cfg_p, 'num_steps', '?'))}")
    print(f"budget at {a.hz:g} Hz       {budget*1000:8.1f} ms  ->  RTF {rtf:.2f}  "
          f"({'real-time' if rtf >= 1 else 'NOT real-time'})")
