"""Where does FLUX 3 spend its chunk time on Thor, and does compiling help?

Times the VAE neighbourhood-attention layers (Natten3D.forward) separately from
the whole select_action, with NATTEN's flex-fna either eager (the default, which
materialises the full score matrix) or torch.compile'd. The compile switch is
applied by monkeypatching from here, so the LeRobot checkout keeps only the
two-line bugfix.
"""
import argparse, time, os
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true", help="torch.compile flex-fna")
ap.add_argument("--steps", type=int, default=4)
a = ap.parse_args()
os.environ.setdefault("F3_NATTEN_BACKEND", "flex-fna")

import lerobot.policies.flux3.f3.video_vae as vv
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

if a.compile:
    _orig = vv._natten_attention_kwargs
    def _with_compile(q, k, v):
        kw = _orig(q, k, v)
        if kw.get("backend") == "flex-fna":
            kw["torch_compile"] = True
        return kw
    vv._natten_attention_kwargs = _with_compile

acc = {"t": 0.0, "n": 0}
_fwd = vv.Natten3D.forward
def timed_forward(self, x):
    torch.cuda.synchronize(); t = time.time()
    out = _fwd(self, x)
    torch.cuda.synchronize(); acc["t"] += time.time() - t; acc["n"] += 1
    return out
vv.Natten3D.forward = timed_forward

REPO = "black-forest-labs/flux-3-action-so101"
cfg = PreTrainedConfig.from_pretrained(REPO)
policy = get_policy_class(cfg.type).from_pretrained(REPO).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=REPO)
feats = policy.config.input_features

def obs():
    o = {}
    for k, f in feats.items():
        s = tuple(f.shape)
        o[k] = (np.random.randint(0, 256, (s[1], s[2], s[0]), dtype=np.uint8)
                if "image" in k else np.zeros(s, dtype=np.float32))
    return o

print(f"mode: flex-fna {'COMPILED' if a.compile else 'eager'}")
rows = []
for i in range(a.steps):
    policy.reset(); acc["t"] = 0.0; acc["n"] = 0
    b = pre(prepare_observation_for_inference(obs(), torch.device("cuda"), "pick up the cube", "so101_follower"))
    torch.cuda.synchronize(); t = time.time()
    with torch.inference_mode():
        act = policy.select_action(b)
    torch.cuda.synchronize(); total = time.time() - t
    rows.append((total, acc["t"], acc["n"]))
    print(f"  step {i}: total {total*1000:7.0f} ms   VAE natten {acc['t']*1000:7.0f} ms "
          f"({100*acc['t']/total:4.1f}%)  over {acc['n']} calls")

st = rows[1:] or rows
tot = np.mean([r[0] for r in st]); vae = np.mean([r[1] for r in st])
print(f"\nsteady: total {tot*1000:.0f} ms, VAE natten {vae*1000:.0f} ms ({100*vae/tot:.1f}%), "
      f"rest {(tot-vae)*1000:.0f} ms")
print(f"peak GPU memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
