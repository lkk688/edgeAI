"""Do the Orin optimisations change pi05-so100_101's actions? Checked on the real
checkpoint (needs the patched LeRobot venv, ~/Developer/lerobot-pi05so).

Same observation (2 real cameras, slots 2-3 padded), same flow noise; each variant
is compared with the as-loaded bf16 model. The scale is the model's own spread:
the baseline with a different noise seed.

  vision_bf16   SigLIP tower + projector in bf16 instead of fp32
  drop_pad      the two padded camera slots removed from the input entirely
  int8_trunk    Gemma-2B trunk weights rounded to int8 (per-output-channel scale),
                i.e. exactly the numbers int8 weight-only storage would give
  all           all three
"""
import argparse, copy
import numpy as np
import torch
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="hqfang/pi05-so100_101")
ap.add_argument("--task", default="pick up the cube and place it in the box")
a = ap.parse_args()

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PaliGemmaWithExpertModel
import lerobot.policies.pi05.processor_pi05  # noqa: F401  (registers the custom processor)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

root = snapshot_download(a.repo)
cfg = PI05Config.from_pretrained(root)
cfg.device, cfg.compile_model, cfg.dtype = "cuda", False, "bfloat16"
cfg.text_tokenizer_name = f"{root}/tokenizer/tokenizer.model"
policy = PI05Policy.from_pretrained(root, config=cfg).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=root,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})

rng = np.random.RandomState(0)
obs = {"observation.state": np.array([0, -90, 90, 60, 0, 20], dtype=np.float32)}
for i in range(4):
    img = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8) if i < 2 else np.zeros((224, 224, 3), np.uint8)
    obs[f"observation.images.camera_{i}"] = img
base_batch = pre(prepare_observation_for_inference(obs, torch.device("cuda"), a.task, "so101_follower"))
for i in range(4):
    base_batch[f"observation.images.camera_{i}_is_pad"] = torch.tensor([i >= 2], device="cuda")

orig_embed = PaliGemmaWithExpertModel.embed_image
orig_feats = copy.deepcopy(policy.config.input_features)
trunk_backup = None

def configure(vision_bf16=False, drop_pad=False, int8_trunk=False):
    global trunk_backup
    pwe = policy.model.paligemma_with_expert
    vis = [pwe.paligemma.model.vision_tower, pwe.paligemma.model.multi_modal_projector]
    for mod in vis:
        mod.to(torch.bfloat16 if vision_bf16 else torch.float32)
    if vision_bf16:
        def embed_image(self, image, **kw):
            out = self.paligemma.model.get_image_features(image.to(torch.bfloat16))
            out = getattr(out, "pooler_output", out)
            return out[0] if isinstance(out, (list, tuple)) else out
        PaliGemmaWithExpertModel.embed_image = embed_image
    else:
        PaliGemmaWithExpertModel.embed_image = orig_embed
    feats = copy.deepcopy(orig_feats)
    if drop_pad:
        for i in (2, 3):
            feats.pop(f"observation.images.camera_{i}", None)
    policy.config.input_features = feats
    lin = [m for n, m in pwe.paligemma.model.language_model.named_modules() if isinstance(m, torch.nn.Linear)]
    if int8_trunk and trunk_backup is None:
        trunk_backup = [m.weight.data.clone() for m in lin]
        for m in lin:
            w = m.weight.data.float()                       # [out, in]
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
            m.weight.data = ((w / s).round().clamp(-127, 127) * s).to(m.weight.dtype)
    elif not int8_trunk and trunk_backup is not None:
        for m, w in zip(lin, trunk_backup):
            m.weight.data = w
        trunk_backup = None

def chunk(seed=0, **opts):
    configure(**opts)
    batch = dict(base_batch)
    if opts.get("drop_pad"):
        for i in (2, 3):
            batch.pop(f"observation.images.camera_{i}", None)
            batch.pop(f"observation.images.camera_{i}_is_pad", None)
    policy.reset()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        out = post(policy.predict_action_chunk(batch))
    return out.float().cpu().numpy().reshape(-1, 6)

ref = chunk()
def row(name, x):
    d = np.abs(x - ref)
    print(f"{name:<22} mean {d.mean():6.3f} deg   max {d.max():6.3f} deg")
print(f"pi05-so100_101, chunk {ref.shape}, compared with the as-loaded bf16 model")
row("noise seed 1 (scale)", chunk(seed=1))
row("vision_bf16", chunk(vision_bf16=True))
row("drop_pad", chunk(drop_pad=True))
row("int8_trunk", chunk(int8_trunk=True))
row("all three", chunk(vision_bf16=True, drop_pad=True, int8_trunk=True))
