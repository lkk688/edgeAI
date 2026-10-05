"""Estimate full-size pi05 latency on a device too small to hold it.

pi05 = SigLIP (fixed) + PaliGemma Gemma-2B (18 layers) + Gemma-300M action expert
(18 layers, run once per denoising step). Latency is linear in depth:

    T(L) = T_fixed + L * T_layer

This builds the REAL LeRobot pi05 network (random weights, so no checkpoint has
to fit) with L in --depths, times `sample_actions`, fits the line, and
extrapolates to L=18. Run it on a big device too and compare with a real
checkpoint there to check the extrapolation (see SO101_TRAINING.md).

    python pi05_layer_scaling.py --depths 2 4 6 --cams 4 --steps 10
"""
import argparse, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6])
ap.add_argument("--cams", type=int, default=4, help="image slots fed to the VLM (pi05-so100_101 uses 4)")
ap.add_argument("--steps", type=int, default=10, help="flow denoising steps")
ap.add_argument("--lang", type=int, default=200, help="tokenizer_max_length (padded prompt length)")
ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16"], help="build dtype (pi05 runs bf16)")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--cuda-graph", action="store_true",
                help="capture the whole sample_actions in a torch.cuda.CUDAGraph and replay it "
                     "(no Triton needed, works on Orin); removes per-kernel launch overhead")
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--full", type=int, default=18)
ap.add_argument("--fit-only", nargs="+", metavar="L=MS", help="skip measuring; fit these points")
ap.add_argument("--hz", type=float, default=30.0)
ap.add_argument("--vision", default="fp32", choices=["fp32", "bf16"],
                help="fp32: whole SigLIP tower + projector in fp32, as LeRobot b6ec006 / "
                     "pi05-so100_101 load it. bf16: the whole tower in bf16 (0.4.4 keeps only the "
                     "tiny patch embedding fp32). Set here so every LeRobot version computes the same thing")
ap.add_argument("--small-vocab", action="store_true",
                help="shrink the 257k-entry token embedding and the (unused) lm_head to 16k rows. "
                     "Each is a single 1.05 GB allocation that NvMap refuses on an 8 GB Orin; "
                     "neither costs compute in sample_actions (lookup / not called)")
ap.add_argument("--chunk", type=int, default=50)
a = ap.parse_args()

import lerobot.policies.pi05.modeling_pi05 as m
if a.small_vocab:
    _Emb, _Lin = torch.nn.Embedding.__init__, torch.nn.Linear.__init__
    def emb_init(self, num, dim, *args, **kw): _Emb(self, min(num, 16384), dim, *args, **kw)
    def lin_init(self, i, o, *args, **kw): _Lin(self, i, min(o, 16384) if o > 100_000 else o, *args, **kw)
    torch.nn.Embedding.__init__, torch.nn.Linear.__init__ = emb_init, lin_init
from lerobot.policies.pi05.configuration_pi05 import PI05Config

if a.vision == "bf16":
    # LeRobot 0.4.4's embed_image: SigLIP + projector straight through, in bf16.
    # Newer versions cast the image to fp32 first, which a bf16 tower rejects.
    def embed_image(self, image, **kw):
        out = self.paligemma.model.get_image_features(image.to(torch.bfloat16))
        out = getattr(out, "pooler_output", out)
        return out[0] if isinstance(out, (list, tuple)) else out
    m.PaliGemmaWithExpertModel.embed_image = embed_image

orig = m.get_gemma_config
dt = getattr(torch, a.dtype)
dev = torch.device("cuda")
print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}  {a.dtype}  cams {a.cams}  "
      f"steps {a.steps}  lang {a.lang}  compile {a.compile}  small-vocab {a.small_vocab}  vision {a.vision}  cuda-graph {a.cuda_graph}")

def build(depth):
    def patched(variant):
        g = orig(variant); g.depth = depth; return g
    m.get_gemma_config = patched
    cfg = PI05Config(dtype="bfloat16", compile_model=False, num_inference_steps=a.steps,
                     chunk_size=a.chunk, n_action_steps=a.chunk, tokenizer_max_length=a.lang)
    # Build in half precision (Orin's RAM is shared with the GPU and cannot hold
    # an fp32 build), then restore the checkpoint's own mixed precision: the
    # projections outside PaliGemma stay fp32, and PaliGemma's own
    # to_bfloat16_for_selected_params decides which of its parts stay fp32.
    torch.set_default_dtype(dt)
    with torch.device(dev):
        model = m.PI05Pytorch(cfg)
    torch.set_default_dtype(torch.float32)
    for name, child in model.named_children():
        if name != "paligemma_with_expert":
            child.float()
    keep32 = ["input_layernorm", "post_attention_layernorm", "model.norm"] + (
        ["vision_tower", "multi_modal_projector"] if a.vision == "fp32" else [])
    for name, p in model.paligemma_with_expert.named_parameters():
        p.data = p.data.to(torch.float32 if any(k in name for k in keep32) else torch.bfloat16)
    return model.eval(), cfg

results = {int(k): float(v) / 1000 for k, v in (x.split("=") for x in (a.fit_only or []))}
for L in ([] if a.fit_only else a.depths):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, cfg = build(L)
    fn = torch.compile(model.sample_actions) if a.compile else model.sample_actions
    imgs = [torch.rand(1, 3, 224, 224, device=dev) * 2 - 1 for _ in range(a.cams)]
    masks = [torch.ones(1, dtype=torch.bool, device=dev) for _ in range(a.cams)]
    tok = torch.randint(0, 1000, (1, a.lang), device=dev)
    tmask = torch.zeros(1, a.lang, dtype=torch.bool, device=dev); tmask[:, :min(60, a.lang)] = True
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=dev)
    if a.cuda_graph:
        # pi05 builds constant attention masks with torch.tensor(list, device=cuda): a
        # host->device copy, which CUDA forbids during graph capture. Their values depend
        # only on input shapes, so cache them during warm-up and hand the cached GPU
        # tensors back during capture (the graph then reads them as constants).
        _tensor, cache = torch.tensor, {}
        def cached_tensor(data, *args, device=None, **kw):
            if device is None or torch.device(device).type != "cuda":
                return _tensor(data, *args, device=device, **kw)
            key = (repr(data), repr(args), repr(sorted(kw.items())), str(device))
            if key not in cache:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("uncached torch.tensor during capture")
                cache[key] = _tensor(data, *args, device=device, **kw)
            return cache[key]
        torch.tensor = cached_tensor
        # Inputs live in static buffers; a real rollout copies each new observation
        # into them and calls g.replay().
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(2):
                model.sample_actions(imgs, masks, tok, tmask, noise=noise, num_steps=a.steps)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            static_out = model.sample_actions(imgs, masks, tok, tmask, noise=noise, num_steps=a.steps)
        torch.tensor = _tensor
        with torch.inference_mode():
            ref = model.sample_actions(imgs, masks, tok, tmask, noise=noise, num_steps=a.steps)
        g.replay(); torch.cuda.synchronize()
        print(f"  graph vs eager max |diff| {(static_out.float() - ref.float()).abs().max().item():.2e}")
        fn = lambda *args, **kw: g.replay()
    lat = []
    for i in range(a.runs + 1):
        torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            fn(imgs, masks, tok, tmask, noise=noise, num_steps=a.steps)
        torch.cuda.synchronize(); lat.append(time.time() - t)
    results[L] = float(np.median(lat[1:]))
    print(f"  depth {L:2d}: {results[L]*1000:8.1f} ms   peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    del model, fn

Ls = np.array(sorted(results)); Ts = np.array([results[L] for L in Ls])
b, c = np.polyfit(Ls, Ts, 1)
full = c + b * a.full
budget = a.chunk / a.hz
print(f"fit: fixed {c*1000:.1f} ms + {b*1000:.1f} ms/layer  (residual max "
      f"{np.abs(np.polyval([b, c], Ls) - Ts).max()*1000:.1f} ms)")
print(f"extrapolated depth {a.full}: {full*1000:.0f} ms   budget {a.chunk} @ {a.hz:g} Hz = "
      f"{budget*1000:.0f} ms -> RTF {budget/full:.2f}")
