# policy_bench

Scripts behind the measured numbers in `../SO101_TRAINING.md`. Each prints the
steady chunk latency, peak memory, and a real-time verdict (RTF, the time one
chunk lasts at `--hz` divided by the time it takes to compute).

On Jetson Thor, set these first for anything that compiles Triton kernels
(`torch.compile`, flash-attn rotary). Triton's bundled ptxas does not know sm_110:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
```

| script | environment on Thor | what it does |
|---|---|---|
| `policy_probe.py` | `~/Developer/lerobot/.venv` | Imports every LeRobot policy and builds its default config. |
| `policy_infer.py <repo>` | `~/Developer/lerobot/.venv` (or `lerobot-pi05so`) | Loads a LeRobot checkpoint and runs the rollout path (`prepare_observation_for_inference` -> preprocessor -> `select_action` -> postprocessor) on observations built from the checkpoint's own input contract. `--compat` rewrites a stale hub `config.json` to the installed field names (local snapshot, weights symlinked). `--set KEY=JSON` overrides config fields. `--dump f.npy --noise-seed N` saves one full chunk for numerical comparisons. |
| `compare_chunks.py a b [c]` | any | Per-joint difference between dumped chunks, against a seed-to-seed baseline `c`. |
| `flux3_profile.py` | `~/Developer/lerobot/.venv` | FLUX 3 only: VAE NATTEN time vs whole chunk. |
| `cosmos3_bench.py` | `~/Developer/vla/cosmos_bench/.venv` (diffusers main) | Cosmos3 policy through `Cosmos3OmniPipeline`. `--latent` skips decoding the co-generated video, `--profile` splits time by component. |
| `hyvla_bench.py` | `~/Developer/vla/Hy-Embodied-0.5-VLA/.venv` | Hy-Embodied-0.5-VLA `forward_evaluate`, as in its `quick_start.py`. |
| `pi05_layer_scaling.py` | any LeRobot with pi05 (Orin: `~/lerobot-py310-cuda`) | Real pi05 network with L of 18 layers, random weights; fits latency vs L and extrapolates. For devices that cannot hold the model. `--vision {fp32,bf16} --cams N --cuda-graph --small-vocab`. |
| `pi05_opt_check.py` | `~/Developer/lerobot-pi05so` | On the real pi05-so100_101 checkpoint: action change from bf16 vision, dropping padded camera slots, int8 trunk. |
| `int8_mm_bench.py`, `int8_dequant_bench.py` | any CUDA torch | W8A8 `torch._int_mm` vs bf16 GEMM, and the cost of weight-only int8 dequant, at pi05's shapes. |
| `lingbot_bench.py` | run **from** `~/Developer/vla/lingbot-vla` (or `lingbot-vla-v2` with `--v2`) | LingBot-VLA through its own deploy class. `--compile`, `--fp32`. |

```bash
# LeRobot policies (main venv)
python policy_infer.py lerobot/smolvla_base
python policy_infer.py lerobot/MolmoAct2-SO100_101-LeRobot --compat --set dtype='"bfloat16"'

# pi05-so100_101 (patched venv ~/Developer/lerobot-pi05so)
P=$(ls -d ~/.cache/huggingface/hub/models--hqfang--pi05-so100_101/snapshots/*/)
python policy_infer.py hqfang/pi05-so100_101 --set text_tokenizer_name="\"${P}tokenizer/tokenizer.model\"" \
  --set dtype='"bfloat16"' --set compile_model=true

# bf16 vs fp32 on the same observation and noise
python policy_infer.py <repo> --set dtype='"float32"'  --steps 1 --dump f32_s0.npy
python policy_infer.py <repo> --set dtype='"bfloat16"' --steps 1 --dump b16_s0.npy
python policy_infer.py <repo> --set dtype='"float32"'  --steps 1 --dump f32_s1.npy --noise-seed 1
python compare_chunks.py f32_s0.npy b16_s0.npy f32_s1.npy

# Non-LeRobot models, each in its own venv
python cosmos3_bench.py --steps 4 --guidance 1.0 --latent --profile
python hyvla_bench.py
cd ~/Developer/vla/lingbot-vla    && python ~/Developer/robotics/policy_bench/lingbot_bench.py --compile
cd ~/Developer/vla/lingbot-vla-v2 && python ~/Developer/robotics/policy_bench/lingbot_bench.py --v2 --compile
```

## Environments on Thor

All use the stock PyTorch `torch==2.11.0 torchvision==0.26.0` from
`https://download.pytorch.org/whl/cu130`. Each non-LeRobot repo pins an older
torch; that pin is overridden, not followed.

| venv | extra pieces |
|---|---|
| `~/Developer/lerobot` | LeRobot main; `natten 0.21.6+torch2110cu130` for FLUX 3 (needs `F3_NATTEN_BACKEND=flex-fna` + the `video_vae.py` fix) |
| `~/Developer/lerobot-pi05so` | LeRobot at `b6ec006` + `hqfang/pi05-so100_101`'s `code/lerobot.patch`, `pip install -e '.[pi]'` |
| `~/Developer/vla/cosmos_bench` | `diffusers @ git+https://github.com/huggingface/diffusers`, transformers 5.x |
| `~/Developer/vla/Hy-Embodied-0.5-VLA` | flash-attn 2.8.4 aarch64 wheel from `pypi.jetson-ai-lab.io/sbsa/cu130`, `transformers<5`, `timm==1.0.21`, `-e . --no-deps` |
| `~/Developer/vla/lingbot-vla` | lerobot 0.4.2 and transformers 4.51.3 (torch overridden), same flash-attn wheel, submodules, `ipdb`, `torchdata` |
| `~/Developer/vla/lingbot-vla-v2` | its `requirements.txt` minus the torch pins, lerobot 0.4.2 `--no-deps`, same flash-attn wheel |

The flash-attn wheel on the Jetson AI Lab index works with torch 2.11 on Thor.
It was checked with a real `flash_attn_func` call against SDPA: max error
0.008 in bf16.

## Known failures kept on purpose

- `flux3_profile.py --compile` fails: NATTEN refuses to compile Flex Attention
  for correctness reasons.
- `cosmos3_bench.py --cudnn-benchmark` changes nothing. The 2.1 s VAE encode is
  the pipeline repeat-padding one frame to 33 and encoding all of them, not a
  bad convolution algorithm.
