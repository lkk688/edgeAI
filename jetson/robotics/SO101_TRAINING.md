# SO-ARM101 Policy Training: Choosing and Training a Policy

Last updated: 2026-09-30

Companion to `SO101_TELEOP_TUTORIAL.md`, which covers bring-up, teleoperation
and recording. This document covers what to do with the data afterwards:
which pretrained policies actually run on an SO-101, what each one demands, and
how to train them on a remote GPU.

Collection and QA run on the Jetson; training is submitted to a GPU box with
`lerobot_datapipeline.py`.

## Which foundation models run on an SO-101

Every SO-101 tutorial starts with ACT, which trains from scratch and cannot
benefit from pretraining. That is no longer the only option — several pretrained
policies accept the SO-101's 6-DoF joint action space directly.

Surveyed from the Hub (`filter=so101&pipeline_tag=robotics`, sorted by
downloads), excluding task-specific fine-tunes:

| model | weights to load | cameras expected | action | code | 16 GB GPU | Thor: chunk latency, RTF @ 30 Hz |
|---|---|---|---|---|---|---|
| **SmolVLA** (`lerobot/smolvla_base`) | **0.9 GB** | `camera1..3` | 6D joint | LeRobot 0.4.4+ | comfortable | 203 ms, RTF 8.2, **real-time** |
| **MolmoAct2-SO100_101** (`lerobot/MolmoAct2-SO100_101-LeRobot`) | 21.8 GB (fp32) | `cam0` + `cam1` | 6D abs joint, SO-100/101 | LeRobot 0.6.1+ (config fix on main) | inference only | 397 ms bf16, RTF 2.5, **real-time** |
| **pi05-so100_101** (`hqfang/pi05-so100_101`) | 16.6 GB (fp32) | `camera_0..3` slots | 6D abs joint, SO-100/101 | LeRobot + bundled patch | inference only | 252 ms bf16 compiled, RTF 6.6, **real-time** |
| LingBot-VLA 4B / 2.0 6B (`robbyant/...`) | 16.8 / 25.5 GB | 3 (dual-arm RoboTwin) | joint, no SO-101 checkpoint | own repo on LeRobot 0.4.2 | no | 253 / ~500 ms compiled, RTF 3.3 / 1.7, real-time |
| Hy-Embodied-0.5-VLA (`tencent/...-UMI`) | 9.1 GB | 3 x 6-frame history | **end-effector delta**, dual arm | own repo | no | 976 ms, RTF 1.7, real-time |
| Cosmos3-Edge-Policy (SO-101 conversion is community-only) | 8.2 GB | concat view | 6D (community conversion) | private fork / diffusers | — | 3.8 s measured (NVIDIA: 1.53 s), **not real-time** |
| FLUX 3 Action SO-101 | ~25 GB incl. encoders | `scene` + `wrist` @ 256x256 | 6D joint delta | LeRobot main only | no | 5.1 s, **not real-time** |
| `lerobot/pi05_base` | 14.5 GB | `base_0_rgb`, `left/right_wrist_0_rgb` | 6D | LeRobot 0.4.4+ | no | same network as pi05-so100_101 |

RTF (real-time factor) is the time one chunk lasts at 30 Hz divided by the time
it takes to compute; below 1.0 the arm has to stop and wait. Details, settings
and caveats are in "Five more candidates on Thor" below. "Inference only" means
the bf16 peak (8.9 GB, 12.1 GB) fits a 16 GB card but training does not.

FLUX 3's 13.9 GB checkpoint is only the trunk. Its frozen encoders load from
`black-forest-labs/flux-3-action-base`: a text encoder (8.9 GB) and a video VAE
(2.5 GB). Trunk + VAE alone is 16.4 GB, so it cannot even run inference on a
16 GB card, let alone train. Jetson Thor's 122 GB of unified memory holds all of
these comfortably, which makes it the machine for trying the large models.

MolmoAct2 has by far the most downloads (7.5k), FLUX 3 Action the most likes.
Both are too large to train on a 16 GB card without heavy tricks; SmolVLA is the
only one that is comfortable.

LeRobot 0.6.1 also added `lingbot_va`, `eo1`, `evo1`, `vla_jepa`, `fastwam`,
`multi_task_dit` and `molmoact2`. Upgrading from 0.4.4 unlocks all of them at
once, but is not required for SmolVLA.

## Camera keys decide which checkpoints you can ever use

A pretrained checkpoint declares exact observation keys. FLUX 3 Action SO-101
declares:

```json
"observation.images.scene": {"shape": [3, 256, 256]},
"observation.images.wrist": {"shape": [3, 256, 256]},
"observation.state":        {"shape": [6]}
```

LeRobot drops any feature not shared across datasets (see the community-data
section of the teleop tutorial), and it only warns. So a dataset recorded with a
`desk` camera cannot feed a checkpoint expecting `scene` without renaming.

For that reason the recorder writes the RealSense stream as **`scene`**, not
`desk`:

```bash
python ~/so101_unified_teleop.py record \
  --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1 \
  --repo-id local/so101_pick --task "Pick up the cube" \
  --cameras wrist,scene --fps 30 \
  --streaming-encoding true --encoder-threads 2
```

`--cameras wrist,desk` still works as an alias and still produces the key
`scene`, so old commands keep running. The dataset ends up with exactly
`observation.images.wrist` + `observation.images.scene`, which matches FLUX 3
Action and stays reasonable for everything else.

Record at **30 fps**: FLUX 3 Action executes at 30 Hz and most SO-101 community
datasets are 30 fps.

## Option A: SmolVLA (start here)

450M parameters, ~0.9 GB. Its dynamic padding accepts any action space up to
32 dimensions, so the SO-101's 6 joints work with no kinematics conversion. It
runs in the LeRobot 0.4.4 environment already installed on the GPU box.

```bash
python ~/lerobot_datapipeline.py train \
  --repo-id local/so101_pick \
  --policy smolvla \
  --remote cmpe28803 \
  --steps 20000
```

That syncs the dataset, submits the run detached under `nohup`, and records it
in `~/.lerobot_runs.json`. Follow and collect it with:

```bash
python ~/lerobot_datapipeline.py status --run <run-id> --follow
python ~/lerobot_datapipeline.py fetch  --run <run-id>
```

Community practice is ~50 episodes and ~20k steps for a single task. Check the
data first — `check` reports joints that never really moved, which no amount of
extra episodes will fix.

### What the pipeline does for a pretrained policy

Choosing a pretrained policy only helps if training actually starts from its
weights. `train` handles three things that are easy to get wrong by hand:

**1. It fine-tunes with `--policy.path`, not `--policy.type`.** `--policy.type`
builds a fresh config and leaves the action head randomly initialised, which
throws away the pretraining. Defaults per policy:

```text
smolvla -> lerobot/smolvla_base
pi0     -> lerobot/pi0_base
pi05    -> lerobot/pi05_base
flux3   -> black-forest-labs/flux-3-action-so101  (+ its lora.json recipe)
act, diffusion -> from scratch
```

Pass `--pretrained <hub-id>` to use another checkpoint, or `--pretrained none`
to deliberately train from scratch.

**2. It maps camera keys onto the checkpoint's.** Checkpoints name their inputs
differently, and the names must line up:

```text
smolvla_base   camera1, camera2, camera3                       no meaning, by order
pi0/pi05_base  base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb by meaning
flux3-so101    scene, wrist                                    matches ours
```

`train` reads both sides and builds `--rename_map` itself — semantically first
(`wrist` to a wrist key, `scene`/`front`/`top` to a base/scene key), then by
order — and prints what it chose:

```text
  dataset cameras   : scene, wrist
  checkpoint expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
  rename    scene -> base_0_rgb
  rename    wrist -> left_wrist_0_rgb
```

When the keys already match (FLUX 3) it passes no map. Override with
`--rename-map '<json>'`. A checkpoint expecting more cameras than you recorded
is fine for SmolVLA: it skips missing image keys and only errors if all are
missing.

**3. It passes `--policy.push_to_hub=false`.** `--policy.path` inherits
`push_to_hub` from the checkpoint's `config.json`; `smolvla_base` stores `true`,
and `lerobot-train` then refuses to start without `--policy.repo_id`. Publish
deliberately with `--push-to-hub --hub-repo-id myuser/so101_smolvla`.

All three were verified by feeding the generated argv to LeRobot's own config
parser and `validate()`, not just by reading the command.

## Option B: FLUX 3 Action SO-101

A 7B **world-action** model: it takes camera frames, robot state and a text
instruction, and denoises the next action chunk *jointly with the next video
frames*. Trained on the SO-101 episodes of `lerobot/community_dataset_v3`.

Its contract, from the checkpoint's own `config.json` and model card:

```text
cameras     observation.images.scene, then observation.images.wrist, 256x256
state       6 dims;  action 6 dims, joint delta with absolute gripper
history     8 observations, 2 visual snapshots, past-command conditioning
horizon     predicts 42 actions, executes 32, replans at 30 Hz
sampling    4 Euler steps, shift 6.93, guidance 3, seed 42, BF16
weights     13.9 GB safetensors
adaptation  rank-32 LoRA, bf16, batch 2, grad accumulation 4, 10k steps
```

### It needs its own environment

`flux3` is **not in any released LeRobot** — not 0.5.x, not 0.6.0, not 0.6.1. It
exists only on `main`, and BFL's documentation says so explicitly. It also wants
Python 3.12, torch 2.11 + cu128, and `natten`.

Keep it separate from the working 0.4.4 conda env:

```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot-main
cd ~/lerobot-main
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install 'torch==2.11.0' 'torchvision==0.26.0' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[training,flux3,peft,diffusion]'
uv pip install 'natten==0.21.6+torch2110cu128' --find-links https://whl.natten.org/
```

### LoRA fine-tuning

```bash
hf download black-forest-labs/flux-3-action-so101 lora.json --local-dir models/so101
lerobot-train --config_path=models/so101/lora.json \
  --policy.path=black-forest-labs/flux-3-action-so101 \
  --dataset.repo_id=YOUR_DATASET \
  --output_dir=outputs/so101-lora --job_name=so101-lora
```

The recipe inherits the camera layout, normalization and control conventions
from the checkpoint, so the dataset must already use `scene` + `wrist`.

### Memory rules out a 16 GB card

Not "tight" — impossible. The checkpoint's 13.9 GB is only the trunk; the text
encoder (8.9 GB) and video VAE (2.5 GB) load from the base repository on top,
about 25 GB in all. Run it on Jetson Thor (see below) or a 32 GB+ GPU.

BFL's own full fine-tuning documentation describes 8-GPU and 32xH200
configurations — a different path from the single-GPU LoRA recipe, but an
indication that the model is not light.

### Safety note from the model card

FLUX 3 Action outputs joint targets and bounds nothing itself:

> Nothing in the model bounds joint velocity, force or workspace; the
> application must enforce those limits and keep a hardware stop within reach.

Keep `--max-relative-target` set and stay on the power switch.

## Running the newer policies on Jetson Thor

LeRobot 0.6.1 and main add many policies that 0.4.4 lacks (`molmoact2`,
`lingbot_va`, `eo1`, `evo1`, `vla_jepa`, `fastwam`, `lawam`, `flux3`, ...). They
were brought up on Jetson Thor, whose 122 GB of unified memory holds the large
checkpoints a 16 GB card cannot.

Environment: LeRobot main (0.6.2-dev, commit `e0d5021`, 2026-09-29) in a `uv`
venv at `~/Developer/lerobot`, with the stock PyTorch aarch64 cu130 wheel. Setup
steps are in `JETSON_THOR.md` section 7.

### Every policy imports

All 19 policies import and build a default config on Thor:

```text
act diffusion smolvla pi0 pi05 pi0_fast groot xvla wall_x molmoact2
eo1 evo1 vla_jepa fastwam lawam lingbot_va multi_task_dit gaussian_actor flux3
```

`eo1` initially failed with a `PermissionError` on the Hugging Face cache. That
was a root-owned `~/.cache/huggingface`, not an aarch64 problem; see
`JETSON_THOR.md`. `torchcodec` is installed but fails to load on Thor and
LeRobot falls back to `pyav` for video decoding — harmless, possibly slower.

Importing is not running. The checks below load real weights.

### SmolVLA on Thor

Loaded `lerobot/smolvla_base` and ran the exact rollout path
(`prepare_observation_for_inference` -> preprocessor -> `select_action` ->
postprocessor) on synthetic observations built from the checkpoint's own
input contract:

```text
loaded in 16.2s   weights resident 0.86 GB   peak 0.90 GB
first call  808 ms (warm-up)
steady      203 ms per action-chunk inference
```

203 ms per chunk is well inside real-time: SmolVLA emits a chunk of actions per
inference, so at 30 Hz a 50-step chunk lasts 1.7 s.

### FLUX 3 needs NATTEN, and NATTEN needs a workaround on Thor

FLUX 3's video VAE requires NATTEN (neighbourhood attention). NATTEN publishes an
exactly matching wheel for this environment:

```bash
uv pip install 'natten==0.21.6+torch2110cu130' --find-links https://whl.natten.org/
```

It installs, but its compiled kernels do not include Thor. Tested per backend on
Thor (compute capability 11.0):

```text
flex-fna       OK     PyTorch FlexAttention, compiled at runtime by Triton
cutlass-fna    FAIL   no kernel image is available for execution on the device
blackwell-fna  FAIL   expects compute capability 100 or 103, Thor is 110
hopper-fna     FAIL   expects 90
```

The prebuilt aarch64 wheels target datacenter SBSA parts (GH200 sm_90,
GB200 sm_100), not Thor's sm_110. Worse, `can_run_cutlass_fna()` still returns
`True` on Thor — it checks the compute capability, not whether the binary has a
kernel for it — so NATTEN auto-selects the one backend that crashes.

FLUX's VAE has an override for exactly this, the `F3_NATTEN_BACKEND`
environment variable. **But on LeRobot main it has no effect**, because of a bug
in `lerobot/policies/flux3/f3/video_vae.py`. The backend is chosen by
`_natten_attention_kwargs()`, whose result is passed as:

```python
na2d(q, k, v, kernel_size=..., attention_kwargs=_natten_attention_kwargs(q, k, v))
```

In NATTEN, `attention_kwargs` configures a *different* operator (the FMHA used to
merge additional keys). The neighbourhood-attention backend is the top-level
`backend=` argument, which is never passed, so NATTEN picks its own and lands on
`cutlass-fna`. The function also returns `run_persistent_kernel`, which is itself
a top-level `na2d` argument — the dictionary was clearly meant to be unpacked.
A minimal reproduction on Thor:

```text
attention_kwargs={"backend": "flex-fna"}   CRASH  no kernel image is available
backend="flex-fna"                         OK
```

The fix is two lines — unpack instead of nesting:

```diff
-                attention_kwargs=_natten_attention_kwargs(q2, k2, v2),
+                **_natten_attention_kwargs(q2, k2, v2),
...
-                attention_kwargs=_natten_attention_kwargs(q, k, v),
+                **_natten_attention_kwargs(q, k, v),
```

Applied to the local checkout at `~/Developer/lerobot` (`git diff` shows it; revert
with `git checkout -- src/lerobot/policies/flux3/f3/video_vae.py`). Then run with:

```bash
export F3_NATTEN_BACKEND=flex-fna
```

This bug affects any GPU where NATTEN's auto-selection is wrong, not just Thor.
It is worth reporting upstream to `huggingface/lerobot`.

### FLUX 3 on Thor: it runs, but not in real time

With the fix and `F3_NATTEN_BACKEND=flex-fna`, the full rollout path runs:

```text
loaded in 44.4s   weights resident 23.63 GB   peak 26.99 GB
first call  6954 ms (warm-up)
steady      5095 ms per action-chunk inference
```

The model card's control scheme is *predict 42 actions, execute 32 at 30 Hz,
then replan* — 32 actions last 1.07 s. A 5.1 s inference means the arm would
execute for about a second and then wait about four for the next chunk.
Roughly **5x too slow for closed-loop control at the intended rate**.

Where the time goes, measured by timing the VAE's neighbourhood-attention layers
separately (`policy_bench/flux3_profile.py`):

```text
steady total      5387 ms
VAE natten        1015 ms  (18.8%, 46 calls)   <- the flex-fna workaround
everything else   4372 ms  (81%)               <- 7B trunk, 4 Euler steps, guidance
```

So the NATTEN workaround is **not** the bottleneck. Compiling it is not an option
either: NATTEN refuses `torch_compile` for Flex Attention, because it cannot
verify correctness in every case, and explicitly discourages overriding that.
Even a free VAE would leave ~4.4 s per chunk, so overriding would trade a stated
correctness risk for at most ~1 s. It was left alone.

What FLUX 3 on Thor is good for: evaluating the checkpoint, LoRA fine-tuning
(memory is not the constraint here), and open-loop or slow tasks. For real-time
rollout, a faster GPU is needed, or a smaller policy.

## Five more candidates on Thor

Jetson Thor is the fastest embedded platform for this arm, so it sets the bar:
**a policy that cannot keep up on Thor is dropped.** Five more were tested
against it on 2026-09-30.

### How "keeps up" was measured

A chunked policy computes a chunk of actions, then executes `n_action_steps` of
them. At 30 Hz (the rate our data is recorded at) that gives a time budget per
chunk:

```text
budget = n_action_steps / 30 Hz        RTF = budget / measured chunk latency
```

RTF >= 1 means the next chunk is ready before the current one runs out.

Every number below:

- was measured on Thor in its default **120 W** power mode. NVIDIA's own
  Thor numbers use MAXN (`sudo nvpmodel -m 0`), so these are conservative.
- is the steady-state latency after warm-up. One-time costs (loading,
  `torch.compile`) are listed separately.
- uses synthetic observations built from each checkpoint's own input contract,
  run through the same path a real rollout uses. They are latency and
  memory numbers, **not** task success.

| policy | precision / mode | executed per chunk | chunk latency | budget | RTF | peak mem |
|---|---|---|---|---|---|---|
| MolmoAct2-SO100_101 | fp32 (checkpoint default) | 30 | 1748 ms | 1000 ms | 0.57 | 22.0 GB |
| MolmoAct2-SO100_101 | **bf16** (LeRobot default) | 30 | **397 ms** | 1000 ms | **2.5** | 12.1 GB |
| pi05-so100_101 | fp32 eager (checkpoint default) | 50 | 1934 ms | 1667 ms | 0.86 | 15.8 GB |
| pi05-so100_101 | bf16 eager | 50 | 575 ms | 1667 ms | 2.9 | 8.9 GB |
| pi05-so100_101 | **bf16 compiled** | 50 | **252 ms** | 1667 ms | **6.6** | 8.9 GB |
| LingBot-VLA 4B (RoboTwin) | bf16 eager | 25 | 627 ms | 833 ms | 1.3 | 8.0 GB |
| LingBot-VLA 4B (RoboTwin) | **bf16 compiled** | 25 | **253 ms** | 833 ms | **3.3** | 8.1 GB |
| LingBot-VLA 2.0 6B (RoboTwin) | bf16 eager | 25 | 893 ms | 833 ms | 0.93 | 12.1 GB |
| LingBot-VLA 2.0 6B (RoboTwin) | **bf16 compiled** | 25 | **~500 ms** | 833 ms | **~1.7** | 12.2 GB |
| Hy-Embodied-0.5-VLA (UMI) | bf16 | 50 | 976 ms | 1667 ms | 1.7 | 8.8 GB |
| Cosmos3-Edge-Policy-DROID | bf16, 4 steps, no CFG, no video decode | 32 | 3815 ms | 1067 ms | 0.28 | 9.5 GB |
| Cosmos3-Edge-Policy-DROID | bf16, 1 step, no CFG, no video decode | 32 | 2580 ms | 1067 ms | 0.41 | 9.5 GB |
| FLUX 3 Action SO-101 (earlier) | bf16, flex-fna | 32 | 5095 ms | 1067 ms | 0.21 | 27.0 GB |
| SmolVLA (earlier) | fp32 | 50 | 203 ms | 1667 ms | 8.2 | 0.9 GB |

One-time costs: `torch.compile` takes 111 s on the first call for LingBot 4B,
~4 min for LingBot 2.0, and 11.7 min for pi05 (its `max-autotune` mode; later
starts reuse the Inductor cache). Loading takes 40–112 s.

### bf16 is safe for MolmoAct2 and pi05

Both checkpoints ship fp32 weights, and the bf16 rows are 3–4x faster. Thor's
fp32 throughput is far below its bf16. LingBot 2.0's README warns that bf16
"can produce materially different success rates" for that model, so bf16 was
checked rather than assumed.

Method: predict one whole chunk from the same observation with the same
flow-matching noise, once in fp32 and once in bf16. The scale for comparison is
the model's own sampling spread: the same fp32 model with a different noise
seed. Joint units are degrees (`policy_bench/compare_chunks.py`):

```text
MolmoAct2 (30 x 6)   bf16 vs fp32     mean 0.07 deg   max 0.26 deg
                     seed 0 vs seed 1  mean 2.81 deg   max 11.2 deg
pi05      (50 x 6)   bf16 vs fp32     mean 0.08 deg   max 0.43 deg
                     seed 0 vs seed 1  mean 1.46 deg   max 18.6 deg
```

The precision change is 20–40x smaller than the model's own run-to-run
variation, so bf16 is the right setting for both. This was not repeated for
LingBot 2.0. If LingBot is ever used, test bf16 against fp32 on real task success
first, as its authors advise.

### MolmoAct2-SO100_101: keep

The strongest candidate. It was trained on SO-100/101 community data, outputs
absolute 6-D joint targets, and is in LeRobot. It runs 2.5x faster than
real-time in bf16.

- **Config fix needed on LeRobot main.** The hub checkpoint was converted in June,
  and main has since renamed fields: `model_dtype` became `dtype`, and
  `enable_lora_vlm` became `train_mode_vlm`. Loading fails with
  `DecodingError: The fields ... are not valid for MolmoAct2Config`. If
  `enable_lora_vlm=false` is simply dropped, the new default `train_mode_vlm="lora"`
  wraps the VLM in PEFT, and the saved weights no longer match (`Missing key(s)
  ... lora_A`). The correct mapping is `enable_lora_vlm=false` -> `train_mode_vlm="fft"`.
  `policy_infer.py --compat` writes a local snapshot with both fixes. The weights
  are symlinked and nothing on the hub is changed. On Thor it is at
  `~/.cache/policy_bench/lerobot--MolmoAct2-SO100_101-LeRobot`.
- **The checkpoint saves `device: cpu`.** Override the preprocessor's device step
  (`preprocessor_overrides={"device_processor": {"device": "cuda"}}`), or the
  batch stays on the CPU.
- **Calibration.** Its processor applies `joint_signs [1,-1,1,1,1,1]` and
  `joint_offsets [0,90,90,0,0,0]`, which convert the LeRobot calibration (mid-range
  zero, PR #777) to the convention the training data used. Our arm was calibrated
  with LeRobot 0.4.4, which is already on the new convention. `so101_unified_teleop.py`
  records with `use_degrees=true`, matching the degree offsets. So the shipped
  transform applies to our data as-is.
- **Cameras.** `cam0` is the primary view and `cam1` the secondary. Map
  `scene -> cam0` and `wrist -> cam1` with `--rename_map`.

### pi05-so100_101: keep

This is pi05 fine-tuned by AllenAI on the same 1,209-dataset SO-100/101 mixture
(36,877 episodes), for 150k steps. There is no task-success evaluation in the
card. It runs 6.6x faster than real-time once compiled.

- **It needs its own LeRobot.** The card pins commit `b6ec006` plus a bundled
  patch (`code/lerobot.patch`) that adds a SentencePiece tokenizer and
  camera-slot padding. Rebuilt on Thor in `~/Developer/lerobot-pi05so`, a
  separate venv with torch 2.11 cu130, as the card instructs. Do not apply it to
  the main checkout.
- **Tokenizer path.** `config.json` points at the author's cluster. Override
  `text_tokenizer_name` with `<snapshot>/tokenizer/tokenizer.model`.
- **Cameras.** There are four anonymous slots, `camera_0..3`. Real views go in
  the first slots, and the unused slots get zero images with
  `observation.images.camera_N_is_pad = true`. Padding only masks tokens, so
  with two cameras the cost is the same as with four. Keep the assignment fixed
  within a rollout.
- **Compile.** `--set compile_model=true` needs the Triton ptxas override below.

### LingBot-VLA: fast, but no SO-101 checkpoint

Both generations keep up once compiled, and the action space is joints, so an
SO-101 fine-tune is possible in principle. Against it:

- **Only dual-arm checkpoints exist:** pretraining, plus RoboTwin (14-D).
- **It uses its own training stack.** It is built on LeRobot 0.4.2 (v1) or
  VeOmni (v2). It reads LeRobot v3 datasets through a per-robot feature-mapping
  YAML, so our data would need a new `configs/robot_configs/so101.yaml` and
  normalization stats.
- **v2 is only just real-time when compiled** (RTF ~1.7), and its authors
  validate in fp32.

Measured through each repo's own deploy class (`LingbotVLAServer`), which runs
FeatureTransform, the Qwen-VL processor, `sample_actions` and un-normalization,
with the real-robot setting `use_length=25`. Setup gotchas are in
`policy_bench/README.md`: the flash-attn wheel and the Triton ptxas override.

Verdict: **not now.** Revisit if an SO-100/101 checkpoint appears, or if
MolmoAct2 and pi05 both fall short.

### Hy-Embodied-0.5-VLA: drop for the SO-101

It keeps up (RTF 1.7), but it predicts **end-effector deltas** (xyz + rot6d +
gripper per arm), not joint angles. The released checkpoints are dual-arm UMI and
RoboTwin. Using it would need an IK layer for a 5-DoF arm that cannot reach
arbitrary orientations, plus fine-tuning through a data loader that has no
LeRobot format. That is a research project, not a policy choice.

Notes: the backbone imports `flash_attn` unconditionally, and
`scripts/quick_start.py` points at a repo name (`tencent/Hy-VLA-RoboTwin`) that
no longer exists. Use `tencent/Hy-Embodied-0.5-VLA-UMI` or `-RoboTwin`.

### Cosmos3-Edge SO-101: drop for now

- **There is no official SO-101 checkpoint.** `geonmin-kim/Cosmos3-Edge-Policy-SO101-init`
  is a community conversion with weights bit-identical to
  `nvidia/Cosmos3-Edge-Policy-DROID`. Its policy class lives in
  `nota-github/xpu-lerobot`, which returns 404 (private or removed). LeRobot's own
  Cosmos3 integration (PR #3745) is still a design draft.
- **It is not real-time at 30 Hz.** The DROID weights (same compute) were run
  through the official diffusers `Cosmos3OmniPipeline` (`policy_bench/cosmos3_bench.py`).
  Every call also generates 33 future video frames. Profiled per component:

  ```text
  4 UniPC steps, no CFG, video decode skipped:  3815 ms
      vae.encode   2146 ms   <- the one conditioning frame, repeat-padded to 33 frames
      transformer  1605 ms   (~400 ms per step)
  decoding the generated video adds ~6.5 s; CFG (guidance 3) adds ~1.6 s
  ```

  NVIDIA's model card reports **1.53 s** on Thor T5000 at MAXN with its own
  optimised policy server (`cosmos-framework`, 4 steps, guidance 3). Even that is
  RTF 0.70 against a 32-step chunk at 30 Hz. It is real-time only at 15 Hz
  (budget 2.13 s), and our data is 30 Hz. Rebuilding NVIDIA's server on Thor
  needs transformer-engine and torch 2.13, and its answer is already published,
  so it was not rebuilt.
- The community card says its 1-step "Drift" variant runs 0.25 s vs 0.62 s on a
  B200. That might make 30 Hz, but it needs the unavailable fork.

Revisit when Cosmos3 lands in LeRobot.

### FLUX 3 Action: drop

Measured earlier at 5.1 s per chunk (RTF 0.21). By the rule above, FLUX 3 is out
as a controller. It stays useful only for offline evaluation.

### Triton on Thor: point it at the system ptxas

Any path that JIT-compiles Triton kernels fails on Thor: `torch.compile`, and
flash-attn's rotary embedding, which LingBot uses even in eager mode. The error:

```text
ptxas-blackwell fatal : Value 'sm_110a' is not defined for option 'gpu-name'
```

The ptxas bundled with Triton 3.6 predates sm_110. CUDA 13.0's own ptxas knows
it. Triton reads a separate variable for Blackwell-family GPUs, so set both:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
```

## Can pi05 or MolmoAct2 run on the Jetson Orin Nano?

The arm is wired to an **Orin Nano 8 GB Super** (JetPack 6.2 / L4T R36.4.7,
`MAXN_SUPER`, 7.6 GB visible). Measured on 2026-09-30, the short answer is:

- **pi05-so100_101: yes, for compute, with three optimisations. Memory forces
  int8 weight storage**, which is designed but not yet built end-to-end.
- **MolmoAct2-SO100_101: no.** It does not fit in memory and would not keep up
  if it did.

### How Orin was measured without fitting the model

The bf16 checkpoint does not fit, so the real LeRobot pi05 network was built with
random weights and only L of its 18 layers, for L = 1..3, and the line
`T(L) = fixed + L * per_layer` was extrapolated to 18 (`policy_bench/pi05_layer_scaling.py`).
Two tables of no compute value were shrunk, each a single 1.05 GB allocation that
NvMap refuses on this board (`NvMapMemAllocInternalTagged ... error 12`): the
257k-row token embedding (a lookup) and the lm_head (never called by
`sample_actions`). Each depth runs in a fresh process, because GPU memory
fragments across builds.

The method was checked on Thor against the real checkpoint:

```text
Thor, extrapolated from L=2,4,6     571 ms
Thor, random weights at L=18        571 ms
Thor, real hqfang/pi05-so100_101    575 ms
```

### Results on Orin (50-action chunk, budget 1667 ms at 30 Hz)

| configuration | fixed | per layer | 18 layers | RTF |
|---|---|---|---|---|
| as pi05-so100_101 loads (fp32 vision), 4 slots, eager | 866 ms | 91 ms | ~2.5 s | 0.67 |
| bf16 vision, 4 slots, eager | 244 ms | 89 ms | 1.85 s | 0.90 |
| bf16 vision, **2 slots**, eager | 161 ms | 61 ms | 1.27 s | 1.32 |
| bf16 vision, 4 slots, **CUDA graph** | 193 ms | 68 ms | 1.42 s | 1.17 |
| **bf16 vision, 2 slots, CUDA graph** | 100 ms | 43 ms | **0.88 s** | **1.90** |
| ... + int8 trunk storage (dequant cost added) | | | **~1.18 s** | **~1.4** |

For comparison, the same optimised configuration on Thor is ~220 ms (eager
baseline 575 ms, compiled 252 ms). Orin is roughly 4x slower than Thor at equal
settings.

What each optimisation is:

1. **Vision tower in bf16.** The LeRobot version this checkpoint needs (b6ec006)
   keeps the whole SigLIP tower in fp32, a choice made for training. On Orin
   that alone costs ~620 ms per chunk (on Thor ~220 ms). The model was trained
   under bf16 autocast, so bf16 vision is no further from training than fp32.
2. **Feed only the real cameras.** pi05-so100_101 has 4 camera slots, and padded
   slots are masked but still run through SigLIP and the trunk. Masked tokens
   neither attend nor advance positions, so removing them from the input is
   equivalent. With two cameras this cuts image tokens from 1024 to 512.
3. **CUDA graph of `sample_actions`.** Each of the 10 denoising steps runs the
   300M expert on only 50 tokens, so the step is launch-bound (~70 ms per step on
   Orin, eager). Orin's PyTorch has no Triton, so `torch.compile` is unavailable,
   but a plain `torch.cuda.CUDAGraph` works. One obstacle: pi05 builds its
   constant attention masks with `torch.tensor(list, device=cuda)`, which is
   forbidden during capture. The bench caches those during warm-up. Graph output
   was bit-identical to eager (max diff 0).

Not used: 5 denoising steps instead of 10 (0.85 s eager) and a shorter padded
prompt (64 instead of 200 tokens). The first changes the model's sampling. The
second gains only 2%.

### Memory is the real constraint

Weights that must stay resident:

```text
Gemma-2B trunk layers    1.98 B params   3.96 GB bf16 | 1.98 GB int8
Gemma-300M expert        0.43 B          0.85 GB bf16   (runs 10x per chunk, keep bf16)
SigLIP + projector       0.41 B          0.83 GB bf16
token embedding          0.53 B          1.05 GB bf16 (lookup only; prunable)
lm_heads                 0.79 B          not needed for inference -> drop
```

In bf16 that is 6.7 GB before the CUDA context, activations and the OS. The desktop
session alone uses 1.7 GB now, so **bf16 cannot fit in 7.6 GB.**

Storing the trunk in int8 brings the weights to ~4.9 GB. Even then it only fits
headless or close to it.

Of the int8 options, only one was usable on Orin:

- **W8A8 with `torch._int_mm` is slower.** It was 2.4–4x slower than bf16 at
  pi05's shapes (`int8_mm_bench.py`). Without Triton or TensorRT, the
  activation quantise and rescale cannot be fused into the GEMM.
- **Weight-only int8 works.** Each trunk layer is dequantised into a bf16 buffer
  just before use. The trunk runs once per chunk, so this costs ~307 ms per
  chunk (`int8_dequant_bench.py`) and saves 1.85 GB.

### The optimisations do not change the actions

Checked on Thor with the real checkpoint (`pi05_opt_check.py`). The same
observation (2 real cameras + 2 padded slots) and the same noise were compared
against the as-loaded bf16 model:

```text
noise seed 1 (model's own spread)   mean 6.22 deg   max 80.4 deg
vision in bf16                      mean 0.25 deg   max 5.1 deg
padded slots removed                mean 0.23 deg   max 1.4 deg
trunk rounded to int8               mean 0.29 deg   max 4.7 deg
all three                           mean 0.29 deg   max 4.2 deg
```

All of these are about 20x below the model's own run-to-run variation.

### MolmoAct2 on Orin: no

- **Memory.** Its resident core is ~4.6 B params: 9.2 GB in bf16, more than
  the whole board. Even int8 (4.6 GB) plus the vision stack does not leave room
  for the runtime.
- **Speed.** On Thor it already runs bf16 with CUDA graphs (397 ms per 30-action
  chunk). At the measured ~4–5x Orin/Thor ratio, that is ~1.6–2.1 s against a
  1.0 s budget, an RTF of ~0.5. The cheap wins pi05 had (fp32 vision, padded
  slots, eager launches) are already taken in MolmoAct2.

### Deployment options, in order of effort

1. **Plug the arm and cameras into Thor and run the policy there.** No porting,
   and both models are real-time on it.
2. **Orin drives the arm and Thor serves the policy over LeRobot async inference**
   (`policy_server` / `robot_client`, gRPC). The two boards are on different
   networks today: Orin is on `192.168.4.0/22` and Thor on `10.x`. Tailscale
   could not connect them directly and relays through DERP (sfo), with a 35–357 ms
   round trip. That is too jittery to rely on. Put both on the same LAN or switch
   first.
3. **pi05 on the Orin alone.** Needs the int8 weight-only trunk (not built yet),
   bf16 vision, two camera slots, a CUDA-graph rollout, and a headless boot. The
   estimate is ~1.2 s per 50-action chunk, which leaves the robot process
   ~0.4 s of slack. That is feasible, but it is a porting project. TensorRT
   (FP16 + INT8 weight-only) is the production version of this route.

The Orin LeRobot env (`~/lerobot-py310-cuda`, 0.4.4) had no `transformers`. For
these measurements the openpi branch LeRobot 0.4.4's pi0/pi05 needs was installed
(`transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi`,
reports 4.53.3). Teleop and recording do not use it.

## Testing pi05-so100_101 and MolmoAct2 on the real arm (Jetson Thor)

### What the two checkpoints were trained on

Both models were trained on the same mixture: 1,220 community SO-100/101 LeRobot
datasets from 377 users. That is 37,459 episodes, each with an AllenAI-annotated
instruction (`allenai/MolmoAct2-SO100_101-Dataset`). The instructions were
counted directly (`~/Developer/models/molmo_so_ds` on Thor):

```text
verbs    grasp 70%  place 65%  move 57%  lift 30%  pick 7%  stack 2%  push 2%  open/close/pour ~1%
objects  box 22%  block 22%  bin 19%  cup 10%  cube 9%  lego 6%  bowl 5%  pawn 4%  duck 4%  ball 4%
colours  blue 20%  red 18%  green 15%  yellow 12%  white 9%  black 9%
top instructions
   476  grasp purple pengrip, place in cup.
   204  grasp red block, place in bin.
   145  grasp red duck, place in box.
   130  grasp green cube, place on red cube.
```

So the scenario is **single-arm tabletop pick-and-place**: a small coloured
object into a box, bin, cup or bowl. Instructions are short and templated.
Cameras vary per dataset, and MolmoAct2's card says camera order does not
matter. Neither card reports real-robot task success, so zero-shot results on
our arm are unknown until tried.

The MolmoAct2 card's sample state (`shoulder_lift 189 deg`) is in the old
calibration convention. The LeRobot conversion's `joint_signs` and `joint_offsets`
map our LeRobot >= PR #777 degrees onto it.

### One-time setup on Thor

Already done:

- `lerobot[feetech]` and `pyrealsense2 2.58` are installed in both
  `~/Developer/lerobot` and `~/Developer/lerobot-pi05so`. LeRobot pins
  pyrealsense2 below 2.57, but the first aarch64 cp312 wheel is 2.58.
- `lerobot[dataset]` is installed in `lerobot-pi05so`.
- The follower and leader calibration was copied from the Orin to
  `~/.cache/huggingface/lerobot/calibration/`. It is the same path and format
  LeRobot main reads (`robots/so_follower/so101_follower.json`).

Needs sudo, once:

```bash
sudo usermod -aG dialout lkk          # serial access to /dev/ttyACM0; log out and back in
# RealSense via pip uses libusb; without these rules only root can open it
sudo curl -fsSL -o /etc/udev/rules.d/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo nvpmodel -m 0                    # MAXN (optional, every number here is from 120 W)
```

Then plug in the follower, the wrist UVC camera and the RealSense. Find them with
`ls /dev/ttyACM* /dev/video*` and `python ~/Developer/robotics/jetson_devices.py camera list`.
Check the calibration was picked up with
`python ~/Developer/robotics/so101_unified_teleop.py check-motors --follower-port /dev/ttyACM0`.

### Rollout commands

Both commands were dry-parsed by LeRobot's own `RolloutConfig` on Thor (robot,
cameras, policy overrides and rename map all resolved).

**Quote `--rename_map` in single quotes.** Without them, the shell strips the
inner quotes and YAML silently parses `{a:b}` into the key `"a:b"` with value
`None`. The command then fails much later, saying the cameras do not match.

```bash
CAMS='{ wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG},
        scene: {type: intelrealsense, serial_number_or_name: "<SERIAL>", width: 640, height: 480, fps: 30} }'

# MolmoAct2 (main venv; compat snapshot from policy_infer.py --compat)
source ~/Developer/lerobot/.venv/bin/activate
lerobot-rollout --strategy.type=base \
  --policy.path=$HOME/.cache/policy_bench/lerobot--MolmoAct2-SO100_101-LeRobot --policy.dtype=bfloat16 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower \
  --robot.max_relative_target=10 --robot.cameras="$CAMS" \
  --rename_map='{"observation.images.scene": "observation.images.cam0", "observation.images.wrist": "observation.images.cam1"}' \
  --task="grasp red block, place in box." --duration=30

# pi05-so100_101 (patched venv). Missing camera_2/3 are padded and masked automatically.
source ~/Developer/lerobot-pi05so/.venv/bin/activate
P=$(ls -d ~/.cache/huggingface/hub/models--hqfang--pi05-so100_101/snapshots/*/)
lerobot-rollout --strategy.type=base \
  --policy.path=hqfang/pi05-so100_101 --policy.dtype=bfloat16 --policy.compile_model=false \
  --policy.text_tokenizer_name=${P}tokenizer/tokenizer.model \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower \
  --robot.max_relative_target=10 --robot.cameras="$CAMS" \
  --rename_map='{"observation.images.scene": "observation.images.camera_0", "observation.images.wrist": "observation.images.camera_1"}' \
  --task="grasp red block, place in box." --duration=30
```

Notes:

- **`max_relative_target=10`** caps each tick's joint jump at 10 deg for the first
  runs, tighter than teleop's 30. Keep a hand on the power switch.
- **pi05 starts eager** (575 ms per chunk, RTF 2.9). `compile_model=true` gets 252 ms
  but takes ~12 min to compile at the first start. Turn it on once things work.
- **Sync inference is enough on Thor.** `--inference.type=rtc` (real-time
  chunking) exists for policies that are only borderline real-time.
- **To record trials for review**, switch to `--strategy.type=episodic` with a
  `--dataset.repo_id`.

### Test protocol

Stay inside the training distribution first, then step out of it one variable at
a time. Run 10 trials per row and count successes. A success is the object
ending inside the container, with no human touch.

| # | setup | instruction | tests |
|---|---|---|---|
| 1 | one red block, open box, fixed spots near the centre of the workspace | `grasp red block, place in box.` | does it work at all on our arm and cameras |
| 2 | same, block placed anywhere reachable | same | position generalisation |
| 3 | red + green + blue blocks | `grasp green block, place in box.` | language grounding (picks the named one) |
| 4 | swap the box for a cup or bowl | `grasp red block, place in cup.` | container generalisation |
| 5 | an object absent from the mix (e.g. a marker cap) | `grasp the cap, place in box.` | novel object |

Run both models on the identical setup per row. Rows 1–3 decide whether to
fine-tune one of them or start from SmolVLA. Our own episodes recorded with
`so101_unified_teleop.py record` can use the same instructions, so that
fine-tuning later stays in-distribution.

## Can LingBot-VLA be fine-tuned on cmpe28803 (RTX 5080 16 GB)?

**No, not in any configuration that is supported.** And re-training on the
1,220-dataset mixture is not feasible on one consumer GPU in any case.

- **Model size.** LingBot-VLA-4B has 4.20 B params: Qwen2.5-VL LLM 3.09 B,
  ViT 0.67 B, action expert 0.44 B. Full fine-tuning with AdamW mixed precision
  needs about 16 bytes per param, i.e. ~67 GB plus activations. The official
  smallest recipe is **4 x A6000 48 GB with FSDP sharding, at ~47 GB per GPU**
  (`configs/vla/Training_Config.md`). FSDP CPU offload cannot rescue it either,
  because cmpe28803 has 30 GB of RAM.
- **No LoRA.** `lingbotvla/utils/lora_utils.py` exists, but no training script
  calls it.
- **Expert only does not fit either.** With `train_expert_only`, the frozen VLM
  is 7.5 GB in bf16 and the fp32 expert plus Adam is ~7 GB. That is 14.6 GB
  before activations and the CUDA context, which overflows 16 GB.
- **Only dual-arm checkpoints exist.** Pretraining is dual-arm, plus RoboTwin.
  An SO-101 run needs a `configs/robot_configs/so101.yaml` feature map and new
  normalization stats.
- **The mixture itself.** The 1,220 repos are ~0.55 TB (100 sampled repos,
  mean 0.45 GB). cmpe28803 has 117 GB free. AllenAI's pi05 run on it was 38.4 M
  sampled frames, on 4 x GB200. Even at an optimistic 10 samples/s, a single
  5080 would need ~44 days.

That mixture training is exactly what `pi05-so100_101` and `MolmoAct2-SO100_101`
already are. The productive path is to fine-tune one of those on our own
episodes, on Thor, where memory is not the limit. MolmoAct2's LeRobot port has
`train_mode_vlm=freeze|lora`, and pi05 has `train_expert_only`, which might
bring them to 16 GB. Neither has been tried on the 5080.

## How NVIDIA got Cosmos3 to 1.53 s on Thor

From the `nvidia/Cosmos3-Edge` and `-Policy-DROID` cards and the
`cosmos-framework` policy server (`action_policy_server_robolab.py`):

- **It is not quantisation.** The card says only BF16 is tested, and FP4, FP8
  and FP16 are not supported.
- **4 UniPC denoising steps** (the server default) instead of the notebook's 30.
- **CFG only on part of the schedule.** `--guidance-interval 960 1001` runs the
  unconditional branch only for timesteps in that range, which is essentially
  the first step. Guidance 3.0 then costs about one extra forward pass, not four.
  The server also has batched cond+uncond and a shared text K/V cache.
- **No video decode.** `decode_video=False` is the default, so the co-generated
  rollout video stays latent. In our diffusers profile, decoding was ~6.5 s.
- **Persistent warm server**, one request at a time over a persistent connection.
  Their numbers follow a discarded warm-up.
- **MAXN** at a 1575 MHz GPU clock. Our runs were at 120 W.
- **Their own runtime.** NVIDIA's stack is PyTorch with transformer-engine, plus
  NATTEN built for sm_110 (`natten==0.21.6+cu130.torch213`). It also encodes the
  full repeat-T conditioning clip, so the speed-up is not from skipping frames.
  Against our diffusers run at the same settings (4 steps, no decode: 3.8 s, of
  which VAE encode 2.1 s and transformer 1.6 s), their pipeline is ~2.5x faster
  end to end. Which kernels account for that cannot be attributed without
  rebuilding their stack.

The model card's 1.53 s is against a 2.13 s budget at 15 Hz with 32-step
chunks. That is why it says "real-time at 15 Hz". At our 30 Hz the budget is
1.07 s, so it still does not keep up.

## Two training machines

`lerobot_datapipeline.py` ships with both configured in `~/.lerobot_pipeline.json`:

```text
cmpe28803   RTX 5080 16 GB, conda env `lerobot`, LeRobot 0.4.4
            -> SmolVLA, ACT, diffusion
jetsonthor  Thor 122 GB unified, uv venv ~/Developer/lerobot, LeRobot main
            -> MolmoAct2 and the other 0.6.x policies
jetsonthor_pi05so
            same machine, venv ~/Developer/lerobot-pi05so (patched LeRobot)
            -> hqfang/pi05-so100_101 only
```

Pick with `--remote`:

```bash
python ~/lerobot_datapipeline.py train --repo-id local/so101_pick --policy smolvla --remote cmpe28803

# MolmoAct2: local --compat snapshot (see above), explicit cameras, bf16 training
python ~/lerobot_datapipeline.py train --repo-id local/so101_pick --policy molmoact2 --remote jetsonthor \
  --pretrained ~/.cache/policy_bench/lerobot--MolmoAct2-SO100_101-LeRobot \
  --rename-map '{"observation.images.scene":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \
  --extra-arg=--policy.dtype=bfloat16

# pi05-so100_101: patched venv, tokenizer from the snapshot
python ~/lerobot_datapipeline.py train --repo-id local/so101_pick --policy pi05 --remote jetsonthor_pi05so \
  --pretrained hqfang/pi05-so100_101 \
  --extra-arg=--policy.text_tokenizer_name=<snapshot>/tokenizer/tokenizer.model
```

The pipeline prints these requirements itself (`CHECKPOINT_NOTES`) when it sees
either checkpoint. The MolmoAct2 command was checked by parsing the generated
arguments with LeRobot main's own `TrainPipelineConfig` + `validate()`: it
resolved to `MolmoAct2Config`, `train_mode_vlm=fft`, with the rename map intact.
Without `--policy.dtype=bfloat16` it trains in the snapshot's fp32, which is 4x
slower on Thor and needs twice the memory. Training throughput on Thor has not
been measured for either model.

The two differ in how their environment is activated (conda vs a venv), which
the `RemoteBackend` handles through the `conda_env` and `activate` fields; the
Thor entry also sets `F3_NATTEN_BACKEND=flex-fna`. Both were checked through the
pipeline's own backend:

```text
jetsonthor  OK  0.6.2 2.11.0+cu130 True
cmpe28803   OK  0.4.4 2.10.0+cu128 True
```

An existing `~/.lerobot_pipeline.json` is not overwritten, so on a machine that
already has one, add the `jetsonthor` block by hand (see `DEFAULT_CONFIG` in the
script).

## Recommended order

1. Record 30–50 episodes at 30 fps with `--cameras wrist,scene`, varying object
   placement so every joint is exercised.
2. `lerobot_datapipeline.py check` until no joint reports `barely used`.
3. **Before training anything, try MolmoAct2-SO100_101 and pi05-so100_101
   zero-shot on Thor.** Both were trained on >1,000 SO-100/101 datasets, output
   our joint space, and run in real time in bf16. If either already does the
   task roughly, fine-tuning it is the best path.
4. Train **SmolVLA** on cmpe28803 as the small, fast baseline.
5. Keep an **ACT** baseline on the same data as a control. Fine-tuning a
   pretrained VLA on a small real dataset does not always beat training from
   scratch.
6. Fine-tune whichever of MolmoAct2 / pi05 did best zero-shot, on Thor. Its
   memory holds both, but training throughput there has not been measured yet.

Dropped by the "must keep up on Thor" rule: FLUX 3 Action and Cosmos3-Edge (not
real-time at 30 Hz). Dropped for fit: Hy-VLA (end-effector actions). On hold:
LingBot-VLA (no SO-101 checkpoint).
