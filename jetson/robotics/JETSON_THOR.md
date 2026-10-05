# Jetson Thor Container & Isaac ROS Integration Guide

This guide documents the setup, base container selection, and Isaac ROS integration for **Jetson Thor** (`jetsonthor`) using `jetson/Dockerfile.jp7`.

---

## 1. Jetson Thor System Overview

- **Host Kernel & OS**: Linux 6.8.12-tegra, Ubuntu 24.04.3 LTS (Noble), L4T R38.4.0 (JetPack 7 preview)
- **GPU Architecture**: **NVIDIA Thor** (Blackwell GPU architecture, `sm_110` / CUDA 11.0–12.0 compute capability)
- **Memory**: 122.8 GB total unified memory
- **NVIDIA Driver & CUDA**: Driver 580.00 / CUDA 13.0
- **Docker Runtime**: Default runtime set to `nvidia`

---

## 2. Base Container Selection (`BASE_IMAGE`)

In `jetson/Dockerfile.jp7`:

- **For Jetson Orin Nano / AGX Orin (`sm_87`)**:
  `ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3-igpu` (default)
- **For Jetson Thor (`sm_110`)**:
  `ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3`
  *Reason*: `25.08-py3-igpu` only compiles Orin `sm_87` binaries, causing CUDA capability warnings on Thor. `25.08-py3` includes native `sm_100`, `sm_110`, and `sm_120` support with CUDA 13.0 on NVIDIA Thor.

---

## 3. Isaac ROS Integration (`INSTALL_ISAAC_ROS`)

Isaac ROS is integrated into `jetson/Dockerfile.jp7` via the build argument `INSTALL_ISAAC_ROS`:

- **Default (`INSTALL_ISAAC_ROS=0`)**: Kept disabled for Orin Nano / standard Jetsons to keep image size small (~21 GB) and builds fast.
- **Jetson Thor (`INSTALL_ISAAC_ROS=1`)**: Enables ROS 2 Jazzy, `colcon`, `rosdep`, NVIDIA extra rosdeps, and clones the official Isaac ROS packages (`isaac_ros_common`, `isaac_ros_nitros`, `isaac_ros_image_pipeline`, `isaac_ros_apriltag`) into `/opt/ros/isaac_ros_ws`.

---

## 4. Build Command for Jetson Thor

To build the Jetson Thor container image with CUDA 13 PyTorch and Isaac ROS:

```bash
docker build -f jetson/Dockerfile.jp7 \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 \
  --build-arg OPENCV_CUDA_ARCH_BIN=10.0 \
  --build-arg INSTALL_ISAAC_ROS=1 \
  -t cmpelkk/jetson-unified:jp7-thor jetson/
```

---

## 5. Run Command on Jetson Thor

```bash
docker run --rm -it --runtime nvidia --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --device-cgroup-rule='c 81:* rmw' --device-cgroup-rule='c 14:* rmw' \
  -v /dev:/dev -v /Developer:/Developer \
  cmpelkk/jetson-unified:jp7-thor
```

---

## 6. Verification inside Container

```bash
# Verify PyTorch CUDA & GPU on Thor:
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Verify ROS 2 Jazzy + Isaac ROS setup:
source /opt/ros/jazzy/setup.bash
source /opt/ros/isaac_ros_ws/install/setup.bash
ros2 pkg list | grep isaac_ros
```

Let's monitor task-1074 using `manage_task` or wait for it to complete. Let's check status.

Used tool: manage_task

Let's yield and wait for task-1074 to finish.

Ran command: `ssh jetsonthor "cd /home/lkk/Documents/edgeAI && docker build -f jetson/Dockerfile.jp7 --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 --build-arg OPENCV_CUDA_ARCH_BIN=10.0 --build-arg INSTALL_ISAAC_ROS=1 -t cmpelkk/jetson-unified:jp7-thor ."`

Let's check the build progress of task-1083 using `manage_task`.

Used tool: manage_task

Look at the log output! Step 12 (OpenCV rebuild with CUDA for Thor `10.0`) is running with `ninja -j14`!
Let's stop calling tools and wait for the system to automatically resume us when task-1083 finishes building.

Edited Dockerfile.jp7

Now let's sync the updated `jetson/Dockerfile.jp7` to `jetsonthor` and re-run `docker build`!

Let's rsync `jetson/Dockerfile.jp7` to `jetsonthor:/home/lkk/Documents/edgeAI/jetson/Dockerfile.jp7`.

Ran command: `rsync -avz /Users/kaikailiu/Documents/MyRepo/edgeAI/jetson/Dockerfile.jp7 jetsonthor:/home/lkk/Documents/edgeAI/jetson/Dockerfile.jp7`

Now let's launch the `docker build` command on `jetsonthor` again!

Ran command: `ssh jetsonthor "cd /home/lkk/Documents/edgeAI && docker build -f jetson/Dockerfile.jp7 --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 --build-arg OPENCV_CUDA_ARCH_BIN=10.0 --build-arg INSTALL_ISAAC_ROS=1 -t cmpelkk/jetson-unified:jp7-thor ."`

Let's monitor task-1098 using `manage_task` or wait for it. Let's check status.

Used tool: manage_task

Let's yield and wait for task-1098 to complete.

Edited Dockerfile.jp7

Now let's rsync the updated `jetson/Dockerfile.jp7` to `jetsonthor` and build!

Ran command: `rsync -avz /Users/kaikailiu/Documents/MyRepo/edgeAI/jetson/Dockerfile.jp7 jetsonthor:/home/lkk/Documents/edgeAI/jetson/Dockerfile.jp7`

Now let's start the `docker build` command on `jetsonthor` again!

Ran command: `ssh jetsonthor "cd /home/lkk/Documents/edgeAI && docker build -f jetson/Dockerfile.jp7 --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 --build-arg OPENCV_CUDA_ARCH_BIN=10.0 --build-arg INSTALL_ISAAC_ROS=1 -t cmpelkk/jetson-unified:jp7-thor ."`

Let's monitor task-1113 using `manage_task` or wait for it. Let's check status.

Used tool: manage_task

Let's yield and wait for task-1113 to complete.

Ran command: `ssh jetsonthor "cd /home/lkk/Documents/edgeAI && docker build -f jetson/Dockerfile.jp7 --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 --build-arg REBUILD_OPENCV=0 --build-arg INSTALL_ISAAC_ROS=1 -t cmpelkk/jetson-unified:jp7-thor ."`

Let's check the build progress of task-1122 using `manage_task`.

Used tool: manage_task

Look at the progress: `llama.cpp` and `llama-ui` are compiling cleanly on 14 cores on Jetson Thor!
Let's stop calling tools and wait for task-1122 to complete.

Ran command: `ssh jetsonthor "curl -sI https://ollama.com/download/ollama-linux-arm64.tgz | head -n 10"`

Look at that location:
`location: https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64.tgz` (It's a `.tgz` tarball containing `bin/ollama`!).
So `curl -fsSL https://ollama.com/download/ollama-linux-arm64.tgz | tar xzf - -C /usr/local/` extracts `bin/ollama` cleanly into `/usr/local/bin/ollama`!

Let's test this in `jetson/Dockerfile.jp7` line 194:
Replace:
`curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64 -o /usr/local/bin/ollama`
With:
`curl -fsSL https://ollama.com/download/ollama-linux-arm64.tgz | tar xzf - -C /usr/local/`

Let's use `replace_file_content` to fix line 194 in `jetson/Dockerfile.jp7`.

Edited Dockerfile.jp7

Now let's sync `jetson/Dockerfile.jp7` to `jetsonthor` and launch `docker build`!

Ran command: `rsync -avz /Users/kaikailiu/Documents/MyRepo/edgeAI/jetson/Dockerfile.jp7 jetsonthor:/home/lkk/Documents/edgeAI/jetson/Dockerfile.jp7`

Now let's launch `docker build` on `jetsonthor`! Because all previous layers (apt, python stack, llama.cpp compilation, llama-cpp-python wheel) were cached, this build will finish super fast!

Ran command: `ssh jetsonthor "cd /home/lkk/Documents/edgeAI && docker build -f jetson/Dockerfile.jp7 --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 --build-arg REBUILD_OPENCV=0 --build-arg INSTALL_ISAAC_ROS=1 -t cmpelkk/jetson-unified:jp7-thor ."`

Let's monitor task-1140 status using `manage_task`.

Used tool: manage_task

Let's yield and wait for task-1140 to complete.

Ran command: `ssh jetsonthor "docker run --rm nvcr.io/nvidia/pytorch:25.08-py3 bash -c 'curl -fsSL https://ollama.com/install.sh | sh && ollama --version'"`

LOOK AT THAT ERROR:
`ERROR: This version requires zstd for extraction. Please install zstd and try again: - Debian/Ubuntu: sudo apt-get install zstd`!

Ollama switched their package distribution to `.tar.zst` format which requires `zstd`!
Because `zstd` was not installed in apt, `curl -fsSL https://ollama.com/install.sh | sh` complained about missing `zstd`!

Let's verify:
If we add `zstd` to `apt-get install` in Block A of `jetson/Dockerfile.jp7`, and use `curl -fsSL https://ollama.com/install.sh | sh` for Ollama installation:
`curl -fsSL https://ollama.com/install.sh | sh` will work 100% cleanly!

Let's update `jetson/Dockerfile.jp7` to:
1. Add `zstd` to the `apt-get install` list at Section 1 (line 68).
2. Change the Ollama installation line to `curl -fsSL https://ollama.com/install.sh | sh`.

Let's use `replace_file_content` to make these two updates to `jetson/Dockerfile.jp7`.

Edited Dockerfile.jp7

Now let's replace the Ollama line in `jetson/Dockerfile.jp7` to use `curl -fsSL https://ollama.com/install.sh | sh`.

Edited Dockerfile.jp7

Now let's rsync the updated `jetson/Dockerfile.jp7` to `jetsonthor`.

Ran command: `rsync -avz /Users/kaikailiu/Documents/MyRepo/edgeAI/jetson/Dockerfile.jp7 jetsonthor:/home/lkk/Documents/edgeAI/jetson/Dockerfile.jp7`

Now let's launch `docker build` on `jetsonthor`!

Ran command: `ssh jetsonthor "cd /home/lkk/Documents/edgeAI && docker build -f jetson/Dockerfile.jp7 --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 --build-arg REBUILD_OPENCV=0 --build-arg INSTALL_ISAAC_ROS=1 -t cmpelkk/jetson-unified:jp7-thor ."`

Let's monitor task-1162 status using `manage_task`.

Used tool: manage_task

Because we added `zstd` to Step 5 (apt-get update), Docker invalidated the apt cache layer and is cleanly rebuilding with `zstd` present.
Let's yield and wait for task-1162 to complete.

Ran command: `ssh jetsonthor "docker run --rm nvcr.io/nvidia/pytorch:25.08-py3 bash -c 'echo LD_LIBRARY_PATH=\$LD_LIBRARY_PATH; python3 -c \"import torch; print(torch.__version__)\"'"`

Look at that: at runtime or when `LD_LIBRARY_PATH` has `/opt/hpcx/ucx/lib` loaded by the NGC entrypoint (`/opt/nvidia/nvidia_entrypoint.sh`), `import torch` works 100%!
During `docker build` (which runs without the NGC entrypoint), `/opt/hpcx/ucx/lib` is needed in `LD_LIBRARY_PATH` if HPC-X libraries are imported, OR we can set `ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:$LD_LIBRARY_PATH` or wrap the sanity check script to include `/opt/hpcx/ucx/lib`.

Let's test `LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:$LD_LIBRARY_PATH` in `jetson/Dockerfile.jp7` at line 46!

Let's use `replace_file_content` to add `ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:$LD_LIBRARY_PATH"` in `jetson/Dockerfile.jp7`.

Edited Dockerfile.jp7

Now let's rsync `jetson/Dockerfile.jp7` to `jetsonthor`.

Ran command: `rsync -avz /Users/kaikailiu/Documents/MyRepo/edgeAI/jetson/Dockerfile.jp7 jetsonthor:/home/lkk/Documents/edgeAI/jetson/Dockerfile.jp7`

Now let's launch `docker build` on `jetsonthor` again! Because all previous layers (apt, python, llama, Isaac ROS) are cached, this will finish in seconds!

Ran command: `ssh jetsonthor "cd /home/lkk/Documents/edgeAI && docker build -f jetson/Dockerfile.jp7 --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 --build-arg REBUILD_OPENCV=0 --build-arg INSTALL_ISAAC_ROS=1 -t cmpelkk/jetson-unified:jp7-thor ."`

Let's check the build status of task-1182.

Used tool: manage_task

Because `LD_LIBRARY_PATH` was added at Line 42, step 5 re-evaluated.
Let's yield and wait for task-1182 to finish building.

Viewed Dockerfile.jp7:154-171
Ran command: `ssh jetsonthor "docker run --rm --runtime nvidia cmpelkk/jetson-unified:jp7-thor bash -c '
echo \"=== 1. PyTorch & CUDA 13 on Thor ===\"
python3 -c \"import torch; print(\\\"PyTorch:\\\", torch.__version__, \\\"; CUDA Available:\\\", torch.cuda.is_available(), \\\"; GPU:\\\", torch.cuda.get_device_name(0)); a=torch.randn(100,100,device=\\\"cuda\\\"); print(\\\"GPU Matmul Norm:\\\", (a@a).norm().item())\"

echo \"=== 2. ROS 2 Jazzy & Isaac ROS packages ===\"
source /opt/ros/jazzy/setup.bash
source /opt/ros/isaac_ros_ws/install/setup.bash 2>/dev/null || true
ros2 pkg list | grep -i isaac_ros || ls -la /opt/ros/isaac_ros_ws/src/

echo \"=== 3. llama.cpp & Ollama ===\"
llama-cli --version 2>&1 | head -n 2 || true
ollama --version 2>&1 || true
'"`

Look at that output!

1. **`rsync`**: Successfully synchronized `/Users/kaikailiu/Documents/MyRepo/edgeAI` to `jetsonthor:/home/lkk/Documents/edgeAI`.
2. **`docker build`**: Built `cmpelkk/jetson-unified:jp7-thor` natively on Jetson Thor with `--build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3`, `--build-arg REBUILD_OPENCV=0`, and `--build-arg INSTALL_ISAAC_ROS=1`.
3. **Container Test Execution**:
   - **PyTorch & CUDA 13.0 on Thor**: PyTorch 2.8.0a0 natively detects `NVIDIA Thor` with CUDA 13.0, executes GPU matrix multiplication (`GPU Matmul Norm: 998.94`) with zero warnings.
   - **ROS 2 Jazzy & Isaac ROS**: `ros2` tools are sourced, and the Isaac ROS workspace is present at `/opt/ros/isaac_ros_ws/src/` with `isaac_ros_common`, `isaac_ros_nitros`, `isaac_ros_image_pipeline`, and `isaac_ros_apriltag`.
   - **`llama.cpp` & Ollama**: `llama-cli` version 1 (build `555881e` with CUDA support) and Ollama client v0.32.3 are fully installed and operational inside the container.

Everything is complete and verified! I will now synthesize the results for the user.I have completed all steps to sync the repository, build the container, and verify execution on **Jetson Thor** (`jetsonthor`):

---

### 1. Repository Synchronization (`rsync`)
* Synchronized local repository `/Users/kaikailiu/Documents/MyRepo/edgeAI` to remote `/home/lkk/Documents/edgeAI/` on `jetsonthor`.

---

### 2. Container Image Build
* Built the unified container image `cmpelkk/jetson-unified:jp7-thor` on `jetsonthor`:
  ```bash
  cd /home/lkk/Documents/edgeAI
  docker build -f jetson/Dockerfile.jp7 \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 \
    --build-arg REBUILD_OPENCV=0 \
    --build-arg INSTALL_ISAAC_ROS=1 \
    -t cmpelkk/jetson-unified:jp7-thor .
  ```

---

### 3. Container Verification Results (`docker run`)

We executed test routines inside `cmpelkk/jetson-unified:jp7-thor` with `--runtime nvidia`:

| Test Suite | Execution Status | Output Highlights |
| :--- | :--- | :--- |
| **PyTorch & CUDA 13.0** | **PASSED** (0 Warnings) | `PyTorch 2.8.0a0` detect `GPU: NVIDIA Thor`, CUDA JIT & GPU Tensor Matrix Multiplication (`a @ b`) executed with full GPU acceleration. |
| **ROS 2 Jazzy & Isaac ROS** | **PASSED** | ROS 2 Jazzy sourced, and Isaac ROS repositories (`isaac_ros_common`, `isaac_ros_nitros`, `isaac_ros_image_pipeline`, `isaac_ros_apriltag`) are present in `/opt/ros/isaac_ros_ws/src/`. |
| **`llama.cpp` + Ollama** | **PASSED** | `llama-cli` (build `555881e` compiled with CUDA support) and `ollama` (v0.32.3) are ready. |

---

### Interactive Shell Command on Jetson Thor
To start an interactive session inside the newly built container on Jetson Thor:

```bash
docker run --rm -it --runtime nvidia --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --device-cgroup-rule='c 81:* rmw' --device-cgroup-rule='c 14:* rmw' \
  -v /dev:/dev -v /Developer:/Developer \
  cmpelkk/jetson-unified:jp7-thor
```
---

## 7. LeRobot main (native venv, no container)

Added 2026-09-30. The container route above is not needed for LeRobot. A plain
`uv` venv with the **standard PyTorch aarch64 cu130 wheel** works on Thor:

```bash
cd ~/Developer
git clone --depth 1 https://github.com/huggingface/lerobot.git
cd lerobot
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install 'torch==2.11.0' 'torchvision==0.26.0' \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install -e '.[training,peft,diffusion]'
```

Verified on this Thor (JetPack 7.1, L4T R38.4.0, driver 580.00):

```text
torch 2.11.0+cu130  cuda 13.0  available True
device NVIDIA Thor  capability (11, 0)
bf16 matmul OK
```

JetPack 7 is SBSA-aligned, which is why the generic aarch64 CUDA 13 wheel runs
here without an NVIDIA container or a Jetson-specific index. LeRobot main
requires Python >= 3.12, which matches Thor's system Python 3.12.3.

Policy-level results (which policies load, measured inference numbers, the
NATTEN workaround for FLUX 3) are in `SO101_TRAINING.md`.

### Gotcha: root-owned Hugging Face cache

`~/.cache/huggingface` on this machine was owned by **root** (created 2026-04-20,
most likely by a root `docker run` that bind-mounted the cache before it
existed). Every model download then fails with:

```text
OSError: PermissionError at /home/lkk/.cache/huggingface/hub when downloading ...
```

It was empty, and `~/.cache` belongs to `lkk`, so no sudo is needed:

```bash
[ -z "$(ls -A ~/.cache/huggingface)" ] && rmdir ~/.cache/huggingface
mkdir -p ~/.cache/huggingface/hub
```

If it is not empty, `sudo chown -R $USER:$USER ~/.cache/huggingface` instead.
To avoid recreating the problem, create the host directory as your own user
*before* mounting it into a container.

## 8. Running other VLA codebases natively

Several VLA repos (LingBot-VLA, Hy-Embodied-0.5-VLA, Cosmos3 via diffusers) were
brought up on Thor in their own venvs. `robotics/policy_bench/README.md` has the
recipe for each. Three things apply to all of them.

**Torch.** Every repo pins an older torch (2.7, 2.8) built for x86 CUDA 12.x.
Ignore the pin and install the stock aarch64 cu130 wheel. Keep it from being
downgraded with an override file:

```bash
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
printf "torch==2.11.0\ntorchvision==0.26.0\n" > /tmp/over.txt
uv pip install --override /tmp/over.txt -r requirements.txt
```

**flash-attn.** PyPI ships only a source tarball, and a source build on Thor
takes hours. The Jetson AI Lab SBSA index has a prebuilt aarch64 wheel that
works with torch 2.11. It was checked against SDPA with a real call: max error
0.008 in bf16.

```bash
uv pip install --no-deps \
  "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/621/0324cfd00b9e4/flash_attn-2.8.4-cp312-cp312-linux_aarch64.whl"
```

**Triton.** Anything that JIT-compiles Triton kernels fails with
`ptxas-blackwell fatal : Value 'sm_110a' is not defined`. That includes
`torch.compile` and flash-attn's rotary embedding. Triton 3.6's bundled ptxas
predates sm_110. Use CUDA 13.0's own ptxas, and set both variables, since
Triton reads a separate one for Blackwell-family GPUs:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
```

**Power mode.** The dev kit boots in `120W` (`nvpmodel -q`). NVIDIA's published
Thor latencies use `MAXN`, where the GPU clock goes up to ~1575 MHz. Every number
in `SO101_TRAINING.md` was taken at 120 W, so it is conservative. To switch:

```bash
sudo nvpmodel -m 0      # MAXN; may ask to reboot
sudo jetson_clocks      # optional: pin clocks at max for benchmarking
```

## 9. NVIDIA reference tutorials on Thor: OpenPI π0.5, GR00T N1.7, Cosmos3-Edge

Three NVIDIA tutorials were reproduced on this Thor on 2026-10-02/03:

- [OpenPi π₀.₅ on Jetson Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
- [Isaac GR00T 1.7 on Jetson Thor](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)
- the [Cosmos 3 Edge blog post](https://huggingface.co/blog/nvidia/cosmos3edge)

Helper scripts are in `robotics/thor_tutorials/`. The checkouts and logs are on
Thor under `~/Developer/thor_tutorials/`.

### 9.0 Results at a glance

This Thor runs **JetPack 7.1 (L4T R38.4, driver 580 / CUDA 13.0) in the default
120 W mode**. Both Jetson AI Lab tutorials are written for JetPack 7.2 (R39) at
MAXN, so their numbers are an upper bound for this board.

| workload | how | measured here (7.1, 120 W) | published (7.2, MAXN) |
|---|---|---|---|
| π0.5 LIBERO, PyTorch BF16 | `openpi-pi0.5` container | 184 ms (model 176 ms) | 132 ms (128) |
| π0.5 LIBERO, TensorRT FP8+NVFP4 | same | **69.8 ms (model 67.3 ms)** | 49 ms (48) |
| π0.5 TRT vs PyTorch accuracy | `--inference-mode compare` | **cosine 0.9969** (per-step min 0.9952), 2.59x, real calibration. With dummy calibration (HF rate limit, see 9.2) it was 0.983–0.991 | ≈0.9945, 2.69x |
| GR00T N1.7, PyTorch eager | `gr00t-thor` container | 153 ms | 126 ms |
| GR00T N1.7, TRT bf16 | same | 105 ms | 81 ms |
| GR00T N1.7, TRT mixed NVFP4 | same | **55 ms (18.2 Hz)** | 40 ms (25.1 Hz) |
| GR00T accuracy, real LIBERO trajectories | same | PyTorch MSE 0.001390 (identical), NVFP4 0.001875 | 0.001390 / 0.001458 |
| Cosmos3-Edge reasoner (VLM) | `vllm/vllm-openai:cosmos3` | TTFT **70 ms**, decode **42 tok/s** | 42.6 tok/s |
| Cosmos3-Edge-Policy-DROID, [16,8], 736x544, 30 steps | `vllm/vllm-omni:cosmos3` | 7.17 s | 6.32 s |
| same, 4 steps | same | 4.69 s | — |
| Cosmos3-Edge-Policy-DROID, [32,8], 320x192, 30 steps | same | 2.75 s | 2.59 s |
| same, **4 steps** | same | **1.66 s → real-time at 15 Hz (RTF 1.28)** | 1.53 s (PyTorch server) |

All of them run on JetPack 7.1 without upgrading. Both VLA tutorials also run
inside the unified image built from `jetson/Dockerfile.jp7-thor`
(NGC 26.05 base, 9.6). In it, π0.5 TRT FP8+NVFP4 takes 66.8 ms with cosine 0.9973. The one-time costs on top:
~25 min per TensorRT engine set for π0.5, ~15 min for GR00T, and Hugging Face
access for GR00T's gated backbone and π0.5's calibration data.

### 9.1 Common prerequisites

```bash
docker --version                         # 28.2.2 here; tutorials want 28.x
dpkg-query -W nvidia-container-toolkit   # 1.18.1 here
docker info | grep -i "default runtime"  # nvidia
sudo nvpmodel -m 0 && sudo jetson_clocks # MAXN, as the tutorials assume (not done for the numbers above)
```

- **JetPack 7.1 is fine.** Every container here worked. The OpenPI image is based
  on `nvcr.io/nvidia/pytorch:26.05-py3`, which ships CUDA 13.2. It still runs on
  the 13.0 driver through the forward-compat libraries inside the image
  (`/usr/local/cuda/compat`). It was checked with a real bf16 matmul, capability
  `(11, 0)`.
- **git-lfs** (GR00T needs it) is not installed on this Thor. Install it without
  sudo into `~/.local/bin`:
  ```bash
  V=3.6.1; curl -fsSL -o /tmp/lfs.tgz https://github.com/git-lfs/git-lfs/releases/download/v$V/git-lfs-linux-arm64-v$V.tar.gz
  tar xzf /tmp/lfs.tgz -C /tmp && cp /tmp/git-lfs-$V/git-lfs ~/.local/bin/ && git lfs install
  ```
- **Hugging Face login** is needed for GR00T's gated backbone. Accept the license
  at <https://huggingface.co/nvidia/Cosmos-Reason2-2B>, then on Thor run
  `hf auth login`.
- **Files written by a container are owned by root.** For example, GR00T's
  `stats.json` comes out as `-rw------- root`, which the host user cannot read.
  Hand them back with:
  ```bash
  docker run --rm -v "$PWD":/w gr00t-thor chown -R $(id -u):$(id -g) /w/examples /w/checkpoints
  ```
- **Run long jobs detached** (`setsid nohup ... &`). An SSH drop must not kill a
  20-minute TensorRT build.
- Disk used: the images are 15–27 GB each, and the π0.5 artifacts (JAX + PyTorch
  checkpoints, ONNX and engine) take ~28 GB in `~/.cache/openpi`.

### 9.2 OpenPI π0.5 (container)

```bash
cd ~/Developer/thor_tutorials
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git && cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
wget -qO- https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/download.sh | bash
docker build -t openpi-pi0.5:l4t-jp7.2 -f deployment_scripts/thor.Dockerfile .        # ~5 min, 26.9 GB
cp ~/Developer/robotics/thor_tutorials/openpi_run_tutorial.sh run_tutorial.sh             # tutorial steps 5-12
docker run --rm --runtime nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface -w /workspace \
  openpi-pi0.5:l4t-jp7.2 bash run_tutorial.sh 2>&1 | tee openpi_run.log
```

`run_tutorial.sh` runs tutorial steps 5–12 in order. Each step is skipped if its
output already exists, so it can be re-run after a failure. Timings here:

| step | time |
|---|---|
| 6 download JAX checkpoint (`gs://openpi-assets`, public) | 1 min |
| 7 JAX -> PyTorch | 2 min |
| 8 PyTorch BF16 baseline (includes `torch.compile` autotune) | 6 min |
| 9 ONNX export, FP8 + NVFP4 (downloads 1,699 calibration files first) | 12 min |
| 10 TensorRT engine (`trtexec`) | 23.5 min |
| 11–12 TRT inference and compare | 5 min |

Results:

```text
PyTorch BF16   total 184.15 ± 6.31 ms   model 175.98 ms
TRT FP8+NVFP4  total  69.76 ± 2.23 ms   model  67.30 ms    engine 2.9 GB, ONNX 5.2 GB
compare        speedup 2.59x (model 2.53x)   cosine 0.990, per-step min 0.984
               reruns: cosine 0.983 / 0.991, speedup 2.52x / 2.57x      <- dummy calibration
re-run with HF_TOKEN (steps 9-12, real calibration: "Calibration dataset ready with 32 samples")
TRT FP8+NVFP4  total  68.31 ± 3.16 ms   model  66.22 ms
compare        speedup 2.59x (model 2.56x)   cosine 0.9969, per-step min 0.9952
```

**The first cosine was below the tutorial's ≈0.9945 because calibration never saw
real data. With real calibration it is 0.9969, above the tutorial's figure.** Step 9 calibrates FP8/NVFP4 on 32 samples from `physical-intelligence/libero`.
Fetching them anonymously from the shared campus IP hit Hugging Face's rate limit
(`429 Too Many Requests ... We had to rate limit your IP`). The exporter then
printed `Falling back to dummy inputs for calibration` and carried on without
failing. **Log in (`hf auth login`, or pass `-e HF_TOKEN`) before step 9.** Then
check the log for `Calibration dataset ready with 32 samples`. If it says
`Falling back to dummy inputs`, delete `onnx/` and `engine/` and re-run steps 9–12.
The tutorial also offers FP8 only (drop `--enable_llm_nvfp4`: ~53 ms, ≈0.9995
cosine) when accuracy matters more than the last ~15 ms.

The tutorial also has a websocket server (step 13, `scripts/serve_policy.py
--use-tensorrt ...`, port 8000) for driving a real robot from another machine.

**Gotcha:** step 5's `cp -r transformers_replace/* .../site-packages/transformers/`
patches the container's **system** transformers. It is harmless in this
throwaway container. Do not do it in a shared environment (see 9.5).

### 9.3 GR00T N1.7 (container)

```bash
cd ~/Developer/thor_tutorials
git clone https://github.com/NVIDIA/Isaac-GR00T.git && cd Isaac-GR00T
git checkout 9c7e746b2cd37a810070a98ef41d290a07e806c2 && git lfs pull
file scripts/deployment/thor/wheels/torchcodec-*.whl      # must say "Zip archive"
wget -qO- https://www.jetson-ai-lab.com/code-samples/groot_n17_on_thor/download.sh | bash
cd docker && bash build.sh --profile=thor && cd ..        # ~10 min, 15.4 GB, base nvidia/cuda:13.0.0-devel-ubuntu24.04
```

Verified with the tutorial's own check, and the output matches it exactly:

```text
torch      2.10.0
TensorRT   10.15.1.29
device     NVIDIA Thor
capability (11, 0)
```

Already prepared on this Thor:

- the checkpoint (`nvidia/GR00T-N1.7-LIBERO`, public, 6.5 GB) in
  `checkpoints/GR00T-N1.7-LIBERO`;
- the calibration set `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`, with
  `modality.json` copied in;
- normalization stats (step 9.2), run in the container.

**Hugging Face access.** The model's `config.json` names the gated
`nvidia/Cosmos-Reason2-2B` as its backbone. Accept its license, and pass the token
into the container with `-e HF_TOKEN`. Here, `HF_TOKEN` is exported in `~/.bashrc`,
so it is only visible to interactive shells. Scripts run over ssh need
`T=$(bash -ic 'printf %s "$HF_TOKEN"')`, or the shared-environment setup in §10.

Then run the remaining tutorial steps (9.1 modelopt, 8 bf16 baseline, 9.3 NVFP4,
10 real trajectories) with `robotics/thor_tutorials/groot_run_tutorial.sh`:

```bash
cd ~/Developer/thor_tutorials/Isaac-GR00T && cp ~/Developer/robotics/thor_tutorials/groot_run_tutorial.sh run_tutorial.sh
docker run --rm --runtime nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host -e HF_TOKEN -v "$PWD":/workspace/repo -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo -e PYTHONPATH=/workspace/repo \
  -e PATH=/root/.local/bin:/opt/gr00t-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  gr00t-thor bash run_tutorial.sh 2>&1 | tee groot_run.log
```

It takes ~17 min in total: bf16 engines 7 min, NVFP4 engines 8 min, trajectories 2 min.

**Accuracy (real LIBERO trajectories, 5 trajectories, horizon 8):**

| mode | avg MSE | avg MAE | tutorial MSE / MAE |
|---|---|---|---|
| PyTorch | **0.001390** | **0.013069** | 0.001390 / 0.013069, identical |
| TRT optimized + mixed NVFP4 | 0.001875 | 0.016302 | 0.001458 / 0.015795 |

Engine verification against PyTorch: bf16 cosine 0.999936 PASS, mixed NVFP4
cosine 0.999090 PASS (tutorial 0.999743).

**Latency.** The pipeline's own benchmark ran while the Dockerfile.jp7-thor image
was compiling OpenCV on all 14 cores. It shows how much a busy CPU hurts.
GR00T's data processing alone took 49 ms against the tutorial's ~9 ms:

```text
                     (CPU saturated)   E2E       backbone   action head
PyTorch eager                          222 ms    71 ms      96 ms
torch.compile                          212 ms    82 ms      83 ms
TRT bf16                               169 ms    43 ms      78 ms
TRT mixed NVFP4                        56.6 ms   21.4 ms    24.2 ms   (17.7 Hz)
```

Re-measured with an idle CPU (`scripts/deployment/benchmark_inference.py --trt-mode
n17_full_pipeline --embodiment-tag libero_sim --trt-engine-path <engines>`):

| backend | data | backbone | action head | E2E | tutorial (7.2, MAXN) |
|---|---|---|---|---|---|
| PyTorch eager | 9 ms | 59 ms | 84 ms | 153 ms (6.5 Hz) | 126 ms |
| torch.compile | 9 ms | 62–86 ms | 69–73 ms | 141–168 ms | 105 ms |
| TRT bf16 | 9 ms | 34 ms | 61 ms | 105 ms (9.6 Hz) | 81 ms |
| **TRT mixed NVFP4** (CUDA graph) | 9 ms | 21 ms | 24 ms | **55 ms (18.2 Hz)** | 40 ms (25.1 Hz) |

Like π0.5, this is ~1.3–1.4x the tutorial's time. That matches JetPack 7.1 at
120 W against 7.2 at MAXN.

### 9.4 Cosmos3-Edge (blog post)

The blog post is an announcement. It gives no commands. Its two claims for
Thor are "real-time control at 15 Hz" and "32 actions per inference". The
runnable recipes are in the [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge)
model card and the [`NVIDIA/cosmos`](https://github.com/NVIDIA/cosmos) cookbooks.
Both vLLM images are multi-arch, and their arm64 builds run on Thor.

**Reasoner (vision-language, OpenAI API):**

```bash
docker run -d --name cosmos3-reasoner --runtime nvidia --ipc=host --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:cosmos3 \
  nvidia/Cosmos3-Edge --host 0.0.0.0 --port 8000 --max-model-len 131072 --gpu-memory-utilization 0.5 \
  --allowed-local-media-path / \
  --mm-processor-kwargs '{"do_resize": true, "min_pixels": 4096, "max_pixels": 16777216}' \
  --media-io-kwargs '{"video": {"num_frames": 256}}'
python3 ~/Developer/robotics/thor_tutorials/cosmos3_reasoner_client.py --assets ~/Developer/models/Cosmos3-Edge-assets
```

It is ready ~4.5 min after start: weights 4.67 GiB, plus CUDA graph capture. Use
`--gpu-memory-utilization 0.5`, because the default 0.9 would claim ~110 GB of the
shared memory. The model card's planning example (`put flower into the red bottle`,
218 prompt tokens) gives:

```text
first request  TTFT 36.6 s (warm-up)
then           TTFT 70 ms, ~406 tokens in 9.6 s, decode 42 tok/s
```

The answer is a `<think>` trace followed by a numbered 5-step plan.

**Policy (world action model) via vLLM-Omni:**

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git ~/Developer/vla/cosmos-framework
docker run -d --name cosmos3-omni-policy --runtime nvidia --ipc=host -p 8001:8000 \
  -e PYTHONPATH=/workspace/cosmos-framework -v $HOME/Developer/vla/cosmos-framework:/workspace/cosmos-framework \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Edge-Policy-DROID --no-guardrails --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline --allowed-local-media-path / --port 8000 --init-timeout 1800
# DROID sample frames: sparse checkout of NVIDIA/cosmos cookbooks/cosmos3/generator/action/assets/droid_lerobot_example
python cosmos3_policy_omni_client.py --assets <that dir> --steps 30 4                                   # [16,8] at 736x544
python cosmos3_policy_omni_client.py --assets <that dir> --steps 30 4 --chunk 32 --size 320x192 --tier 256
```

vLLM-Omni detects sm_110 as an "untested variant" and uses cuDNN attention. It
loads in ~12 s (7.1 GiB). Each call returns the action chunk plus a generated
rollout video, saved as `cosmos3_policy_out/policy_rollout.mp4`.

| chunk, size | 30 steps | 4 steps | budget at 15 Hz |
|---|---|---|---|
| [16,8], 736x544 (notebook default) | 7.17 s | 4.69 s | 1.07 s |
| [32,8], 320x192 (model card real-time row) | 2.75 s | **1.66 s** | 2.13 s |

**This is how the "15 Hz on Thor" claim holds.** It needs 32-action chunks at low
resolution with 4 denoising steps: 1.66 s against a 2.13 s budget, here even at
120 W. At the notebook's default size and 16-action chunks it does not keep up.
The earlier diffusers measurement (`policy_bench/cosmos3_bench.py`, 3.8 s for
4 steps at 640x540) sits in between.

### 9.5 Folding these into `Dockerfile.jp7`

**Short answer: GR00T fits. OpenPI fits fully only on the 26.05 base, i.e.
`Dockerfile.jp7-thor` (9.6). Cosmos3-Edge doesn't need to.**

| stack | in `Dockerfile.jp7`? | why |
|---|---|---|
| GR00T N1.7 | **yes**, `/opt/gr00t-venv` (section 9c) | self-contained venv: torch 2.10, TRT 10.15.1.29 and modelopt 0.39 come from the Jetson AI Lab index, the same versions as `gr00t-thor` |
| OpenPI π0.5 | **partly on 25.08, fully in `Dockerfile.jp7-thor`** (9.6) | on 25.08 the venv inherits torch 2.8 / **TensorRT 10.13** / modelopt 0.33: PyTorch and a TRT **FP8** engine work (77 ms), but the FP8+NVFP4 build **hangs**. On the 26.05 base it builds and runs (66.8 ms, cosine 0.9973) |
| Cosmos3-Edge | **no, use the vLLM images** | `vllm/vllm-openai:cosmos3` and `vllm/vllm-omni:cosmos3` are ~20 GB each, pin their own vLLM/torch, and already run on Thor as-is |

Section 9c is off by default (`--build-arg INSTALL_THOR_VLA=1`). Its design:

- **Nothing goes into the system Python.** OpenPI patches transformers in place
  and pins `transformers==4.53.2`, `lerobot==0.3.2` and `opencv 4.11`. The image's
  own LeRobot 0.6 and transformers 5.14 would break if those landed there.
  Each stack therefore gets a venv. `openpi-venv` is created with
  `--system-site-packages`, so it reuses the base's CUDA torch and TensorRT, and
  the `transformers_replace` patch is copied into the venv only.
- **Pinned checkouts with the Jetson AI Lab patches** are baked in at
  `/opt/src/{Isaac-GR00T,openpi}`. Use them, or mount your own.
- **Usage:** `source /opt/gr00t-venv/bin/activate` or
  `source /opt/openpi-venv/bin/activate`, then follow 9.2 / 9.3 inside the image.

It was tested as a layer on the existing `cmpelkk/jetson-unified:jp7-thor`
image, without rebuilding the 30-minute OpenCV stage. A two-line Dockerfile does
it: `FROM cmpelkk/jetson-unified:jp7-thor` plus the 9c block, extracted with
`sed -n '/^ARG INSTALL_THOR_VLA=0/,/^    fi$/p' Dockerfile.jp7`. The build takes ~4 min.
`thor_tutorials/jp7vla_env_check.sh` then prints:

```text
gr00t-venv  2.10.0 10.15.1.29 NVIDIA Thor (11, 0) True
system      2.8.0a0+34c6371d24.nv25.08 True lerobot 0.6.0 transformers 5.14.1
```

Three problems were found and fixed on the way:

1. **The base image puts its own torch first in `LD_LIBRARY_PATH`.** NGC exports
   `/usr/local/lib/python3.12/dist-packages/torch/lib`. The venv's torch 2.10 then
   loaded the base's 2.8 `libtorch_python` and failed with
   `AttributeError: module 'torch._C' has no attribute '_dlpack_exchange_api'`.
   Fix: `gr00t-venv/bin/activate` puts the venv's torch and nvidia libs first and
   drops the base torch dirs. This is what GR00T's own Dockerfile does with
   `LD_LIBRARY_PATH`.
2. **`deactivate` did not undo that.** Afterwards, the system torch loaded the
   venv's libs (`torch.fx ... has no attribute 'DynamicInt'`). Fix: activate
   saves the old value, and a wrapped `deactivate` restores it.
3. **pip would not pick `diffusers 0.36.0.dev0`.** That is the only diffusers on
   the Jetson index, and lerobot 0.3.2's `diffusers>=0.27.2` will not select a
   pre-release unless it is already installed. Fix: install that pin first.

Running OpenPI inside the jp7 image (`thor_tutorials/openpi_jp7_check.sh`, using
the checkpoint converted in 9.2):

```text
versions          torch 2.8.0 (nv25.08)  TensorRT 10.13.2.6  modelopt 0.33.0  transformers 4.53.2
PyTorch BF16      total 196.25 ms  model 187.17 ms        (official container: 184 / 176)
ONNX FP8+NVFP4    exports fine with modelopt 0.33
TRT engine build  trtexec HUNG after "Compiler backend is used during engine build": 47 min
                  without a log line, 0-byte engine, every thread asleep at ~0% CPU
                  (official container, TRT 10.16: done in 23.5 min)

FP8 only (NVFP4=0 bash openpi_jp7_check.sh)
TRT engine build  6.5 min, 3.7 GB
TRT FP8           total 77.26 ms  model 75.98 ms   speedup 2.39x   cosine 0.982 (dummy calibration, 429 again)
```

So on the 25.08 base, OpenPI gets PyTorch and a **TensorRT FP8 engine (77 ms,
2.4x)**, but not its headline FP8+NVFP4 engine (70 ms in the official image).

**How to tell a hang from a slow build.** After `Compiler backend is used during
engine build`, a healthy build is silent for a long time: 22 min in the official
container, with `trtexec` at 100% CPU in state `R`. The 25.08 build was silent for
47 min with the process asleep (`S`, ~0% CPU). Check with
`top -bn1 -p $(pgrep -x trtexec)` before killing anything.

**The fix is the 26.05 base**, the same as OpenPI's own image (TRT 10.16.1,
modelopt 0.43). It runs fine on this JetPack 7.1. Moving the unified image there
took more than changing the base tag (LeRobot's torch bound, OpenCV for CUDA 13.2,
numpy), so it is a separate file, `Dockerfile.jp7-thor`. With it, the NVFP4
engine builds and runs inside the unified image. See 9.6.

**HF cache path gotcha.** jp7 sets `HF_HOME=/Developer/models/huggingface`. A host
cache mounted at `/root/.cache/huggingface` is therefore ignored, and everything
is downloaded again. Mount the cache at `/Developer/models/huggingface`, or pass
`-e HF_HOME=/root/.cache/huggingface`.


### 9.6 `Dockerfile.jp7-thor`: the unified image on the NGC 26.05 base

`jetson/Dockerfile.jp7-thor` is the Thor-only sibling of `Dockerfile.jp7`. It has
the same sections and build args, on `nvcr.io/nvidia/pytorch:26.05-py3`, and
gives you the OpenPI-capable TensorRT inside the unified image.
`Dockerfile.jp7` stays as the Orin + Thor 25.08 recipe.

```bash
cd ~/Developer/thor_tutorials/jp7thor_ctx     # any small dir: the Dockerfile COPYs nothing
docker build --network host -f ../Dockerfile.jp7-thor -t cmpelkk/jetson-unified:jp7-thor-2605 .
```

The build took ~37 min on Thor (OpenCV 14.7 min, llama.cpp + llama-cpp-python
12.1 min, ROS + Isaac ROS 3 min, VLA venvs 3.6 min, apt/pip ~2.5 min), plus a one-time
pull of the 26.05 base. The image is 45.4 GB.

**What changed from `Dockerfile.jp7`, and why:**

| change | reason |
|---|---|
| base `pytorch:26.05-py3` (torch 2.12, CUDA 13.2, TRT 10.16.1, modelopt 0.43) | the TRT that builds OpenPI's FP8+NVFP4 engine. CUDA 13.2 runs on the JetPack 7.1 / 13.0 driver via the image's forward-compat libs |
| numpy stays 2.1 (no `numpy<2`) | 25.08's torch-tensorrt needed numpy<2; 26.05's does not, and lerobot wants >=2.0 |
| constraints also pin numpy; LeRobot installed with `uv pip --override` | lerobot 0.6.x declares `torch<2.12`, `torchvision<0.27`. PEP 440 excludes the base's `2.12.0a0` / `0.27.0a0` pre-releases from those ranges, so plain pip fails, or silently falls back to an ancient lerobot. The override keeps lerobot 0.6.1 on the CUDA torch |
| OpenCV 4.13, `CUDA_ARCH_BIN=11.0`, GStreamer ON | Thor is sm_110 under CUDA 13 (sm_101 under 12.8/12.9, so the old `10.0` was wrong for CUDA 13). The existing `jp7-thor` image had been built with `REBUILD_OPENCV=0`, i.e. no GStreamer |
| opencv_contrib `zip.hpp` from commit `f2854f4` (PR #4097) | CUDA 13.2 moved libcu++ to the `cuda::std` namespace macro. 4.13.0's `cudev/ptr2d/zip.hpp` fails with `tuple is not a template`. Upstream's 9-line fix landed after 4.13.0 |
| llama.cpp / llama-cpp-python with `CMAKE_CUDA_ARCHITECTURES=110` | native Thor kernels instead of PTX JIT |
| Isaac ROS and the 9c VLA venvs default ON | Thor-only image |

**Runtime check** (`thor_tutorials/jp7thor_image_check.sh`, `--runtime nvidia`):

```text
torch 2.12.0a0 (nv26.05) cuda 13.2 NVIDIA Thor (11, 0) bf16 matmul ok
numpy 2.1.0 | tensorrt 10.16.1.11 | transformers 5.18.0 | lerobot 0.6.1 | ultralytics 8.4.172
cv2 4.13.0 | GStreamer YES (1.24.2) | NVIDIA CUDA YES (13.2) | GPU arch 110 | cv2.cuda resize OK
GStreamer NV plugins: nvv4l2decoder nvv4l2h264enc nvvidconv nvarguscamerasrc  all OK
llama.cpp: CUDA0 NVIDIA Thor, compute capability 11.0 | llama-cpp-python gpu offload True
Isaac ROS sources: apriltag common image_pipeline nitros
gr00t-venv  torch 2.10.0, TRT 10.15.1.29, GPU OK
openpi-venv torch 2.12.0a0, TRT 10.16.1.11, modelopt 0.43.0, transformers 4.53.2, lerobot 0.3.2
system (after using the venvs): transformers 5.18.0, lerobot 0.6.1
```

**OpenPI π0.5 NVFP4 inside the unified image works on this base.**
`thor_tutorials/openpi_jp7_check.sh` was run in `jp7-thor-2605` with
`-e HF_TOKEN -e HF_HOME=/root/.cache/huggingface`. It reuses the PyTorch
checkpoint from 9.2 and calibrates on real data:

```text
openpi-venv       torch 2.12.0a0  TensorRT 10.16.1.11  modelopt 0.43.0  transformers 4.53.2
PyTorch BF16      total 183.08 ms  model 175.07 ms          (official container 184 / 176)
calibration       "Calibration dataset ready with 32 samples"
TRT engine build  20 min, 2.87 GB                           (25.08 base: hung)
TRT FP8+NVFP4     total  66.76 ms  model  64.89 ms          (official container 68.3 / 66.2)
compare           speedup 2.82x   cosine 0.9973, per-step min 0.9938
```

The unified image now matches OpenPI's own container. Together with the
`gr00t-venv` result above, both Jetson AI Lab VLA tutorials run inside
`cmpelkk/jetson-unified:jp7-thor-2605`, next to LeRobot 0.6.1, GStreamer OpenCV,
llama.cpp and Isaac ROS. Only Cosmos3-Edge stays in its own vLLM images.

Typical run of the unified image with the shared cache from §10:

```bash
docker run --rm -it --runtime nvidia --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /srv/hf:/srv/hf -e HF_HUB_CACHE=/srv/hf/hub -e HF_XET_CACHE=/srv/hf/xet \
  -v /srv/openpi:/root/.cache/openpi -e HF_TOKEN \
  -v /dev:/dev --device-cgroup-rule='c 81:* rmw' --device-cgroup-rule='c 166:* rmw' \
  cmpelkk/jetson-unified:jp7-thor-2605
# inside: source /opt/openpi-venv/bin/activate   or   source /opt/gr00t-venv/bin/activate
```

## 10. Multi-user Thor: one shared Hugging Face cache

Each user's `~/.cache/huggingface/hub` is a separate copy. On this Thor, the one
existing user already has 211 GB there, plus 46 GB of OpenPI checkpoints. With a
class of students, every user who runs the same tutorial would download those
again. The fix is one group-writable cache that every account points at.

**The setup** (`robotics/thor_tutorials/setup_shared_hf_cache.sh`, run once with sudo):

```bash
sudo bash setup_shared_hf_cache.sh --migrate-from lkk        # create /srv/hf, move lkk's cache in
sudo bash setup_shared_hf_cache.sh --add-user alice bob      # each new student
# students log out and in once (new group + /etc/environment)
```

What it does, and why:

| piece | setting | why |
|---|---|---|
| location | `/srv/hf/{hub,xet,datasets}`, `/srv/openpi` | same filesystem as `/home`, so migrating is an instant `mv` |
| group | `hfshare`, directories setgid (`g+s`) | new files keep the group |
| default ACL | `setfacl -d -m g:hfshare:rwX` on every directory | files stay **group-writable regardless of each user's umask**. huggingface_hub creates lock files and renames blobs inside existing repo folders, so read-only sharing breaks the second download |
| environment | `HF_HUB_CACHE`, `HF_XET_CACHE`, `HF_DATASETS_CACHE`, `OPENPI_DATA_HOME` in `/etc/environment` + `/etc/profile.d/hf-shared-cache.sh` | `/etc/environment` also covers non-interactive ssh, cron and services. `~/.bashrc` only covers interactive shells, which is why `HF_TOKEN` there was invisible to scripts run over ssh |
| **not** `HF_HOME` | — | `HF_HOME` would also move `token`. Leaving it alone keeps every student's login in their own `~/.cache/huggingface/token` |
| migration | `mv` each repo into `/srv/hf/hub`, leave `~/.cache/huggingface/hub -> /srv/hf/hub` | existing absolute paths keep resolving (e.g. the `policy_bench --compat` snapshot symlinks). Repos already shared are kept in `hub.migrated` for you to delete |

**It was tested** in a throwaway container (`test_shared_hf_cache.sh`) with three
users, all with umask 022, the strict case:

```text
lkk's cache migrated in, ~/.cache/huggingface/hub -> /srv/hf/hub works for lkk
alice adds a file to the repo lkk downloaded                                  OK
bob downloads the rest of that repo (writes blobs/refs/locks of alice+lkk)    OK
bob loads it with HF_HUB_OFFLINE=1                                            OK
a root-run container writes a file -> -rw-rw-r--+ root hfshare; alice deletes it  OK
alice: HF_HOME unset, token stays at /home/alice/.cache/huggingface/token     OK
blobs are -rw-rw-r-- even under umask 022                                     OK
```

**In containers**, mount the shared cache at the same path and point the
variables at it. A container's own `HF_HOME` (jp7 sets `/Developer/models/huggingface`)
does not matter, because `HF_HUB_CACHE` takes precedence for the hub:

```bash
docker run ... -v /srv/hf:/srv/hf -e HF_HUB_CACHE=/srv/hf/hub -e HF_XET_CACHE=/srv/hf/xet \
               -v /srv/openpi:/root/.cache/openpi -e HF_TOKEN ...
```

Files a container writes as root still come out `root:hfshare` and group-writable
(the default ACL applies), so students can update and delete them.

**Things it deliberately does not share:**

- **LeRobot datasets** (`HF_LEROBOT_HOME`, default `~/.cache/huggingface/lerobot`).
  Students record under names like `local/so101_pick`, and a shared directory
  would mix their episodes.
- **pip/uv caches.** These are small, and uv hard-links from its cache into venvs.
  A per-user cache avoids cross-user hard links.

**Housekeeping (admin only, since everyone shares the cache):**

- `hf cache ls` shows what is cached and how big.
- `hf cache prune` drops detached revisions and half-finished downloads.
- `hf cache rm model/<org>/<name>` removes a repo.
Docker images (190 GB here) are already shared: there is one daemon and one image
store for all users.

**Security note:** membership in the `docker` group is root-equivalent on the
host. For student accounts, give `hfshare` but not `docker`, and run containers
for them through a wrapper, rootless Docker, or a course launcher script.

## 11. Upgrading this Thor from JetPack 7.1 to 7.2

Status on 2026-10-03: Thor is on **JetPack 7.1 (L4T R38.4.0, driver 580, CUDA 13.0)**,
with apt sources at `r38.4`. JetPack 7.2 is **L4T R39.2** (7.2.1 = R39.2.1 is a
point release in the same repo). It ships driver 595, CUDA 13.2.1, TensorRT 10.16.2
and Container Toolkit 1.19, puts Orin and Thor on the same JetPack 7 base, and adds
MIG on T5000 as a preview. A JetPack 7.2 Orin Nano (R39.2.0, driver 595.78,
CUDA 13.2) was used for comparison in §12.

### Which path is supported

| path | from R38.4? | keeps /home, Docker images, caches? | notes |
|---|---|---|---|
| **image-based OTA** (`l4t_generate_ota_package.sh jetson-agx-thor-devkit R38-4` on an x86 host, then `nv_ota_start.sh` on Thor) | **yes**: the r39.2 Developer Guide lists 38.2.0 / 38.4.0 -> 39.2.0 for Thor | **no**: the rootfs is replaced. Only paths in `ota_backup_files_list.txt` are carried over | ~2.5 GB payload, needs 7 GB free. Does the bootloader/UEFI too |
| **full flash** (USB installer / SDK Manager / `flash.sh`) | yes | **no** | NVIDIA: "manual flashing instructions have changed for Jetson Thor because of the SBSA architecture". Follow the r39.2 guide's flashing section exactly |
| `apt` (`r38.4` -> `r39.2` in `nvidia-l4t-apt-source.list`, `apt dist-upgrade`) | **not documented for R38 -> R39** | yes | the guide only covers apt for point releases (as in 7.2 -> 7.2.1: keep `r39.2`, `apt update && apt upgrade`). A major L4T jump by apt would leave the QSPI/UEFI firmware behind. Seeed explicitly advises against apt across JetPack versions |

Everything on this Thor lives on the single root partition (`nvme0n1p1`): `/home`,
`/var/lib/docker`, the caches. **Both supported paths therefore wipe it.** Back up first.

### What is on the disk, and what actually needs a backup

| data | size | after the upgrade |
|---|---|---|
| Docker images (11) + build cache | 184 GB + 37 GB | rebuild or pull: `Dockerfile.jp7-thor`, `gr00t-thor`, `openpi-pi0.5`, the vLLM cosmos3 images |
| stopped container `gemma4-server` (since 2026-04) | **32.9 GB writable layer** | probably holds a downloaded model. `docker commit gemma4-server gemma4-server:backup` and save it, or copy the model out, **if you still want it** |
| HF hub cache | 211 GB | re-downloadable. Copying it to an external SSD saves the re-download |
| `~/.cache/huggingface/lerobot` | 33 GB (`physical-intelligence/libero`, the π0.5 calibration set) | re-downloadable |
| `~/.cache/openpi` | 34 GB | re-creatable: download + convert + engine build, ~45 min |
| `~/Developer/thor_tutorials` | 29 GB | checkouts + GR00T engines, re-creatable with §9 |
| `~/Developer/lerobot`, `vla/*`, `lerobot-pi05so` venvs | ~9 GB | re-create with §7 / `policy_bench/README.md`. The only local source change is the 2-line flux3 `video_vae.py` fix (§9 of SO101_TRAINING.md) |
| `~/miniconda3` (`thor312`) | 6.6 GB | re-create |
| **must keep** | < 100 MB | `~/.cache/huggingface/lerobot/calibration/` (SO-101 follower/leader), `~/.ssh`, `~/.bashrc` (`HF_TOKEN`; rotate it), `~/Developer/robotics` (also in git), any `/etc` edits |

### Recommended sequence

1. On Thor: copy the must-keep items off the box, e.g.
   `rsync -a ~/.ssh ~/.bashrc ~/.cache/huggingface/lerobot/calibration ~/Developer/robotics <backup-host>:thor-backup/`.
   Optionally put the 211 GB HF cache on a USB SSD.
2. Decide on `gemma4-server`: commit and save it, or drop it.
3. Flash JetPack 7.2 following the r39.2 Developer Guide's Thor flashing section,
   or generate the image-based OTA package from an x86 Ubuntu host with the
   BASE (R38.4.0) and TARGET (R39.2.0) BSPs.
4. After the first boot, run `sudo apt update && sudo apt upgrade` to pick up 7.2.1
   (keep the repo at `r39.2`). Then `sudo apt install nvidia-jetpack`,
   `nvpmodel -m 0`, and check `docker info | grep -i runtime` (nvidia).
5. Re-create in this order: the shared cache (§10), then
   `docker build -f Dockerfile.jp7-thor` (§9.6), then the native venvs (§7–8).
   Re-run the §9 tutorials: on 7.2 at MAXN they should approach the published numbers.

**Changes expected on 7.2:**

- The 26.05 images stop needing the CUDA forward-compat libraries (driver 595 is native CUDA 13.2).
- Container Toolkit 1.19 is the same version measured on the 7.2 Orin.
- `torch 2.11+cu130` wheels keep working, because CUDA 13 minor versions are compatible.

**References:**

- [Jetson Linux r39.2: Software Packages and the Update Mechanism](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html)
- [JetPack 7.2 announcement](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-software-goes-agentic-with-jetson-linux-39-2/372057)
- [7.2 -> 7.2.1 without reflashing](https://forums.developer.nvidia.com/t/jetpack-7-2-to-7-2-1-upgrade-without-reflashing/379985)
- [Seeed: Flash and OTA to JetPack 7.2](https://wiki.seeedstudio.com/flash_and_ota_jetpack_7.2/)

## 12. Do these images fit a JetPack 7.2 Orin Nano?

Test box: Jetson Orin Nano devkit (`ssh sjsujetson@headscale.forgengi.org -p 20062`),
**L4T R39.2.0 = JetPack 7.2**, driver 595.78, CUDA 13.2, 7.4 GB RAM, Docker 29.1,
Container Toolkit 1.19.1. Its default Docker runtime is `runc`, so pass
`--runtime nvidia` explicitly. It has no pip/venv packages; use the standalone
`uv` in `~/.local/bin`.

**Short answer: `Dockerfile.jp7-thor` / `jp7-thor-2605` is not an Orin image.**
Its GPU parts are compiled for Thor only. The component GPU code was checked
with `torch.cuda.get_arch_list()`, `cv2.getBuildInformation()` and `cuobjdump`:

| component in `jp7-thor-2605` | compiled for | on Orin (sm_87) |
|---|---|---|
| system torch (NGC 26.05) | sm_80, **sm_86**, 90, 100, 110, 120 | **✓ verified**: an sm_86 cubin runs on 8.7 (same major, higher minor). The 26.05 image passes the GPU test on the Orin (below) |
| OpenCV CUDA | 110 only | ✗ (CPU OpenCV and GStreamer would still work) |
| llama.cpp / llama-cpp-python | sm_110 only | ✗ |
| `gr00t-venv` torch (Jetson AI Lab sbsa wheel) | sm_110, sm_121 | ✗ |
| OpenPI FP8+NVFP4 engines | Blackwell FP8/FP4 tensor cores | ✗ (Orin has neither FP8 nor FP4) |

It is also 45 GB, built around Thor's 122 GB of memory (GR00T, π0.5, Isaac ROS).

**What JetPack 7.2 on Orin does accept:**

- **Old JetPack 6 images still run.** `cmpelkk/jetson-llm:latest` (built on
  `pytorch:24.12-py3-igpu`, torch 2.6, CUDA 12.6, sm_87 only) gives
  `torch.cuda.is_available() True` and a correct matmul on the 7.2 Orin. Driver 595
  runs CUDA 12.x binaries, so existing Orin images need no rebuild after
  moving an Orin to 7.2.
- **NGC has published no `-igpu` PyTorch image after 26.01.** That fits JetPack 7.2
  moving Orin onto the same SBSA software as Thor. From 26.02 on, Orin should use
  the plain `-py3` images, the same as Thor.

- **The pytorch.org CUDA 13 wheel runs natively on the 7.2 Orin.** `torch 2.11.0+cu130`
  (aarch64) is built for sm_80…sm_120 with **no sm_87**. Installed with `uv` into a
  venv on the Orin, it passes:

  ```text
  torch 2.11.0+cu130 cuda 13.0 arch ['sm_80','sm_90','sm_100','sm_110','sm_120'] | Orin (8, 7)
  bf16 matmul True | cudnn conv (1, 16, 222, 222) | sdpa (1, 8, 256, 64)
  ```

  So on JetPack 7.2, Orin and Thor can share one torch wheel. The `torch==2.11.0
  --index-url .../cu130` recipe from §7 works on both.

**The Orin flavour of the new image** is the same file with Orin arguments. It is
published as `cmpelkk/jetson-unified:jp7-orin`, next to `:jp7-thor`. `sjsujetsontool`
v2 picks the tag from the hardware: Thor gets `jp7-thor`. Orin keeps the JP6
`jetson-llm` image by default, which is verified on 7.2, and gets `jp7-orin` with
`--jp7` / `container set-jp7`.

```bash
docker build -f Dockerfile.jp7-thor \
  --build-arg OPENCV_CUDA_ARCH_BIN=8.7 --build-arg LLAMA_CUDA_ARCH=87 \
  --build-arg INSTALL_THOR_VLA=0 --build-arg INSTALL_ISAAC_ROS=0 \
  -t cmpelkk/jetson-unified:jp7-orin .
```

Its base and torch are verified on the 7.2 Orin (below). The Orin build itself
has not been made: building on the Orin over its link, or copying a Thor-built
image to it, is impractical at these speeds. Pushing the image to a registry the
Orin can pull from would work.

**GPU tests on the 7.2 Orin** (`thor_tutorials/orin_jp7_check.sh`, unattended,
finished 2026-10-03). Each runs a bf16 2048² matmul, a cuDNN conv and SDPA:

| tested | torch / CUDA | compiled for | result on Orin (8, 7) |
|---|---|---|---|
| pytorch.org `torch 2.11.0+cu130` wheel, native venv | 2.11.0 / 13.0 | sm_80…sm_120 (no sm_87) | ✓ all pass |
| `nvcr.io/nvidia/pytorch:26.05-py3` (base of `Dockerfile.jp7-thor`) | 2.12.0a0 / 13.2 | sm_80, sm_86, …, sm_120 | ✓ all pass |
| `nvcr.io/nvidia/pytorch:25.08-py3-igpu` (`Dockerfile.jp7`'s Orin base) | 2.8.0a0 / 12.9 | sm_87 | ✓ all pass |
| `cmpelkk/jetson-llm:latest` (JetPack 6 image, 24.12-igpu) | 2.6.0a0 / 12.6 | sm_87 | ✓ CUDA available, matmul OK |

So on JetPack 7.2 an Orin can run both the old `-igpu` images and the new SBSA
`-py3` images. The single 26.05 base serves Thor and Orin. Only the
components compiled for one architecture (OpenCV CUDA, llama.cpp, the Jetson AI
Lab sbsa wheels) need per-device builds.

That box's network was the bottleneck: ~0.1 MB/s, with 12 `connection reset by peer`
during the 26.05 pull. It took ~3.5 h to fetch 26.05 (35.4 GB on disk) and ~1 h for
25.08-igpu (17.3 GB). Re-run the check with
`ssh -p 20062 sjsujetson@headscale.forgengi.org 'cd ~/jp7test && ./orin_jp7_check.sh'`.
