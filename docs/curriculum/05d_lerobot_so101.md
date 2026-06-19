# 🦾 LeRobot & SO-ARM101 on Jetson Orin Nano

**Author:** Dr. Kaikai Liu, Ph.D. — San Jose State University

This tutorial brings the Jetson into **physical robotics**: teleoperating a low‑cost
**SO‑ARM101** robot arm and collecting imitation‑learning datasets with Hugging Face's
[**LeRobot**](https://huggingface.co/docs/lerobot). It follows the [ROS 2 & Isaac ROS](05c_ros2_isaac_ros_jetson.md)
tutorial — where ROS 2 is the *middleware* for perception graphs, LeRobot is an *end‑to‑end
learning* stack: teleoperate → record → train a policy → replay.

> Hardware reference for everything below is a real lab box (`cmpe-jetson`, Orin Nano,
> JetPack 6.2 / L4T R36.4.7, Ubuntu 22.04, glibc 2.35). The companion hardware/camera
> log lives in [`jetson/robotics/JETSON_ORIN_NANO_SETUP.md`](../../jetson/robotics/JETSON_ORIN_NANO_SETUP.md),
> and the teleop helper in [`jetson/robotics/so101_unified_teleop.py`](../../jetson/robotics/so101_unified_teleop.py).

---

## 🤖 What is SO‑ARM101 + LeRobot?

**SO‑ARM101** is an open‑source, 3D‑printed **6‑DOF** robot arm (shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper) driven by **Feetech** serial bus servos over a
single USB‑serial link. It comes as a **leader** (you move it by hand) + **follower** (mirrors
the leader) pair — the classic setup for collecting human demonstrations.

**LeRobot** is Hugging Face's robotics library: a common robot/teleoperator API, dataset format,
and training/eval CLIs (`lerobot-calibrate`, `lerobot-teleoperate`, `lerobot-record`,
`lerobot-train`, `lerobot-eval`, …). The standard learning loop is:

```text
calibrate ─▶ teleoperate ─▶ record dataset ─▶ train policy ─▶ eval / replay
```

The arm + cameras are the sensors; LeRobot turns recorded `(observation, action)` pairs into a
trained visuomotor policy (ACT, Diffusion Policy, etc.).

---

## 🧰 Environments on the Orin Nano (two venvs, on purpose)

The Orin Nano's system Python is **3.10** and the OS ships **glibc 2.35**. That constrains which
wheels load, so we keep **two** `uv` virtual environments with distinct jobs:

| Env | Python | Role | Key fact |
|-----|--------|------|----------|
| `~/lerobot-py312` | 3.12 | latest LeRobot 0.5.x, **CPU**, host‑side data collection | clean isolated venv; `torch==2.x+cpu` |
| `~/lerobot-py310-cuda` | 3.10 | **CUDA** torch + TensorRT + RealSense SDK + Feetech | `--system-site-packages` so it sees JetPack's TensorRT |

> Why two? The Jetson CUDA wheel index has no compatible **Python 3.12** torch, and
> `pyrealsense2` 3.12 wheels need glibc 2.38 (this host has 2.35). So Python 3.10 is the
> hardware/CUDA env; Python 3.12 is for trying the newest LeRobot on CPU.

### Create the CPU collection env (Python 3.12)
```bash
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv --python 3.12 ~/lerobot-py312
~/.local/bin/uv pip install --python ~/lerobot-py312/bin/python pip
~/.local/bin/uv pip install --python ~/lerobot-py312/bin/python "lerobot[feetech]==0.5.1"
source ~/lerobot-py312/bin/activate
lerobot-info        # verify
```

### Create the CUDA/RealSense env (Python 3.10)
```bash
~/.local/bin/uv venv --system-site-packages --python /usr/bin/python3 ~/lerobot-py310-cuda
# CUDA torch from the Jetson AI Lab wheel index (cu126)
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
# torch 2.8 needs libcudss.so.0 (not a JetPack system lib) — install + isolate it
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python nvidia-cudss-cu12==0.7.1.6
mkdir -p ~/lerobot-py310-cuda/cudss-lib
cp -a ~/lerobot-py310-cuda/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss*.so* \
  ~/lerobot-py310-cuda/cudss-lib/
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python "lerobot[feetech]==0.4.4"
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python \
  numpy==1.26.4 opencv-python-headless==4.11.0.86 pyrealsense2==2.58.1.10581 pygame==2.6.1
```
Activate it (note the `LD_LIBRARY_PATH` for the isolated cuDSS lib — **only** that dir, so it
doesn't shadow JetPack's native CUDA 12.6 libraries):
```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
```
Verified on `cmpe-jetson`: `torch 2.8.0`, `torch.cuda.is_available() = True` (GPU: Orin),
TensorRT 10.3 imported from JetPack, `pyrealsense2` import OK, D435i detected.

> ⚠️ **Do not** manually upgrade glibc to satisfy a 3.12 `pyrealsense2` wheel — it can break apt,
> Python, and NVIDIA user‑space libraries on JetPack 6. Keep RealSense SDK work in the 3.10 env,
> or build `librealsense` from source against glibc 2.35.

### 🆕 JetPack 7 boxes (Python 3.12): toward a *single* env

The two‑venv split above is a **JetPack 6** artifact: that box's system Python is 3.10, and its
glibc 2.35 is below the 2.38 the py312 `pyrealsense2` wheel needs. On a **JetPack 7** node
(e.g. `jetson61`: Ubuntu 24.04, **glibc 2.39**, system **Python 3.12**) *both* reasons disappear.
Verified on `jetson61`:

- A single **py312** env holds `lerobot 0.5.1` + `pyrealsense2 2.58` + OpenCV + `numpy<2` together —
  `pyrealsense2` now **imports cleanly** (no `GLIBC_2.38` error); it only needs the system lib
  `libusb-1.0-0` (`sudo apt install -y libusb-1.0-0`).

```bash
python3 -m venv ~/lerobot-py312 && source ~/lerobot-py312/bin/activate
pip install -U pip
pip install "lerobot[feetech]" pyrealsense2 "numpy<2"
sudo apt install -y libusb-1.0-0        # provides libusb-1.0.so.0 for pyrealsense2
```

> ⚠️ **CPU‑torch caveat for the host venv above.** `lerobot` needs `torchvision ≥ 0.21` (0.5.1 wants
> `torch ≥ 2.7`), and there's no CUDA `torch ≥ 2.7` for jp7/py312 on PyPI or the Jetson AI Lab wheel
> index (`jp7/cu130` → *"stage could not be found"*; only `jp6/cu126` is populated). So a plain
> `pip install` venv gets **CPU** torch. That's fine for host‑side tasks (teleop, calibration,
> dataset record/replay) — none need CUDA — but not for GPU training/inference.

#### ✅ The GPU single‑env: build from an NGC `-igpu` base

For a **single py312 env with CUDA**, don't fight PyPI — base a container on NVIDIA's NGC PyTorch
`-igpu` image, which already ships CUDA torch and runs on JetPack 7 via CUDA forward‑compat. The
repo's [`jetson/Dockerfile.jp7`](../../jetson/Dockerfile.jp7) does exactly this and is
**verified on `jetson61`**:

```bash
cd jetson && docker build -f Dockerfile.jp7 -t jetson-unified:jp7 .          # base: nvcr.io/nvidia/pytorch:25.08-py3-igpu
docker run --rm -it --runtime nvidia --network host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all -v /Developer:/Developer jetson-unified:jp7
```
Verified runtime on `jetson61` (Orin, L4T r39) — **torch 2.8.0a0 / CUDA 12.9**:
```text
torch.cuda.is_available() = True (Orin)
LeRobot 0.5.1 | PyTorch built with CUDA: True | CUDA 12.9
YOLO (ultralytics) on GPU: 6 boxes, device=cuda:0
pyrealsense2 2.58 import OK
```
The build pins the base's CUDA `torch`/`torchvision` via a constraints file so installing
`lerobot[feetech]` can't swap in a CPU wheel (it fails loudly instead, telling you to bump the NGC
tag). One tradeoff: the base's `torch-tensorrt` wants `numpy<2` while lerobot/rerun want `≥2`; the
image keeps `numpy<2` (everything works except `rerun-sdk` viz). See the Dockerfile header for details.

> **Bottom line:** on JetPack 7 a single py312 env does *all* LeRobot tasks — **CPU** via a quick
> host venv, or **GPU** via `Dockerfile.jp7`. The JetPack 6 two‑venv split is no longer needed.

---

## 🔌 Connect & calibrate the arm

Find the Feetech bus serial port (unplug/replug to identify):
```bash
lerobot-find-port            # e.g. /dev/ttyACM0 (follower), /dev/ttyACM1 (leader)
```
Calibrate each arm once (writes a calibration file LeRobot reuses):
```bash
lerobot-calibrate --help | grep -n "so101" -A 20
```
> Always start with calibration and short **dry‑run** recordings before any long dataset run.

---

## 🎮 Unified teleop helper — `so101_unified_teleop.py`

[`so101_unified_teleop.py`](../../jetson/robotics/so101_unified_teleop.py) wraps several teleop
paths behind one CLI and works with **both LeRobot 0.4.4 and 0.5.x** (it delegates to the stable
`lerobot-teleoperate` for leader mode, and uses the shared `SOFollower` API for the rest). Run it
from the Python 3.10 CUDA env:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
python ~/so101_unified_teleop.py --help
```

| Mode | What it does |
|------|--------------|
| `leader` | Physical leader arm drives the follower (delegates to `lerobot-teleoperate`) |
| `keyboard` | Terminal joint jogging — **no X11/pynput needed** (great over SSH) |
| `remote-server` | UDP + HTTP server on the Jetson; calls `robot.send_action()` |
| `mac-ps5-client` | Runs on a Mac, reads a PS5 pad via `pygame`, sends UDP to the Jetson |
| `gamepad-local` | LeRobot's local gamepad teleop for a pad wired/paired to the Jetson |

### Leader → follower
```bash
python ~/so101_unified_teleop.py leader \
  --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1 \
  --robot-id so101_follower --leader-id so101_leader
```

### Keyboard jogging (headless‑friendly)
```bash
python ~/so101_unified_teleop.py keyboard \
  --follower-port /dev/ttyACM0 --robot-id so101_follower
```
```text
q/a: shoulder_pan   w/s: shoulder_lift   e/d: elbow_flex
r/f: wrist_flex     t/g: wrist_roll      y/h: gripper      x or ESC: exit
```

### Remote server (Jetson) + HTTP control
```bash
python ~/so101_unified_teleop.py remote-server \
  --follower-port /dev/ttyACM0 --bind-host 0.0.0.0 --require-deadman true
# listens on UDP 8766 and HTTP 8765
```
Send a one‑off delta over HTTP:
```bash
python ~/so101_unified_teleop.py api-post \
  --url http://jetsonorin:8765/command \
  --json '{"delta": {"shoulder_pan.pos": 2.0}, "deadman": true}'
```

### Mac PS5 client → Jetson
```bash
python -m pip install pygame
python so101_unified_teleop.py mac-ps5-client \
  --jetson-host jetsonorin --deadman-button 5 --print-events
```
Use `--print-events` first to confirm the PS5 axis/button mapping, then adjust `--axis-lx`,
`--axis-ly`, `--axis-rx`, `--axis-ry`, `--axis-l2`, `--axis-r2` if needed. For a first real
collection, the **Mac PS5 path is easiest** — the controller is known‑good on the Mac and the
Jetson only receives network commands.

> PS5 **directly** on the Jetson (`gamepad-local`) works after Bluetooth pairing
> (`bluetoothctl` → pair/trust/connect). If Python can't read `/dev/input/event*`, add yourself to
> the `input` group: `sudo usermod -aG input $USER` and re‑login.

---

## 📷 Cameras for LeRobot

LeRobot's OpenCV camera path wants a readable **color V4L2** device. On this Jetson:

- **Easiest RGB:** Intel **RealSense D435i** color node `/dev/video6` (UVC) or any USB webcam.
- **CSI (Arducam IMX219):** `/dev/video0`/`/dev/video1` expose **raw Bayer RG10** — OpenCV can't read
  them directly. Capture via Argus/GStreamer (`nvarguscamerasrc`, `flip-method=2`) and bridge into
  virtual color devices with `v4l2loopback`, then point LeRobot at those:
  ```bash
  sudo apt install -y v4l2loopback-dkms v4l2loopback-utils
  sudo modprobe v4l2loopback devices=2 video_nr=20,21 \
    card_label="CSI Front","CSI Side" exclusive_caps=1
  env -u DISPLAY gst-launch-1.0 \
    nvarguscamerasrc sensor-id=0 tnr-mode=2 ee-mode=2 aeantibanding=3 ! \
    "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1" ! \
    nvvidconv flip-method=2 ! "video/x-raw,format=YUY2" ! \
    v4l2sink device=/dev/video20 sync=false
  # → LeRobot/OpenCV camera config uses /dev/video20, /dev/video21
  ```
  `env -u DISPLAY` avoids Argus `FrameConsumer` errors over SSH.
- **Avoid L515 RGB** (comes out essentially black here); L515 depth works via raw V4L2 `Z16`.
- The Orin Nano in this setup has **no hardware H.264 encoder** (`nvv4l2h264enc` absent) — use
  software `x264enc`.

List/inspect cameras:
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video6 --list-formats-ext
```

---

## 🔁 Suggested SO‑ARM101 workflow

```bash
# 1. activate (CPU collection env is fine for teleop+record; use 3.10-cuda if you need RealSense SDK)
ssh jetsonorin && source ~/lerobot-py312/bin/activate

# 2. ports + calibration
lerobot-find-port
lerobot-calibrate --help | grep -n "so101" -A 20

# 3. teleoperate to warm up (leader, keyboard, or PS5)
python ~/so101_unified_teleop.py keyboard --follower-port /dev/ttyACM0 --robot-id so101_follower

# 4. record a small dataset (start short!), then train + eval
lerobot-record  --help | grep -n "so101" -A 20
lerobot-train   --help
lerobot-eval    --help
```
Heavy **training** belongs on a workstation GPU, Jetson Thor, or a CUDA container — the Orin Nano is
the data‑collection / inference node. Rebooting before a long run reduces Jetson unified‑memory
fragmentation.

---

## 🧯 Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `torch.cuda.is_available() == False` in py312 | Expected — 3.12 has CPU torch. Use `~/lerobot-py310-cuda` for GPU. |
| `pyrealsense2` import: `GLIBC_2.38 not found` | 3.12 wheel needs newer glibc; use the 3.10 env's `pyrealsense2==2.58.1.10581`. |
| `libcudss.so.0` missing | Install `nvidia-cudss-cu12`, copy only `libcudss*.so*` into a dir, prepend just that to `LD_LIBRARY_PATH`. |
| CSI camera black/empty in OpenCV | Raw Bayer — use Argus + `v4l2loopback` (above), not direct `/dev/video0`. |
| PS5 pad pairs but no input | `sudo usermod -aG input $USER`, re‑login. |
| Argus `FrameConsumer` error over SSH | Prefix the pipeline with `env -u DISPLAY`. |

---

## 🔗 References

- LeRobot install: <https://huggingface.co/docs/lerobot/main/en/installation>
- LeRobot PyPI: <https://pypi.org/project/lerobot/>
- SO‑ARM101 (TheRobotStudio): <https://github.com/TheRobotStudio/SO-ARM100>
- Arducam IMX219 on Jetson: <https://docs.arducam.com/Nvidia-Jetson-Camera/Native-Camera/imx219/>
- PyTorch for Jetson: <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>
- RealSense on Jetson: <https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md>

→ **Prev:** [ROS 2 & NVIDIA Isaac ROS on Jetson](05c_ros2_isaac_ros_jetson.md)
