# Jetson Orin Nano SO-ARM101 and Camera Setup

Last updated: 2026-08-02

This note records the current Jetson Orin Nano setup for SO-ARM101 / LeRobot data collection, including the CSI cameras, Intel RealSense cameras, encoding behavior, and the dedicated LeRobot Python environment.

## Device Summary

Host:

```bash
ssh jetsonorin
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
lerobot-info
```

Observed system:

```text
hostname: cmpe-jetson
OS: Ubuntu 22.04.5 LTS
L4T: R36.4.7
kernel: 5.15.148-tegra
architecture: aarch64
Docker: installed
disk: /dev/nvme0n1p1, about 1.7T free during setup
```

Important consequence: this host uses glibc 2.35. Some Python 3.12 aarch64 binary wheels, especially `pyrealsense2`, currently require newer glibc and do not load here.

## LeRobot Environment

We created a new dedicated `uv` environment for current LeRobot instead of modifying the existing `~/.venv`.

Environment path:

```text
/home/cmpe/lerobot-py312
```

Creation commands:

```bash
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv --python 3.12 ~/lerobot-py312
~/.local/bin/uv pip install --python ~/lerobot-py312/bin/python pip
~/.local/bin/uv pip install --python ~/lerobot-py312/bin/python "lerobot[feetech]==0.5.1"
```

Activate:

```bash
source ~/lerobot-py312/bin/activate
```

Verification:

```bash
~/lerobot-py312/bin/lerobot-info
~/lerobot-py312/bin/python -m pip check
```

Current verified package state:

```text
Python: 3.12.13
LeRobot: 0.5.1
torch: 2.10.0+cpu
torchvision: 0.25.0
opencv-python-headless: 4.13.0.92
av: 15.1.0
datasets: 4.8.5
rerun-sdk: 0.26.2
feetech-servo-sdk: 1.0.0
pyserial: 3.5
pip check: No broken requirements found.
```

Installed LeRobot CLI entry points:

```text
lerobot-calibrate
lerobot-dataset-viz
lerobot-edit-dataset
lerobot-eval
lerobot-find-cameras
lerobot-find-joint-limits
lerobot-find-port
lerobot-imgtransform-viz
lerobot-info
lerobot-record
lerobot-replay
lerobot-setup-can
lerobot-setup-motors
lerobot-teleoperate
lerobot-train
lerobot-train-tokenizer
```

### CUDA Status

The Python 3.12 LeRobot env installed a CPU-only PyTorch wheel:

```text
PyTorch version: 2.10.0+cpu
Is PyTorch built with CUDA support?: False
```

The Jetson CUDA wheel index checked during setup did not provide compatible Python 3.12 `torch` / `torchvision` wheels for this host. Treat `~/lerobot-py312` as the host-side robot/data-collection environment, not the training environment.

Recommended split:

- Prefer `~/lerobot-py310-cuda` for the Orin Nano because it has CUDA torch, TensorRT, Feetech, and RealSense SDK support.
- Keep `~/lerobot-py312` only for checking latest LeRobot 0.5.x behavior on CPU, or for code that does not need JetPack system Python packages.
- Use Jetson Thor, a Jetson-compatible container, or a workstation GPU for heavier training.

### RealSense Python Caveat

`pyrealsense2` was tested in the Python 3.12 LeRobot env and removed because both available Python 3.12 aarch64 wheels failed to import:

```text
ImportError: /lib/aarch64-linux-gnu/libm.so.6: version `GLIBC_2.38' not found
```

The existing Python 3.10 env remains the better RealSense SDK env:

```text
/home/cmpe/.venv
pyrealsense2==2.58.1.10581
```

Use the Python 3.10 CUDA env or the older `/home/cmpe/.venv` for RealSense SDK depth/RGB access.

## CUDA-Capable Python 3.10 Environment

`uv venv --system-site-packages` can import JetPack system packages such as TensorRT, but only when the Python ABI matches the system Python. On this Jetson, system Python is 3.10, so a Python 3.10 venv can see `/usr/lib/python3/dist-packages/tensorrt`; a Python 3.12 venv cannot import those Python 3.10 bindings.

We created a second env for CUDA PyTorch / TensorRT tests:

```text
/home/cmpe/lerobot-py310-cuda
```

Creation and install pattern:

```bash
~/.local/bin/uv venv --system-site-packages --python /usr/bin/python3 ~/lerobot-py310-cuda
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python nvidia-cudss-cu12==0.7.1.6
mkdir -p ~/lerobot-py310-cuda/cudss-lib
cp -a ~/lerobot-py310-cuda/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss*.so* \
  ~/lerobot-py310-cuda/cudss-lib/
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python "lerobot[feetech]==0.4.4"
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python \
  numpy==1.26.4 opencv-python-headless==4.11.0.86
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python pyrealsense2==2.58.1.10581
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python pygame==2.6.1
```

Use it with:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
```

Verified:

```text
Python: 3.10.12
LeRobot: 0.4.4
TensorRT: 10.3.0, imported from JetPack system packages
pyrealsense2: 2.58.1.10581 import OK
torch: 2.8.0
torch.version.cuda: 12.6
torch.cuda.is_available(): True
GPU: Orin
CUDA elementwise add: OK
CUDA matmul/cuBLAS: OK
RealSense D435i detected by SDK: Intel RealSense D435I, serial 834412072073
pygame: 2.6.1
```

Why `nvidia-cudss-cu12` is handled this way:

- `torch==2.8.0` needed `libcudss.so.0`, which was not installed as a JetPack system library.
- Installing `nvidia-cudss-cu12` provides that library.
- Do not put the whole `site-packages/nvidia/cu12/lib` directory first in `LD_LIBRARY_PATH`, because that can override JetPack's native CUDA 12.6 libraries with PyPI CUDA runtime wheels.
- We copied only the `libcudss*.so*` files into `~/lerobot-py310-cuda/cudss-lib` and prepend only that directory.

Notes:

- `torch==2.10.0` from the same Jetson wheel index imported with CUDA but failed cuBLAS handle creation in this setup. `torch==2.8.0` passed CUDA matmul.
- Because this env uses `--system-site-packages`, `pip check` can report unrelated inherited system-package conflicts. Treat this env as the main Orin Nano hardware environment, but expect dependency resolution to be less pristine than a fully isolated venv.
- If this env is used for long-running training/inference, rebooting before the run can help reduce Jetson unified-memory fragmentation.

## glibc and RealSense

Do not manually upgrade glibc in-place on the Jetson Orin Nano just to satisfy the Python 3.12 `pyrealsense2` wheel.

Current host glibc:

```text
ldd (Ubuntu GLIBC 2.35-0ubuntu3.13) 2.35
```

The Python 3.12 `pyrealsense2` wheels tested here require `GLIBC_2.38`. glibc is the core C runtime for the OS; replacing it underneath Ubuntu 22.04 / JetPack 6 can break apt, Python, NVIDIA user-space libraries, and system services.

Safer options:

1. Keep RealSense SDK work in the existing Python 3.10 env where `pyrealsense2==2.58.1.10581` works for the D435i.
2. Build `librealsense` / `pyrealsense2` from source on this Jetson against the host's glibc 2.35.
3. Run RealSense code in a container whose user space has a compatible glibc, while passing through `/dev/bus/usb`, `/dev/video*`, and udev permissions.
4. Move RealSense Python 3.12 work to a newer Jetson OS image that officially ships the newer glibc, rather than replacing glibc manually.

For this Orin Nano, the practical recommendation is: do not update glibc manually. Use Python 3.10 for RealSense SDK access, or build the SDK locally.

## Unified SO-ARM101 Teleop Helper

Script in this repo:

```text
mylerobot/scripts/so101_unified_teleop.py
```

Copied to the Jetson for convenience:

```text
/home/cmpe/so101_unified_teleop.py
```

For a step-by-step bring-up walkthrough (connect the arm, find the port,
calibrate, keyboard test, PS5 test, leader teleop), see:

```text
jetson/robotics/SO101_TELEOP_TUTORIAL.md
```

Or run the guided menu, which launches the real LeRobot CLIs for find-port,
info, calibrate and teleoperate:

```bash
python ~/so101_unified_teleop.py interactive
```

This script is designed to work with both LeRobot 0.4.4 and 0.5.x:

- `interactive` is a guided menu over all of the below plus the LeRobot CLIs.
- `ps5-local` reads a DualSense paired to the Jetson via evdev and sends
  joint-space goals, with L1 as a deadman and Circle as an E-STOP.
- `leader` delegates to `lerobot-teleoperate`, which exists in both versions.
- `keyboard` uses the shared `SOFollower` API directly and does not need X11 or `pynput`.
- `remote-server` runs a UDP/HTTP bridge on the Jetson and calls `robot.send_action()`.
- `mac-ps5-client` runs on a Mac, reads a PS5 joystick with `pygame`, and sends UDP commands to the Jetson.
- `gamepad-local` delegates to LeRobot's local `gamepad` teleoperator for a controller connected directly to the Jetson.

Recommended Orin Nano environment:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
python ~/so101_unified_teleop.py --help
```

Leader-arm mode:

```bash
python ~/so101_unified_teleop.py leader \
  --follower-port /dev/ttyACM0 \
  --leader-port /dev/ttyACM1 \
  --robot-id so101_follower \
  --leader-id so101_leader
```

Terminal keyboard joint jogging:

```bash
python ~/so101_unified_teleop.py keyboard \
  --follower-port /dev/ttyACM0 \
  --robot-id so101_follower
```

Keyboard mapping:

```text
q/a: shoulder_pan
w/s: shoulder_lift
e/d: elbow_flex
r/f: wrist_flex
t/g: wrist_roll
y/h: gripper
x or ESC: exit
```

Remote server on Jetson:

```bash
python ~/so101_unified_teleop.py remote-server \
  --follower-port /dev/ttyACM0 \
  --bind-host 0.0.0.0 \
  --require-deadman true
```

The server listens on:

```text
UDP:  8766
HTTP: 8765
```

HTTP command example:

```bash
python ~/so101_unified_teleop.py api-post \
  --url http://jetsonorin:8765/command \
  --json '{"delta": {"shoulder_pan.pos": 2.0}, "deadman": true}'
```

Mac PS5 client:

```bash
python -m pip install pygame
python mylerobot/scripts/so101_unified_teleop.py mac-ps5-client \
  --jetson-host jetsonorin \
  --deadman-button 5 \
  --print-events
```

Use `--print-events` first to verify the PS5 controller axis/button mapping on the Mac. If the sticks or triggers are mapped differently, adjust `--axis-lx`, `--axis-ly`, `--axis-rx`, `--axis-ry`, `--axis-l2`, and `--axis-r2`.

### PS5 Controller Directly on Jetson

Confirmed working on 2026-08-02. The easiest path is now:

```bash
python ~/jetson_devices.py bluetooth ps5      # guided scan + pair + trust + connect
python ~/jetson_devices.py bluetooth gamepad  # live button/axis check
```

See the `jetson_devices.py` section above for the three pitfalls that break
manual pairing (readline control bytes in `bluetoothctl` output, the inbound
HID service authorization timeout, and `br-connection-create-socket` when the
controller is asleep).

Original observations:

```text
Bluetooth controller: present
bluetooth.service: active
rfkill Bluetooth: unblocked
pygame: installed in ~/lerobot-py310-cuda
```

Pairing flow:

```bash
bluetoothctl
power on
agent on
default-agent
pairable on
scan on
```

Put the PS5 controller in pairing mode by holding **PS + Create** until the light flashes. In `bluetoothctl`, then run:

```text
pair <controller-mac>
trust <controller-mac>
connect <controller-mac>
quit
```

Then try:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
python ~/so101_unified_teleop.py gamepad-local \
  --follower-port /dev/ttyACM0 \
  --robot-id so101_follower
```

A permission caveat was expected here but did not apply. `/dev/input/event*` is
owned by `root:input` with mode 0660 and user `cmpe` is not in the `input`
group, yet reading works: systemd-logind attaches a `user:cmpe:rw-` uaccess ACL
to newly appearing input devices because `cmpe` holds the active local seat
session. Verified after the DualSense connected:

```text
$ getfacl /dev/input/event7
user::rw-
user:cmpe:rw-
group::rw-
other::---
```

Only add the user to the `input` group if there is no local seat session (for
example a fully headless boot with no console login), in which case the ACL is
never applied:

```bash
sudo usermod -aG input cmpe    # then log out and back in
```

For first real collection, the remote Mac PS5 path is usually easier because the controller is known to work on the Mac and the Jetson only receives network commands.

## Connected Cameras

State observed 2026-08-02 (D435i plus a USB UVC camera, L515 not attached):

```text
/dev/video0  Arducam IMX219 CSI, raw Bayer RG10, Argus sensor-id=0
/dev/video1  Arducam IMX219 CSI, raw Bayer RG10, Argus sensor-id=1

D435i (8086:0b3a, USB3 5G, serial 850123050151):
  /dev/video2  depth Z16
  /dev/video3  metadata
  /dev/video4  IR: GREY / UYVY / Y8I / Y12I
  /dev/video5  metadata
  /dev/video6  RGB color YUYV
  /dev/video7  metadata

Innomaker U20CAM-1080p (0c45:6366, USB2 480M):
  /dev/video8  MJPG up to 1920x1080@30, YUYV 1920x1080@5
  /dev/video9  metadata
```

Node numbering shifts with what is plugged in, so resolve cameras by serial or
by `jetson_devices.py camera list` rather than hardcoding `/dev/videoN`.

Earlier USB and V4L2 state while D435i and L515 were both plugged in:

```text
Bus 002 Device 006: ID 8086:0b64 Intel RealSense 515
Bus 002 Device 004: ID 8086:0b3a Intel RealSense Depth Camera 435i
```

Both RealSense devices were on USB3 at `5000M`.

V4L2 mapping:

```text
/dev/video0: Arducam IMX219 CSI camera, raw Bayer RG10
/dev/video1: Arducam IMX219 CSI camera, raw Bayer RG10

D435i:
  /dev/video2 - depth/IR related node
  /dev/video3 - metadata
  /dev/video4 - depth/IR related node
  /dev/video5 - metadata
  /dev/video6 - RGB color YUYV
  /dev/video7 - metadata

L515:
  /dev/video8  - greyscale/intensity
  /dev/video9  - metadata
  /dev/video10 - depth Z16
  /dev/video11 - metadata
  /dev/video12 - confidence CNF4
  /dev/video13 - metadata
  /dev/video14 - RGB color YUYV
  /dev/video15 - metadata
```

Do not assume `/dev/video10` and `/dev/video11` are free for virtual CSI cameras when the L515 is plugged in.

## Arducam IMX219 CSI Cameras

Hardware:

```text
2x Arducam 8MP Camera V2.2
Sensor: IMX219
Interface: CSI
```

Status:

- Both cameras are enabled and visible.
- They expose raw Bayer `RG10` through V4L2.
- Useful modes include `1920x1080@30`, `1280x720@60`, and full-resolution modes.
- Direct OpenCV access to `/dev/video0` and `/dev/video1` is not enough because these nodes are raw Bayer and capture timed out in earlier tests.
- Use Argus/GStreamer (`nvarguscamerasrc`) for clean CSI capture.

Important SSH/Argus detail:

```bash
env -u DISPLAY gst-launch-1.0 ...
```

Unsetting `DISPLAY` avoided Argus `FrameConsumer` errors over SSH.

### Flipped Recording

Both CSI images needed to be flipped upside down. Use `nvvidconv flip-method=2`.

Single-camera pattern:

```bash
env -u DISPLAY gst-launch-1.0 -e \
  nvarguscamerasrc sensor-id=0 tnr-mode=2 ee-mode=2 aeantibanding=3 num-buffers=300 ! \
  "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1" ! \
  nvvidconv flip-method=2 ! "video/x-raw,format=I420" ! \
  x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 ! \
  h264parse ! mp4mux ! filesink location=/home/cmpe/camera_samples/imx219_cam0_flipped_1080p30.mp4
```

Use `sensor-id=1` for the second CSI camera.

Known good sample files:

```text
/home/cmpe/camera_samples/imx219_cam0_flipped_1080p30_20260526_192853.mp4
/home/cmpe/camera_samples/imx219_cam1_flipped_1080p30_20260526_192853.mp4
/home/cmpe/camera_samples/imx219_dual_flipped_side_by_side_1080p30_20260526_192853.mp4
```

### CSI CPU Use

Observed rough CPU use:

```text
Dual CSI 1080p30 + flip + software H.264:
  total CPU about 53-55%
  gst process about 240-248% CPU

Dual CSI capture only, no encode:
  total CPU about 19-22%
  gst process about 50-56% CPU

Single CSI 1080p30 + flip + software H.264:
  total CPU about 30-34%
  gst process about 140-148% CPU
```

## Hardware H.264 Encoding

The Orin Nano does not expose the hardware H.264 encoder in this setup:

```text
gst-inspect-1.0 nvv4l2h264enc: not found
/dev/nvhost-msenc: not present
gst-inspect-1.0 x264enc: available
```

Use software encoding (`x264enc`) or store raw/less-compressed data when CPU budget allows. This is expected behavior for Orin Nano class hardware without NVENC.

## Intel RealSense D435i

Status:

- Detected on USB3.
- RGB node: `/dev/video6`.
- Python 3.10 `pyrealsense2==2.58.1.10581` detects it.
- SDK serial reported earlier: `834412072073`.
- V4L by-id serial shown by Linux: `850123050151`.

Known good RGB sample:

```text
/home/cmpe/camera_samples/realsense_d435i_color_1080p30_20260526_193431.mp4
```

Observed CPU:

```text
D435i RGB 1080p30 + software H.264:
  total CPU about 33-36%
  gst process about 170-180% CPU

D435i RGB 1080p30 capture only:
  total CPU about 6%
  gst process about 18-19% CPU
```

Recommended use:

- For RGB-only LeRobot collection in Python 3.12, try OpenCV/UVC via `/dev/video6`.
- For depth, alignment, intrinsics, and point clouds, use `pyrealsense2` from the Python 3.10 env or a container where the RealSense wheel/build is compatible.

## Intel RealSense L515

Status:

- Detected on USB3 as `8086:0b64`.
- V4L2 depth works.
- Python RealSense SDK support is problematic on this host:
  - Python 3.10 `pyrealsense2==2.58.1.10581` detected the D435i but not the L515.
  - Older `pyrealsense2==2.55.1.6486` segfaulted in a temporary Python 3.10 test.
  - Python 3.12 `pyrealsense2` wheels fail due to glibc 2.38 requirement.

Depth node:

```text
/dev/video10
format: Z16
modes: 480x640@30, 768x1024@30, 240x320@30, 480x1024@30
```

Depth streaming result:

```text
480x640 Z16 @ 30 fps: stable at 30.01 fps
768x1024 Z16 @ 30 fps: stable after initial empty buffers
```

Known good depth files:

```text
/home/cmpe/camera_samples/l515_depth_colormap_480x640_30fps_20260526_205215.mp4
/home/cmpe/camera_samples/l515_depth_z16_480x640_30fps_20260526_205144.raw
/home/cmpe/camera_samples/l515_depth_stats_20260526_205215.txt
```

Depth stats from the 480x640 test:

```text
frames=150
valid_pixels_mean=177703.6 / 307200 (57.85%)
raw_p50_mean=7626.28
raw_min_nonzero=1960
raw_max=38446
```

RGB node:

```text
/dev/video14
format: YUYV
modes include 1920x1080@30, 1280x720@60/30, 640x480@60/30
```

RGB result:

- Default L515 RGB output was essentially black.
- Raw frame luminance confirmed this was not an MP4 encoding issue.
- Very long exposure plus max gain produced visible frames but dropped to about 1 fps.

RGB samples:

```text
/home/cmpe/camera_samples/l515_rgb_640x480_30fps_20260526_205248.mp4
/home/cmpe/camera_samples/l515_rgb_visible_exp1000_gain4096_20260527_004101.mp4
/home/cmpe/camera_samples/l515_rgb_visible_exp10000_gain4096_20260527_004138.mp4
```

Recommendation:

- Do not rely on L515 RGB for SO-ARM101 collection.
- L515 depth is usable through V4L2 raw Z16.
- If synchronized RGB+depth is needed, prefer D435i or an Orbbec device with a working SDK path.

## Orbbec Cameras

Cameras discussed:

```text
Orbbec Gemini 336
Orbbec Femto Bolt Depth+RGB
```

Official Orbbec SDK v2 documentation lists Linux ARM64 support and tested Jetson platforms including Jetson Orin Nano and Jetson Thor. `pyorbbecsdk2` is the current Python package name.

Package availability checked:

```text
pyorbbecsdk2==2.1.1 is available
```

Not installed in `~/lerobot-py312` yet. Install later only when one of the Orbbec cameras is plugged in for testing:

```bash
~/.local/bin/uv pip install --python ~/lerobot-py312/bin/python pyorbbecsdk2
```

The SDK setup script may need sudo for udev/environment rules.

## CSI Cameras in LeRobot

The CSI cameras are not normal color UVC cameras. LeRobot's OpenCV camera path expects readable color frames from a V4L2 device, while `/dev/video0` and `/dev/video1` expose raw Bayer `RG10`.

Practical options:

1. Use RealSense D435i or another UVC camera for RGB in LeRobot.
2. Use `v4l2loopback` to bridge Argus CSI frames into virtual color cameras.
3. Write a small custom LeRobot camera wrapper around a GStreamer/Argus appsink pipeline.

If using `v4l2loopback`, choose high video numbers because L515 already uses `/dev/video10` and `/dev/video11` when plugged in:

```bash
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils
sudo modprobe v4l2loopback devices=2 video_nr=20,21 card_label="CSI Front","CSI Side" exclusive_caps=1
```

Feed CSI camera 0:

```bash
env -u DISPLAY gst-launch-1.0 \
  nvarguscamerasrc sensor-id=0 tnr-mode=2 ee-mode=2 aeantibanding=3 ! \
  "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1" ! \
  nvvidconv flip-method=2 ! "video/x-raw,format=YUY2" ! \
  v4l2sink device=/dev/video20 sync=false
```

Feed CSI camera 1:

```bash
env -u DISPLAY gst-launch-1.0 \
  nvarguscamerasrc sensor-id=1 tnr-mode=2 ee-mode=2 aeantibanding=3 ! \
  "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1" ! \
  nvvidconv flip-method=2 ! "video/x-raw,format=YUY2" ! \
  v4l2sink device=/dev/video21 sync=false
```

Then point LeRobot/OpenCV camera configs to `/dev/video20` and `/dev/video21`.

## Suggested SO-ARM101 Workflow

Activate the LeRobot env:

```bash
ssh jetsonorin
source ~/lerobot-py312/bin/activate
```

Find the motor USB port:

```bash
lerobot-find-port
```

Inspect available SO-101 CLI options:

```bash
lerobot-record --help | grep -n "so101" -A 20
lerobot-calibrate --help | grep -n "so101" -A 20
```

Start with calibration and short dry-run recordings before long dataset collection.

For cameras:

- Easiest RGB path: D435i RGB through `/dev/video6` or a normal USB camera.
- Best CSI image quality: Argus pipeline with `flip-method=2`; bridge to LeRobot through `v4l2loopback` or a custom camera class.
- Avoid L515 RGB.
- Use L515 depth only if a custom V4L2 depth reader is acceptable.

## Unified Device Tool: `jetson_devices.py`

Script in this repo:
synced to jetsonorin:/home/cmpe/jetson_devices.py
```text
jetson/robotics/jetson_devices.py
```

Copied to the Jetson for convenience:

```text
/home/cmpe/jetson_devices.py
```

One CLI that detects and tests everything attached to the Orin Nano: CSI, USB
UVC, RealSense and Orbbec cameras, plus Bluetooth game controllers. Core paths
are standard library only; optional features (RealSense SDK streams, gamepad
input) auto re-exec into an interpreter that has the needed module.

Recommended environment:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=/home/cmpe/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
python ~/jetson_devices.py --help
```

### Camera subcommands

```bash
python ~/jetson_devices.py camera list --formats   # detect + describe everything
python ~/jetson_devices.py camera test             # capture smoke test, all cameras
python ~/jetson_devices.py camera snap   -c csi:0 --out shot.jpg
python ~/jetson_devices.py camera record -c usb -d 5
python ~/jetson_devices.py camera show   -c rs:depth --mode stream --port 8090
python ~/jetson_devices.py camera formats /dev/video8
python ~/jetson_devices.py camera interactive      # menu-driven
```

Camera selectors accepted by `--camera`:

```text
0, 1, 2 ...        index from `camera list`
csi:0, csi:1       CSI cameras by Argus sensor-id
/dev/videoN        explicit V4L2 node
usb, csi, rs       first camera of that kind
rs:color           RealSense SDK color stream
rs:depth           RealSense SDK depth, JET colormapped
rs:ir              RealSense SDK left infrared
rs:rgbd            RealSense SDK color + depth side by side
```

`camera list` reports kind, USB vendor/product with a vendor name, serial, link
speed, bus, Argus sensor-id, which node is color vs depth, which nodes are
metadata-only, and the full V4L2 format/resolution/fps table with `--formats`.
When `pyrealsense2` is importable it also prints SDK-level device info and
stream profiles.

Live view has two modes. `--mode window` uses a GStreamer video sink and needs
`DISPLAY` (`nveglglessink` locally, `ximagesink` over X11 forwarding).
`--mode stream` needs no display at all: it serves MJPEG over HTTP so the feed
opens in a browser on another machine.

```text
http://<jetson-ip>:8090/           HTML page with the live image
http://<jetson-ip>:8090/stream     raw multipart MJPEG
http://<jetson-ip>:8090/snapshot   single JPEG
```

`--mode auto` (the default) picks `window` when `DISPLAY` is set and `stream`
otherwise.

### Bluetooth subcommands

```bash
python ~/jetson_devices.py bluetooth status          # adapter, bonds, input nodes
python ~/jetson_devices.py bluetooth scan -t 15      # classify nearby controllers
python ~/jetson_devices.py bluetooth ps5             # PS5 control panel (interactive)
python ~/jetson_devices.py bluetooth quest           # guided Quest 3 attempt
python ~/jetson_devices.py bluetooth pair <mac>      # pair + trust + connect
python ~/jetson_devices.py bluetooth gamepad         # live button/axis dashboard
python ~/jetson_devices.py bluetooth interactive     # menu-driven
```

`scan` classifies known controller signatures (DualSense, DualShock 4, Xbox,
8BitDo, Nintendo, Meta) by advertised name and USB vendor/product id.

`bluetooth ps5` with no flags opens an interactive control panel that shows the
controller's current state at the top and offers pair, connect, disconnect,
unpair, live input test, and the button mapping. Every action is also a
one-shot flag so the command stays scriptable:

```bash
python ~/jetson_devices.py bluetooth ps5 --status      # state + battery + input node
python ~/jetson_devices.py bluetooth ps5 --pair        # guided pairing
python ~/jetson_devices.py bluetooth ps5 --connect     # wake + connect a known pad
python ~/jetson_devices.py bluetooth ps5 --disconnect
python ~/jetson_devices.py bluetooth ps5 --unpair
python ~/jetson_devices.py bluetooth ps5 --test        # mapping table + live panel
```

The live input test prints the full physical/evdev mapping table and then a
panel with normalized values and bar graphs:

```text
  Left stick   LX -0.84 ..#########|
               LY -0.92 .##########|
  Right stick  RX +0.80            |#########..
               RY +0.00 ...........|
  Triggers     L2  0.00 ......................
               R2  0.78 #################.....
  D-pad        < . . .   LEFT

  Held         Square  R1
  Last press   Square           [BTN_SOUTH]
```

The panel redraws in place on a tty. Over a plain SSH pipe it falls back to
rate-limited event lines; use `ssh -t jetsonorin` to get the panel.

### PS5 button and axis mapping on this host

Because `hid_playstation` is missing, the DualSense enumerates as a generic HID
gamepad: HID buttons 1..15 land on the generic `BTN_SOUTH..BTN_THUMBR` block in
report order, which does not match those names' usual meanings. Both scripts
translate back to physical labels using that report order.

```text
physical          evdev         code      physical    evdev      kind
Square            BTN_SOUTH     304       LX          ABS 0      stick
Cross             BTN_EAST      305       LY          ABS 1      stick
Circle            BTN_C         306       RX          ABS 2      stick
Triangle          BTN_NORTH     307       L2          ABS 3      trigger
L1                BTN_WEST      308       R2          ABS 4      trigger
R1                BTN_Z         309       RY          ABS 5      stick
L2 button         BTN_TL        310       D-pad X     ABS 16     hat
R2 button         BTN_TR        311       D-pad Y     ABS 17     hat
Create            BTN_TL2       312
Options           BTN_TR2       313
L3 stick click    BTN_SELECT    314
R3 stick click    BTN_START     315
PS                BTN_MODE      316
Touchpad          BTN_THUMBL    317
Mute              BTN_THUMBR    318
```

Confirmed on hardware: Circle, Triangle, L1, R1 report the codes above, and
pressing the L2/R2 buttons raised `ABS 3` / `ABS 4` in sync. The trigger axes
were identified from `absinfo`: `ABS_RX`/`ABS_RY` rest at 0 (triggers) while
`ABS_Z`/`ABS_RZ` rest at mid-scale (right stick).

### Driving the SO-ARM101 from the DualSense

`so101_unified_teleop.py` gained a `ps5-local` mode that reads the controller
directly on the Jetson through evdev and sends joint-space goals:

```bash
python ~/so101_unified_teleop.py ps5-local --follower-port /dev/ttyACM0 --dry-run
python ~/so101_unified_teleop.py ps5-local --follower-port /dev/ttyACM0
```

L1 is a deadman (the arm only moves while held), Circle is an E-STOP and
Options clears it. See `SO101_TELEOP_TUTORIAL.md` for the full walkthrough.

Note that LeRobot's own `gamepad` teleoperator (the `gamepad-local` mode) emits
end-effector deltas and expects a robot with inverse kinematics, so it does not
pair with a plain `so101_follower`, which consumes joint positions. Use
`ps5-local` for joint control.

### Whole-system report

```bash
python ~/jetson_devices.py doctor
```

Prints host/L4T info, required CLI tools, GStreamer element availability,
which Python modules each interpreter on the box provides, the camera
inventory, and Bluetooth state.

### Verified on this Orin Nano

Camera capture smoke test, 2026-08-02, all six sources passed:

```text
PASS  CSI camera 0 (IMX219)          1280x720, Argus sensor-id=0
PASS  CSI camera 1 (IMX219)          1280x720, Argus sensor-id=1
PASS  RealSense D435i via V4L2       /dev/video6 color
PASS  USB UVC Innomaker U20CAM       /dev/video8, MJPG 1280x720
PASS  RealSense SDK color            pyrealsense2
PASS  RealSense SDK depth            pyrealsense2
```

Camera notes found during that run:

- Both CSI images need `flip-method=2`; the tool applies it by default for
  `csi:*` and `--flip` overrides it.
- The Innomaker USB camera exposes `MJPG` and `YUYV`. The tool prefers `MJPG`,
  which avoids a re-encode and gets 1080p at 30 fps instead of YUYV's 5 fps.
- MJPEG streaming sustained 30 fps at 1280x720 from a CSI camera.
- RealSense RGB auto-exposure needs roughly 15 frames to converge, so short
  captures look black. `camera snap` discards 15 frames for SDK sources.
- The RealSense depth node enumerates its pixel format as `Z16 ` with a
  trailing space.

### Bluetooth findings

PS5 DualSense pairing was verified end to end on 2026-08-02:

```text
name:      DualSense Wireless Controller
address:   A0:FA:9C:8B:2B:6F
modalias:  usb:v054Cp0CE6
result:    Paired: yes, Trusted: yes, Connected: yes
input:     /dev/input/event7 and /dev/input/js0
```

Three non-obvious things that break naive DualSense pairing on this host:

1. `bluetoothctl` wraps its ANSI colors in readline's `\x01` / `\x02` markers,
   so `[CHG]` arrives as `[\x01...\x02CHG\x01...\x02]`. Stripping ANSI alone is
   not enough; those two control bytes must be removed or all scan output
   parses as zero devices.
2. Immediately after bonding, the controller connects back *inbound* and BlueZ
   asks the agent to authorize the HID service. If that prompt is not answered
   within about 30 seconds, bluetoothd logs `auth_callback() Access denied` and
   drops the link. Trust the device early and answer the agent prompt promptly.
3. A bonded controller that has gone to sleep will not answer an outbound page;
   `bluetoothctl connect` fails with
   `org.bluez.Error.Failed br-connection-create-socket`. Press the PS button to
   wake it and let it connect inbound.

Input device permissions turned out to be a non-issue here. `/dev/input/event*`
is `root:input 0660` and user `cmpe` is not in the `input` group, but
systemd-logind grants a `user:cmpe:rw-` uaccess ACL to newly appearing input
devices because `cmpe` owns the active local seat session. `evdev` therefore
works over SSH with no group change. `/dev/input/js0` is additionally
world-readable, and the tool falls back to the classic joydev API when event
nodes are not readable.

This kernel does not ship `hid_playstation`:

```text
modinfo hid_playstation: not found
```

So the DualSense enumerates as a generic HID gamepad and its buttons do not map
to the printed PS labels. The tool reports the kernel's canonical evdev names
(`BTN_SOUTH`, `BTN_NORTH`, ...) rather than guessing at physical labels, and
says so at the start of a gamepad test.

Meta Quest 3 Touch Plus controllers bond to the headset over a proprietary BLE
link and do not advertise a standard Bluetooth HID gamepad profile, so a Linux
host generally cannot use them as a joystick. The `bluetooth quest` subcommand
scans, reports Meta vendor `0x2833` devices, attempts pairing, and checks
whether any HID input node appeared, so the outcome is visible rather than
guessed. For Quest input on the Jetson, prefer sending poses from a headset-side
app over UDP/WebSocket, matching the pattern in `so101_unified_teleop.py`.

## Useful Test Commands

List cameras:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id
```

List formats:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video1 --list-formats-ext
v4l2-ctl -d /dev/video6 --list-formats-ext
```

Test D435i RGB:

```bash
gst-launch-1.0 -e \
  v4l2src device=/dev/video6 num-buffers=300 ! \
  "video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1" ! \
  videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 ! \
  h264parse ! mp4mux ! filesink location=/home/cmpe/camera_samples/d435i_rgb_test.mp4
```

Test L515 depth:

```bash
v4l2-ctl -d /dev/video10 \
  --set-fmt-video=width=480,height=640,pixelformat="Z16 " \
  --stream-mmap=4 --stream-count=60 --stream-poll --verbose
```

Check encoders:

```bash
gst-inspect-1.0 x264enc
gst-inspect-1.0 nvv4l2h264enc
test -e /dev/nvhost-msenc && echo yes || echo no
```

## Reference Links

- LeRobot installation: https://huggingface.co/docs/lerobot/main/en/installation
- LeRobot PyPI: https://pypi.org/project/lerobot/
- Arducam IMX219 on Jetson: https://docs.arducam.com/Nvidia-Jetson-Camera/Native-Camera/imx219/
- NVIDIA JetPack 6.2: https://developer.nvidia.com/embedded/jetpack-sdk-62
- NVIDIA PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- RealSense Jetson notes: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
- Orbbec SDK v2: https://orbbec.github.io/OrbbecSDK_v2/
