# 🤖 ROS 2 & NVIDIA Isaac ROS on Jetson Orin Nano
**Author:** Dr. Kaikai Liu, Ph.D. · San Jose State University

> **Prerequisite:** [05b — Object Detection on Jetson](05b_yolo_vlm_object_detection.md). This chapter turns the `jetson_object_detection_toolkit.py` detector into a **ROS 2 node** so it can be part of a robotics pipeline.

## 🧠 What is ROS 2?

**ROS 2 (Robot Operating System 2)** is the de‑facto middleware for robots: a set of libraries + tools for building **nodes** (small programs) that talk over **topics** (named, typed pub/sub streams), **services**, and **actions**. It uses **DDS** for discovery and transport, so nodes find each other automatically on a network. The current LTS is **Humble Hawksbill** (Ubuntu 22.04 / 24.04), which is what Jetson + Isaac ROS use.

Key concepts you'll use here:
- **Node** — a process (e.g. a camera driver, a detector, a planner).
- **Topic** — a typed channel, e.g. `/image_raw` carrying `sensor_msgs/Image`.
- **Message** — a typed payload, e.g. `vision_msgs/Detection2DArray` for detections.
- **`colcon`** — the ROS 2 build tool; **workspace** — a `src/` folder of packages built with `colcon build`.

## ⚡ What is NVIDIA Isaac ROS?

[**NVIDIA Isaac ROS**](https://github.com/NVIDIA-ISAAC-ROS) is a collection of **GPU‑accelerated ROS 2 packages ("GEMs")** for perception and navigation on Jetson — e.g. stereo depth, visual SLAM (`isaac_ros_visual_slam`), AprilTags, DNN inference (`isaac_ros_dnn_inference`, `isaac_ros_yolov8`), and image processing. Its secret sauce is **NITROS** (NVIDIA Isaac Transport for ROS): type‑adapted, zero‑copy transport that keeps data in GPU memory between nodes, avoiding the CPU copies that normally bottleneck ROS image pipelines. Isaac ROS ships as **containers** built on CUDA/TensorRT bases.

| | Plain ROS 2 | Isaac ROS |
|---|---|---|
| Transport | DDS, CPU copies | DDS + **NITROS** (GPU zero‑copy) |
| Perception | you bring your own | prebuilt accelerated GEMs |
| Best for | learning, custom nodes (this chapter) | production GPU perception |

---

## 🧭 Hardware support & the lightweight choice

Isaac ROS support is **hardware‑ and JetPack‑specific** — check before you invest:

| Isaac ROS line | Target Jetson | JetPack / L4T | Ubuntu / ROS 2 |
|---|---|---|---|
| **Current** | **Jetson Thor** (T5000/T4000) | JetPack 7.1 (R38.4) | 24.04 / **Jazzy** |
| **Previous (3.x)** | **Orin family** (AGX Orin, Orin NX, **Orin Nano**) | JetPack 6.x | 22.04 / **Humble** |

> ⚠️ **Isaac ROS is too heavy for the Jetson Orin Nano (8 GB).** Even where it's "supported" (Orin Nano on JetPack 6), the GEMs are built for the bigger Orin/Thor modules: the Docker images are multi‑GB, and the GPU perception graphs (stereo depth, DNN inference, multiple NITROS nodes) routinely exhaust the Orin Nano's 8 GB of shared CPU/GPU memory. On **JetPack 7 / L4T r39** Orin Nano (like our lab boxes) it's worse — that config isn't a supported Isaac ROS target at all (the new line is Thor‑only).

**So on an Orin Nano, default to plain ROS 2** (the container below) with our `ros2_jetson_detect` node — our toolkit already gives accelerated detection (YOLO+TensorRT from 05b) without NITROS, and it fits comfortably in 8 GB. Reach for Isaac ROS only on a **Thor** (or a bigger Orin) when you specifically need an accelerated GEM (VSLAM, stereo depth, AprilTags…).

**Lightweight options, heaviest → lightest:**
1. Full Isaac ROS Docker dev container (`isaac_ros_common`) — multi‑GB; bigger Orin / Thor only.
2. Isaac ROS **Debian/apt** packages — `sudo apt install ros-humble-isaac-ros-<gem>` (JetPack‑6 Orin) — installs just one GEM, no container build.
3. Isaac ROS CLI **`venv`/`baremetal`** modes (Thor) — lighter than Docker.
4. **Plain ROS 2 + our node** — the right fit for Orin Nano (used in the rest of this chapter).

---

## 🐳 Why containers (and which one)

On Jetson, ROS 2 and Isaac ROS are run in **Docker containers** — this avoids polluting the L4T host and gives you the exact CUDA/TensorRT/ROS versions that match your JetPack. Two well‑trodden paths:

1. **`jetson-containers`** (Dusty‑NV) — the easiest way to get ROS 2 **plus** the ML stack (PyTorch, Ultralytics, OpenCV) in one image. Recipes live under [`packages/physicalAI/ros`](https://github.com/dusty-nv/jetson-containers/tree/master/packages/physicalAI/ros). **We use this for our detector node** (it needs PyTorch + Ultralytics).
2. **Isaac ROS dev container** (`isaac_ros_common`) — the official environment for the Isaac ROS GEMs.

> ✅ **Verified on a JetPack 7 (L4T r39) Orin Nano** (`sjsujetson-61`): ROS 2 Humble runs in a container, rclpy pub/sub works, and our node's `vision_msgs` messages build correctly. Notes below mark what was verified vs. reference steps.

### Quick sanity check (no GPU, ~1 GB) — confirm ROS 2 runs at all
```bash
docker pull ros:humble-ros-base
docker run --rm -it ros:humble-ros-base bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 run demo_nodes_cpp talker'   # Ctrl-C to stop
```
✅ *Verified:* on jetson‑61 this image runs and a Python rclpy publisher→subscriber round‑trips a message. This proves your Docker + ROS 2 install is healthy before pulling the much larger GPU images.

---

## 📦 Install the ROS 2 container with `jetson-containers`

```bash
# 1. Install the tooling (one time; needs sudo for apt + adds 'jetson-containers'/'autotag')
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# 2. See which prebuilt ROS image matches your JetPack/L4T
autotag ros                      # e.g. -> dustynv/ros:humble-desktop-r36.x  (or builds if none)

# 3. Run it, mounting the course repo so the toolkit is available inside
jetson-containers run -v /Developer:/Developer $(autotag ros)
```
`jetson-containers run` adds `--runtime nvidia`, `--network host`, device mounts, and X11 forwarding automatically. Inside the container you now have ROS 2 Humble on the GPU.

> If `autotag` has no prebuilt image for a brand‑new L4T (e.g. r39/JetPack 7), build a combined image once:
> ```bash
> jetson-containers build --name=ros_ml ros:humble-desktop pytorch opencv
> ```
> Then `jetson-containers run -v /Developer:/Developer ros_ml`.

### NVIDIA Isaac ROS path — Jetson Thor (JetPack 7.1, ROS 2 Jazzy)
> Use this **only on a Jetson Thor** (T5000/T4000), *not* the Orin Nano. Confirm the platform first:
> ```bash
> cat /etc/nv_tegra_release          # expect: R38 (release), REVISION: 4.0   (JetPack 7.1)
> ```

The current Isaac ROS uses a single CLI (`isaac-ros-cli`) with three isolation modes — **`docker`** (recommended), **`venv`** (apt + isolated Python, lighter), or **`baremetal`** (advanced). Per the official **[Isaac ROS getting‑started](https://nvidia-isaac-ros.github.io/getting_started/index.html)** docs:
```bash
# 1. add the NVIDIA Isaac ROS apt repo + install the CLI (one time)
sudo apt-get update && sudo apt-get install -y isaac-ros-cli

# 2. initialize an environment (pick ONE mode)
sudo isaac-ros init docker        # full container (recommended)
# sudo isaac-ros init venv        # lighter: system apt + isolated Python venv
# sudo isaac-ros init baremetal   # advanced: system-level install

# 3. enter the environment and pull/build the GEM(s) you need, e.g. accelerated YOLOv8
#    (each GEM has a Quickstart under github.com/NVIDIA-ISAAC-ROS)
```
On Thor, the accelerated GEMs (NITROS, TensorRT) have the memory/compute headroom the Orin Nano lacks. The older `isaac_ros_common` + `./scripts/run_dev.sh` flow is the JetPack‑6/Orin (Humble) equivalent.

---

## ▶️ Play existing examples inside the container

Open two shells into the **same** running container (`docker exec -it <name> bash`, then `source /opt/ros/humble/setup.bash` in each).

**ROS 2 demos (talker / listener):**
```bash
# shell 1
ros2 run demo_nodes_cpp talker
# shell 2
ros2 run demo_nodes_py listener        # prints "I heard: Hello World: N"
ros2 topic list                        # /chatter
ros2 topic echo /chatter --once
```

**turtlesim (needs an X display — `jetson-containers run` forwards it):**
```bash
ros2 run turtlesim turtlesim_node &
ros2 run turtlesim turtle_teleop_key    # drive the turtle with arrow keys
```

**Isaac ROS example (reference):** each GEM has a Quickstart, e.g. **`isaac_ros_yolov8`** — run an accelerated YOLOv8 on an image and view `vision_msgs/Detection2DArray`, or **`isaac_ros_visual_slam`** for VSLAM from a stereo/RGB‑D camera. See the per‑package READMEs under [NVIDIA-ISAAC-ROS](https://github.com/NVIDIA-ISAAC-ROS).

---

## 🛠️ Sample package: wrap our detector as a ROS 2 node

We ship a ready package, [`jetson/ros2_jetson_detect`](../../jetson/ros2_jetson_detect/), that wraps `jetson_object_detection_toolkit.py`. It **subscribes** to an image topic, runs any toolkit detector on the GPU, and **publishes** results:

| Topic | Type |
|---|---|
| `~/detections` | `vision_msgs/Detection2DArray` (boxes + class + score) |
| `~/detections_image` | `sensor_msgs/Image` (annotated, for RViz/rqt) |

The whole wrapper is ~120 lines — the relevant core:
```python
results = self.toolkit.detect(frame, **self._detect_kwargs())   # reuse the 05b toolkit
for box, score, name in zip(results['boxes'], results['scores'], results['class_names']):
    x1, y1, x2, y2 = (float(v) for v in box)
    det = Detection2D()
    det.bbox.center.position.x = (x1 + x2) / 2.0   # vision_msgs (Humble) layout
    det.bbox.center.position.y = (y1 + y2) / 2.0
    det.bbox.size_x, det.bbox.size_y = x2 - x1, y2 - y1
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id, hyp.hypothesis.score = str(name), float(score)
    det.results.append(hyp); arr.detections.append(det)
```
*(✅ This `vision_msgs` layout was verified against real Humble messages on jetson‑61.)*

### Build it — on a GPU‑PyTorch base (the smart choice)
The node needs **both** the toolkit's deps (PyTorch, Ultralytics, OpenCV) and ROS deps (`vision_msgs`, `cv_bridge`). Rather than install CUDA PyTorch from scratch, **start from the `jetson-llm` container `sjsujetsontool` already uses** — it ships CUDA PyTorch — and just add ROS 2 on top. Since that image is Ubuntu 24.04, the matching ROS 2 distro is **Jazzy** (our node's `vision_msgs` layout is the same on Humble and Jazzy):

```bash
# 1. a GPU container from the jetson-llm image, with the repo mounted
docker run -d --name ros2gpu --runtime nvidia --network host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all -v /Developer:/Developer jetson-llm:v1 sleep infinity

# 2. add ROS 2 Jazzy + the message/bridge packages, and YOLO
docker exec ros2gpu bash -c '
  apt-get update && apt-get install -y curl gnupg
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list
  apt-get update && apt-get install -y ros-jazzy-ros-base ros-jazzy-vision-msgs ros-jazzy-cv-bridge
  pip install ultralytics "numpy<2"      # see gotcha below
'
```
> ⚠️ **numpy gotcha:** `pip install ultralytics` pulls **numpy 2.x**, which breaks the image's prebuilt OpenCV (`ImportError: numpy.core.multiarray failed to import`). Pin **`numpy<2`** (as above) to match the prebuilt cv2/torch.

Then build the workspace (or run the node directly):
```bash
docker exec -it ros2gpu bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ros2_ws/src && ln -s /Developer/edgeAI/jetson/ros2_jetson_detect ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select ros2_jetson_detect && source install/setup.bash
```

### Run it
Our package includes a **`still_image_publisher`** (ROS's `image_publisher` mishandles a single `.jpg` — it treats it as an image sequence), so the launch file works on one image:
```bash
ros2 launch ros2_jetson_detect detect.launch.py model:=yolo image:=/Developer/models/bus.jpg
# inspect in another shell:
ros2 topic echo /jetson_detector/detections --once
ros2 run rqt_image_view rqt_image_view            # pick /jetson_detector/detections_image
```
With a **live camera** instead:
```bash
ros2 run usb_cam usb_cam_node_exe &               # or isaac_ros_argus for CSI cameras
ros2 launch ros2_jetson_detect detect.launch.py model:=yolo image_topic:=/image_raw
```
Switch detectors with `model:=` (`faster-rcnn`, `maskrcnn`, `detr`, `owl-vit` + `prompts:="person,car"`) — exactly as in 05b.

#### ✅ Verified GPU result (jetson‑61, Orin, ROS 2 Jazzy, YOLOv8n on CUDA)
```text
[jetson_detector]: Loading detector 'yolo' on cuda ...
[jetson_detector]: Detector ready.
[jetson_detector]: Subscribed to '/image_raw', publishing '~/detections'.
[jetson_detector]: 6 detections (1232 ms)

RECEIVED Detection2DArray with 6 detections:
  bus          0.87  center=(414,494) size=(782x526)
  person       0.87  center=(147,651) size=(197x504)
  person       0.85  center=(740,635) size=(140x485)
  person       0.83  center=(283,632) size=(123x452)
  person       0.26  center=(32,712)  size=(63x323)
  stop sign    0.25  center=(16,290)  size=(32x70)
```
YOLOv8 runs on the **Orin GPU inside the ROS 2 node** and publishes `vision_msgs/Detection2DArray` — ready to feed any downstream planner/visualizer. *(For a one‑shot smoke test without `colcon`, run [`test_gpu_detect.py`](../../jetson/ros2_jetson_detect/test_gpu_detect.py).)*

---

## 🧯 Troubleshooting

| Symptom | Fix |
|---|---|
| `colcon: command not found` | you're not in the ROS container, or didn't `source /opt/ros/humble/setup.bash` |
| `ModuleNotFoundError: jetson_object_detection_toolkit` | set `-p toolkit_path:=/Developer/edgeAI/jetson` or mount `/Developer` |
| `No module named torch/ultralytics` | use a `jetson-containers` image that includes PyTorch, or `pip install` inside it |
| `cv_bridge`/`vision_msgs` import error | `apt-get install ros-humble-cv-bridge ros-humble-vision-msgs` |
| rqt/turtlesim: no display | launch with `jetson-containers run` (sets up X11), or `xhost +local:` on the host |
| node loads but 0 detections | lower `confidence:=0.1`; for `owl-vit`/`grounding-dino` you must pass `prompts:=` |

## 📌 Summary
- ROS 2 = nodes + topics + messages; **Isaac ROS** adds GPU‑accelerated perception GEMs with **NITROS** zero‑copy transport.
- Run ROS 2 / Isaac ROS in **containers** via `jetson-containers` (easiest for ROS + ML) or `isaac_ros_common`.
- We wrapped the 05b detector as a ROS 2 node ([`ros2_jetson_detect`](../../jetson/ros2_jetson_detect/)) that publishes `vision_msgs/Detection2DArray` — drop it into any robot pipeline.

→ **Next:** [LeRobot & SO-ARM101 Teleop / Data Collection](05d_lerobot_so101.md)
