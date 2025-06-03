# ✅ Jetson Camp Student Setup Checklist

## 🖥️ Hardware Requirements

- Jetson Orin Nano with Ubuntu 22.04 (JetPack 5.x+)
- Monitor, keyboard, mouse
- Internet access (Wi-Fi or Ethernet)
- USB webcam (optional)
- Robotics hardware (if used): breadboard, wires, motors, sensors

---

## 🧰 Required Software on Host

### 1. System Update

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Docker

```bash
sudo apt install docker.io -y
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

> 🔁 Log out and back in after adding your user to the `docker` group.

### 3. NVIDIA Container Toolkit

```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 🖼️ GUI Setup for X11 Apps

```bash
xhost +local:docker
```

(Optional) Add to `~/.bashrc`:

```bash
export DISPLAY=:0
```

---

## 📦 Download the Docker Image

**Option A: From TAR File (offline)**

```bash
gunzip -c jetsoncamp-docker.tar.gz | docker load
```

**Option B: From Docker Hub (online)**

```bash
docker pull yourorg/jetsoncamp:latest
```

---

## 🚀 Launch the Container

```bash
bash run.sh
```

Includes:
- CUDA + PyTorch
- ROS 2 Humble ready
- OpenCV, LangChain, Transformers
- Network tools: tcpdump, nmap, fail2ban

---

## 📂 Volume Mount Setup

Ensure the following directories exist:

```bash
mkdir -p ~/ros2_ws/src
mkdir -p ~/isaac_ros_ws/src
mkdir -p ./curriculum
```

---

## 🧪 Daily Workflow Inside Container

```bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build
source install/setup.bash
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| Docker permission denied | Run `usermod -aG docker $USER` then reboot |
| GUI apps don't display | Run `xhost +local:docker` |
| ROS commands not found | Run `source /opt/ros/humble/setup.bash` |

Happy hacking!
