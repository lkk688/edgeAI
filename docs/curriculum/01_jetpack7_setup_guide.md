# 🚀 JetPack 7.2 Installation & Post-Setup Guide

**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

---

## 📌 Overview

This guide covers the complete process of upgrading a Jetson Orin Nano Developer Kit from **JetPack 6.x** to **JetPack 7.2** (L4T R39.2), followed by a full post-installation setup including:

- NVIDIA Container Runtime (CUDA-enabled Docker)
- NVIDIA PyTorch container
- `sjsujetsontool` CLI
- `uv` Python virtual environment with TensorRT and CUDA-enabled PyTorch

> ⚠️ **Important**: JetPack 7.2 is a **full OS reinstall** (Ubuntu 22.04 → Ubuntu 24.04). The NVMe SSD will be **completely erased**. Back up everything before proceeding.

---

## 🧭 Part 1: Pre-Upgrade Checklist

### 1.1 Verify Your Current System

SSH into your Jetson and verify the current JetPack version:

```bash
cat /etc/nv_tegra_release
dpkg-query --show nvidia-jetpack
```

Expected output for JetPack 6.x (upgrade-ready):
```
# R36 (release), REVISION: 4.7, ...
nvidia-jetpack    6.2.1+b38
```

> ✅ L4T **R36.x** (JetPack 6.x) UEFI firmware is required before upgrading to JetPack 7.2. If your device has firmware **older than R36.0**, follow the [JetPack 6.x Update Path](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html) first.

### 1.2 Back Up Critical Data

The NVMe SSD root partition will be **wiped**. Back up before proceeding:

```bash
# Back up Docker images to /Developer or external storage
docker save jetson-llm:v1 | gzip > /Developer/jetson-llm-v1.tar.gz
docker save cmpelkk/jetson-llm:latest | gzip > /Developer/jetson-llm-latest.tar.gz

# Back up home directory and custom configs
tar czf /Developer/home-backup.tar.gz \
  /home/sjsujetson/ \
  /etc/docker/daemon.json \
  ~/.bashrc ~/.ssh/

# Back up /Developer/models if space allows (these are large)
# du -sh /Developer/models/
```

> 💡 The `sjsujetsontool` script is already saved to GitHub — no need to back it up separately.

### 1.3 What You Need

| Item | Requirement |
|---|---|
| USB flash drive | ≥ 16 GB |
| Laptop/PC storage | ≥ 25 GB free (for ISO download) |
| Target storage | NVMe SSD already installed ✅ |
| Display + keyboard | Required for UEFI/GRUB interaction |
| Power supply | 19V DC barrel jack (not USB-C) |

---

## 🖥️ Part 2: Installing JetPack 7.2 via ISO

### 2.1 Download the Jetson ISO (on your Mac/PC)

Download the Jetson ISO for JetPack 7.2 (L4T r39.2):

```bash
# Direct download link (check https://developer.nvidia.com/embedded/jetpack for latest)
# Jetson ISO r39.2 (~4GB download)
curl -L -o jetsoninstaller-r39.2.0.iso \
  "https://developer.nvidia.com/downloads/embedded/L4T/r39_Release_v2.0/iso/jetsoninstaller-r39.2.0-2026-06-01-23-53-13-arm64.iso"
```

Or download via browser from the [JetPack Download Page](https://developer.nvidia.com/embedded/jetpack/downloads).

### 2.2 Flash ISO to USB Drive (on your Mac/PC)

Use **Balena Etcher** — do NOT use simple file copy:

1. Download [Balena Etcher](https://etcher.balena.io/)
2. Open Etcher → **Flash from file** → select the `.iso`
3. Select your USB flash drive as target
4. Click **Flash!** and wait for verification

> ⚠️ Do NOT write the ISO directly to the NVMe SSD. The USB is the **installer medium**; the SSD is the **install target**.

**Alternative — command line on Mac:**
```bash
# Find your USB disk (e.g., /dev/disk4)
diskutil list
diskutil unmountDisk /dev/disk4
sudo dd if=jetsoninstaller-r39.2.0.iso of=/dev/rdisk4 bs=1m status=progress
```

**Alternative — command line on Linux:**
```bash
# Find your USB disk (e.g., /dev/sdb)
lsblk
sudo dd if=jetsoninstaller-r39.2.0.iso of=/dev/sdb bs=4M status=progress oflag=sync
```

### 2.3 Boot the Jetson from USB

1. Connect: DisplayPort monitor, USB keyboard, USB flash drive
2. Make sure the NVMe SSD is installed in the M.2 slot
3. Connect the 19V power supply — the Jetson powers on automatically

**Select the USB in UEFI Boot Manager:**
1. When the NVIDIA boot splash appears, press `Esc` repeatedly
2. In the UEFI menu, select **Boot Manager**
3. Select your USB flash drive from the list → press `Enter`

### 2.4 QSPI Capsule Update (if prompted)

If the installer detects outdated QSPI firmware, it will prompt:

```
QSPI capsule update available. Press Y to accept.
```

> Press **`Y`** to confirm. The capsule update runs **twice** and the device may reboot. This is normal — do not power off.

### 2.5 Install Jetson Linux onto NVMe SSD

After the GRUB menu appears:

1. Select **"Install Jetson ISO r39.2"**
2. When prompted for target storage, select **NVMe SSD** (e.g., `nvme0n1`)
3. Confirm the install — **this erases the SSD**
4. Wait for installation to complete (~10–20 minutes)
5. When prompted, remove the USB drive and press `Enter` to reboot

### 2.6 First-Boot Setup

On first boot from the NVMe SSD, complete Ubuntu 24.04 setup:

1. Accept the NVIDIA Jetson software EULA
2. Select language, keyboard layout, timezone
3. Create username and password (e.g., `sjsujetson` / your password)
4. Connect to network (Wi-Fi or Ethernet)
5. Log in to the Ubuntu desktop

### 2.7 Unlock Maximum Performance

```bash
# Enable MAXN SUPER power mode for best AI performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

Or via desktop: Click power mode in top bar → **Power Mode** → **MAXN SUPER**

---

## ⚙️ Part 3: Verify the New Installation

Open a terminal and verify JetPack 7.2 is installed correctly:

```bash
# Check L4T version
cat /etc/nv_tegra_release
# Expected: # R39 (release), REVISION: 2.0, ...

# Check JetPack version
dpkg-query --show nvidia-jetpack
# Expected: nvidia-jetpack    7.2.x

# Check CUDA
ls /usr/local/ | grep cuda
nvcc --version
# Expected: Cuda compilation tools, release 12.8 (or newer)

# Check OS version
lsb_release -a
# Expected: Ubuntu 24.04 LTS
```

### Add CUDA to PATH (permanent)

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

---

## 🐳 Part 4: Install CUDA-Enabled Docker

JetPack 7.2 installs Docker, but you need to configure the **NVIDIA Container Runtime** so GPU access works inside containers.

### 4.1 Install Docker (if not already present)

```bash
# Check if Docker is installed
docker --version

# If not installed:
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### 4.2 Install NVIDIA Container Toolkit

```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 4.3 Configure Docker to Use NVIDIA Runtime

```bash
# Configure Docker daemon for NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify configuration
cat /etc/docker/daemon.json
```

Expected `daemon.json`:
```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

### 4.4 Test GPU Access in Docker

```bash
# Test: should print GPU info
docker run --rm --runtime=nvidia \
  nvcr.io/nvidia/l4t-base:r39.0 \
  nvidia-smi

# Or test with a quick CUDA container
docker run --rm --runtime=nvidia \
  --env NVIDIA_VISIBLE_DEVICES=all \
  nvcr.io/nvidia/l4t-base:r39.0 \
  nvcc --version
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...     Driver Version: ...     CUDA Version: 12.8               |
+-----------------------------------------------------------------------------+
```

### 4.5 Configure Docker Without sudo (Optional but Recommended)

```bash
sudo usermod -aG docker $USER
# Log out and back in, or run:
newgrp docker

# Verify
docker run --rm hello-world
```

---

## 🔥 Part 5: Pull the NVIDIA PyTorch Container

NVIDIA provides pre-built PyTorch containers optimized for Jetson with full CUDA support. These are the recommended way to run deep learning workloads.

### 5.1 Find the Right Container Tag

For JetPack 7.2 (L4T R39.2), use containers from the `nvcr.io/nvidia/l4t-pytorch` or `dustynv/` repositories:

```bash
# NVIDIA official L4T PyTorch container (check nvcr.io for latest r39.x tag)
docker pull nvcr.io/nvidia/l4t-pytorch:r39.0-pth2.6-py3

# Or use dusty-nv's community containers (often has more up-to-date options)
docker pull dustynv/pytorch:2.6-r39.0
```

> 💡 Check [dusty-nv's jetson-containers](https://github.com/dusty-nv/jetson-containers) for the most current community builds.

### 5.2 Verify PyTorch + CUDA in the Container

```bash
docker run --rm --runtime=nvidia \
  -v /usr/local/cuda:/usr/local/cuda \
  nvcr.io/nvidia/l4t-pytorch:r39.0-pth2.6-py3 \
  python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

Expected output:
```
PyTorch version: 2.6.0a0+...
CUDA available: True
CUDA version: 12.8
GPU: Orin (nvgpu)
```

### 5.3 Run Interactive PyTorch Shell

```bash
docker run --rm -it --runtime=nvidia \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /Developer:/Developer \
  -v /home/$USER:/home/$USER \
  nvcr.io/nvidia/l4t-pytorch:r39.0-pth2.6-py3 \
  bash
```

---

## 🛠️ Part 6: Install `sjsujetsontool`

Our custom Jetson CLI tool manages containers, AI models, and system tasks.

### 6.1 One-Line Install

```bash
curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh | bash
```

### 6.2 Add to PATH and Apply

```bash
# The installer adds ~/.local/bin to PATH in ~/.bashrc
source ~/.bashrc

# Verify
sjsujetsontool version
```

Expected output:
```
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
📦 JetPack Version: 7.2.x
🏷️  L4T BSP Revision: R39.2
⚙️  CUDA Version: 12.8
🧬 cuDNN Version: 9.x.x
🤖 TensorRT Version: 10.x.x
🧾 sjsujetsontool Script Version: v1.0.0
```

### 6.3 Pull and Update the Container

```bash
# Pull the latest container image (this takes a while — ~17GB)
sjsujetsontool update-container

# Then update the script itself
sjsujetsontool update-script

# Run a system health check
sjsujetsontool healthcheck
```

### 6.4 Create Developer Directories

```bash
sudo mkdir -p /Developer/models /Developer/edgeAI
sudo chown -R $USER:$USER /Developer

# Clone the edgeAI repo
git clone https://github.com/lkk688/edgeAI.git /Developer/edgeAI
```

---

## 🐍 Part 7: Install `uv` Python Environment Manager

`uv` is a fast Rust-based Python package manager that replaces `pip`/`venv`/`conda`. On Jetson, it's the recommended way to manage Python environments for AI development without conflicting with system packages.

### 7.1 Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
source $HOME/.local/bin/env
# Or add to ~/.bashrc:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
uv --version
# Expected: uv 0.x.x
```

### 7.2 Create the AI Development Virtual Environment

```bash
# Create a project directory
mkdir -p ~/ai-dev && cd ~/ai-dev

# Create a virtual environment using Python 3.12
uv venv .venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate
```

### 7.3 Install System-Backed NVIDIA Packages (TensorRT, cuDNN)

JetPack installs TensorRT and cuDNN as **system packages** in `/usr/lib/` and `/usr/include/`. To use them inside a `uv` venv, we link them properly and install the Python wheel.

```bash
# Step 1: Install the TensorRT Python wheel from the system install
# Find the wheel location (JetPack installs it here):
find /usr -name "tensorrt*.whl" 2>/dev/null
# Typically: /usr/local/lib/python3.12/dist-packages/ or via dpkg

# Install TensorRT Python bindings
pip install tensorrt

# Or install directly from JetPack's wheel directory:
pip install \
  /usr/lib/python3/dist-packages/tensorrt*.whl 2>/dev/null || \
  pip install tensorrt --extra-index-url https://pypi.nvidia.com

# Step 2: Link system CUDA/cuDNN libraries into the venv search path
# This allows packages to find libcuda.so, libcudnn.so, etc.
echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf
echo "/usr/lib/aarch64-linux-gnu/nvidia" | sudo tee -a /etc/ld.so.conf.d/jetson-nvidia.conf
sudo ldconfig
```

### 7.4 Install CUDA-Enabled PyTorch

On Jetson (aarch64), PyTorch must be installed from NVIDIA's JetPack wheel index — the standard PyPI wheels do **not** have CUDA support for aarch64.

```bash
# Activate venv (if not already active)
source ~/ai-dev/.venv/bin/activate

# Install PyTorch from NVIDIA's JetPack index (adjust for your JetPack 7.x release)
# Check https://developer.download.nvidia.com/compute/redist/jp/ for available wheels
pip install torch torchvision torchaudio \
  --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v72/pytorch/
```

> 💡 **Wheel index versions**: Replace `v72` with the JetPack version:
> - JetPack 7.2 → `jp/v72/`
> - JetPack 6.2 → `jp/v62/`
>
> Check available packages at: `https://developer.download.nvidia.com/compute/redist/jp/`

### 7.5 Install Core ML and AI Packages

```bash
# Inside the activated venv
uv pip install \
  numpy \
  scipy \
  scikit-learn \
  matplotlib \
  pillow \
  opencv-python-headless \
  transformers \
  accelerate \
  datasets \
  huggingface-hub \
  onnx \
  onnxruntime-gpu \
  langchain \
  langchain-community \
  fastapi \
  uvicorn \
  gradio \
  jupyter \
  ipykernel
```

### 7.6 Verify CUDA is Available in the venv

```bash
# Run this inside the activated venv
python3 - <<'EOF'
import sys
print("=" * 60)
print(f"Python: {sys.version}")

import platform
print(f"Platform: {platform.platform()}")

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Quick GPU tensor test
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    print(f"GPU matrix multiply: ✅ ({y.shape})")
else:
    print("❌ CUDA not available — check wheel installation")

try:
    import tensorrt as trt
    print(f"\nTensorRT: {trt.__version__}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    print("TensorRT Builder: ✅")
except ImportError as e:
    print(f"\nTensorRT: ❌ {e}")

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"\nONNX Runtime: {ort.__version__}")
    print(f"Providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("CUDA EP: ✅")
    else:
        print("CUDA EP: ❌ (only CPU available)")
except ImportError as e:
    print(f"\nONNX Runtime: ❌ {e}")

print("=" * 60)
EOF
```

Expected output:
```
============================================================
Python: 3.12.x ...
Platform: Linux-5.xx-tegra-aarch64-with-glibc2.39

PyTorch: 2.6.0a0+...
CUDA available: True
CUDA version: 12.8
GPU: Orin (nvgpu)
GPU memory: 7.4 GB
GPU matrix multiply: ✅ (torch.Size([1000, 1000]))

TensorRT: 10.x.x
TensorRT Builder: ✅

ONNX Runtime: 1.x.x
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
CUDA EP: ✅
============================================================
```

### 7.7 Register the venv as a Jupyter Kernel

```bash
# Install this venv as a Jupyter kernel named "ai-dev"
source ~/ai-dev/.venv/bin/activate
python3 -m ipykernel install --user --name=ai-dev --display-name="AI Dev (JetPack 7.2)"

# Launch JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Access from your laptop at: `http://<jetson-ip>:8888`

---

## 📦 Part 8: Install Additional Acceleration Packages

### 8.1 ONNX Runtime with TensorRT Execution Provider

```bash
# Install onnxruntime-gpu built for JetPack
pip install onnxruntime-gpu \
  --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v72/onnxruntime/
```

### 8.2 Torch-TensorRT (TorchTRT)

Allows compiling PyTorch models to TensorRT engines directly:

```bash
pip install torch-tensorrt \
  --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v72/pytorch/
```

### 8.3 CuPy — GPU-Accelerated NumPy

```bash
# Install CuPy for CUDA 12.x
pip install cupy-cuda12x
```

Test:
```bash
python3 -c "import cupy as cp; a = cp.array([1,2,3]); print('CuPy GPU:', a)"
```

### 8.4 Ollama (for Local LLMs)

```bash
# Install Ollama inside a container or natively
curl -fsSL https://ollama.ai/install.sh | sh

# Start and test
ollama serve &
ollama pull llama3.2
ollama run llama3.2 "What is NVIDIA Jetson?"
```

### 8.5 llama.cpp (GGUF inference)

```bash
# Install inside the container (sjsujetsontool provides this)
sjsujetsontool shell
# Inside container:
llama-cli --version
```

---

## 🔧 Part 9: System Configuration

### 9.1 Set Up Persistent Swap

JetPack 7.2 uses zram swap by default. Add disk swap for large model loading:

```bash
# Create 8GB swap on the SSD
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make persistent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

### 9.2 Set Hostname (for Lab Deployment)

```bash
sjsujetsontool set-hostname sjsujetson-XX
sudo reboot
```

### 9.3 Join Headscale VPN

```bash
sjsujetsontool tailscale up
```

### 9.4 Install SSH Keys from GitHub

```bash
sjsujetsontool setup-ssh your_github_username
```

### 9.5 Set Up NFS Mount (if needed)

```bash
sjsujetsontool mount-nfs nfs-server.local /srv/nfs/shared /mnt/nfs/shared
```

---

## ✅ Part 10: Final Verification Checklist

Run the built-in health check to confirm everything is working:

```bash
sjsujetsontool healthcheck
```

Manual checklist:

```bash
# 1. OS and JetPack
cat /etc/nv_tegra_release         # Should show R39.x
dpkg-query --show nvidia-jetpack  # Should show 7.2.x

# 2. CUDA
nvcc --version                    # Should show release 12.8 or newer
nvidia-smi                        # Should show GPU + driver

# 3. Docker with GPU
docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-base:r39.0 nvidia-smi

# 4. Python venv
source ~/ai-dev/.venv/bin/activate
python3 -c "import torch; print(torch.cuda.is_available())"  # True
python3 -c "import tensorrt as trt; print(trt.__version__)"  # 10.x.x

# 5. sjsujetsontool
sjsujetsontool version
sjsujetsontool tailscale status
```

---

## 🗂️ Quick Reference: Important Paths

| Path | Contents |
|---|---|
| `/usr/local/cuda-12.x/` | CUDA toolkit |
| `/usr/include/cudnn_version.h` | cuDNN headers |
| `/usr/lib/aarch64-linux-gnu/nvidia/` | NVIDIA system libs |
| `/Developer/` | Models, datasets, projects |
| `/Developer/models/` | GGUF and HF model files |
| `~/.local/bin/sjsujetsontool` | CLI tool |
| `~/ai-dev/.venv/` | Python virtual environment |
| `/etc/docker/daemon.json` | Docker NVIDIA runtime config |

---

## 🆘 Troubleshooting

### CUDA not available in Python
```bash
# Verify system libs are accessible
ldconfig -p | grep libcuda
# If missing, rebuild ldconfig:
sudo ldconfig
```

### TensorRT import fails in venv
```bash
# Ensure the system TRT libs are visible
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH
# Add permanently to ~/.bashrc
```

### Docker GPU access fails
```bash
# Restart Docker after toolkit install
sudo systemctl restart docker
# Test with explicit runtime flag
docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-base:r39.0 nvidia-smi
```

### ISO boots to black screen
> Older QSPI firmware cannot boot JetPack 7.2 ISO. Follow the [JetPack 6.x Update Path](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html) first.

### Out of disk space after install
```bash
# Remove old snap packages
sudo snap remove --purge chromium gnome-42-2204 gtk-common-themes
# Clean apt cache
sudo apt-get autoremove && sudo apt-get clean
```

---

## 📚 References

- [NVIDIA Jetson Orin Nano Quick Start Guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html)
- [JetPack 6.x Update Path](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)
- [NVIDIA JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [jetson-containers (dusty-nv)](https://github.com/dusty-nv/jetson-containers)
- [NVIDIA L4T PyTorch Wheels](https://developer.download.nvidia.com/compute/redist/jp/)
- [uv — Fast Python Package Manager](https://docs.astral.sh/uv/)
- [sjsujetsontool Guide](./00_sjsujetsontool_guide.md)

---

Made with 💻 by [Kaikai Liu](mailto:kaikai.liu@sjsu.edu) — [GitHub Repo](https://github.com/lkk688/edgeAI)
