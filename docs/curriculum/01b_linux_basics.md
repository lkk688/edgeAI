# 🧠 Introduction to Linux Basics on Jetson

This document introduces the fundamentals of the Linux operating system with a focus on practical usage for students working on NVIDIA Jetson devices like the Orin Nano. You'll also learn about NVIDIA's custom Linux version: L4T.

---

## 📚 What is Linux?

Linux is an open-source operating system that powers everything from phones to servers. Jetson uses a specialized version of Ubuntu Linux.

**Key Characteristics:**

* Open-source and customizable
* Multi-user, multi-tasking environment
* Command-line and graphical interfaces
* High performance and low resource usage

---

## 🏗️ Linux System Architecture

Linux architecture is typically divided into:

### 1. **Kernel**

* Core of the OS that interfaces with hardware
* Manages memory, processes, file systems, device drivers

### 2. **Shell**

* Interface that accepts user commands (CLI)
* Common shells: `bash`, `zsh`, `sh`

### 3. **File System Hierarchy**

* `/`: Root of the filesystem
* `/bin`, `/sbin`: Essential binaries
* `/usr`: Secondary hierarchy for user-installed software
* `/etc`: Configuration files
* `/home`: Personal user directories
* `/dev`, `/proc`, `/sys`: Virtual files for devices and kernel state

### 4. **User Space and Daemons**

* System and user-level applications
* Daemons provide background services (e.g., `systemd`, `networkd`)

---

## 🧩 What is Jetson L4T?

L4T (Linux for Tegra) is NVIDIA’s embedded Linux distribution tailored for Jetson SoCs (System-on-Chip). It extends standard Ubuntu with:

### 📦 Components:

* **Ubuntu Base Image**: L4T usually uses Ubuntu 20.04 or 22.04 LTS
* **Tegra Drivers**: GPU, ISP, CSI camera, I2C, SPI, PWM
* **Bootloader Stack**: U-Boot, CBoot, extlinux
* **CUDA Toolkit**: GPU-accelerated computing framework
* **TensorRT**: Inference optimizer and runtime
* **cuDNN**: Deep neural network acceleration libraries
* **Multimedia API**: V4L2, GStreamer, OpenMAX for camera and audio

### 🧬 Architecture:

* Jetson boots via UEFI or CBoot into Linux kernel
* Kernel loads NVIDIA drivers (GPU, CPU governors, DeepSleep)
* Device tree manages hardware layout (CPU/GPU/I/O)
* Userland launches graphical UI or SSH terminal

### 🔍 L4T Version Check:

```bash
head -n 1 /etc/nv_tegra_release
```

This shows JetPack version and internal driver details.

---

## 💻 Popular Linux Commands

| Command            | Description                          |
| ------------------ | ------------------------------------ |
| `ls`               | List files in a directory            |
| `cd`               | Change directory                     |
| `pwd`              | Print working directory              |
| `cp` / `mv` / `rm` | Copy, move, remove files             |
| `top` / `htop`     | Monitor system processes             |
| `cat` / `less`     | View file content                    |
| `sudo`             | Run command with admin privileges    |
| `apt`              | Install or update software packages  |
| `chmod` / `chown`  | Change file permissions/ownership    |
| `journalctl`       | View system logs (via systemd)       |
| `dmesg`            | Print kernel ring buffer (boot logs) |
| `lscpu`, `lsblk`   | List hardware info (CPU/disk)        |

---

## 🧪 Lab Exercise: Advanced Linux Practice on Jetson

### Step 1: File System Exploration

```bash
cd /
ls -lh
ls -lh /dev /proc /sys /boot /media
```

### Step 2: System Monitoring and Processes

```bash
top
htop  # If not installed: sudo apt install htop
ps aux | grep python
```

### Step 3: Examine Kernel and Drivers

```bash
uname -r
lsmod | grep nvgpu
cat /proc/device-tree/model
```

### Step 4: Explore L4T Features

```bash
head -n 1 /etc/nv_tegra_release
ls /usr/lib/aarch64-linux-gnu/tegra
nvpmodel -q
tegrastats
```

### Step 5: Create a Custom Script

```bash
mkdir -p ~/jetson_labs
cd ~/jetson_labs
echo -e "#!/bin/bash\necho Welcome to Jetson L4T!\nuname -a\ntegrastats --interval 1000" > check_system.sh
chmod +x check_system.sh
./check_system.sh
```

### Step 6: Modify System Settings (SAFE)

```bash
# Change hostname (do not reboot in lab)
echo "jetson-labtest" | sudo tee /etc/hostname
# Create a new non-sudo user for labs
sudo useradd -m labstudent
sudo passwd labstudent
```


