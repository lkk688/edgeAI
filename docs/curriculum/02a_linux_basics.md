# 🧠 Introduction to Linux Basics on Jetson
**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

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
sjsujetson@sjsujetson-01:~$ head -n 1 /etc/nv_tegra_release
# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025
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
htop  # Already installed, If not installed: sudo apt install htop
ps aux | grep python
```

### Step 3: Examine Kernel and Drivers

```bash
sjsujetson@sjsujetson-01:~$ uname -r
5.15.148-tegra
sjsujetson@sjsujetson-01:~$ lsmod | grep nvgpu
nvgpu                2654208  23
host1x                180224  6 host1x_nvhost,host1x_fence,nvgpu,tegra_drm,nvidia_drm,nvidia_modeset
mc_utils               16384  3 nvidia,nvgpu,tegra_camera_platform
nvmap                 204800  79 nvgpu
sjsujetson@sjsujetson-01:~$ cat /proc/device-tree/model
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
```

### Step 4: Explore L4T Features

```bash
sjsujetson@sjsujetson-01:~$ head -n 1 /etc/nv_tegra_release
# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025
sjsujetson@sjsujetson-01:~$ ls /usr/lib/aarch64-linux-gnu/tegra
sjsujetson@sjsujetson-01:~$ nvpmodel -q
NV Power Mode: MAXN_SUPER
2
sjsujetson@sjsujetson-01:~$ tegrastats
07-14-2025 10:29:36 RAM 2147/7620MB (lfb 2x4MB) SWAP 1476/3810MB (cached 1MB) CPU [0%@729,0%@729,0%@729,0%@729,0%@729,0%@729] GR3D_FREQ 0% cpu@47.687C soc2@46.718C soc0@47.468C gpu@48.625C tj@48.625C soc1@48.25C VDD_IN 4556mW/4556mW VDD_CPU_GPU_CV 483mW/483mW VDD_SOC 1451mW/1451mW
```

<!-- ### Step 5: Create a Custom Script

```bash
mkdir -p ~/jetson_labs
cd ~/jetson_labs
echo -e "#!/bin/bash\necho Welcome to Jetson L4T!\nuname -a\ntegrastats --interval 1000" > check_system.sh
chmod +x check_system.sh
./check_system.sh
``` -->

<!-- ### Step 6: Modify System Settings (SAFE)

```bash
# Change hostname (do not reboot in lab)
echo "jetson-labtest" | sudo tee /etc/hostname
# Create a new non-sudo user for labs
sudo useradd -m labstudent
sudo passwd labstudent
``` -->

---

## 🔬 Deep Dive: Operating System Fundamentals

### 🧠 What is an Operating System?

An Operating System (OS) is a complex software layer that acts as an intermediary between computer hardware and application programs. It provides essential services and manages system resources.

#### Core Functions:

1. **Process Management**: Creating, scheduling, and terminating processes
2. **Memory Management**: Allocating and deallocating memory space
3. **File System Management**: Organizing and accessing data on storage devices
4. **Device Management**: Controlling and coordinating hardware devices
5. **Security and Access Control**: Protecting system resources and user data
6. **User Interface**: Providing command-line and graphical interfaces

#### OS Types:

- **Monolithic Kernel**: Single large kernel (Linux, Windows)
- **Microkernel**: Minimal kernel with services in user space (QNX, L4)
- **Hybrid Kernel**: Combination approach (macOS, Windows NT)
- **Real-time OS**: Deterministic response times (FreeRTOS, VxWorks)

### 🏛️ Linux Kernel Architecture Deep Dive

The Linux kernel is a monolithic kernel with modular design:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Space Applications                   │
├─────────────────────────────────────────────────────────────┤
│                    System Call Interface                     │
├─────────────────────────────────────────────────────────────┤
│  Process    │  Memory     │  File       │  Network    │ I/O │
│  Scheduler  │  Manager    │  System     │  Stack      │ Mgr │
├─────────────────────────────────────────────────────────────┤
│                    Device Drivers                           │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Abstraction                     │
└─────────────────────────────────────────────────────────────┘
```

#### Kernel Subsystems:

**1. Process Scheduler (CFS - Completely Fair Scheduler)**
```bash
# Check process scheduling policy
chrt -p $$

# Set real-time priority (requires sudo)
#sudo chrt -f -p 50 $$
```

**2. Memory Management Unit (MMU)**
```bash
# View memory information
cat /proc/meminfo
cat /proc/vmstat
cat /proc/buddyinfo  # Buddy allocator info

# Check memory mapping of a process
cat /proc/self/maps
cat /proc/self/smaps  # Detailed memory usage
```

**3. Virtual File System (VFS)**
```bash
# View mounted filesystems
cat /proc/mounts
mount | column -t

# File system statistics
cat /proc/filesystems
df -Th  # Disk usage by filesystem type
```

**4. Network Stack**
```bash
# Network interface statistics
cat /proc/net/dev
cat /proc/net/netstat

# Socket information
ss -tuln  # TCP/UDP listening sockets, need apt install iproute2
cat /proc/net/tcp
```

### 🔧 System Calls and Kernel Interface

#### 🧠 What Are System Calls?

A **system call** is how a user-space program requests a service from the operating system's **kernel** (the core part of the OS with full control over hardware).

Think of it like this:
- **User space** = your app or code
- **Kernel space** = the powerful, protected core of the OS
- **System call** = a formal way to knock on the kernel’s door and ask it to do something only it can do (like read a file, allocate memory, or send data).


#### 🧰 Common System Call Categories (with Use Cases)

| Category              | System Calls                              | What They Do                                                                 |
|-----------------------|-------------------------------------------|------------------------------------------------------------------------------|
| **Process Control**   | `fork()`, `exec()`, `wait()`, `exit()`     | Manage how processes are created, replaced, and terminated.                 |
| **File Operations**   | `open()`, `read()`, `write()`, `close()`   | Interact with files – similar to opening, reading from, and saving documents. |
| **Memory Management** | `mmap()`, `brk()`, `munmap()`              | Request and manage memory for applications.                                 |
| **IPC**               | `pipe()`, `socket()`, `shmget()`           | Allow processes to talk to each other or share memory.                      |

📦 **Example**: When you run a C program that reads a file:
```c
int fd = open("data.txt", O_RDONLY); // file open
read(fd, buffer, 100);               // file read
close(fd);                           // file close
```
This actually causes the kernel to get involved three times!

User App
   |
   |  open("file.txt")
   |
======== System Call Boundary ========
   |
   |---> Kernel handles VFS (Virtual File System)
              |
              |---> Filesystem driver
                        |
                        |---> Disk I/O


#### Tracing System Calls:
`strace` lets you watch how programs talk to the kernel.

```bash
# Install strace
apt install strace

# Trace system calls of a command
strace ls -la

# Trace specific system calls
strace -e trace=open,read,write cat /etc/passwd

# Trace system calls of running process
sudo strace -p <PID>

# Count system calls
strace -c ls -la
```

---

## 🚀 Jetson L4T: In-Depth Technical Analysis

### 🏗️ L4T Architecture Overview

L4T (Linux for Tegra) is NVIDIA's comprehensive embedded Linux solution:

```
┌─────────────────────────────────────────────────────────────┐
│                    Applications & Frameworks                 │
│  TensorRT │ cuDNN │ OpenCV │ GStreamer │ ROS │ Docker      │
├─────────────────────────────────────────────────────────────┤
│                    CUDA Runtime & Libraries                  │
│  CUDA Toolkit │ cuBLAS │ cuFFT │ Thrust │ NPP             │
├─────────────────────────────────────────────────────────────┤
│                    L4T User Space                           │
│  Ubuntu 20.04/22.04 │ systemd │ NetworkManager │ X11/Wayland│
├─────────────────────────────────────────────────────────────┤
│                    L4T Kernel Space                         │
│  Linux Kernel │ NVIDIA GPU Driver │ Tegra Drivers          │
├─────────────────────────────────────────────────────────────┤
│                    Bootloader & Firmware                    │
│  CBoot │ U-Boot │ TOS │ BPMP-FW │ Device Tree             │
├─────────────────────────────────────────────────────────────┤
│                    Tegra SoC Hardware                       │
│  ARM CPU │ NVIDIA GPU │ ISP │ VIC │ NVENC │ NVDEC │ I/O    │
└─────────────────────────────────────────────────────────────┘
```

### 📦 L4T Components Deep Dive

#### 1. **Bootloader Chain**

```bash
# Check bootloader information
sjsujetson@sjsujetson-01:~$ cat /proc
root=PARTUUID=2eea7ca9-2b58-4100-80a1-ba1b97bdc52b rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0 nospectre_bhb video=efifb:off console=tty0 bl_prof_dataptr=2031616@0x271E10000 bl_prof_ro_ptr=65536@0x271E00000

# View boot configuration
sjsujetson@sjsujetson-01:~$ ls -la /boot/extlinux/
total 24
drwxr-xr-x 2 root root  4096 Jun 22 14:24 .
drwxr-xr-x 5 root root 12288 Jun 22 15:19 ..
-rw-r--r-- 1 root root   938 Jun 22 14:24 extlinux.conf
-rw-r--r-- 1 root root   727 Apr 21 10:59 extlinux.conf.nv-update-extlinux-backup

sjsujetson@sjsujetson-01:~$ cat /boot/extlinux/extlinux.conf
TIMEOUT 30
DEFAULT primary

MENU TITLE L4T boot options

LABEL primary
      MENU LABEL primary kernel
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND ${cbootargs} root=PARTUUID=2eea7ca9-2b58-4100-80a1-ba1b97bdc52b rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0 nospectre_bhb video=efifb:off console=tty0

# Check boot partition
lsblk -f
```

**Boot Sequence:**
1. **BootROM**: Hardware initialization, loads CBoot
2. **CBoot**: NVIDIA's bootloader, loads kernel and device tree
3. **Linux Kernel**: Initializes hardware, loads drivers
4. **systemd**: User space initialization

#### 2. **Device Tree and Hardware Configuration**

```bash
# View device tree information
sjsujetson@sjsujetson-01:~$ ls /proc/device-tree/
'#address-cells'       dsu-pmu2             regulator-vdd-1v1-hub    soc1-throttle-alert
 aliases               firmware             regulator-vdd-1v8-ao     soc2-throttle-alert
 bpmp                  gpio-keys            regulator-vdd-1v8-hs     soctherm-oc-event
 bus@0                 gpu-throttle-alert   regulator-vdd-1v8-sys    sound
 camera-ivc-channels   hot-surface-alert    regulator-vdd-3v3-ao     sram@40000000
 chosen                interrupt-parent     regulator-vdd-3v3-pcie   __symbols__
 compatible            mgbe-vm-irq-config   regulator-vdd-3v3-sd     tegra-capture-vi
 cpus                  model                regulator-vdd-3v3-sys    tegra-carveouts
 cpu-throttle-alert    name                 regulator-vdd-5v0-sys    tegra-hsp@b950000
 cv0-throttle-alert    nvpmodel             reserved-memory          tegra_mce@e100000
 cv1-throttle-alert    opp-table-cluster0   rtcpu@bc00000            tegra-rtcpu-trace
 cv2-throttle-alert    opp-table-cluster1   scf-pmu                  thermal-zones
 dce@d800000           opp-table-cluster2   serial                   timer
 display@13800000      pmu                  serial-number            tsc_sig_gen@c6a0000
 dsu-pmu0              psci                '#size-cells'             vm-irq-config
 dsu-pmu1              pwm-fan              soc0-throttle-alert

sjsujetson@sjsujetson-01:~$ cat /proc/device-tree/model
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super

sjsujetson@sjsujetson-01:~$ cat /proc/device-tree/compatible
nvidia,p3768-0000+p3767-0005-supernvidia,p3767-0005nvidia,tegra234

```

#### 3. **NVIDIA GPU Driver Architecture**

```bash
# Check GPU driver version
sjsujetson@sjsujetson-01:~$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  540.4.0  Release Build  (buildbrain@mobile-u64-6336-d8000)  Tue Jan  7 17:35:12 PST 2025
GCC version:  collect2: error: ld returned 1 exit status

# View GPU device information
sjsujetson@sjsujetson-01:~$ ls -la /dev/nvidia*
crw-rw-rw- 1 root root 195,   0 Dec 31  1969 /dev/nvidia0
crw-rw-rw- 1 root root 195, 255 Dec 31  1969 /dev/nvidiactl
crw-rw-rw- 1 root root 195, 254 Dec 31  1969 /dev/nvidia-modeset

```

#### 4. **Tegra-Specific Drivers and Services**

```bash
# List Tegra-specific kernel modules
sjsujetson@sjsujetson-01:~$ lsmod | grep tegra

sjsujetson@sjsujetson-01:~$ lsmod | grep nvgpu
nvgpu                2654208  23
host1x                180224  6 host1x_nvhost,host1x_fence,nvgpu,tegra_drm,nvidia_drm,nvidia_modeset
mc_utils               16384  3 nvidia,nvgpu,tegra_camera_platform
nvmap                 204800  79 nvgpu

sjsujetson@sjsujetson-01:~$ lsmod | grep nvhost

# Check Tegra driver information
modinfo tegra_xudc
modinfo nvgpu

# View Tegra-specific devices
ls -la /dev/tegra*
ls -la /dev/nvhost*
```

### 🔌 Hardware Access and Driver Interface
`/opt/nvidia/jetson-io/jetson-io.py` is a tool to configure 40-pin header from Nvidia.

#### 1. **GPIO (General Purpose Input/Output)**
GPIO (General Purpose Input/Output) pins allow the Jetson board to interface with external hardware such as LEDs, buttons, sensors, and relays. Each pin can be configured as either:
    - Input – to read digital signals (e.g. button press)
    - Output – to send digital signals (e.g. turn on an LED)

On modern Jetson platforms, using libgpiod is generally better and more future-proof than using the older sysfs (/sys/class/gpio) GPIO interface.


```bash
# Install gpiod via sudo apt install gpiod (already installed)
#Discover GPIO chips:
sjsujetson@sjsujetson-01:~$ gpiodetect
gpiochip0 [tegra234-gpio] (164 lines)
gpiochip1 [tegra234-gpio-aon] (32 lines)
#List lines on a chip:
sjsujetson@sjsujetson-01:~$ gpioinfo gpiochip0
gpiochip0 - 164 lines:
	line   0:      "PA.00" "regulator-vdd-3v3-sd" output active-high [used]
	line   1:      "PA.01"       unused   input  active-high
    ....
```
gpiochip0 → 164 lines (main Tegra GPIO); gpiochip1 → 32 lines (AON = always-on GPIO, often used for wake, power buttons, etc.)

Physical Pin 7 on the 40-pin header is connected to GPIO3_PBB.00. That corresponds to GPIO line 84 on gpiochip0 (according to NVIDIA’s Orin Nano mapping).
```bash
sjsujetson@sjsujetson-01:~$ gpioinfo gpiochip0 | grep -i 84
	line  84:      "PN.00"       unused   input  active-high
```
tells us that:
	•	GPIO line 84 maps to PN.00 (port N, pin 0)
	•	It’s currently configured as an input
	•	It is unused, so safe to control

```bash
#Temporarily Set Line 84 to Output & Drive High:
gpioset gpiochip0 84=1 #automatically requests the line as output, sets it HIGH, and releases it after execution.
#To hold the output state (e.g., keep LED on), use the --mode=wait flag
gpioset --mode=wait gpiochip0 84=1
```

if use Python (with libgpiod): `sudo apt install python3-libgpiod`

`Jetson.GPIO` is another popular GPIO python library developed by Nvidia, but it uses sysfs (deprecated), which is deprecated in the Linux kernel since 4.8, and removed in Linux 5.10+.

#### 2. **I2C (Inter-Integrated Circuit)**

```bash
# List I2C buses
sjsujetson@sjsujetson-01:~$ ls /dev/i2c-*
/dev/i2c-0  /dev/i2c-1  /dev/i2c-2  /dev/i2c-4  /dev/i2c-5  /dev/i2c-7  /dev/i2c-9

sjsujetson@sjsujetson-01:~$ i2cdetect -l
i2c-0	i2c       	3160000.i2c                     	I2C adapter
i2c-1	i2c       	c240000.i2c                     	I2C adapter
i2c-2	i2c       	3180000.i2c                     	I2C adapter
i2c-4	i2c       	Tegra BPMP I2C adapter          	I2C adapter
i2c-5	i2c       	31b0000.i2c                     	I2C adapter
i2c-7	i2c       	c250000.i2c                     	I2C adapter
i2c-9	i2c       	NVIDIA SOC i2c adapter 0        	I2C adapter

# Scan I2C bus for devices (bus 1)
i2cdetect -y 1

```
You can use the `smbus2` library to use Python for I2C.


#### 3. **SPI (Serial Peripheral Interface)**

```bash
# Check SPI devices
sjsujetson@sjsujetson-01:~$ ls /dev/spidev*
/dev/spidev0.0  /dev/spidev0.1  /dev/spidev1.0  /dev/spidev1.1

```
To read from an SPI device on your Jetson Orin Nano using Python, you typically use the spidev module, which provides access to the SPI bus via /dev/spidev*.

#### 4. **PWM (Pulse Width Modulation)**

```bash
# Check PWM chips
sjsujetson@sjsujetson-01:~$ ls /sys/class/pwm/
pwmchip0  pwmchip1  pwmchip2  pwmchip3  pwmchip4
```

Controlling PWM (Pulse Width Modulation) on NVIDIA Jetson (e.g., Orin Nano, Xavier NX, Nano) in Python requires configuring the correct pins and using the Linux PWM sysfs interface or pwmchip character devices.

### 📹 Camera and Multimedia Subsystem

#### 1. **Camera Interface (CSI)**

```bash
# List video devices
ls /dev/video*
#sudo apt install v4l-utils
v4l2-ctl --list-devices

# Get camera capabilities
v4l2-ctl -d /dev/video0 --all
v4l2-ctl -d /dev/video0 --list-formats-ext

# Capture image using GStreamer
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920,height=1080' ! nvjpegenc ! filesink location=test.jpg

# Live camera preview
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720' ! nvvidconv ! xvimagesink
```

**Python Camera Example:**
```python
#!/usr/bin/env python3
import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Create camera capture
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display frame
        cv2.imshow('Jetson Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
```

### ⚡ Power Management and Performance

#### 1. **Power Models (nvpmodel)**

```bash
# Check current power model
sjsujetson@sjsujetson-01:~$ nvpmodel -q
NV Power Mode: MAXN_SUPER
2

# List available power models
nvpmodel -q --verbose

# Set power model (requires sudo)
sudo nvpmodel -m 0  # Maximum performance
sudo nvpmodel -m 1  # Balanced
sudo nvpmodel -m 2  # Power saving

# Check power model configuration
cat /etc/nvpmodel.conf
```

#### 2. **CPU Frequency Scaling**

```bash
# Check CPU frequency information
sjsujetson@sjsujetson-01:~$ lscpu | grep MHz
CPU max MHz:                        1728.0000
CPU min MHz:                        115.2000

# View CPU frequency scaling governors
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_available_governors

# Set CPU governor (requires sudo)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check current CPU frequencies
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

#### 3. **GPU Frequency and Power**

```bash
# Check GPU frequency
cat /sys/kernel/debug/bpmp/debug/clk/gpcclk/rate
cat /sys/kernel/debug/bpmp/debug/clk/gpcclk/state

# Monitor GPU usage
sudo tegrastats --interval 1000

# Check thermal zones
cat /sys/class/thermal/thermal_zone*/type
cat /sys/class/thermal/thermal_zone*/temp
```

---

## 🛠️ Advanced System Administration

### 📊 System Monitoring and Debugging

#### 1. **Process and Resource Monitoring**

```bash
# Advanced process monitoring
top -H  # Show threads
htop -t  # Tree view
iotop   # I/O monitoring
iftop   # Network monitoring

# Process tree
pstree -p
pstree -u  # Show users

# Detailed process information
ps aux --forest
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu

# Memory analysis
cat /proc/meminfo
free -h
vmstat 1 5  # Virtual memory statistics

# Disk I/O statistics
iostat -x 1 5
lsof +D /path/to/directory  # Files open in directory
```

#### 2. **System Logs and Debugging**

```bash
# systemd journal
journalctl -f  # Follow logs
journalctl -u service_name  # Service-specific logs
journalctl --since "2024-01-01" --until "2024-01-02"
journalctl -p err  # Error level and above

# Kernel logs
dmesg | tail -50
dmesg -T  # Human-readable timestamps
dmesg -l err,warn  # Error and warning levels

# System logs
tail -f /var/log/syslog
tail -f /var/log/kern.log

# Boot analysis
systemd-analyze
systemd-analyze blame  # Service startup times
systemd-analyze critical-chain  # Critical path
```

#### 3. **Network Diagnostics**

```bash
# Network interface configuration
ip addr show
ip route show
ip link show

# Network connectivity
ping -c 4 google.com
traceroute google.com
mtr google.com  # Continuous traceroute

# Port scanning and connections
nmap -sT localhost
ss -tuln  # Socket statistics
netstat -tuln  # Network connections
lsof -i :22  # Processes using port 22

# Network traffic analysis
sudo tcpdump -i eth0 -n
sudo tcpdump -i any port 22
```

### 🔧 Package Management and Software Installation

#### 1. **APT Package Manager**

```bash
# Update package lists
sudo apt update

# Upgrade packages
sudo apt upgrade
sudo apt full-upgrade

# Search packages
apt search keyword
apt show package_name

# Install packages
sudo apt install package_name
sudo apt install ./local_package.deb

# Remove packages
sudo apt remove package_name
sudo apt purge package_name  # Remove config files too
sudo apt autoremove  # Remove unused dependencies

# Package information
dpkg -l | grep package_name
dpkg -L package_name  # List files in package
dpkg -S /path/to/file  # Find package containing file
```

#### 2. **Snap Package Manager**

```bash
# List installed snaps
snap list

# Search snaps
snap find keyword

# Install snap
sudo snap install package_name

# Update snaps
sudo snap refresh
sudo snap refresh package_name

# Remove snap
sudo snap remove package_name

# Snap information
snap info package_name
```

#### 3. **Building from Source**

```bash
# Install build tools
sudo apt install build-essential cmake git

# Example: Building a simple C program
cat > hello.c << EOF
#include <stdio.h>
int main() {
    printf("Hello, Jetson!\n");
    return 0;
}
EOF

# Compile
gcc -o hello hello.c
./hello

# Example: CMake project
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

---

## 🔐 Security and User Management

### 👥 User and Group Management

```bash
# User information
id
whoami
groups
w  # Who is logged in
last  # Login history

# Create user
sudo useradd -m -s /bin/bash newuser
sudo passwd newuser

# Modify user
sudo usermod -aG sudo newuser  # Add to sudo group
sudo usermod -s /bin/zsh newuser  # Change shell

# Delete user
sudo userdel -r newuser  # Remove home directory too

# Group management
sudo groupadd newgroup
sudo usermod -aG newgroup username
sudo gpasswd -d username groupname  # Remove from group
```

### 🔒 File Permissions and Security

```bash
# File permissions
ls -la
stat filename

# Change permissions
chmod 755 filename  # rwxr-xr-x
chmod u+x filename  # Add execute for user
chmod g-w filename  # Remove write for group
chmod o=r filename  # Set read-only for others

# Change ownership
sudo chown user:group filename
sudo chown -R user:group directory/

# Special permissions
chmod +t directory/  # Sticky bit
chmod g+s directory/  # Set GID
chmod u+s filename   # Set UID (SUID)

# Access Control Lists (ACL)
getfacl filename
setfacl -m u:username:rw filename
setfacl -x u:username filename
```

### 🛡️ System Security

```bash
# Firewall (UFW)
sudo ufw status
sudo ufw enable
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw deny 23/tcp   # Telnet
sudo ufw delete allow 80/tcp

# SSH security
sudo systemctl status ssh
sudo nano /etc/ssh/sshd_config
# Recommended changes:
# PermitRootLogin no
# PasswordAuthentication no
# PubkeyAuthentication yes
sudo systemctl restart ssh

# Generate SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub  # Public key

# System updates and security
sudo apt update && sudo apt upgrade
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```


