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
# View scheduler information
cat /proc/sched_debug
cat /sys/kernel/debug/sched_features

# Check process scheduling policy
chrt -p $$

# Set real-time priority (requires sudo)
sudo chrt -f -p 50 $$
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
ss -tuln  # TCP/UDP listening sockets
cat /proc/net/tcp
```

### 🔧 System Calls and Kernel Interface

System calls are the primary interface between user space and kernel space:

#### Common System Call Categories:

```c
// Process control
fork()    // Create new process
exec()    // Execute program
wait()    // Wait for child process
exit()    // Terminate process

// File operations
open()    // Open file
read()    // Read from file
write()   // Write to file
close()   // Close file

// Memory management
mmap()    // Map memory
munmap()  // Unmap memory
brk()     // Change data segment size

// Inter-process communication
pipe()    // Create pipe
socket()  // Create socket
shmget()  // Get shared memory
```

#### Tracing System Calls:

```bash
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
sudo cat /proc/device-tree/chosen/bootargs
sudo dmesg | grep -i boot

# View boot configuration
ls -la /boot/extlinux/
cat /boot/extlinux/extlinux.conf

# Check boot partition
sudo fdisk -l | grep boot
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
ls /proc/device-tree/
cat /proc/device-tree/model
cat /proc/device-tree/compatible

# Check specific hardware nodes
find /proc/device-tree -name "*gpu*" -type d
find /proc/device-tree -name "*camera*" -type d

# Decode device tree binary
sudo dtc -I fs -O dts /proc/device-tree > current_dt.dts
head -50 current_dt.dts
```

#### 3. **NVIDIA GPU Driver Architecture**

```bash
# Check GPU driver version
cat /proc/driver/nvidia/version
nvidia-smi  # If available

# View GPU device information
ls -la /dev/nvidia*
cat /proc/driver/nvidia/gpus/*/information

# Check GPU memory usage
cat /proc/driver/nvidia/gpus/*/used_memory

# CUDA device query
/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
```

#### 4. **Tegra-Specific Drivers and Services**

```bash
# List Tegra-specific kernel modules
lsmod | grep tegra
lsmod | grep nvgpu
lsmod | grep nvhost

# Check Tegra driver information
modinfo tegra_xudc
modinfo nvgpu

# View Tegra-specific devices
ls -la /dev/tegra*
ls -la /dev/nvhost*
```

### 🔌 Hardware Access and Driver Interface

#### 1. **GPIO (General Purpose Input/Output)**

```bash
# List available GPIO chips
ls /sys/class/gpio/
cat /sys/kernel/debug/gpio

# Export GPIO pin (example: pin 18)
echo 18 | sudo tee /sys/class/gpio/export
ls /sys/class/gpio/gpio18/

# Configure GPIO as output
echo out | sudo tee /sys/class/gpio/gpio18/direction

# Set GPIO value
echo 1 | sudo tee /sys/class/gpio/gpio18/value
echo 0 | sudo tee /sys/class/gpio/gpio18/value

# Read GPIO value
cat /sys/class/gpio/gpio18/value

# Unexport GPIO
echo 18 | sudo tee /sys/class/gpio/unexport
```

**Python GPIO Example:**
```python
#!/usr/bin/env python3
import Jetson.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BOARD)

# Setup GPIO pin 18 as output
GPIO.setup(18, GPIO.OUT)

try:
    while True:
        GPIO.output(18, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(18, GPIO.LOW)
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
```

#### 2. **I2C (Inter-Integrated Circuit)**

```bash
# List I2C buses
ls /dev/i2c-*
i2cdetect -l

# Scan I2C bus for devices (bus 1)
i2cdetect -y 1

# Read from I2C device (address 0x48)
i2cget -y 1 0x48 0x00

# Write to I2C device
i2cset -y 1 0x48 0x00 0xFF
```

**Python I2C Example:**
```python
#!/usr/bin/env python3
import smbus
import time

# Create I2C bus object
bus = smbus.SMBus(1)  # I2C bus 1

# Device address
device_addr = 0x48

try:
    # Read byte from register 0x00
    data = bus.read_byte_data(device_addr, 0x00)
    print(f"Read data: 0x{data:02X}")
    
    # Write byte to register 0x01
    bus.write_byte_data(device_addr, 0x01, 0xFF)
    print("Data written successfully")
    
except Exception as e:
    print(f"I2C Error: {e}")
finally:
    bus.close()
```

#### 3. **SPI (Serial Peripheral Interface)**

```bash
# Check SPI devices
ls /dev/spidev*

# SPI configuration in device tree
cat /proc/device-tree/spi*/status
```

**Python SPI Example:**
```python
#!/usr/bin/env python3
import spidev
import time

# Create SPI object
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Device 0

# Configure SPI
spi.max_speed_hz = 1000000  # 1 MHz
spi.mode = 0

try:
    # Send data
    data_to_send = [0x01, 0x02, 0x03, 0x04]
    response = spi.xfer2(data_to_send)
    print(f"Sent: {data_to_send}")
    print(f"Received: {response}")
    
except Exception as e:
    print(f"SPI Error: {e}")
finally:
    spi.close()
```

#### 4. **PWM (Pulse Width Modulation)**

```bash
# Check PWM chips
ls /sys/class/pwm/

# Export PWM channel 0
echo 0 | sudo tee /sys/class/pwm/pwmchip0/export

# Configure PWM
echo 1000000 | sudo tee /sys/class/pwm/pwmchip0/pwm0/period    # 1ms period
echo 500000 | sudo tee /sys/class/pwm/pwmchip0/pwm0/duty_cycle # 50% duty cycle
echo 1 | sudo tee /sys/class/pwm/pwmchip0/pwm0/enable          # Enable PWM

# Disable PWM
echo 0 | sudo tee /sys/class/pwm/pwmchip0/pwm0/enable
echo 0 | sudo tee /sys/class/pwm/pwmchip0/unexport
```

### 📹 Camera and Multimedia Subsystem

#### 1. **Camera Interface (CSI)**

```bash
# List video devices
ls /dev/video*
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
nvpmodel -q

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
cat /proc/cpuinfo | grep MHz
lscpu | grep MHz

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

**Python System Monitor Example:**
```python
#!/usr/bin/env python3
import psutil
import time
import subprocess

def get_gpu_temp():
    """Get GPU temperature from thermal zone"""
    try:
        with open('/sys/class/thermal/thermal_zone1/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
        return temp
    except:
        return None

def get_cpu_freq():
    """Get current CPU frequencies"""
    freqs = []
    for i in range(psutil.cpu_count()):
        try:
            with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq', 'r') as f:
                freq = int(f.read().strip()) / 1000  # Convert to MHz
                freqs.append(freq)
        except:
            freqs.append(0)
    return freqs

def monitor_system():
    """Monitor system performance"""
    print("System Performance Monitor")
    print("=" * 50)
    
    while True:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freqs = get_cpu_freq()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # GPU temperature
            gpu_temp = get_gpu_temp()
            
            # Display information
            print(f"\rCPU Usage: {[f'{p:5.1f}%' for p in cpu_percent]}")
            print(f"CPU Freqs: {[f'{f:6.0f}' for f in cpu_freqs]} MHz")
            print(f"Memory: {memory.percent:5.1f}% ({memory.used//1024//1024:,} MB / {memory.total//1024//1024:,} MB)")
            if gpu_temp:
                print(f"GPU Temp: {gpu_temp:5.1f}°C")
            print("\033[4A", end='')  # Move cursor up 4 lines
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n" * 4)
            print("Monitoring stopped.")
            break

if __name__ == "__main__":
    monitor_system()
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

---

## 🧪 Comprehensive Lab: Linux System Mastery

### Lab Objectives

By completing this lab, you will:

1. **Master System Architecture**: Understand kernel subsystems and hardware interfaces
2. **Implement Driver Access**: Work with GPIO, I2C, SPI, and camera interfaces
3. **Optimize Performance**: Configure power management and monitoring
4. **Secure the System**: Implement security best practices
5. **Debug and Monitor**: Use advanced troubleshooting techniques

### Lab Setup

```bash
# Create lab directory
mkdir -p ~/linux_mastery_lab
cd ~/linux_mastery_lab

# Install required packages
sudo apt update
sudo apt install -y htop iotop iftop strace tcpdump i2c-tools \
    python3-pip python3-dev build-essential cmake git

# Install Python packages
pip3 install psutil smbus2 spidev opencv-python
```

### Exercise 1: System Architecture Analysis

```bash
#!/bin/bash
# Create system_analysis.sh
cat > system_analysis.sh << 'EOF'
#!/bin/bash

echo "=== Jetson System Architecture Analysis ==="
echo

echo "1. Hardware Information:"
echo "Model: $(cat /proc/device-tree/model)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo "L4T Version: $(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo 'Not available')"
echo

echo "2. Memory Information:"
free -h
echo

echo "3. Storage Information:"
df -h
echo

echo "4. GPU Information:"
if [ -f /proc/driver/nvidia/version ]; then
    cat /proc/driver/nvidia/version
else
    echo "NVIDIA driver not found"
fi
echo

echo "5. Loaded Kernel Modules (Tegra-specific):"
lsmod | grep -E "(tegra|nvgpu|nvhost)" | head -10
echo

echo "6. Device Tree Information:"
echo "Compatible: $(cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' ')"
echo

echo "7. Power Model:"
nvpmodel -q 2>/dev/null || echo "nvpmodel not available"
echo

echo "8. Thermal Information:"
for zone in /sys/class/thermal/thermal_zone*/; do
    if [ -f "$zone/type" ] && [ -f "$zone/temp" ]; then
        type=$(cat "$zone/type")
        temp=$(cat "$zone/temp")
        temp_c=$((temp / 1000))
        echo "$type: ${temp_c}°C"
    fi
done
EOF

chmod +x system_analysis.sh
./system_analysis.sh
```

### Exercise 2: Hardware Interface Programming

**GPIO Control Script:**
```python
#!/usr/bin/env python3
# gpio_control.py
import time
import sys
import os

class GPIOController:
    def __init__(self, pin):
        self.pin = pin
        self.gpio_path = f"/sys/class/gpio/gpio{pin}"
        self.exported = False
        
    def export(self):
        """Export GPIO pin"""
        if not os.path.exists(self.gpio_path):
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(self.pin))
            self.exported = True
            time.sleep(0.1)  # Wait for export
            
    def unexport(self):
        """Unexport GPIO pin"""
        if self.exported and os.path.exists(self.gpio_path):
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(self.pin))
            self.exported = False
            
    def set_direction(self, direction):
        """Set GPIO direction (in/out)"""
        with open(f"{self.gpio_path}/direction", "w") as f:
            f.write(direction)
            
    def set_value(self, value):
        """Set GPIO value (0/1)"""
        with open(f"{self.gpio_path}/value", "w") as f:
            f.write(str(value))
            
    def get_value(self):
        """Get GPIO value"""
        with open(f"{self.gpio_path}/value", "r") as f:
            return int(f.read().strip())
            
    def __enter__(self):
        self.export()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unexport()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 gpio_control.py <pin_number>")
        sys.exit(1)
        
    pin = int(sys.argv[1])
    
    try:
        with GPIOController(pin) as gpio:
            # Configure as output
            gpio.set_direction("out")
            
            print(f"Blinking GPIO pin {pin}. Press Ctrl+C to stop.")
            
            while True:
                gpio.set_value(1)
                print(f"GPIO {pin}: HIGH")
                time.sleep(1)
                
                gpio.set_value(0)
                print(f"GPIO {pin}: LOW")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping GPIO control.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

**I2C Scanner:**
```python
#!/usr/bin/env python3
# i2c_scanner.py
import smbus
import sys

def scan_i2c_bus(bus_number):
    """Scan I2C bus for devices"""
    try:
        bus = smbus.SMBus(bus_number)
        devices = []
        
        print(f"Scanning I2C bus {bus_number}...")
        print("     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f")
        
        for row in range(0, 8):
            print(f"{row*16:02x}: ", end="")
            
            for col in range(0, 16):
                addr = row * 16 + col
                
                if addr < 0x03 or addr > 0x77:
                    print("   ", end="")
                    continue
                    
                try:
                    bus.read_byte(addr)
                    print(f"{addr:02x} ", end="")
                    devices.append(addr)
                except:
                    print("-- ", end="")
                    
            print()
            
        bus.close()
        return devices
        
    except Exception as e:
        print(f"Error scanning I2C bus {bus_number}: {e}")
        return []

def main():
    # Scan common I2C buses
    buses = [0, 1, 2]
    
    for bus_num in buses:
        try:
            devices = scan_i2c_bus(bus_num)
            if devices:
                print(f"\nDevices found on bus {bus_num}: {[hex(d) for d in devices]}")
            else:
                print(f"\nNo devices found on bus {bus_num}")
            print("-" * 50)
        except:
            print(f"Bus {bus_num} not available")

if __name__ == "__main__":
    main()
```

### Exercise 3: System Performance Monitor

```python
#!/usr/bin/env python3
# performance_monitor.py
import psutil
import time
import json
import argparse
from datetime import datetime

class JetsonMonitor:
    def __init__(self):
        self.data = []
        
    def get_cpu_info(self):
        """Get CPU information"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        
        cpu_info = {
            'usage_percent': cpu_percent,
            'frequencies': [f.current for f in cpu_freq] if cpu_freq else [],
            'load_average': psutil.getloadavg()
        }
        
        return cpu_info
        
    def get_memory_info(self):
        """Get memory information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
        
    def get_disk_info(self):
        """Get disk information"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        return {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': (disk_usage.used / disk_usage.total) * 100,
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0
        }
        
    def get_network_info(self):
        """Get network information"""
        net_io = psutil.net_io_counters()
        
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
    def get_thermal_info(self):
        """Get thermal information"""
        thermal_info = {}
        
        try:
            # Read thermal zones
            import glob
            for zone_path in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
                zone_name = zone_path.split('/')[-2]
                with open(zone_path, 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    thermal_info[zone_name] = temp
        except:
            pass
            
        return thermal_info
        
    def get_gpu_info(self):
        """Get GPU information (if available)"""
        gpu_info = {}
        
        try:
            # Try to read GPU memory usage
            with open('/proc/driver/nvidia/gpus/0000:00:00.0/used_memory', 'r') as f:
                gpu_info['memory_used'] = int(f.read().strip())
        except:
            pass
            
        return gpu_info
        
    def collect_data(self):
        """Collect all system data"""
        timestamp = datetime.now().isoformat()
        
        data_point = {
            'timestamp': timestamp,
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'disk': self.get_disk_info(),
            'network': self.get_network_info(),
            'thermal': self.get_thermal_info(),
            'gpu': self.get_gpu_info()
        }
        
        self.data.append(data_point)
        return data_point
        
    def display_data(self, data_point):
        """Display data in human-readable format"""
        print(f"\n=== System Monitor - {data_point['timestamp']} ===")
        
        # CPU
        cpu = data_point['cpu']
        print(f"CPU Usage: {[f'{p:5.1f}%' for p in cpu['usage_percent']]}")
        if cpu['frequencies']:
            print(f"CPU Freq:  {[f'{f:6.0f}' for f in cpu['frequencies']]} MHz")
        print(f"Load Avg:  {cpu['load_average']}")
        
        # Memory
        mem = data_point['memory']
        print(f"Memory:    {mem['percent']:5.1f}% ({mem['used']//1024//1024:,} MB / {mem['total']//1024//1024:,} MB)")
        if mem['swap_total'] > 0:
            print(f"Swap:      {mem['swap_percent']:5.1f}% ({mem['swap_used']//1024//1024:,} MB / {mem['swap_total']//1024//1024:,} MB)")
        
        # Disk
        disk = data_point['disk']
        print(f"Disk:      {disk['percent']:5.1f}% ({disk['used']//1024//1024//1024:,} GB / {disk['total']//1024//1024//1024:,} GB)")
        
        # Thermal
        thermal = data_point['thermal']
        if thermal:
            temps = [f"{zone}: {temp:.1f}°C" for zone, temp in thermal.items()]
            print(f"Thermal:   {', '.join(temps)}")
        
        # GPU
        gpu = data_point['gpu']
        if gpu:
            if 'memory_used' in gpu:
                print(f"GPU Mem:   {gpu['memory_used']} bytes")
                
    def save_data(self, filename):
        """Save collected data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Data saved to {filename}")
        
    def monitor(self, duration=60, interval=1, save_file=None):
        """Monitor system for specified duration"""
        print(f"Starting system monitoring for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                data_point = self.collect_data()
                self.display_data(data_point)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            
        if save_file:
            self.save_data(save_file)

def main():
    parser = argparse.ArgumentParser(description='Jetson System Performance Monitor')
    parser.add_argument('-d', '--duration', type=int, default=60, help='Monitoring duration in seconds')
    parser.add_argument('-i', '--interval', type=int, default=1, help='Sampling interval in seconds')
    parser.add_argument('-s', '--save', type=str, help='Save data to JSON file')
    
    args = parser.parse_args()
    
    monitor = JetsonMonitor()
    monitor.monitor(duration=args.duration, interval=args.interval, save_file=args.save)

if __name__ == "__main__":
    main()
```

### Exercise 4: Security Hardening Script

```bash
#!/bin/bash
# security_hardening.sh

echo "=== Jetson Security Hardening Script ==="
echo

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "Please run as root (use sudo)"
        exit 1
    fi
}

# Function to backup configuration files
backup_config() {
    local file=$1
    if [ -f "$file" ]; then
        cp "$file" "${file}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "Backed up $file"
    fi
}

# Update system
update_system() {
    echo "1. Updating system packages..."
    apt update && apt upgrade -y
    apt autoremove -y
    echo "System updated."
    echo
}

# Configure SSH security
secure_ssh() {
    echo "2. Securing SSH configuration..."
    backup_config "/etc/ssh/sshd_config"
    
    # SSH hardening
    sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
    sed -i 's/#Protocol 2/Protocol 2/' /etc/ssh/sshd_config
    
    # Add additional security settings
    echo "" >> /etc/ssh/sshd_config
    echo "# Security hardening" >> /etc/ssh/sshd_config
    echo "MaxAuthTries 3" >> /etc/ssh/sshd_config
    echo "ClientAliveInterval 300" >> /etc/ssh/sshd_config
    echo "ClientAliveCountMax 2" >> /etc/ssh/sshd_config
    echo "X11Forwarding no" >> /etc/ssh/sshd_config
    
    systemctl restart ssh
    echo "SSH secured and restarted."
    echo
}

# Configure firewall
setup_firewall() {
    echo "3. Setting up UFW firewall..."
    
    # Install UFW if not present
    apt install -y ufw
    
    # Reset UFW to defaults
    ufw --force reset
    
    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow 22/tcp
    
    # Allow common services (uncomment as needed)
    # ufw allow 80/tcp   # HTTP
    # ufw allow 443/tcp  # HTTPS
    # ufw allow 8080/tcp # Alternative HTTP
    
    # Enable firewall
    ufw --force enable
    
    echo "Firewall configured and enabled."
    echo
}

# Set up automatic security updates
setup_auto_updates() {
    echo "4. Setting up automatic security updates..."
    
    apt install -y unattended-upgrades
    
    # Configure unattended upgrades
    cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};

Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

    # Enable automatic updates
    cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF

    echo "Automatic security updates configured."
    echo
}

# Secure shared memory
secure_shared_memory() {
    echo "5. Securing shared memory..."
    
    backup_config "/etc/fstab"
    
    # Add tmpfs mount for /tmp if not present
    if ! grep -q "/tmp" /etc/fstab; then
        echo "tmpfs /tmp tmpfs defaults,noexec,nosuid,nodev,size=1G 0 0" >> /etc/fstab
    fi
    
    # Secure /dev/shm
    if ! grep -q "/dev/shm" /etc/fstab; then
        echo "tmpfs /dev/shm tmpfs defaults,noexec,nosuid,nodev 0 0" >> /etc/fstab
    fi
    
    echo "Shared memory secured."
    echo
}

# Set up fail2ban
setup_fail2ban() {
    echo "6. Setting up fail2ban..."
    
    apt install -y fail2ban
    
    # Configure fail2ban for SSH
    cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
EOF

    systemctl enable fail2ban
    systemctl restart fail2ban
    
    echo "fail2ban configured and started."
    echo
}

# Main execution
main() {
    check_root
    
    echo "Starting security hardening process..."
    echo "This script will:"
    echo "1. Update system packages"
    echo "2. Secure SSH configuration"
    echo "3. Set up UFW firewall"
    echo "4. Configure automatic security updates"
    echo "5. Secure shared memory"
    echo "6. Set up fail2ban"
    echo
    
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    
    update_system
    secure_ssh
    setup_firewall
    setup_auto_updates
    secure_shared_memory
    setup_fail2ban
    
    echo "=== Security Hardening Complete ==="
    echo "Please review the changes and reboot the system."
    echo "Backup files have been created with timestamps."
    echo
    echo "Important notes:"
    echo "- SSH root login has been disabled"
    echo "- Password authentication has been disabled"
    echo "- Make sure you have SSH key access before rebooting"
    echo "- UFW firewall is now active"
    echo "- fail2ban is monitoring SSH attempts"
}

# Run main function
main "$@"
```

### Lab Deliverables

1. **System Analysis Report**: Complete output from system_analysis.sh
2. **Hardware Interface Demos**: Working GPIO and I2C examples
3. **Performance Data**: JSON output from performance monitor
4. **Security Configuration**: Documentation of applied security measures
5. **Custom Scripts**: All developed Python and Bash scripts

### Assessment Criteria

- **Technical Implementation** (40%): Correct execution of all exercises
- **System Understanding** (30%): Demonstration of Linux architecture knowledge
- **Security Awareness** (20%): Proper implementation of security measures
- **Documentation** (10%): Clear documentation and code comments

---

## 📚 Summary and Next Steps

This comprehensive tutorial has covered:

✅ **Operating System Fundamentals**: Core concepts, kernel architecture, and system calls
✅ **Linux Architecture**: Deep dive into kernel subsystems and hardware abstraction
✅ **Jetson L4T Analysis**: Detailed exploration of NVIDIA's embedded Linux distribution
✅ **Hardware Interfaces**: Practical examples for GPIO, I2C, SPI, and camera access
✅ **System Administration**: Advanced monitoring, debugging, and security techniques
✅ **Hands-on Labs**: Comprehensive exercises with real-world applications

**Key Takeaways:**
- Linux provides a robust foundation for embedded AI applications
- Jetson L4T extends standard Linux with specialized drivers and optimizations
- Understanding system architecture is crucial for performance optimization
- Security hardening is essential for production deployments
- Practical experience with hardware interfaces enables IoT integration

**Future Learning Paths:**
- Advanced kernel programming and driver development
- Real-time systems and deterministic computing
- Container orchestration and edge deployment
- Custom Linux distribution creation
- Performance tuning and optimization techniques

This foundation prepares you for advanced topics in edge AI, embedded systems, and high-performance computing on Jetson platforms.


