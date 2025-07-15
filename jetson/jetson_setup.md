# 🚀 Jetson Orin Nano Setup Guide with SSD
## Jetson Orin Nano Super Developer Kit
The [Jetson Orin Nano Super Developer Kit] (https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) is shown in ![Diagram](./docs/figures/jetson-nano-dev-kit.png)

| Mark. | Name                                  | Note                     |
|-------|---------------------------------------|--------------------------|
| 1     | microSD card slot                     |                          |
| 2     | 40-pin Expansion Header               |                          |
| 3     | Power Indicator LED                   |                          |
| 4     | USB-C port                            | For data only            |
| 5     | Gigabit Ethernet Port                 |                          |
| 6     | USB 3.2 Type-A ports (x4)             | 10Gbps                   |
| 7     | DisplayPort Output Connector          |                          |
| 8     | DC Power Jack                         | 5.5mm x 2.5mm            |
| 9     | MIPI CSI Camera Connectors (x2)       | 22pin, 0.5mm pitch       |
| 10    | M.2 Slot (Key-M, Type 2280)           | PCIe 3.0 x4              |
| 11    | M.2 Slot (Key-M, Type 2230)           | PCIe 3.0 x2              |
| 12    | M.2 Slot (Key-E, Type 2230) (populated) |                          |

The Jetson Orin Nano 8GB Module has NVIDIA Ampere architecture with 1024 CUDA cores and 32 tensor cores, delivers up to 67 INT8 TOPS of AI performance, 8GB 128-bit LPDDR5 (102GB/s memory bandwidth), and 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU 1.5MB L2 + 4MB L3 (1.7GHz CPU Frequency). The power range is 7W–25W. You can flash the base L4T BSP on any of these storage medium using SDK Manager: SD card slot (1), external NVMe (2280-size on 10, 2230-size on 11), and USB drive on any USB port (4 or 6). 

Key components of the Carrier Board include: 
 - 2x MIPI CSI-2 camera
connectors (0.5mm pitch 22-pin flex connectors to connect CSI camera modules)
   - 15-pin connector like Raspberry Pi Camera Module v2, a 15-pin to 22-pin conversion cable is required.
   - supports the following: CAM0: CSI 1 x2 lane, CAM1: CSI 1 x2 lane or 1 x4 lane
 - 2x M.2 Key M, M.2 Key E 
   - M.2 Key M slot with x4 PCIe Gen3
   - M.2 Key M slot with x2 PCIe Gen3
   - M.2 Key E slot
 - 4x USB 3.2 Gen2 Type-A
 - USB Type-C for UFP (supports Host, Device and USB Recovery mode), can NOT be used to output display signal. 
   - *In host mode*: You can use this port as a downstream-facing port (DFP), just like the 4 Type-A ports.
   - *Device mode*: You can connect your Jetson to a PC and expose three logical USB device: USB Mass Storage Device (mount L4T-README drive), USB Serial, USB Ethernet (RNDIS) device to form a local area network in between your PC and Jetson (your Jetson being 192.168.55.1)
   - *USB Recovery mode*: use the PC to flash Jetson
 - Gigabit Ethernet
 - DisplayPort (8): 1x DP 1.2 (+MST) connector
 - 40-pin expansion header (UART, SPI, I2S, I2C, GPIO), 12-pin button header, and 4-pin fan header
 - DC power jack for 19V power input
 - Mechanical: 103mm x 90.5mm x 34.77mm

The ![40-pin Expansion Header](docs/figures/jetsonnano40pin.png)

> Reference: 
 - [Jetson Orin Nano Developer Kit User Guide - Hardware Specs](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/hardware_spec.html)
 - [Jetson datasheet](https://nvdam.widen.net/s/zkfqjmtds2/jetson-orin-datasheet-nano-developer-kit-3575392-r2).
 - [Jetson Orin Nano Developer Kit User Guide - Software Setup](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/software_setup.html) 
 - [Jetson Orin Nano Developer Kit Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
 - [Jetson Orin Nano Developer Kit Carrier Board Specification](https://developer.download.nvidia.com/assets/embedded/secure/jetson/orin_nano/docs/Jetson-Orin-Nano-DevKit-Carrier-Board-Specification_SP-11324-001_v1.3.pdf?__token__=exp=1750620110~hmac=a78678cf11fa4e52be5ec5dc4e403f4575431a0cf9a56fffe709f85327f8c267&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9)
 - [Jetson Orin Nano Initial Setup using SDK Manager](https://www.jetson-ai-lab.com/initial_setup_jon_sdkm.html)

## Setup Guide
This guide walks you through preparing a **Jetson Orin Nano Super Dev Kit** with SSD. If NVMe SSD is used for the OS and data, we need to use SDK Manager to flash the latest JetPack on the NVMe SSD. If you are flashing OS to the SD card, you may choose a different [guide](https://www.jetson-ai-lab.com/initial_setup_jon.html), that does not need a host X86 PC. 


The host x86 PC running Ubuntu 22.04 or Ubuntu 20.04 is required for NVIDIA SDK Manager. We also have the following processes:
- Install NVIDIA SDK Manager on host PC: [Documentation](https://docs.nvidia.com/sdk-manager/), [Download SDK Manager](https://docs.nvidia.com/sdk-manager/download-run-sdkm/index.html)
- M.2 NVMe SSD installation
- JetPack OS installation **on SSD**
- System boot
- Auto-run of lab provisioning script

🧰 Prerequisites

| Item | Notes |
|------|-------|
| ✅ Jetson Orin Nano Super Dev Kit | Includes power supply, heatsink, and case |
| ✅ M.2 NVMe SSD (e.g., Crucial P310) | 128GB–1TB recommended |
| ✅ Linux host with internet access | Ubuntu 18.04–22.04 for flashing JetPack |
| ✅ USB-C cable | For flashing from host |
| ✅ Keyboard + HDMI monitor (optional) | For first boot (can be headless later) |
| ✅ JetPack SDK Manager (on host PC) | [Install from NVIDIA](https://developer.nvidia.com/nvidia-sdk-manager) |
| ✅ `jetson_lab_client_setup.sh` | Your automated post-flash setup script |

### 🔧 Step 1: Install the SSD

1. Unbox the Jetson and place it on an anti-static surface.
2. Flip the board to expose the M.2 2280 slot on the **underside**.
3. Insert the **NVMe SSD** into the M.2 socket.
4. Secure with the included screw and standoff.
5. (Optional but recommended) Install a **low-profile SSD heatsink** for thermal control.

### 💻 Step 2: Flash JetPack to SSD (Not SD Card)

Jetson Orin Nano supports **native boot from SSD**.

### On your Ubuntu host:

1. Launch SDK Manager:
   ```bash
   sdkmanager
   ```

2. Login with your NVIDIA developer account. Connect your Jetson developer kit to your Ubuntu PC (via the USB-C port) and power it on in Forced Recovery mode.
   - *Forced Recovery mode*: While shorting pin 9 and pin 10 of J14 header (labeled with FC REC and GND) located below the Jetson module using a jumper pin, insert the DC power supply plug into the DC jack of the carrier board to power it on.
   - You can use `lsusb` command in the host machine to check the jetson device.

3. In the popup window, select " Jetson Orin Nano [8GB developer kit version] " and hit " OK ":
   - Uncheck " Host Machine "
   - Target Hardware is selected and show "Jetson Orin Nano modules"
   - SDK Version: Latest **JetPack version** (e.g., JetPack 6.x)
   - Click " Continue " button to proceed to the next step.

4. Select Software Components to Install. Leave the only " Jetson Linux " component checked, and uncheck everything. Click " Continue " button to proceed to the next step. It will prompt for the sudo command password.

5. SDK Manager will start downloading the "BSP" package and "RootFS" package. Once downloads are complete, it will untar the package and start generating the images to flash in the background. Once images are ready, SDK it will open the prompt for flashing.
   - On the flashing prompt, select " Runtime " for "OEM Configuration".
   - On the flashing prompt, select " NVMe "
   - Click "Flash" and the prompt popup will change like this.
   - Wait Flash successfully completes.

---

### 🧪 Step 3: First Boot on SSD

1. Connect Jetson to the Monitor:
   - If still plugged, *remove the jumper* from header (that was used to put it in Forced Recovery mode)
   - Connect the DisplayPort cable or adapter and USB keyboard and mouse to Jetson Orin Nano Developer Kit, or hook up the USB to TTL Serial cable.
   - *Unplug the power supply and put back in to power cycle*.
   - Jetson should now boot into the Jetson Linux (BSP) of your selected JetPack version from the storage of your choice.

2. Power up Jetson — it will boot from SSD automatically.

3. Complete **initial Ubuntu setup wizard** (username, password, time zone).

4. Verify SSD is rootfs:
   ```bash
   
   df -h /
   # Output should show something like: /dev/nvme0n1p1
   #Identify your NVMe SSD
   sjsujetson@sjsujetson-01:~$ lsblk
   sjsujetson@sjsujetson-01:~$ sudo nvme list
   sjsujetson@sjsujetson-01:~$ sudo nvme smart-log /dev/nvme0
   ```
5. Check SSD speed:
```bash
#Sequential write:
sjsujetson@sjsujetson-01:~$ sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
dd if=/dev/zero of=testfile bs=1G count=1 oflag=direct
3
1+0 records in
1+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.542918 s, 2.0 GB/s
#Sequential read:
sjsujetson@sjsujetson-01:~$ sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
dd if=testfile of=/dev/null bs=1G iflag=direct
3
1+0 records in
1+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.337109 s, 3.2 GB/s
```

6. Check JetPack version
   ```bash
   sjsujetson@sjsujetson-01:~$ dpkg-query --show nvidia-l4t-core
   nvidia-l4t-core	36.4.3-20250107174145
   sjsujetson@sjsujetson-01:~$ dpkg -l | grep nvidia*
   ```
It shows L4T 36.4.3, which corresponds to JetPack 6.2 [Official mapping reference](https://developer.nvidia.com/embedded/jetpack-archive). JetPack 6.2 is the latest production release of JetPack 6. This release includes Jetson Linux 36.4.3, featuring the Linux Kernel 5.15 and an Ubuntu 22.04-based root file system. The Jetson AI stack packaged with JetPack 6.2 includes CUDA 12.6, TensorRT 10.3, cuDNN 9.3, VPI 3.2, DLA 3.1, and DLFW 24.0.

7. Connect to Wifi:
```bash
sudo nmcli device wifi connect ${WIFI_SSID} password ${WIFI_PASSWORD}
#check what IP address your Jetson got assigned
ip addr
```

8. Install common packages
```bash
sudo apt update
sudo apt install nano
sudo apt install python3-pip
pip3 install --upgrade pip
sudo apt install htop iotop iftop nvtop sysstat
```

## 📁 Step 4: CUDA Installation and Samples
**Install CUDA nvcc:** Even though JetPack 6.2 (L4T 36.4.3) includes CUDA 12.6, the nvcc command is not installed by default on Jetson devices starting from JetPack 6.x. CUDA is split into host and device components. On Jetson, only the runtime components of CUDA are installed by default (for deploying and running models). The full CUDA toolkit (including nvcc, compiler, samples, etc.) is now optional.
To get nvcc and other development tools, install the CUDA Toolkit manually:
```bash
sjsujetson@sjsujetson-01:~$ sudo apt update
sjsujetson@sjsujetson-01:~$ sudo apt install cuda-toolkit-12-6
sjsujetson@sjsujetson-01:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```
Add CUDA 12.6 to your path (optional, already set automatically) or add these lines to ~/.bashrc
```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

```bash
#sudo apt install cmake #sudo apt purge cmake
wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-aarch64.tar.gz
tar -xf cmake-3.28.3-linux-aarch64.tar.gz
sjsujetson@sjsujetson-01:~/Developer$ export PATH=$HOME/Developer/cmake-3.28.3-linux-aarch64/bin:$PATH
sjsujetson@sjsujetson-01:~/Developer$ cmake --version
cmake version 3.28.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
#Disable All target_compile_features(...) for CUDA
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ find ../Samples -name CMakeLists.txt -exec sed -i '/target_compile_features.*cuda/d' {} +
#Skip Building nvJPEG Samples
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ mv ../Samples/4_CUDA_Libraries/nvJPEG ../Samples/4_CUDA_Libraries/nvJPEG.skip
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ mv ../Samples/4_CUDA_Libraries/nvJPEG_encoder ../Samples/4_CUDA_Libraries/nvJPEG_encoder.skip
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ sed -i 's|add_subdirectory(nvJPEG)|# add_subdirectory(nvJPEG)|' ../Samples/4_CUDA_Libraries/CMakeLists.txt
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ sed -i 's|add_subdirectory(nvJPEG_encoder)|# add_subdirectory(nvJPEG_encoder)|' ../Samples/4_CUDA_Libraries/CMakeLists.txt
#skip fp16ScalarProduct sample (__hfma2 is not supported on Jetson)
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ mv ../Samples/0_Introduction/fp16ScalarProduct/ ../Samples/0_Introduction/fp16ScalarProduct.skip
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ sed -i 's|add_subdirectory(fp16ScalarProduct)|# add_subdirectory(fp16ScalarProduct)|' ../Samples/0_Introduction/CMakeLists.txt
#
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ mv ../Samples/3_CUDA_Features/graphConditionalNodes/ ../Samples/3_CUDA_Features/graphConditionalNodes.skip
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ sed -i 's|add_subdirectory(graphConditionalNodes)|# add_subdirectory(graphConditionalNodes)|' ../Samples/3_CUDA_Features/CMakeLists.txt
#
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ mv ../Samples/4_CUDA_Libraries/watershedSegmentationNPP/ ../Samples/4_CUDA_Libraries/watershedSegmentationNPP.skip
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ sed -i 's|add_subdirectory(watershedSegmentationNPP)|# add_subdirectory(watershedSegmentationNPP)|' ../Samples/4_CUDA_Libraries/CMakeLists.txt
#Set Jetson Architecture, Removes any existing set(CMAKE_CUDA_ARCHITECTURES ...), Appends set(CMAKE_CUDA_ARCHITECTURES 72 87) (Jetson Xavier NX and Orin) to the bottom of each CMakeLists.txt 
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/$ find Samples -type f -name "CMakeLists.txt" -exec sed -i \
  -e '/set(CMAKE_CUDA_ARCHITECTURES/d' \
  -e '$a\set(CMAKE_CUDA_ARCHITECTURES 72 87)' {} +
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ cmake ../Samples -DCMAKE_CUDA_ARCHITECTURES="72;87"
-- Configuring done (14.4s)
-- Generating done (0.7s)
-- Build files have been written to: /home/sjsujetson/Developer/cuda-samples/build
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ make -j$(nproc)
#Run samples
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ ./0_Introduction/vectorAdd/vectorAdd
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ ./1_Utilities/deviceQuery/deviceQuery
...
Result = PASS
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ ./0_Introduction/matrixMul/matrixMul
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ ./0_Introduction/asyncAPI/asyncAPI #Demonstrates overlapping data transfers and kernel execution using CUDA streams. 
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ ./0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams #Demonstrates using Unified Memory with async memory prefetching
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ ./2_Concepts_and_Techniques/convolutionTexture/convolutionTexture #Demonstrates using CUDA’s texture memory to do efficient image convolution.
sjsujetson@sjsujetson-01:~/Developer/cuda-samples/build$ ./2_Concepts_and_Techniques/histogram/histogram #Computes histograms with/without shared memory. 
```

---

## 📁 Step 5: Docker Setup
Docker is already installed in Jetpack. Check the docker version and nvidia runtime:
```bash
sjsujetson@sjsujetson-01:~$ docker --version
Docker version 28.2.2, build e6534b4
sjsujetson@sjsujetson-01:~$ sudo docker info | grep -i runtime
 Runtimes: io.containerd.runc.v2 nvidia runc
 Default Runtime: runc
#test container
sjsujetson@sjsujetson-01:~$ sudo docker run --rm --runtime=nvidia --network=host   -v /usr/bin/tegrastats:/usr/bin/tegrastats   nvcr.io/nvidia/l4t-base:r36.2.0   bash -c "echo CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES && tegrastats"
```
Jetson often works best when Docker containers share the host network directly, `--network=host` bypasses Docker’s internal bridge networking and avoids the iptables dependency.

Enable docker without sudo:
```bash
sudo usermod -aG docker $USER
sudo reboot
#After logging back in, run:
sjsujetson@sjsujetson-01:~$ docker run --network=host hello-world #t works without sudo
```

NVIDIA now recommends using the standard pytorch container with the iGPU tag for Jetson devices on JetPack 6.x. This image includes CUDA, cuDNN, TensorRT, and PyTorch — no separate l4t-pytorch needed.
```bash
sjsujetson@sjsujetson-01:~$ docker pull nvcr.io/nvidia/pytorch:24.12-py3-igpu
sjsujetson@sjsujetson-01:~$ docker run --rm -it --runtime=nvidia --network=host nvcr.io/nvidia/pytorch:24.12-py3-igpu
root@sjsujetson-01:/workspace# python
Python 3.12.3 (main, Nov  6 2024, 18:32:19) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.__version__, torch.cuda.is_available())
2.6.0a0+df5bbc09d1.nv24.12 True
```

Run the container with GPU, port, and storage mounts:
```bash
docker run -it --rm \
  --runtime nvidia \
  --network host \
  -v ~/Developer:/workspace \
  --name jetson-pytorch \
  nvcr.io/nvidia/pytorch:24.12-py3-igpu
root@sjsujetson-01:/workspace# apt install -y python3-pip
root@sjsujetson-01:/workspace# pip install jupyterlab
root@sjsujetson-01:/workspace# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
#On your Jetson browser (or remote PC via IP):
```

In another terminal (while container is running), or click "control+P+Q" to exit the container without stopping it. Check the container status and commit the new changes, run:
```bash
sjsujetson@sjsujetson-01:~$ docker ps
CONTAINER ID   IMAGE                                   COMMAND                  CREATED          STATUS          PORTS     NAMES
5247e031b6af   nvcr.io/nvidia/pytorch:24.12-py3-igpu   "/opt/nvidia/nvidia_…"   39 minutes ago   Up 39 minutes             jetson-pytorch
sjsujetson@sjsujetson-01:~$ docker commit jetson-pytorch jetson-pytorch-v1
sha256:da28af1b9eede5a3ef5ff7a1a9473fef1cfa2cbdd749aec411b43010837d6e60
sjsujetson@sjsujetson-01:~$ docker images
REPOSITORY                TAG              IMAGE ID       CREATED         SIZE
jetson-pytorch-v1         latest           da28af1b9eed   4 seconds ago   9.71GB
```
Then next time you can run:
```bash
docker run -it --rm \
  --runtime nvidia \
  --network host \
  -v ~/Developer:/workspace \
  jetson-pytorch-v1
```
Stop a running container, if it’s running in the foreground, press: "Ctrl+C". If it’s running in the background (detached mode), find the container name or ID, then stop it:
```bash
docker ps
#docker stop <container_id_or_name>
sjsujetson@sjsujetson-01:~$ docker ps
CONTAINER ID   IMAGE                                   COMMAND                  CREATED          STATUS          PORTS     NAMES
5247e031b6af   nvcr.io/nvidia/pytorch:24.12-py3-igpu   "/opt/nvidia/nvidia_…"   43 minutes ago   Up 43 minutes             jetson-pytorch
sjsujetson@sjsujetson-01:~$ docker stop jetson-pytorch
```

If you want to delete a specific image:
```bash 
docker image ls
docker rmi jetson-pytorch #You can append -f to force removal
```

ROS Container:
```bash
sudo docker pull nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_1c5650affa65caa30889ccf92f639896
```

## Step 6: Jetson Monitoring
Check CPU, GPU, memory, and power usage (Jetson-specific)
```bash
sjsujetson@sjsujetson-01:~$ sudo tegrastats
#View GPU usage with tegrastats summary
watch -n 1 sudo tegrastats
```

htop – CPU & memory usage monitor (like top but better)
```bash
#Show CPU load and memory usage
sudo apt install htop
htop
```

Shows per-disk read/write rates (not per-process)
```bash
sudo apt install sysstat
iostat -xz 1
```

Use btop or bpytop for modern terminal dashboard
```bash
sudo apt install btop
btop
```


Power Management:
```bash
#Check current power mode
sjsujetson@sjsujetson-01:~$ sudo nvpmodel -q
NV Power Mode: MAXN_SUPER
2
#To list available modes:
sudo nvpmodel -q --verbose
#Set power mode: sudo nvpmodel -m 1
#check temperature
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
```

<!-- ## 📁 Step6: All-in-one script
1. From your gateway or a USB drive, copy your lab setup script:
   ```bash
   scp jetson_lab_client_setup.sh student@<jetson-ip>:~/setup.sh
   ```

2. SSH or login and run:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

This script:
- Mounts shared NFS
- Sets up Docker
- Pulls PyTorch container
- Configures model cache
- Installs SSH keys and remote tools -->

---

<!-- ## 🔄 Optional Auto-Run Script on First Boot

To automate script execution after flash:

1. Copy `jetson_lab_client_setup.sh` into `/etc/systemd/system/lab-setup.service` like:

```bash
[Unit]
Description=Jetson Lab First-Time Setup
After=network.target

[Service]
ExecStart=/bin/bash /home/student/jetson_lab_client_setup.sh
Type=oneshot
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```

2. Enable service:
```bash
sudo systemctl enable lab-setup
``` -->
## Clone SSDs
### Step 1: Clone the Master SSD to an Image File. 
Insert the master Jetson SSD using a USB M.2 enclosure and Identify the disk and use dd to clone:
```bash
lsblk
# Example: /dev/sdX (DO NOT use your system drive!)
sudo dd if=/dev/sdX of=jetson_lab_master.img bs=4M status=progress
sync
```
This creates a raw image (.img) of the entire SSD, including bootloader, rootfs, user data.
Repeat the following for each SSD you want to clone:
    1.	Insert a new SSD using the same USB enclosure.
    2.	Identify the device (e.g., /dev/sdY), then:

```bash
sudo dd if=jetson_lab_master.img of=/dev/sdY bs=4M status=progress
sync
```

Make Each Jetson Unique, Add a first-boot script (e.g., in jetson_lab_client_setup.sh) detect first boot and:
```bash
hostnamectl set-hostname jetsonXX
# or use MAC to auto-set hostname
```
Option of Post-clone script (manual or Ansible):
```bash
sudo hostnamectl set-hostname jetsonXX
sudo rm -f /etc/ssh/ssh_host_*
sudo dpkg-reconfigure openssh-server
```
Boot up each Jetson:
    •	Verify SSD is used: `df -h /`
Run hostname, docker, and test containers
    •	Check that shared NFS mounts and SSH keys work

### Step 1: if using Etcher
Step 1: Prepare the “Golden” Jetson SSD
On one Jetson Orin Nano:
    1.	Flash JetPack to the SSD using SDK Manager with rootfs on NVMe
    2.	Boot into Ubuntu and complete:
    •	User setup
    •	Script installs (e.g. jetson_lab_client_setup.sh)
    •	NFS mounts, hostname auto script, etc.

Identify the Device, On the host machine: lsblk
Create the Image Using dd
```bash
sudo dd if=/dev/sdX of=jetson_lab_master.img bs=4M status=progress
sync
```
This creates a raw disk image of the entire SSD.
    •	File size will equal the size of the SSD (e.g. 64GB, 128GB, etc.) You can compress it afterward: `gzip jetson_lab_master.img`

### Step 2: Write the Image to Target SSDs
Download from balena.io/etcher, Plug the M.2 NVMe SSD into a USB enclosure or multi-dock
Click “Flash from file”
    •	Choose your .img or .img.gz file (e.g., jetson_lab_master.img)
    Click “Select target”
    •	Choose the connected SSD (double-check to avoid overwriting your system drive!)

or run the dd command:
```bash
sudo dd if=jetson_lab_master.img of=/dev/sdY bs=4M status=progress
```
### Step 3: Boot device and Assign Unique Hostname
If set-unique-hostname.sh is present, it will:
    •	Detect the MAC address
    •	Rename the device (e.g., jetson14)
    •	Regenerate SSH host keys
    •	Log the name to /var/log/hostname-configured

🔹 Method B: Manual (for admin/debugging)
```bash
# View MAC address
cat /sys/class/net/eth0/address

# Set a new hostname (e.g., jetson22)
sudo hostnamectl set-hostname jetson22

# Update /etc/hosts
sudo sed -i 's/127.0.1.1.*/127.0.1.1\tjetson22/' /etc/hosts

# Regenerate SSH host keys
sudo rm /etc/ssh/ssh_host_*
sudo dpkg-reconfigure openssh-server
```
Verify Configuration
```bash
# Check hostname
hostname

# Check root filesystem is on SSD
df -h /

# Check IP and MAC
ip addr show eth0

# Check NFS mount
df -h | grep /mnt/shared

# Check Docker container alias
type lab

# Launch container
lab   # should start PyTorch container with shared volume
```
Notify Instructor or Log Status
```bash
echo "jetson22 OK" | curl -X POST -d @- http://192.168.100.1:5000/log
```
Or confirm visually via Cockpit or dashboard.
