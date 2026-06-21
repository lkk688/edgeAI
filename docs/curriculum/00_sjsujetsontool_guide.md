# 🧠 NVIDIA Jetson Orin Nano Student Guide

**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

---

## 📌 Overview

This guide introduces the **NVIDIA Jetson Orin Nano**, explains how to install and use our custom Jetson utility script `sjsujetsontool`, and provides step-by-step instructions for development tasks such as launching servers, running AI models, setting up Jupyter, and managing devices.

---

## 🧠 What Is NVIDIA Jetson Orin Nano?

The **Jetson Orin Nano** is a powerful, energy-efficient AI edge computing board by NVIDIA. Key features:

* ✅ 6-core ARM Cortex CPU
* ✅ Ampere GPU with up to 1024 CUDA cores
* ✅ Ideal for robotics, vision, AI model serving, and cyber experiments
* ✅ Supports JetPack SDK with Ubuntu, CUDA, cuDNN, TensorRT

---

## 🌐 Connecting to Jetson via ssh

If you have deployed Jetson devices in the same network of your client device, e.g., Macbook, Windows, Linux, you can connect to Jetson devices via its local IP address, e.g., `ssh username@192.168.xx.xx`

We already enabled the Jetson devices with mDNS, so you can access the device using the `.local` hostname from the client:

```bash
ssh username@jetson-hostname.local
```

For example:

```bash
ssh sjsujetson@sjsujetson-01.local
```

> If this doesn't work, make sure `avahi-daemon` is running on Jetson and that your network supports mDNS.

If you want to enable X11-forwarding, you can use 
```bash
% ssh -X sjsujetson@sjsujetson-01.local
sjsujetson@sjsujetson-01:~$ xclock #test x11
```

Another option is use ethernet over usb, which is already enabled in the Jetson device. You can connect your client device to the Jetson via a USB type C cable. You can use `ifconfig` to check the additional IP address, e.g., `192.168.55.100`. You can ssh into the Jetson device via:
```bash
% ssh sjsujetson@192.168.55.1
```


## 🌐 Mesh VPN Connection
All Jetson devices are connected through an overlay Layer 3 (L3) mesh VPN network, allowing them to communicate with each other using static IP addresses. To access another Jetson device in the mesh, simply use its assigned IP address. The IP address format is: `192.168.100.(10 + <number>)`

Here, <number> corresponds to the numeric suffix xx of the device’s hostname (sjsujetson-xx). For example, to SSH into the device sjsujetson-04, use:
```bash
ssh [username]@192.168.100.14
```
Replace [username] with your actual username.

---

## ⚙️ Installing `sjsujetsontool`

A command-line tool for Jetson-based workflows: container management, model serving, AI apps, and more.

### ✅ One-line install (no sudo required)

```bash
curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh | bash
```

For example:
```bash
student@sjsujetson-62:~$ curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh | bash
⬇️ Downloading sjsujetsontool from GitHub...
✅ Downloaded script.
📦 Installing to /home/student/.local/bin/sjsujetsontool
🛠️  Adding ~/.local/bin to your PATH...
✅ Added to /home/student/.bashrc
👉 Please run: source /home/student/.bashrc
✅ Installed successfully. You can now run: sjsujetsontool
```

After the script installation, run `source /home/student/.bashrc` before using `sjsujetsontool`. You can also run `sjsujetsontool update` to update the local script and container image. The container update takes a long time. For example,
```bash
student@sjsujetson-62:~$ sjsujetsontool update
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Developer Kit
🏷️  L4T BSP Revision: R39.2.0
Warning: xhost command failed. X11 forwarding may not work.
🔄 Running full update (script + setup-check + container)...
  Use 'update-container', 'update-script', or 'setup-check' to run individually.

📜 Step 1/2: Updating script from GitHub...
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Developer Kit
🏷️  L4T BSP Revision: R39.2.0
Warning: xhost command failed. X11 forwarding may not work.
⬇️ Updating sjsujetsontool script from GitHub...
📂 Backup saved to: /home/student/.local/bin/sjsujetsontool.bak
⬇️ Downloading latest script...
################################################################################################### 100.0%
✅ Script downloaded. Replacing current script...
✅ Script updated successfully. Re-run your command to use the new version.
✅ Chat client updated: /home/student/.local/bin/sjsujetsontool-chat.py

🐳 Step 2/2: Updating container image...
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Developer Kit
🏷️  L4T BSP Revision: R39.2.0
Warning: xhost command failed. X11 forwarding may not work.
🔍 Checking Docker image update...
⬇️ Pulling latest image from Docker Hub...
⠸ Downloading latest image... Please waitlatest: Pulling from cmpelkk/jetson-llm
⠴ Downloading latest image... Please waitDigest: sha256:739c319cb2ac5e1696a0fe4948f7a92279d0f22e01dc81ef52da46ecdd8cff24
Status: Image is up to date for cmpelkk/jetson-llm:latest
docker.io/cmpelkk/jetson-llm:latest
✅ Image downloaded successfully.                 
✅ Local container is already up-to-date.
```

> [!TIP]
> **Zero-Downtime Updates:** When a new version of the container image is downloaded, `sjsujetsontool` automatically removes the existing `jetson-dev` container. The next time you run any tool subcommand (such as `shell`, `llama-cli`, etc.), a new container is automatically recreated from the updated image—making updates completely seamless.

You can also update just the script or just the container independently:
```bash
sjsujetsontool update-script      # update only this CLI script from GitHub
sjsujetsontool update-container   # update only the Docker container image
```

Verify and check all available commands:
```bash
sjsujetsontool list
```

You can check the script and system versions:
```bash
student@sjsujetson-62:~$ sjsujetsontool version
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Developer Kit
🏷️  L4T BSP Revision: R39.2.0
Warning: xhost command failed. X11 forwarding may not work.
🧾 sjsujetsontool Script Version: v1.0.0
🧊 Docker Image: jetson-llm:v1
🔍 Image ID: sha256:739c319cb2ac5e1696a0fe4948f7a92279d0f22e01dc81ef52da46ecdd8cff24
```

The `sjsujetsontool` wraps python apps running via container and makes running code inside the container easy to use. `docker` without sudo is already setup in the jetson device. Check existing containers available in the Jetson:
```bash
sjsujetson@sjsujetson-01:~$ docker images
REPOSITORY                TAG              IMAGE ID       CREATED         SIZE
jetson-llm-v1             latest           8236678f7ef1   6 days ago      9.89GB
jetson-pytorch-v1         latest           da28af1b9eed   9 days ago      9.71GB
hello-world               latest           f1f77a0f96b7   5 months ago    5.2kB
nvcr.io/nvidia/pytorch    24.12-py3-igpu   ee796da7f569   6 months ago    9.63GB
nvcr.io/nvidia/l4t-base   r36.2.0          46b8e6a6a6a7   19 months ago   750MB
sjsujetson@sjsujetson-01:~$ sjsujetsontool shell #enter into the container
root@sjsujetson-01:/workspace#
```
if you face errors like "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?", restart docker:
```bash
sudo systemctl start docker
sudo systemctl status docker
```
> [!NOTE]
> On Jetson platforms running newer Ubuntu releases, Docker can fail to start due to an `iptables` mode mismatch (`nf_tables` instead of `iptables-legacy`). If Docker does not start, run the automatic fix tool:
```bash
sjsujetsontool dockerfix
```

The `\Developer` and `\Developer\models` folders in the jetson host are mounted to the container in the path of `\Developer` and `\models`

### ✅ Hostname changes (sudo required)
```bash
sjsujetson@sjsujetson-01:~$ hostname
sjsujetson-01
sjsujetson@sjsujetson-01:~$ sjsujetsontool set-hostname sjsujetson-02
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
🔧 Setting hostname to: sjsujetson-02
[sudo] password for sjsujetson: 
📝 Updating /etc/hosts...
🔄 Resetting machine-id...
🆔 Writing device ID to /etc/device-id
🔁 Please reboot for changes to fully apply.
sjsujetson@sjsujetson-01:~$ sudo reboot
```
You will need to use the new hostname to ssh into the device
```bash
% ssh -X sjsujetson@sjsujetson-02.local
sjsujetson@sjsujetson-02:~$ hostname
sjsujetson-02
```

For TA, run this additional steps:
```bash
sudo chfn -f "Student" student
sudo passwd student
sjsujetson@sjsujetson-02:/Developer/edgeAI$ sjsujetsontool force_git_pull
```
If you’re logged in as student and want to change your own password: `passwd`. You’ll be prompted to enter your current password, then the new password twice.

### ✅ SSD changes (sudo required)
Our default SSD image is 500GB. If the physical SSD devices installed in the jetson is larger than 500GB, you may only get 500GB in the system (shows 456G in the following example, but your `lsblk` command shows 1.8T for nvme0n1):
```bash
sjsujetson@sjsujetson-01:~$ df -h
Filesystem       Size  Used Avail Use% Mounted on
/dev/nvme0n1p1   456G   83G  351G  20% /
tmpfs            3.8G  136K  3.8G   1% /dev/shm
tmpfs            1.5G   27M  1.5G   2% /run
tmpfs            5.0M  4.0K  5.0M   1% /run/lock
/dev/nvme0n1p10   63M  112K   63M   1% /boot/efi
tmpfs            762M   92K  762M   1% /run/user/128
tmpfs            762M   80K  762M   1% /run/user/1000
sjsujetson@sjsujetson-01:~$ lsblk
NAME         MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
...
nvme0n1      259:0    0   1.8T  0 disk 
├─nvme0n1p1  259:1    0 464.3G  0 part /
...
```
You need to run the following command to re-claim the lost space.
```bash
sjsujetson@sjsujetson-01:~$ sudo apt install cloud-guest-utils
sjsujetson@sjsujetson-01:~$ sudo growpart /dev/nvme0n1 1
CHANGED: partition=1 start=3050048 old: size=973723080 end=976773128 new: size=3903979087 end=3907029135
sjsujetson@sjsujetson-01:~$ sudo resize2fs /dev/nvme0n1p1
resize2fs 1.46.5 (30-Dec-2021)
Filesystem at /dev/nvme0n1p1 is mounted on /; on-line resizing required
old_desc_blocks = 59, new_desc_blocks = 233
The filesystem on /dev/nvme0n1p1 is now 487997385 (4k) blocks long.

sjsujetson@sjsujetson-01:~$ df -h
Filesystem       Size  Used Avail Use% Mounted on
/dev/nvme0n1p1   1.8T   83G  1.7T   5% /
tmpfs            3.8G  136K  3.8G   1% /dev/shm
tmpfs            1.5G   27M  1.5G   2% /run
tmpfs            5.0M  4.0K  5.0M   1% /run/lock
/dev/nvme0n1p10   63M  112K   63M   1% /boot/efi
tmpfs            762M   92K  762M   1% /run/user/128
tmpfs            762M   80K  762M   1% /run/user/1000
```
Now, your `/dev/nvme0n1p1` has 1.8T space.

### ✅ Exter the Container Shell
Run the `sjsujetsontool shell` command line to enter into the shell of the container
```bash
sjsujetson@sjsujetson-01:/Developer/edgeAI$ sjsujetsontool shell
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
root@sjsujetson-01:/workspace# pip install transformers==4.37.0 #install transformer package
```

Exit the container via `exit`, and the container is still running
```bash
root@sjsujetson-01:/workspace# exit
exit
sjsujetson@sjsujetson-01:/Developer/edgeAI$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED      STATUS      PORTS     NAMES
c4010b14e9c0   8236678f7ef1   "/opt/nvidia/nvidia_…"   4 days ago   Up 4 days             jetson-dev
```

If you want to stop the container, you can use `sjsujetsontool stop`
```bash
sjsujetson@sjsujetson-01:/Developer/edgeAI$ sjsujetsontool stop
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
🛑 Stopping container...
jetson-dev
sjsujetson@sjsujetson-01:/Developer/edgeAI$ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

---

## 🧪 Common Usage Examples


### 🧾 `sjsujetsontool update`

Updates the script from GitHub (step 1) then pulls the latest Docker container image (step 2). A backup of the current script is saved automatically.

Use `update-script` or `update-container` to update individually.

### 🔬 `sjsujetsontool healthcheck`

Runs a deep system health check and prints a comprehensive diagnostic report. This is the recommended first step when troubleshooting or after receiving a new Jetson device.

```bash
sjsujetsontool healthcheck
```

Output covers:
- 📟 **Hardware & OS**: Jetson model, kernel, Ubuntu version, architecture
- 📦 **NVIDIA JetPack / L4T**: JetPack version (inferred from L4T BSP version if the meta-package is missing), L4T BSP revision (`R36.4.7`), L4T package version
- ⚙️ **CUDA**: Version detected via `nvcc` or via `/usr/local/cuda-*` directory, with PATH tip if nvcc is missing
- 🧬 **cuDNN**: Parsed from `/usr/include/cudnn_version.h` headers (with `dpkg-query` package fallback if development packages are missing)
- 🤖 **TensorRT**: Detected from installed `libnvinfer`/`tensorrt-libs` packages (supports TensorRT 8.x/10.x)
- 💾 **Memory**: RAM and Swap usage
- 💿 **Disk**: Filesystem usage with warnings if >80% full
- 🌡️ **Temperatures**: All thermal zones from `tegrastats`
- ⚡ **Power**: Per-rail power via INA3221 sensors (`VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC`)
- 🐳 **Docker**: Daemon status, version, NVIDIA runtime, available images, running containers, and active `iptables` driver mode
- 🔌 **Key Services**: Port status for JupyterLab (8888), Ollama (11434), llama.cpp (8000), FastAPI (8001), Gradio (7860)
- 📦 **Apt Upgrades**: Count and list of upgradable packages

Example output snippet:
```
════════════════════════════════════════════════════
🔬 Jetson Deep System Health Check
════════════════════════════════════════════════════

📟 Hardware & OS
  Model     : NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
  Kernel    : 5.15.148-tegra
  OS        : Ubuntu 22.04.5 LTS
  Arch      : aarch64

📦 NVIDIA JetPack / L4T
  JetPack   : 6.2.1+b38
  L4T BSP   : R36.4.7 (pkg: 36.4.7-20250918154033)

⚙️  CUDA
  CUDA      : 12.6 (via /usr/local/cuda-12.6, nvcc not in PATH)
  Tip       : Add to ~/.bashrc: export PATH=/usr/local/cuda-12.6/bin:$PATH

🧬 cuDNN
  cuDNN     : 9.3.0

🤖 TensorRT
  TensorRT  : 10.3.0.30-1+cuda12.5

💾 Memory
               total        used        free      ...
  Mem:         7.4Gi       1.8Gi       217Mi

💿 Disk Usage
  /dev/nvme0n1p1   1.8T   91G  1.7T   6% /

🌡️  Temperatures
  cpu@53.25C  soc2@51.09C  gpu@53.43C  tj@54.65C

⚡ Power
  VDD_IN               920mA @ 5048mV = 4644mW
  VDD_CPU_GPU_CV       112mA @ 5040mV = 564mW
  VDD_SOC              288mA @ 5040mV = 1451mW

🐳 Docker
  Status    : ✅ Running  (version 29.4.0)
  Runtime   : io.containerd.runc.v2 nvidia runc

🔌 Key Services
✅ JupyterLab is running on port 8888
❌ Ollama not running (port 11434 closed)

📦 Apt Upgradable Packages
  Available : 124 package(s) upgradable
  Run       : sjsujetsontool sysupgrade   (to apply safe upgrades)

════════════════════════════════════════════════════
✅ Health check complete.
════════════════════════════════════════════════════
```

### 🔄 `sjsujetsontool sysupgrade`

Safely upgrades Ubuntu system packages while **excluding** Jetson/NVIDIA/L4T packages to prevent breaking the JetPack BSP.

```bash
sjsujetsontool sysupgrade
```

> ⚠️ **Important**: Never run bare `sudo apt upgrade` on a Jetson — it can overwrite NVIDIA L4T kernel modules and break GPU support. Always use `sysupgrade` which filters out `nvidia-*`, `cuda-*`, `l4t-*`, and `libnvinfer*` packages.

The command will:
1. Run `apt-get update` to refresh the package index
2. Show you the list of upgradable packages
3. Ask for confirmation before applying
4. Upgrade only non-Jetson packages

### 🐳 `sjsujetsontool dockerfix`

Fixes the Docker daemon startup failure on Jetson by switching the system's `iptables` mode from `nf_tables` (default on Ubuntu 22.04+) to `iptables-legacy` (which the Jetson kernel supports).

```bash
sjsujetsontool dockerfix
```

The command will:
1. Detect current `iptables` mode. If set to `nf_tables`, switches system settings via `update-alternatives` (requires sudo password).
2. Restarts the Docker daemon service.
3. Performs a test pull and execution using NVIDIA GPU runtimes to ensure containerized CUDA works.

### 📋 `sjsujetsontool list`

Displays all available commands with usage examples.

### 🌐 `sjsujetsontool tailscale`

Manages the device's connection to the **SJSU headscale** (self-hosted Tailscale control server) at `headscale.forgengi.org`. This allows all Jetson devices to reach each other over a secure overlay network regardless of physical network location.

**Subcommands:**

| Subcommand | Description |
|---|---|
| `install` | Install Tailscale if not already present |
| `up [--force]` | Join the headscale network (with conflict checking) |
| `status` | Show current Tailscale connection status |
| `down` | Disconnect from the Tailscale network |

#### Check Status

```bash
sjsujetsontool tailscale status
```

Example output:
```
══════════════════════════════════════════════════
🌐 Tailscale Status
══════════════════════════════════════════════════
📦 Version : 1.98.4
🔧 Daemon  : active

✅ State   : Running
   Hostname  : sjsujetson-01
   IPs       : 100.82.159.9, fd7a:115c:a1e0::a636:9f09
   DNS Name  : sjsujetson-01.headscale.forgengi.org.
   Peers     : 8 connected

══════════════════════════════════════════════════
```

#### Install Tailscale (if not present)

```bash
sjsujetsontool tailscale install
```

Downloads and installs Tailscale from `https://tailscale.com/install.sh`. No action taken if already installed.

#### Join the Headscale Network

```bash
sjsujetsontool tailscale up
```

This command performs the following safety checks before joining:

1. ✅ **Verifies Tailscale is installed** (installs automatically if missing)
2. ✅ **Starts `tailscaled` service** if not already running
3. ⚠️ **Detects if already connected** to any Tailscale/Headscale network and warns:
   - If already on this headscale server → exits cleanly
   - If on a **different** network → asks for confirmation before switching
4. 🔍 **Checks for hostname conflicts** on the headscale server via API before joining
5. 🚀 **Joins the network** with `--accept-routes` for full mesh routing

Example — fresh device joining successfully:
```
══════════════════════════════════════════════════
🌐 Joining Headscale Network
══════════════════════════════════════════════════
✅ Tailscale already installed: 1.98.4
🖥️  This device hostname : sjsujetson-03
🌐 Headscale server     : https://headscale.forgengi.org

🔍 Checking for hostname conflicts on headscale...
  ✅ No hostname conflict: 'sjsujetson-03' is available on the headscale server.

🚀 Joining headscale network...

══════════════════════════════════════════════════
✅ Successfully joined headscale network!
   Hostname      : sjsujetson-03
   Tailscale IPs : 100.82.160.12
   Backend State : Running
   Server        : https://headscale.forgengi.org
══════════════════════════════════════════════════
```

Example — hostname conflict detected:
```
🔍 Checking for hostname conflicts on headscale...
  ⚠️  Hostname conflict detected: 'sjsujetson-01' is already registered on the headscale server.
  💡 To avoid conflicts, rename this device first:
       sjsujetsontool set-hostname <new-unique-name>
     Or force re-registration with:
       sjsujetsontool tailscale up --force
```

#### Force Re-registration

Use `--force` to disconnect from any current network and re-join even if a hostname conflict is detected:

```bash
sjsujetsontool tailscale up --force
```

> ⚠️ Use `--force` only when you intentionally want to replace an existing registration (e.g., after cloning a Jetson image). This will overwrite the old entry on the headscale server.

#### Disconnect

```bash
sjsujetsontool tailscale down
```

#### 🔧 Troubleshooting

| Symptom | Fix |
|---|---|
| `❌ Failed to join headscale network` | Check `journalctl -u tailscaled -n 30 --no-pager` |
| Auth key error | The pre-shared authkey may have expired — contact the TA for a new one |
| Hostname conflict | Run `sjsujetsontool set-hostname <new-name>` then retry |
| Can't reach headscale server | Check internet / VPN: `curl https://headscale.forgengi.org` |
| iptables warnings | Known Jetson L4T kernel quirk — does not affect connectivity |

### 🟢 `sjsujetsontool jupyter`

Launches JupyterLab on port 8888 from inside the Jetson's Docker container. It allows interactive Python notebooks for AI model testing, data exploration, and debugging.

```bash
sjsujetson@sjsujetson-01:~$ sjsujetsontool jupyter
....
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://hostname:8888/lab?token=3bbbf2fbea22e917bdbace45cb414bbaeb52f1251163adcf
        http://127.0.0.1:8888/lab?token=3bbbf2fbea22e917bdbace45cb414bbaeb52f1251163adcf
```
you can access the jupyter server via the the provided url. If you want to remote access the jupyter server from another computer, you can replace the hostname with the IP address of the device. 

### 🐍 `sjsujetsontool run <script.py>`

Runs any Python script inside the preconfigured container. Ensures all ML/AI libraries and GPU drivers are properly set up. The path of `script.py` should be accessible by the container, for example, the `\Developer` path:
```bash
sjsujetson@sjsujetson-01:/Developer/models$ sjsujetsontool run /Developer/edgeAI/jetson/test.py 
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
🐍 Running Python script: /Developer/edgeAI/jetson/test.py
📦 Python: 3.12.3 (main, Nov  6 2024, 18:32:19) [GCC 13.2.0]
🧠 Torch: 2.6.0a0+df5bbc09d1.nv24.12
⚙️  CUDA available: True
🖥️  CUDA version: Cuda compilation tools, release 12.6, V12.6.85
📚 Transformers: 4.37.0
🧬 HuggingFace hub: Version: 0.33.2
💡 Platform: Linux-5.15.148-tegra-aarch64-with-glibc2.39
🔍 Ollama: ✅ Ollama installed: ollama version is 0.9.2
```

### 🧠 `sjsujetsontool ollama`

This section introduces its integrated `ollama` command group, which allows you to manage, run, and query large language models inside a Docker container on your Jetson.


`sjsujetsontool ollama <subcommand>` enables local management and interaction with Ollama models from inside a persistent Jetson container.

Alternatively, you can run these top-level **shortcut commands** directly from the host terminal:
* `sjsujetsontool ollama-serve` (Starts the Ollama REST API server)
* `sjsujetsontool ollama-run [model]` (Runs the model interactively; defaults to `gemma4`)

Supported subcommands:

| Subcommand       | Description                                  |
|------------------|----------------------------------------------|
| `serve`          | Start Ollama REST API server (port 11434)    |
| `run <model>`    | Run specified model in interactive CLI       |
| `list`           | List all installed Ollama models             |
| `pull <model>`   | Download a new model                         |
| `delete <model>` | Remove a model from disk                     |
| `status`         | Check if Ollama server is running            |
| `ask`            | Ask model a prompt via REST API              |


🚀 Commands and Usage

1. Start the Ollama Server
Using the subcommand:
```bash
sjsujetsontool ollama serve
```
Or directly from the host using the shortcut:
```bash
sjsujetsontool ollama-serve
```

Starts the Ollama REST server inside the container, listening on http://localhost:11434.


2. Run a Model in CLI Mode
Using the subcommand:
```bash
sjsujetsontool ollama run mistral
```
Or directly from the host using the shortcut (defaults to `gemma4`):
```bash
sjsujetsontool ollama-run gemma4
```
💬 Launching model 'mistral' in CLI...
pulling manifest 
pulling ff82381e2bea: 100% ▕██████████████████▏ 4.1 GB                         
pulling 43070e2d4e53: 100% ▕██████████████████▏  11 KB                         
pulling 1ff5b64b61b9: 100% ▕██████████████████▏  799 B                         
pulling ed11eda7790d: 100% ▕██████████████████▏   30 B                         
pulling 42347cd80dc8: 100% ▕██████████████████▏  485 B                         
verifying sha256 digest 
writing manifest 
success 
>>> Send a message (/? for help)
```
Launches interactive terminal mode using the mistral model. Enter `\exit` to exit.

3. List Installed Models
```bash
$ sjsujetsontool ollama list
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
📃 Installed models:
NAME               ID              SIZE      MODIFIED           
mistral:latest     3944fe81ec14    4.1 GB    About a minute ago    
llama3.2:latest    a80c4f17acd5    2.0 GB    2 hours ago           
qwen2:latest       dd314f039b9d    4.4 GB    9 days ago            
llama3.2:3b        a80c4f17acd5    2.0 GB    9 days ago 
```
Shows a table of downloaded models and their sizes.

4. Download a New Model
```bash
sjsujetsontool ollama pull llama3
```
Pulls the specified model into the container. Examples include:
	•	phi3
	•	mistral
	•	llama3
	•	qwen:7b

5. Delete a Model
```bash
sjsujetsontool ollama delete mistral
```
Frees up disk space by removing the model.


6. Check Server Status
```bash
sjsujetsontool ollama status
```
Checks if the REST API is running on port 11434.

7. Ask a Prompt (with auto-pull + caching)
```bash
sjsujetsontool ollama ask "What is nvidia jetson orin?"
```
Uses the last used model, or you can specify one:
```bash
sjsujetsontool ollama ask --model mistral "Explain transformers in simple terms."
```
	•	Automatically pulls model if not available
	•	Remembers last used model in .last_ollama_model under workspace/


🧪 Example: Simple Chat Session

Pull and run mistral model
```bash
sjsujetsontool ollama pull mistral
sjsujetsontool ollama run mistral
```
Ask directly via REST
```bash
sjsujetsontool ollama ask --model mistral "Give me a Jetson-themed poem."
```

⸻

🧰 Troubleshooting
	•	Port already in use: Run sudo lsof -i :11434 and kill the process if needed.
	•	Model not found: Use sjsujetsontool ollama pull <model> manually before ask or run.
	•	Server not running: Start with sjsujetsontool ollama serve before using REST API.



### 🔬 `sjsujetsontool llama`

Starts the `llama.cpp` server (C++ GGUF LLM inference engine) on port 8000. Loads a `.gguf` model and serves an HTTP API for tokenized prompt completion.

#### ⚙️ Compiling & Updating `llama.cpp` inside Container

A current `llama.cpp` is needed for two reasons: direct Hugging Face loading via `-hf` (no manual
GGUF downloads), and **support for new model architectures** — e.g. **Qwen3.5** uses the `qwen35`
arch, which older builds reject with `unknown model architecture: 'qwen35'`. The build baked into the
container is **b9743** (verified below). To (re)build with CUDA inside the container:

1. **Clone the latest source** (a shallow clone is fine):
   ```bash
   cd /opt && git clone --depth=1 https://github.com/ggml-org/llama.cpp llamacpp-new && cd llamacpp-new
   ```
2. **Configure + build with CUDA for the Orin GPU** (`sm_87`):
   ```bash
   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
   # build the tools we use (text + multimodal/vision):
   cmake --build build --config Release -j"$(nproc)" \
     --target llama-cli llama-server llama-bench llama-completion llama-mtmd-cli
   ```
   > On a Jetson Orin Nano this CUDA build takes **~25 minutes**. Verified: build **b9743**, CUDA 12.6,
   > `sm_87`. Note in recent builds `llama-cli` is interactive-only — use **`llama-completion`** for
   > scriptable one-shot generation and **`llama-mtmd-cli`** for image input.
3. **Install into the path `sjsujetsontool` looks for** (`/opt/llama.cpp/build_cuda/bin`), so the
   `llama` / `llama-cli` shortcuts pick it up automatically:
   ```bash
   mkdir -p /opt/llama.cpp/build_cuda/bin
   cp build/bin/llama-* build/bin/*.so /opt/llama.cpp/build_cuda/bin/
   ```
   *(This is exactly how the prebuilt binaries are placed in the published container image, so all
   devices get the qwen35-capable build after `sjsujetsontool update`.)*

#### 🚀 Serving Gemma 4 E2B via Llama Server
You can launch the Gemma 4 E2B `llama-server` directly from the host system using the top-level shortcut:

```bash
# From the host:
sjsujetsontool llama
```

This shortcut automatically starts the persistent container and launches `llama-server` with CUDA acceleration on port `8080` (configured with the required batch sizes). 

Basic web UI can be accessed via browser at `http://localhost:8080`.

*(Alternatively, to run manually inside the container)*:
1. Enter the container: `sjsujetsontool shell`
2. Start the server: `llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S --host 0.0.0.0 --port 8080 --ubatch-size 2048 --batch-size 2048`

#### 💬 Querying the Model via HTTP API (OpenAI Compatible)
You can run a local chat completion query in another terminal (on the host machine or from within the container) using `curl`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain what is Nvidia Jetson?"}
    ]
  }'
```

#### 🖼️ Vision: ask about an image
Gemma 4 E2B is **multimodal**. Because `sjsujetsontool llama` loads the model with `-hf`, `llama-server` automatically downloads and loads the matching **mmproj** (multimodal projector), so image input works with no extra setup. Send an image with the OpenAI vision format (a base64 `image_url`); the helper script does the encoding:

```bash
# describe your own image (or omit --image to use a generated test image)
python3 jetson/jetson-llm/vision_test.py --image my_photo.jpg -p "What is in this image?"
# → VISION REPLY: A yellow circle on a blue square background.   (built-in test image)
```
See [`jetson/jetson-llm/vision_test.py`](../../jetson/jetson-llm/vision_test.py). The same client also works against a `gputool` server (Qwen3.5) — see the [gputool guide → vision](00c_gputool_guide.md). *(Pass images as base64 data URIs; the CUDA build does not fetch remote image URLs.)*

#### 🖥️ Running CLI Inference
You can run Gemma 4 E2B CLI inference directly from the host using the `llama-cli` shortcut. By default, it runs the **Gemma 4** architecture with full **GPU/CUDA hardware acceleration** enabled (`-ngl 99`):
```bash
# From the host:
sjsujetsontool llama-cli -p "Explain what is Nvidia Jetson"
```

The output will automatically show model loading and GPU status logs (e.g. `offloaded 27/27 layers to GPU`), print the thinking process/final response, and output token performance stats at the end of execution:
```text
Thinking Process:
...
[Prompt: 39.0 t/s | Generation: 19.8 t/s]
```

*(Alternatively, to run manually inside the container)*:
1. Enter the container: `sjsujetsontool shell`
2. Run inference on the GPU: `env LD_LIBRARY_PATH=/opt/llama.cpp/build_cuda/bin llama-cli -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S -ngl 99 -p "Explain what is Nvidia Jetson"`

#### 👁️ Multimodal (Vision) Inference with Gemma 4 E2B
Gemma 4 E2B is a natively multimodal Vision-Language Model (VLM) supporting image input.

##### 1. CLI Inference with Image Input
First, download the example image inside the container to the `/Developer` folder (or download it to `/Developer` on the host, as it maps directly):
```bash
# On the host or inside container:
wget -O /Developer/LoveSJ-hero-4.png https://www.sjsu.edu/visit/pics/LoveSJ-hero-4.png
```

To analyze the image using the host shortcut command (which defaults to GPU offloading and correct attention batch sizes):
> [!IMPORTANT]
> Because Gemma 4's vision encoder uses non-causal attention, the `sjsujetsontool llama-cli` shortcut pre-configures `--ubatch-size 2048` and `--batch-size 2048` to prevent the engine from crashing when processing image tokens.

```bash
# From the host (fully GPU-accelerated by default):
sjsujetsontool llama-cli --image /Developer/LoveSJ-hero-4.png -p "Describe this image and identify its main components."
```

*(Alternatively, to run manually inside the container)*:
```bash
# Inside the container (manually offloading to GPU):
env LD_LIBRARY_PATH=/opt/llama.cpp/build_cuda/bin llama-cli -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S \
  --image /Developer/LoveSJ-hero-4.png \
  --ubatch-size 2048 --batch-size 2048 \
  -ngl 99 \
  -p "Describe this image and identify its main components."
```

##### 2. Serving VLM via Llama Server
Simply start the server using the host shortcut:
```bash
# From the host:
sjsujetsontool llama
```

*(Alternatively, to run manually inside the container)*:
```bash
# Inside the container:
llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S \
  --ubatch-size 2048 --batch-size 2048 \
  --port 8080
```

##### 3. Querying the VLM Server via HTTP API (OpenAI Format)
You can query the server by passing a direct public URL to the image, or by base64-encoding a local image.

**Option A: Direct Public Image URL**
If the image is hosted publicly, pass the image URL directly in the payload:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://www.sjsu.edu/visit/pics/LoveSJ-hero-4.png"
            }
          },
          {
            "type": "text",
            "text": "What text is shown in this picture and what are the main elements?"
          }
        ]
      }
    ]
  }'
```

**Option B: Base64-Encoded Local Image**
If using a local file, convert it to a base64-encoded string:
```bash
# Get base64 string portably using python:
export IMG_B64=$(python3 -c "import base64; print(base64.b64encode(open('/Developer/LoveSJ-hero-4.png', 'rb').read()).decode('utf-8'))")

# Query the server:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,'"$IMG_B64"'"
            }
          },
          {
            "type": "text",
            "text": "Identify the main text and design features in this image."
          }
        ]
      }
    ]
  }'
```

#### 📊 Model comparison: default Gemma 4 E2B vs Qwen3.5 (Jetson Orin Nano)

The default **Gemma 4 E2B** GGUF is multi-GB and **slow to download**. We benchmarked the current
default against smaller [Unsloth **Qwen3.5**](https://huggingface.co/unsloth) GGUFs as lighter
alternatives, all at the same **`Q4_K_S`** quant used by `sjsujetsontool llama`.

> ⚠️ **Two gotchas found during testing:**
> 1. [`unsloth/gemma-4-E2B-it-qat-mobile`](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-mobile)
>    is **not a llama.cpp model** — it ships only `model.safetensors` (a QAT/mobile build), no GGUF —
>    so it cannot be served by `llama-server`. (The GGUF variant is `unsloth/gemma-4-E2B-it-GGUF`.)
> 2. Qwen3.5 uses the **`qwen35`** architecture, which the container's shipped llama.cpp (build 5752)
>    rejects with `unknown model architecture: 'qwen35'`. You must **rebuild llama.cpp** (we used
>    **b9743**, CUDA, `sm_87`) — see *Compiling & Updating llama.cpp* above — until the image ships a newer build.

**Test setup:** jetson-61 = Orin Nano (8 GB), JetPack 6 / L4T R36.4.7, llama.cpp **b9743** CUDA,
`-ngl 99` (all layers on GPU), quant `Q4_K_S`, ~47 Mbps link.

| Model | Params | On-disk | Download (≈5.9 MB/s) | Weights in RAM | Prompt `pp512` | Generation `tg128` | Math (3 Qs) |
|---|---|---|---|---|---|---|---|
| `gemma-4-E2B-it` **(current default)** | 4.65 B | 3.04 GB | ~8.6 min (516 s) | 2.82 GiB | ~644 tok/s | ~20 tok/s | ✅ 3 / 3 |
| `Qwen3.5-0.8B` | 0.77 B | 0.52 GB | ~1.5 min (89 s) | 0.49 GiB | ~1205 tok/s | **~35 tok/s** | ⚠️ 2 / 3 |
| `Qwen3.5-2B` | 1.94 B | 1.26 GB | ~3.6 min (216 s) | 1.17 GiB | ~750 tok/s | **~21 tok/s** | ✅ 3 / 3 |
| `Qwen3.5-4B` | 4.33 B | 2.68 GB | ~7.6 min (456 s) | 2.49 GiB | ~307 tok/s | **~10 tok/s** | ✅ 3 / 3 |

Peak system RAM (incl. ~2–2.9 GB idle baseline + page cache) stayed under ~7.3 GB on the 8 GB board
for every model — so even the 3 GB-class ones fit, with little headroom.

> 💡 **Headline:** the default **Gemma 4 E2B is the *largest* download (3.04 GB) yet not the fastest** —
> `Qwen3.5-2B` matches its accuracy (3/3) and generation speed (~21 vs ~20 tok/s) at **~⅓ the download
> size** — **and it is also multimodal** (`Qwen/Qwen3.5-2B` is a *unified vision-language* model;
> the Unsloth GGUF repo ships an `mmproj`), so Gemma's vision is no longer a unique advantage. See the
> VLM verification below.

**Accuracy** — three high-school questions (`17 × 23 = 391`; `2x + 5 = 17 → x = 6`;
right-triangle legs 6, 8 → hypotenuse `10`), greedy decoding:

- **Gemma 4 E2B**, **Qwen3.5-2B**, **Qwen3.5-4B**: all three correct, clean concise answers — reliable.
  (Gemma and Qwen3.5-2B answer directly; Qwen3.5-4B "thinks" first, then answers.)
- **Qwen3.5-0.8B**: reached the right working (391; x = 6) but **over-thinks badly** — on the
  hypotenuse question it kept second-guessing and never committed to a final answer even at 3072
  tokens. Capable but unreliable at producing a final answer.

**Recommendation:** for a fast-downloading, reliable everyday model on the Orin Nano,
**`Qwen3.5-2B:Q4_K_S`** is the sweet spot — same accuracy, speed, **and vision** as the default
Gemma 4 E2B at **~⅓ the download**. Use Qwen3.5 `4B` for more reasoning headroom (~10 tok/s);
avoid `0.8B` for tasks needing a definite answer.

```bash
# text server (inside the container, with an up-to-date llama.cpp build):
llama-server -hf unsloth/Qwen3.5-2B-MTP-GGUF:Q4_K_S --host 0.0.0.0 --port 8080 -ngl 99
```

##### 👁️ VLM (vision) — verified on Qwen3.5-2B

`Qwen/Qwen3.5-2B` is a **unified vision-language** model. The Unsloth GGUF repo ships a vision
projector (`mmproj-F16.gguf`, 0.67 GB); pair it with the text GGUF and run the multimodal CLI
(`llama-mtmd-cli`) or `llama-server --mmproj`:

```bash
llama-mtmd-cli \
  -m Qwen3.5-2B-Q4_K_S.gguf --mmproj mmproj-F16.gguf -ngl 99 \
  --image /Developer/models/bus.jpg \
  -p "Describe this image in detail, including the main object, its color, and how many people are visible."
```

**Result on `bus.jpg`** (the bus + pedestrians image) — correct and detailed:
> *"It's a **blue and white** electric **minibus** ('100% eléctrico', 'cero emisiones')… I see
> **three people clearly** standing/walking near the bus … plus a partial view of a person on the
> far left edge."*

It even read text off the bus (OCR) and its people count matched the YOLO detector (4 person boxes,
one low-confidence) from the object-detection lab. Vision needs the `mmproj` + a current
`llama-mtmd-cli` (the same b9743 build); the shipped container build cannot load it.

### 🚀 `sjsujetsontool vllm`

This section introduces high-performance, production-grade serving using **vLLM** on NVIDIA Jetson platforms. Compared to Ollama and `llama.cpp` which are highly optimized for CPU/GPU edge hardware, **vLLM** utilizes **PagedAttention** and advanced parallel speculative decoding to maximize GPU throughput and reduce latency under heavy model workloads.

#### ⚙️ How Speculative Decoding Works
Speculative decoding speeds up inference by running a smaller, faster **draft model** (the *speculator*) in parallel with a larger **target model** (the *verifier*). The speculator guesses multiple draft tokens in a single forward pass, which the verifier validates in parallel. 

For example, using `RedHatAI/Qwen3-8B-speculator.eagle3`, the EAGLE-3 method generates candidates rapidly, accelerating generation speed by up to 2x–3x on Jetson platforms.

#### 🚀 Serving with vLLM Host Shortcut
You can start the vLLM server directly from the host system:

```bash
# Start vLLM with the default Qwen3 EAGLE-3 speculator model:
sjsujetsontool vllm RedHatAI/Qwen3-8B-speculator.eagle3
```

> [!TIP]
> **GPU Memory Allocation:** vLLM allocates up to 90% of GPU memory by default. On resource-constrained edge devices (like the 8GB Jetson Orin Nano), the shortcut pre-configures `--gpu-memory-utilization 0.8` to leave headroom. You can customize the memory utilization or add other flags by passing them to the command:
> ```bash
> sjsujetsontool vllm RedHatAI/Qwen3-8B-speculator.eagle3 --gpu-memory-utilization 0.5 --max-model-len 2048
> ```

The server launches inside the optimized NVIDIA Jetson vLLM container (`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`) and runs on port `8000`.

#### 💬 Querying the vLLM Server via API
Once the vLLM server is running, you can perform queries using the OpenAI-compatible HTTP API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "messages": [
      {"role": "user", "content": "What are the benefits of speculative decoding?"}
    ]
  }'
```

### 📦 `sjsujetsontool status`

Displays:

* Docker container state
* GPU stats from `tegrastats`
* Port listening status for key services

### 🔧 `sjsujetsontool set-hostname <name>`

Changes device hostname, regenerates system identity, writes `/etc/device-id`.


### 🛑 `sjsujetsontool stop`

Stops the running Docker container started by previous commands.


### 🔑 `sjsujetsontool setup-nvapi`

Used in [Lesson 11: Building a Next.js AI App with NVIDIA Nemotron](./11_nextjs_nemotron_app.md) to set up and verify the NVIDIA NGC Build API key. 

Running this command will guide you through acquiring your free NGC API Key from [build.nvidia.com](https://build.nvidia.com) and saving it locally into the `.env.local` file inside the Next.js application directory.

#### ⚙️ How to Setup
```bash
sjsujetsontool setup-nvapi
```

During execution, it will prompt you:
```text
🔑 Paste your NVIDIA API Key (nvapi-...): <paste your key here>
```

The script automatically:
1. Validates the prefix format of the key.
2. Identifies the directory structure of the Next.js application and writes or updates the key as `NVIDIA_API_KEY=nvapi-...` in `edgeLLM/nextjs-nemotron-app/.env.local`.
3. Performs a test API query using `curl` against the **Nemotron 3 Nano Omni** reasoning model to verify authentication.

#### 🧪 Connection Test Example output on success:
```text
🧪 Testing connection to NVIDIA Build API using model 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning'...
✅ API Test Succeeded (HTTP 200)!
💬 Model response: Hello! How can I assist you today?
```

### 💬 `sjsujetsontool nv-chat`

Similar to the local `llama` and `llama-cli` shortcuts, `sjsujetsontool` provides a convenient `nv-chat` utility for students to interact directly with the NVIDIA Build API backend. It supports both **interactive chat** and **single-prompt modes**, offers a selection of optimized reasoning and chat models, and returns real-time streaming tokens along with performance statistics (such as token generation speed and Time-To-First-Token).

#### ⚙️ Usage Modes

##### 1. Interactive Chat Mode
Run `nv-chat` without any arguments to launch an interactive session. You will be prompted to select from a list of curated NVIDIA models:
```bash
sjsujetsontool nv-chat
```

Upon launching, the interactive menu presents:
```text
🤖 Select NVIDIA Build Model:
  1) Llama 3.1 Nemotron Nano (8B) [Default]
  2) Llama 3.3 Nemotron Super (49B)
  3) Llama 3.1 Nemotron Ultra (253B)
  4) Nemotron 3 Nano Omni (30B reasoning)
  5) Nemotron 3 Ultra (550B reasoning)
  6) Nemotron 3 Super (120B reasoning)
Select [1-6]:
```

Once selected, you can type your prompts interactively. Type `exit` or `quit` to end the session.

##### 2. Single-Prompt Mode
You can pass a prompt directly as arguments to perform a single query:
```bash
sjsujetsontool nv-chat "Explain the architecture of Nvidia Jetson Orin Nano"
```
Or use the explicit flags to specify a different model or prompt:
```bash
sjsujetsontool nv-chat -p "Explain the architecture of Nvidia Jetson Orin Nano" -m nvidia/llama-3.3-nemotron-super-49b-v1
```

#### ⚡ Performance Metrics
Every response is streamed in real-time. Upon completion, the tool outputs performance statistics:
* **Time-to-first-token (TTFT):** Measures the initial server latency.
* **Generation speed:** Tokens generated per second, along with total generation time and token count.

##### Example Output (Streaming + Performance Logs):
```text
💬 Starting interactive chat with nvidia/llama-3.1-nemotron-nano-8b-v1...
   Type 'exit' to quit.
══════════════════════════════════════════════════
User > What is NVIDIA Jetson?
💬 [Response]: NVIDIA Jetson is a series of embedded computing boards from NVIDIA designed for bringing accelerated AI to the edge. It combines a powerful ARM CPU with an integrated NVIDIA GPU (based on architectures like Ampere, Maxwell, or Pascal) to enable high-performance deep learning inference...

⚡ [Performance]: Time-to-first-token: 0.45s | Generation: 68.4 tokens/sec (124 tokens generated in 1.81s)
```

> [!NOTE]
> * **Reasoning Models:** For reasoning models (e.g., `nemotron-3-nano-omni-30b-a3b-reasoning` or `nemotron-3-super-120b-a12b`), the tool streams the model's internal thinking process under a `🧠 [Thinking Process]` header before streaming the final answer under a `💬 [Response]` header.
> * **Robustness and Timeouts:** To prevent execution from hanging indefinitely on slow or cold-starting cloud endpoints, the tool imposes a strict **15-second request timeout**. If a connection cannot be established or a model is unresponsive, the query fails gracefully with a timeout error.
> * **HTTP 403 Forbidden Errors:** If you receive a `403 Forbidden` error, this indicates that the specific model name is either restricted/unavailable under your free account plan, deprecated/renamed on the NVIDIA Build platform, or your API key's free credit quota has run out. You can resolve this by checking model availability in the [NVIDIA API Catalog](https://build.nvidia.com) or updating/renewing your API key using `sjsujetsontool setup-nvapi`.

### 💬 `sjsujetsontool chat` — one chat client, three backends

`sjsujetsontool chat` is a **unified, streaming terminal chat client** that talks to any OpenAI-compatible endpoint and lets you choose, at launch, **where the model runs**:

| # | Backend | Runs on | Needs |
|---|---------|---------|-------|
| **1** | **Local Jetson llama.cpp** | this Jetson (`http://localhost:8080`) | a local server — start it with `sjsujetsontool llama` |
| **2** | **NVIDIA Build API** | NVIDIA cloud (`integrate.api.nvidia.com`) | an API key — set it with `sjsujetsontool setup-nvapi` |
| **3** | **Our LLM server** | a shared GPU node (e.g. an RTX board) via `https://llm.forgengi.org/<node>` | network access (over Headscale) + the server's API key (sjsugputool) |

It uses the same engine as `gputool chat`: it renders **streaming Markdown** with the [`rich`](https://github.com/Textualize/rich) library when installed (and falls back to a plain pure-stdlib renderer otherwise), and prints per-turn **prefill / generation** token speeds.

> [!TIP]
> For the nicer Markdown/code-highlighted UI: `pip install rich` (optional — the client works without it).

#### Launch
```bash
sjsujetsontool chat            # shows the backend menu, then drops you into a session
```
You'll be asked to pick a backend (and, for NVIDIA, a model). You can also skip the menu:
```bash
sjsujetsontool chat --local    # local Jetson llama.cpp on :8080
sjsujetsontool chat --nvidia   # NVIDIA Build API (then pick a Nemotron model)
sjsujetsontool chat --server   # our shared LLM server (prompts for URL + key)
```

#### Example (backend 3 — our shared server)
```
🤖 Select chat backend:
  1) Local Jetson llama.cpp   (http://localhost:8080 — start it with: sjsujetsontool llama)
  2) NVIDIA Build API         (cloud — needs NVIDIA_API_KEY; setup: sjsujetsontool setup-nvapi)
  3) Our LLM server           (https://llm.forgengi.org/<node> over Headscale)
Select [1-3]: 3
Server base URL [https://llm.forgengi.org/node05/v1]:
API key (blank if none): ********
╭─ Assistant ▸ ──────────────────────────────────────────────────╮
│ NVIDIA Blackwell is the company's latest GPU architecture …     │
╰────────────────────────────────────────────────────────────────╯
(prefill 24 tok @ 765 tok/s · gen 38 tok @ 99.2 tok/s · 0.5s)
```

#### In-chat slash commands
| Command | Action |
|---|---|
| `/exit`, `/quit`, `/q` | Leave the chat |
| `/server` | Switch to a different server URL / API key |
| `/save [file]` | Save the conversation (`.md` default, or `.json`) |
| `/reset` | Clear history (keeps the system prompt) |
| `/system <text>` | Set/clear the system prompt |
| `/think on\|off` | Toggle the model's reasoning output |
| `/temp <v>` | Set sampling temperature (e.g. `/temp 0.7`) |
| `/set <k> <v>` | Set `top_p` / `top_k` / `min_p` / `presence` / `max_tokens` |
| `/preset <name>` | Apply Qwen3.5 presets: `thinking` · `coding` · `instruct` |
| `/config` | Show current sampling settings |
| `/help` | Show the command help |

> **Test different settings live.** Reasoning models like Qwen3.5 behave very differently by temperature and thinking mode. Use `/preset thinking` (creative: temp 1.0) vs `/preset coding` (precise: temp 0.6) vs `/preset instruct` (no thinking: temp 0.7), or fine-tune with `/temp 0.6` and `/set top_p 0.95`. These match [Unsloth's recommended Qwen3.5 settings](https://unsloth.ai/docs/models/qwen3.5).

> [!NOTE]
> **What is "our LLM server"?** A GPU node (e.g. an RTX 5080 board) runs `llama.cpp` and joins the **Headscale** network. The Headscale host (`headscale.forgengi.org`) reverse-proxies it under a friendly name at `https://llm.forgengi.org/<node>/v1` (TLS-terminated, the node's API key still required). This lets every Jetson use a big shared model **by name, over HTTPS — no IP addresses**. See [the gputool guide](00c_gputool_guide.md) for how a node serves the model and how the gateway maps it.

### ⚙️ `sjsujetsontool setup-check`

Used to verify and configure the host `/Developer` directory environment and retrieve/update the `edgeAI` curriculum codebase. This is crucial when setting up brand new Jetson nodes or checking cloned systems to ensure correct container mounts.

#### ⚙️ Why It Is Needed
When you launch the Jetson container shell (`sjsujetsontool shell`), `sjsujetsontool` automatically mounts the host `/Developer` folder inside the container as `/Developer`. 
* **Permission Alignment:** If `/Developer` does not exist on the host, the Docker daemon creates it as `root`-owned with restricted write permissions, preventing students from running git clones, code edits, or saving model weights.
* **Auto Setup Check:** `setup-check` ensures `/Developer` exists on the host, sets permissions to `777` (world-writable), ensures the `/Developer/models` weights directory is set up, and downloads/pulls the `edgeAI` codebase.

> [!TIP]
> The `setup-check` command runs automatically as a lifecycle step inside the **`sjsujetsontool update`** subcommand and whenever **`sjsujetsontool shell`** is launched on a fresh node where the directory structure is missing.

#### 🛠️ How to Run
```bash
sjsujetsontool setup-check
```

Example Output on success:
```text
══════════════════════════════════════════════════
⚙️  Checking /Developer folder and edgeAI git repository...
══════════════════════════════════════════════════
✅ Directory '/Developer' exists.
✅ Directory '/Developer' is writable.
✅ edgeAI repository already exists.
🔄 Pulling latest changes from origin...
✅ Repository updated successfully.
══════════════════════════════════════════════════
```

---

## ⚠️ Safety Guidelines

* 🔌 **Power Supply**: Use a 5A USB-C adapter or official barrel jack for stability.
* 💾 **SSD Cloning**: Change the hostname and machine-id after cloning to prevent network conflicts.
* 🔐 **SSH Security**: Only install SSH keys from trusted GitHub accounts.
* 🧼 **Disk Cleanup**: Remove cache and large datasets before creating system images.
* 📦 **Containers**: Always stop containers with `sjsujetsontool stop` before unplugging.

---

## 🧭 Ready to Learn and Build

You're now equipped to:

* Run AI models (LLaMA, Mistral, DeepSeek, etc.)
* Build and test LLM applications
* Access Jetson remotely with SSH or VS Code
* Run real-time cyber/AI experiments on the edge!

---

Made with 💻 by [Kaikai Liu](mailto:kaikai.liu@sjsu.edu) — [GitHub Repo](https://github.com/lkk688/edgeAI)
