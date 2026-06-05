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

After the script installation, run `sjsujetsontool update` to update the local script and container image. The container update takes a long time.
```bash
sjsujetson@sjsujetson-01:~$ curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh | bash
⬇️ Downloading sjsujetsontool from GitHub...
✅ Downloaded script.
📦 Installing to /home/sjsujetson/.local/bin/sjsujetsontool
✅ Installed successfully. You can now run: sjsujetsontool
sjsujetson@sjsujetson-01:~$ sjsujetsontool update
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
📦 JetPack Version: 6.2.1+b38
🏷️  L4T BSP Revision: R36.4.7
⚙️  CUDA Version: 12.6
🧬 cuDNN Version: 9.3.0
🤖 TensorRT Version: 10.3.0.30-1+cuda12.5
🔄 Running full update (script + container)...
  Use 'update-container' or 'update-script' to update individually.

📜 Step 1/2: Updating script from GitHub...
📂 Backup saved to: /home/sjsujetson/.local/bin/sjsujetsontool.bak
⬇️ Downloading latest script...
#################################################################################################### 100.0%
✅ Script downloaded. Replacing current script...
✅ Script updated successfully. Re-run your command to use the new version.

🐳 Step 2/2: Updating container image...
🔍 Checking Docker image update...
⬇️ Pulling latest image from Docker Hub...
latest: Pulling from cmpelkk/jetson-llm
....
✓ Pull complete.
📦 New version detected. Updating local image...
✅ Local container updated from Docker Hub.
```

You can also update just the script or just the container independently:
```bash
sjsujetsontool update-script      # update only this CLI script from GitHub
sjsujetsontool update-container   # update only the Docker container image
```

Verify:

```bash
sjsujetsontool list
```

You can check the script and system versions:
```bash
sjsujetson@sjsujetson-01:/Developer/edgeAI$ sjsujetsontool version
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
📦 JetPack Version: 6.2.1+b38
🏷️  L4T BSP Revision: R36.4.7
⚙️  CUDA Version: 12.6
🧬 cuDNN Version: 9.3.0
🤖 TensorRT Version: 10.3.0.30-1+cuda12.5
🧾 sjsujetsontool Script Version: v1.0.0
🧊 Docker Image: jetson-llm:v1
🔍 Image ID: sha256:83e9d42c36f8cae3a1a61fd5aa488b14b9ac11922c512920652212795847280a
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
if you face errors like "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?", restart the docker:
```bash
sudo systemctl start docker
sudo systemctl status docker
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
- 📦 **NVIDIA JetPack / L4T**: JetPack version, L4T BSP revision (`R36.4.7`), L4T package version
- ⚙️ **CUDA**: Version detected via `nvcc` or via `/usr/local/cuda-*` directory, with PATH tip if nvcc is missing
- 🧬 **cuDNN**: Parsed from `/usr/include/cudnn_version.h` headers
- 🤖 **TensorRT**: Detected from installed `libnvinfer` packages
- 💾 **Memory**: RAM and Swap usage
- 💿 **Disk**: Filesystem usage with warnings if >80% full
- 🌡️ **Temperatures**: All thermal zones from `tegrastats`
- ⚡ **Power**: Per-rail power via INA3221 sensors (`VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC`)
- 🐳 **Docker**: Daemon status, version, NVIDIA runtime, available images, running containers
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

### 📋 `sjsujetsontool list`

Displays all available commands with usage examples.

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
```bash
sjsujetsontool ollama serve
```

Starts the Ollama REST server inside the container, listening on http://localhost:11434.


2. Run a Model in CLI Mode
```bash
$ sjsujetsontool ollama run mistral
🧠 Detected Jetson Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
⚙️  CUDA Version: 12.6
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

After entering into the container, you can run a local downloaded model ('build_cuda' folder is the cuda build):
```bash
root@sjsujetson-01:/Developer/llama.cpp# llama-cli -m /models/mistral.gguf -p "Explain what is Nvidia jetson"
....
llama_perf_sampler_print:    sampling time =      34.98 ms /   532 runs   (    0.07 ms per token, 15210.86 tokens per second)
llama_perf_context_print:        load time =    3498.72 ms
llama_perf_context_print: prompt eval time =    2193.93 ms /    17 tokens (  129.05 ms per token,     7.75 tokens per second)
llama_perf_context_print:        eval time =   84805.65 ms /   514 runs   (  164.99 ms per token,     6.06 tokens per second)
llama_perf_context_print:       total time =   92930.78 ms /   531 tokens
```

`llama-server` is a lightweight, OpenAI API compatible, HTTP server for serving LLMs. Start a local HTTP server with default configuration on port 8080: `llama-server -m model.gguf --port 8080`, Basic web UI can be accessed via browser: `http://localhost:8080`. Chat completion endpoint: `http://localhost:8080/v1/chat/completions`
```bash
root@sjsujetson-01:/Developer/llama.cpp# llama-server -m /models/mistral.gguf --port 8080
```

Send request via curl in another terminal (in the host machine or container):
```bash
sjsujetson@sjsujetson-01:~$ curl http://localhost:8080/completion -d '{
  "prompt": "Explain what is Nvidia jetson?",
  "n_predict": 100
}'


### 📦 `sjsujetsontool status`

Displays:

* Docker container state
* GPU stats from `tegrastats`
* Port listening status for key services

### 🔧 `sjsujetsontool set-hostname <name>`

Changes device hostname, regenerates system identity, writes `/etc/device-id`.


### 🛑 `sjsujetsontool stop`

Stops the running Docker container started by previous commands.


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
