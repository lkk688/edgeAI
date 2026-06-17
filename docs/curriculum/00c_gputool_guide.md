# 🚀 gputool User Guide

**gputool** is a lightweight, non-sudo utility CLI designed to manage services and networking in restricted or locked-down environments (such as university lab machines or shared server nodes) where you **do not have root (`sudo`) privileges**.

Unlike the main `sjsujetsontool`, which installs system-level packages and systemd services on NVIDIA Jetson dev kits, `gputool` runs completely inside the user space, saving all of its state, logs, and sockets locally under `$HOME/.gputool`.

---

## 🆚 `sjsujetsontool` vs `gputool`

| Feature | `sjsujetsontool` | `gputool` |
|---|---|---|
| **Privileges Required** | `sudo` (Root) | **No Sudo** (User space only) |
| **Primary Platforms** | NVIDIA Jetson | Generic Linux (Jetson, x86 Servers, etc.) |
| **Tailscale Mode** | System kernel interface (`tailscale0` tun card) | User-space networking (Proxy mode) |
| **Installation** | System-wide binaries & systemd services | Static binaries in user home directory |
| **Download Dependencies** | Requires system `curl` / `wget` | Falls back to **Python 3** if curl/wget are missing |

---

## 📥 Installation

Choose the command that fits your system's network tools:

### Option A: Standard (curl / wget installed)
```bash
curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_gputool.sh | bash
```

or 
```bash
wget https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_gputool.sh
chmod +x install_gputool.sh
./install_gputool.sh
source ~/.bashrc
gputool version
```

### Option B: Locked-down Lab Machines (No `curl` or `wget`)
If the computer has neither `curl` nor `wget` (or they are blocked), use Python 3 to stream the installer:
```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_gputool.sh').read().decode())" | bash
```

---

## 📋 Command Reference

### Core Commands

* **Show Help Menu:**
  ```bash
  gputool help
  ```
* **Print Version:**
  ```bash
  gputool version
  ```
* **Install/Re-install local script:**
  ```bash
  gputool install
  ```
* **Update script from GitHub:**
  ```bash
  gputool update-script
  ```
* **Install Miniconda:**
  ```bash
  gputool install-conda [install_path]
  ```
* **Run complete system check:**
  ```bash
  gputool check [env_name]
  ```
* **Create customized Conda env (PyTorch + HF):**
  ```bash
  gputool setup-env [env_name] [python_ver]
  ```
* **Create LeRobot / PyTorch / HF Conda env:**
  ```bash
  gputool setup-lerobot [env_name]
  ```

---

## 🐍 AI & Machine Learning Setup

`gputool` provides helper utilities to set up localized machine learning virtual environments directly inside your user directory using Conda.

### 📦 Install Miniconda
On a brand new/empty machine (where Conda is not present), you can download and install Miniconda to user space:

```bash
gputool install-conda [install_path]
```
*(If `install_path` is omitted, it defaults to `$HOME/miniconda3`).*

**What it does:**
1. **Downloads Installer:** Downloads the latest Miniconda installer from Anaconda's repositories.
2. **Silent Install:** Performs a silent batch installation in user space (no root required).
3. **ToS Auto-Acceptance:** Automatically accepts the Anaconda Terms of Service to prevent `CondaToSNonInteractiveError` during package setups.
4. **Shell configuration:** Runs `conda init` to automatically configure your `.bashrc` and/or `.zshrc`.

> [!TIP]
> If you run `gputool setup-lerobot` directly on a system where Conda is missing, `gputool` will automatically detect this and trigger the Miniconda auto-installation fallback for you!

### 🤖 Create Conda Env & Install LeRobot / PyTorch / HF
Creates a new Conda environment with Python 3.10 and automatically configures it with the Blackwell-compatible CUDA 12.8 PyTorch build, LeRobot (with extra simulator dependencies), and Hugging Face packages:

```bash
gputool setup-lerobot [env_name]
```
*(If `env_name` is omitted, it defaults to `lerobot`).*

**What it does:**
1. **Locates Conda:** Dynamically scans for and sources the active Conda initialization profile (e.g. `miniconda3`, `anaconda3`, or system paths).
2. **Creates Environment:** Initializes the target Conda virtual environment using Python 3.10.
3. **Installs CMake < 4:** Automatically configures the environment with `cmake<4` from `conda-forge`. This is a critical fallback that prevents build isolation compile failures (`Compatibility with CMake < 3.5 has been removed from CMake`) when compiling older robotics simulation packages like `egl_probe`/`hf-egl-probe` under newer system environments.
4. **Installs PyTorch (Auto-Detected CUDA Build):** Probes the host with `nvidia-smi` and `nvcc`, then automatically selects the matching PyTorch wheel index:
   * **Modern GPUs** (compute capability ≥ 7.0, including Blackwell `sm_120` RTX 5080) → CUDA 12.8 (`cu128`). This is required on Blackwell, as older wheels raise "no kernel image available" exceptions.
   * **Older GPUs** (compute capability < 7.0, e.g. Pascal/Maxwell) → CUDA 11.8 (`cu118`) for compatibility.
   * **No GPU / no CUDA toolkit** → defaults to CUDA 12.8 (`cu128`).
   * If the chosen wheel index fails, it falls back to the default PyPI `torch` build. The detected GPU name, compute capability, nvcc version, and selected build are printed before installation.
5. **Installs LeRobot & HF:** Performs a Pip installation of `huggingface_hub` and the complete LeRobot suite with simulation extras (`lerobot[all]`).
6. **Runs Verification:** Automatically verifies GPU detection, PyTorch CUDA initialization, and imports.

To activate the environment after setup:
```bash
conda activate <env_name>
```

#### Example Verification Output
Upon successful installation, `gputool` outputs the active versions and hardware state:
```
==================================================
🧬 PyTorch Version    : 2.10.0+cu128
🟢 CUDA Available      : True
🖥️  GPU Device Name    : NVIDIA GeForce RTX 5080
⚙️  CUDA Device Arch   : ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
🤗 HF Hub Version     : 0.35.3
🤖 LeRobot Version    : 0.4.4
==================================================
```

### 🐍 Create Customized Conda Env & Install PyTorch / HF
Creates a new Conda environment with a custom Python version (defaulting to name `py312` and Python `3.12`) and automatically configures it with the Blackwell-compatible CUDA 12.8 PyTorch build and Hugging Face:

```bash
gputool setup-env [env_name] [python_ver]
```
*(If parameters are omitted, it defaults to environment name `py312` and Python version `3.12`).*

**What it does:**
1. **Locates Conda:** Dynamically scans for and sources the active Conda initialization profile (e.g., `miniconda3`, `anaconda3`, or system paths).
2. **Creates Environment:** Initializes the target Conda virtual environment using the specified Python version (e.g., `3.12`).
3. **Installs PyTorch (Auto-Detected CUDA Build):** Probes the host GPU and CUDA toolkit and selects the matching PyTorch wheel automatically — CUDA 12.8 (`cu128`) for modern GPUs like the Blackwell `sm_120` RTX 5080, CUDA 11.8 (`cu118`) for older GPUs (compute capability < 7.0), and `cu128` as the default when no GPU/CUDA is found. Falls back to the default PyPI `torch` if the selected index fails.
4. **Installs Hugging Face:** Performs a Pip installation of `huggingface_hub`.
5. **Runs Verification:** Automatically verifies GPU detection, PyTorch CUDA initialization, and imports.

To activate the environment after setup:
```bash
conda activate <env_name>
```

#### Example Verification Output
Upon successful installation, `gputool` outputs the active versions and hardware state:
```
==================================================
🧬 PyTorch Version    : 2.10.0+cu128
🟢 CUDA Available      : True
🖥️  GPU Device Name    : NVIDIA GeForce RTX 5080
⚙️  CUDA Device Arch   : ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
🤗 HF Hub Version     : 0.35.3
==================================================
```

### 🔍 Run Complete System Diagnostic Check
To verify that the system hardware, Conda environment, PyTorch with CUDA, Hugging Face Hub, LeRobot installation, and Userspace Tailscale VPN/Proxy are all correctly configured, run:

```bash
gputool check [env_name]
```
*(If `env_name` is omitted, it defaults to `lerobot`).*

**What it verifies:**
1. **GPU & Driver:** Checks if `nvidia-smi` is available and reports the active GPU name, NVIDIA driver version, and host system CUDA version.
2. **Conda environment:** Verifies if Conda is installed and detects if the target environment (e.g. `lerobot`) exists.
3. **PyTorch & CUDA:** Performs in-environment checks to verify PyTorch is installed, CUDA is active, and prints the compute capability of the GPU (e.g. Blackwell RTX 5080 capability details).
4. **Hugging Face Hub:** Checks Hugging Face Hub library setup, authentication status, and tests network connectivity to `huggingface.co`.
5. **LeRobot:** Confirms `lerobot` package version and checks key simulation backends (`gymnasium`, `mujoco`, `h5py`).
6. **Tailscale & Proxy:** Queries tailscaled daemon status, checks whether proxy port `1055` is active and listening, and displays assigned Tailscale IPs.

#### Example Output:
```
══════════════════════════════════════════════════
🖥️  System Hardware Check
══════════════════════════════════════════════════
[✅] NVIDIA Driver found via nvidia-smi.
   • GPU Name       : NVIDIA GeForce RTX 5080
   • Driver Version : 550.54.14
   • CUDA Version   : 12.8

══════════════════════════════════════════════════
🐍 Conda Environment Check
══════════════════════════════════════════════════
[✅] Conda is installed.
   • Conda Path    : /home/student/miniconda3/bin/conda
   • Conda Version : 24.1.2
[✅] Conda environment 'lerobot' exists.
[⚙️] Running Python diagnostic checks in Conda env 'lerobot'...

════ PyTorch & CUDA Diagnostic ════
   • PyTorch Installed      : 2.10.0+cu128                   ✅
   • CUDA Available         : True                           ✅
   • CUDA Backend Ver       : 12.8                           ✅
   • GPU Device Name        : NVIDIA GeForce RTX 5080        ✅
   • Compute Capability     : ('12', '0')                    ✅

════ Hugging Face Hub Diagnostic ════
   • HF Hub Installed       : 0.35.3                         ✅
   • HF Auth Status         : Logged In                      ✅
   • HF Hub Connectivity    : Connected                      ✅

════ LeRobot Diagnostic ════
   • LeRobot Installed      : 0.4.4                          ✅
   • Simulation Packages    : gymnasium(OK), mujoco(OK)...   ✅

══════════════════════════════════════════════════
🌐 Userspace Tailscale VPN & Proxy Check
══════════════════════════════════════════════════
[✅] tailscaled background daemon is running.
   • Daemon PID      : 29841
[✅] Proxy port 1055 is listening.
   • Tailscale IPs   : 100.64.0.15
══════════════════════════════════════════════════
```

---

## 🦙 Llama.cpp & Local LLM Serving

`gputool` provides fully integrated commands to compile `llama.cpp` with NVIDIA CUDA support, download optimized GGUF model quantizations from Hugging Face, and manage background `llama-server` instances for local API serving with maximum GPU offloading.

### 1. Compile llama.cpp with CUDA Support
This command clones `llama.cpp` from source, locates the active CUDA compiler (`nvcc`), and builds the binaries (`llama-cli` and `llama-server`) with GPU acceleration enabled:

```bash
gputool setup-llamacpp [env_name]
```
*(If `env_name` is omitted, it defaults to `lerobot`).*

For example:
```bash
(py312) 010796032@coe-cmpe-288-05:~$ gputool setup-llamacpp py312
```

**What it does (robust, self-healing build):**
1. **Locates `nvcc`:** Searches `PATH`, then `/usr/local/cuda/bin`, then any `/usr/local/cuda-*/bin`, and reports the detected CUDA toolkit version, GPU name, and compute capability.
2. **Auto-installs build tools:** Detects whether `cmake` and `ninja` are present *inside the conda env* and installs any that are missing from `conda-forge` (with a `pip` fallback for `cmake`). This means envs created by `gputool setup-env` (which ship PyTorch + HF only) no longer fail with `cmake: command not found`.
3. **Detects GPU architecture:** Auto-configures `nvcc` for your GPU (e.g. Blackwell RTX 5080 → `sm_120a`). Uses the Ninja generator when available for a faster build, and automatically wipes a stale build directory if the generator changed.
4. **Installs binaries + libraries:** Copies `llama-cli`, `llama-server`, **and all shared libraries** (`libllama`, `libggml-cuda`, …) into `~/.gputool/bin/` so the binaries run standalone.
5. **Verifies CUDA:** Runs `llama-cli --list-devices` and confirms the GPU is visible to llama.cpp before finishing.

> [!NOTE]
> On fully locked-down machines with no network during the build, llama.cpp may print a warning that it could not download the embedded web UI assets — this is harmless and the server still builds and serves the API normally.

### 2. Download a GGUF Model
Use Python's native `huggingface_hub` downloader to fetch a GGUF model directly to `~/.gputool/models/`. This avoids symlink compilation issues and handles large file streaming smoothly.

```bash
gputool download-model [repo_id] [filename] [env_name]
```
* **Default Model:** If arguments are omitted, it defaults to the unsloth Q6_K_XL quantization of **Qwen3.5 9B**:
  * Repo ID: `unsloth/Qwen3.5-9B-GGUF`
  * Filename: `Qwen3.5-9B-UD-Q6_K_XL.gguf`

### 3. Serve the LLM via Llama Server
Start, stop, or check status of the local OpenAPI-compatible background serving daemon:

* **Start the server:**
  Offloads all model layers (`-ngl 99`) to the GPU.
  ```bash
  gputool serve-llamacpp start [model_filename_or_path] [port] [--background|--foreground]
  ```
  *(Defaults: model = `Qwen3.5-9B-UD-Q6_K_XL.gguf`, port = `8080`, run mode = background)*

  By default the server **binds to `0.0.0.0`**, so it is reachable from other machines on the same network (e.g. lab peers) at `http://<server-ip>:<port>`. To restrict it to the local machine only, pass `--host 127.0.0.1`.
  ```bash
  gputool serve-llamacpp start                       # LAN-accessible on 0.0.0.0:8080 (default)
  gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080 --host 127.0.0.1   # local-only
  ```
  > [!WARNING]
  > Binding to `0.0.0.0` exposes the model API to every host that can route to this machine. On shared or untrusted networks, either protect it with `--api-key` (below), use `--host 127.0.0.1`, or reach it over an SSH tunnel / the Tailscale VPN.

  **Protect the API with a token (`--api-key`):** Require every request to carry an `Authorization: Bearer <token>` header. Requests without the correct token are rejected with HTTP `401`.
  ```bash
  gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080 --api-key sjsugputool
  ```
  You can also set the token via the `GPUTOOL_LLAMA_API_KEY` environment variable instead of passing it on the command line (which keeps it out of your shell history):
  ```bash
  export GPUTOOL_LLAMA_API_KEY=sjsugputool
  gputool serve-llamacpp start
  ```
  Clients then include the bearer token. With `curl`:
  ```bash
  curl http://10.31.96.155:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sjsugputool" \
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
  ```
  Or, on machines without `curl`, with Python:
  ```python
  import json, urllib.request
  req = urllib.request.Request(
      "http://<server-ip>:8080/v1/chat/completions",
      data=json.dumps({"messages": [{"role": "user", "content": "Hello!"}]}).encode(),
      headers={"Content-Type": "application/json", "Authorization": "Bearer sjsugputool"})
  print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
  ```

  The run mode controls whether the server detaches from your terminal:
  * **`--background` / `-d` (default):** Runs as a detached `nohup` daemon, writes its PID to `~/.gputool/llama-server.pid`, and logs to `~/.gputool/llama-server.log`. The command returns immediately so you can keep using the shell (or close the SSH session) while the server keeps running.
    ```bash
    gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080            # background (default)
    gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080 -d         # explicit background
    ```
  * **`--foreground` / `-f`:** Runs attached to your terminal and streams server logs live. Useful for debugging. Press `Ctrl+C` to stop it.
    ```bash
    gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080 --foreground
    ```

* **Check server status:**
  ```bash
  gputool serve-llamacpp status
  ```
  *(Prints process PID details and runs a connection health check to `/health`. If no tracked PID file exists, it still detects stray/foreground `llama-server` instances via `pgrep`.)*

* **Stop the server:**
  ```bash
  gputool serve-llamacpp stop
  ```
  This stops **the tracked daemon and any stray/orphaned `llama-server` processes** — it collects PIDs from the PID file plus a `pgrep` match on the installed binary, terminates each gracefully (`SIGTERM`, then `SIGKILL` after a 10s grace period), and clears the PID file. Running it when nothing is active simply reports that no servers were found.

#### Example Usage & Inference Query:
```bash
# 1. Start the server (offloads all layers to the GPU with -ngl 99)
gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080

# 2. Query completions via the OpenAI-compatible endpoint
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What GPU architecture is NVIDIA Blackwell?"}
    ]
  }'
```

> [!TIP]
> **No `curl` on locked-down machines?** Many lab nodes ship without `curl`/`wget`. Query the server with Python's built-in `urllib` instead:
> ```python
> import json, urllib.request
> req = urllib.request.Request(
>     "http://localhost:8080/v1/chat/completions",
>     data=json.dumps({"messages": [{"role": "user", "content": "What GPU architecture is NVIDIA Blackwell?"}]}).encode(),
>     headers={"Content-Type": "application/json"})
> print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
> ```

> [!NOTE]
> **Reasoning models (Qwen3.5):** Qwen3.5 is a "thinking" model. By default it spends tokens on internal reasoning (returned in the `reasoning_content` field) before producing the final `content`, so a small `max_tokens` may return empty `content` with `finish_reason: "length"`. Either raise `max_tokens`, or disable thinking for a direct answer by adding `"chat_template_kwargs": {"enable_thinking": false}` to the request body.

### 4. Chat from the Terminal (`gputool chat`)
Instead of crafting raw `curl`/`urllib` requests, `gputool` ships a built-in terminal chat client that talks to any OpenAI-compatible endpoint (your local `llama-server` by default), **streams the response token-by-token**, and shows a clean colored UI with per-turn token/throughput stats. It is written in pure-stdlib Python, so it works even on locked-down machines without `curl` or extra `pip` packages.

```bash
gputool chat [message] [--host <ip>] [--port <port>] [--api-key <token>] [--system <text>] [--think] [--no-stream] [--no-color]
```

* **Interactive session** (multi-turn, keeps conversation history):
  ```bash
  gputool chat
  ```
* **One-shot question** (prints the streamed answer and exits):
  ```bash
  gputool chat "What GPU architecture is NVIDIA Blackwell?"
  ```
* **Talk to a server on another machine, with an API key:**
  ```bash
  gputool chat --host 10.31.96.155 --port 8080 --api-key sjsugputool
  # or set it once: export GPUTOOL_LLAMA_API_KEY=sjsugputool
  ```

**Defaults & options:**
* `--host` / `--port` default to `127.0.0.1:8080`; or pass a full base URL with `--url http://10.31.96.155:8080/v1`.
* `--api-key` defaults to the `GPUTOOL_LLAMA_API_KEY` environment variable (so you don't have to type the token each time).
* `--model` is auto-detected from the server's `/v1/models` endpoint if omitted.
* Thinking/reasoning output is **off by default** for snappy answers; enable it with `--think` (or `/think on` during a session).

**Interactive slash commands:**

| Command | Action |
|---|---|
| `/exit`, `/quit`, `/q` | Leave the chat |
| `/reset`, `/clear` | Clear the conversation history (keeps the system prompt) |
| `/system <text>` | Set (or clear, if empty) the system prompt |
| `/think on\|off` | Toggle the model's reasoning output |
| `/help`, `/?` | Show the command help |

#### Example session
```
══════════════════════════════════════════════════
 🦙 gputool chat — local OpenAI-compatible LLM
══════════════════════════════════════════════════
  Endpoint : http://127.0.0.1:8080/v1/chat/completions
  Model    : Qwen3.5-9B-UD-Q6_K_XL.gguf
  Auth     : on   Streaming: on   Thinking: off
  Commands : /exit  /reset  /system <text>  /think on|off  /help

You ▸ What GPU architecture is NVIDIA Blackwell?
Assistant ▸ NVIDIA Blackwell is the company's latest GPU architecture, succeeding Hopper...
(24 prompt + 38 completion tokens · 92.4 tok/s · 0.5s)
You ▸ /exit
Bye!
```

---

## 🌐 Userspace Tailscale VPN

Since `gputool` runs without root privileges, it cannot create a virtual network interface card (`tailscale0`). Instead, it uses **user-space networking (SOCKS5/HTTP Proxy)** to routing network traffic through the Headscale VPN.

### 1. Download Static Binaries
Downloads the pre-compiled, standalone Tailscale package for your CPU architecture (AMD64, ARM64, or ARM) and configures the binaries in `~/.gputool/tailscale`:
```bash
gputool tailscale setup
```

### 2. Connect to the VPN
Starts the background `tailscaled` daemon in userspace proxy mode and registers this device on the SJSU Headscale network:
```bash
gputool tailscale up
```
*Note: If another device is already registered under the same hostname, use `gputool tailscale up --force` to override.*

### 3. Check Connection & Proxy Details
Displays the current connection state, your assigned Tailscale IP address, and proxy variables:
```bash
gputool tailscale status
```
Example Output:
```
══════════════════════════════════════════════════
🌐 Userspace Tailscale VPN Status
══════════════════════════════════════════════════
📦 Tailscale Version : 1.68.1
🔧 Daemon PID        : 12845
🔌 Socket Path       : /home/student/.gputool/tailscaled.sock

🟢 Connection State  : Running
   Device Hostname   : lab-machine-01
   Tailscale IPs     : 100.64.0.15
   Connected Peers   : 8

🛡️  User-Space Proxy Configuration:
   • SOCKS5 Proxy    : localhost:1055
   • HTTP Proxy      : localhost:1055
══════════════════════════════════════════════════
```

### 4. Disconnect and Stop Daemon
Logs out of the Headscale server and stops the background tailscaled process:
```bash
gputool tailscale down
```

---

## 🔌 How to Access Peers (Proxy Usage)

Because the userspace client does not have a virtual TUN adapter, outgoing traffic is routed through local proxies running on **port 1055**.

### Web Requests & APIs (curl, python, wget)
To access web interfaces or APIs on other Headscale nodes, prefix or export the HTTP/HTTPS proxy:

* **Single command:**
  ```bash
  curl -x http://localhost:1055 http://<peer-tailscale-ip>:<port>
  ```
* **Export for session:**
  ```bash
  export http_proxy=http://localhost:1055
  export https_proxy=http://localhost:1055
  # Now commands run transparently through the proxy
  curl http://<peer-tailscale-ip>:<port>
  ```

### Secure Shell (SSH)
To SSH into another machine on the Headscale network, route the connection via the SOCKS5 proxy command:
```bash
ssh -o ProxyCommand="nc -X 5 -x localhost:1055 %h %p" username@<peer-tailscale-ip>
```

---

## ⚠️ Key Limitations of Non-Sudo Mode

* **Inbound Access is Transparent:** Other machines on the Headscale network can ping or access ports on this computer directly via its Tailscale IP without any proxy setup.
* **Outbound Access Requires Proxies:** This machine *cannot* ping or access other nodes directly unless configured to use the proxy on `localhost:1055` (as shown in the examples above).
