# 🧠 SJSU Jetson Tool Cheatsheet

A quick reference guide for using the `sjsujetsontool` utility on NVIDIA Jetson devices. Full tutorial: [00_sjsujetsontool_guide.md](https://github.com/lkk688/edgeAI/blob/main/docs/curriculum/00_sjsujetsontool_guide.md)

## 📋 Commands by Category

### 📥 Installation

| Command | Description |
|---------|-------------|
| `curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh \| bash` | Install (no sudo) |
| `sjsujetsontool update` | Update container & script |
| `sjsujetsontool update-container` | Update container only |
| `sjsujetsontool update-script` | Update script only |

### 🔍 Basic Commands

| Command | Description |
|---------|-------------|
| `sjsujetsontool list` | Show all commands |
| `sjsujetsontool version` | Show versions |
| `sjsujetsontool status` | Show container & GPU stats |
| `sjsujetsontool debug` | Run diagnostics |
| `sjsujetsontool setup-check` | Check /Developer folder and clone/pull edgeAI git repo |

### 🐳 Container Management

| Command | Description |
|---------|-------------|
| `sjsujetsontool shell` | Enter container shell |
| `exit` | Exit shell (container keeps running) |
| `sjsujetsontool stop` | Stop container completely |

### 🐍 Python & Jupyter

| Command | Description |
|---------|-------------|
| `sjsujetsontool run /path/to/script.py` | Run Python script in container |
| `sjsujetsontool jupyter` | Launch JupyterLab (port 8888) |

### 🧠 Ollama LLM Commands

| Command | Description |
|---------|-------------|
| `sjsujetsontool ollama-serve` | Start server (shortcut for `ollama serve`, port 11434) |
| `sjsujetsontool ollama-run gemma4` | Run model interactively (shortcut for `ollama run`) |
| `sjsujetsontool ollama list` | List installed models |
| `sjsujetsontool ollama pull gemma4` | Download model |
| `sjsujetsontool ollama delete gemma4` | Remove model |
| `sjsujetsontool ollama status` | Check server status |
| `sjsujetsontool ollama ask "What is NVIDIA Jetson?"` | Ask question (auto-pulls) |

### 🔬 Llama.cpp & Gemma 4 E2B (VLM) Commands

| Command | Description |
|---------|-------------|
| `sjsujetsontool llama` | Start Gemma 4 E2B `llama-server` (port 8080) |
| `sjsujetsontool llama-cli -p "prompt"` | Run Gemma 4 E2B `llama-cli` text query |
| `sjsujetsontool llama-cli --image /Developer/LoveSJ-hero-4.png -p "Describe"` | Run Gemma 4 E2B `llama-cli` image query |
| `curl http://localhost:8080/v1/chat/completions -d '...'` | Query OpenAI-compatible Chat completions API |
| `http://localhost:8080` | Access web UI dashboard |

### 🚀 vLLM Speculative Decoding Commands

| Command | Description |
|---------|-------------|
| `sjsujetsontool vllm [model]` | Start vLLM serve engine (defaults to Qwen3-8B-speculator on port 8000) |
| `curl http://localhost:8000/v1/chat/completions -d '...'` | Query vLLM OpenAI-compatible REST API |

### 🔑 NVIDIA Build API / NGC Cloud Inference

| Command | Description |
|---------|-------------|
| `sjsujetsontool setup-nvapi` | Setup NVIDIA NGC API Key in `.env.local` & perform validation test |
| `sjsujetsontool nv-chat` | Start interactive chat with curated model selection (Llama 3.x, Nemotron) |
| `sjsujetsontool nv-chat "prompt"` | Run single prompt query with default model and performance metrics |
| `sjsujetsontool nv-chat -p "prompt" -m [model]` | Run single query with a customized model (e.g. Nemotron Omni) |


## 📂 Important Information

### Mounted Paths
- Host directories mounted in container:
  - `/Developer` → `/Developer` in container
  - `/Developer/models` → `/models` in container

### SSH Connectivity
- Connect via mDNS hostname: `ssh username@jetson-hostname.local`
- Example: `ssh sjsujetson@sjsujetson-01.local`
- For X11 forwarding: `ssh -X sjsujetson@sjsujetson-01.local`

## ⚠️ Safety Tips

- **Power Supply**: Use a 5A USB-C adapter or official barrel jack for power stability
- **Containers**: Always stop containers with `sjsujetsontool stop` before unplugging
- **SSD Cloning**: Change hostname and machine-id after cloning to prevent network conflicts
- **SSH Security**: Only install SSH keys from trusted GitHub accounts
- **Disk Cleanup**: Remove cache and large datasets before creating system images