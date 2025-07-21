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
| `sjsujetsontool ollama serve` | Start server (port 11434) |
| `sjsujetsontool ollama run qwen2` | Run CLI mode (use `\exit`) |
| `sjsujetsontool ollama list` | List installed models |
| `sjsujetsontool ollama pull mistral` | Download model |
| `sjsujetsontool ollama delete mistral` | Remove model |
| `sjsujetsontool ollama status` | Check server status |
| `sjsujetsontool ollama ask "What is NVIDIA Jetson?"` | Ask question (auto-pulls) |
| `sjsujetsontool ollama ask --model mistral "Explain transformers."` | Ask with specific model |

### 🔬 Llama.cpp Commands

| Command | Description |
|---------|-------------|
| `sjsujetsontool llama` | Start server (port 8000) |
| `./build_cuda/bin/llama-cli -m /models/mistral.gguf -p "prompt"` | Run model directly |
| `./build_cuda/bin/llama-server -m /models/mistral.gguf --port 8080` | Start HTTP server |
| `curl http://localhost:8080/completion -d '{"prompt":"...","n_predict":100}'` | Query API |
| `http://localhost:8080` | Access web UI |

### 🔧 System Management

| Command | Description |
|---------|-------------|
| `sjsujetsontool set-hostname new-hostname` | Change hostname (sudo) |
| `sjsujetsontool mount-nfs <host> <path> <mount>` | Mount NFS share |

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