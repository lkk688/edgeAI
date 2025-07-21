# 🧠 SJSU Jetson Tool Cheatsheet

A quick reference guide for using the `sjsujetsontool` utility on NVIDIA Jetson devices. Full tutorial: [00_sjsujetsontool_guide.md](https://github.com/lkk688/edgeAI/blob/main/docs/curriculum/00_sjsujetsontool_guide.md)

## 📋 Command Reference Table

| Category | Command | Description |
|----------|---------|-------------|
| **📥 Installation** | `curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh \| bash` | One-line install (no sudo required) |
| | `sjsujetsontool update` | Update both container and script (container update takes time) |
| | `sjsujetsontool update-container` | Update only the Docker container |
| | `sjsujetsontool update-script` | Update only the script |
| **🔍 Basic** | `sjsujetsontool list` | Display all available commands with usage examples |
| | `sjsujetsontool version` | Show script version, Docker image details, and CUDA version |
| | `sjsujetsontool status` | Show container state, GPU stats (tegrastats), and port status |
| **🐳 Container** | `sjsujetsontool shell` | Enter container shell (with mounted `/Developer` and `/models` directories) |
| | `exit` | Exit container shell (container keeps running in background) |
| | `sjsujetsontool stop` | Stop the running Docker container completely |
| **🐍 Python** | `sjsujetsontool run /path/to/script.py` | Run a Python script inside the container (script must be in a mounted path like `/Developer`) |
| | `sjsujetsontool jupyter` | Launch JupyterLab on port 8888 (access via http://hostname:8888/lab?token=<token> or replace hostname with device IP for remote access) |
| **🧠 Ollama** | `sjsujetsontool ollama serve` | Start Ollama server on port 11434 |
| | `sjsujetsontool ollama run mistral` | Run model in CLI mode (use `\exit` to exit) |
| | `sjsujetsontool ollama list` | List all installed models with size and modification date |
| | `sjsujetsontool ollama pull llama3` | Download a model (examples: phi3, mistral, llama3, qwen:7b) |
| | `sjsujetsontool ollama delete mistral` | Remove a model from disk to free space |
| | `sjsujetsontool ollama status` | Check if Ollama server is running on port 11434 |
| | `sjsujetsontool ollama ask "What is NVIDIA Jetson?"` | Ask a question (auto-pulls model if needed) |
| | `sjsujetsontool ollama ask --model mistral "Explain transformers."` | Ask with specific model (remembers last used model) |
| **🔬 Llama.cpp** | `sjsujetsontool llama` | Start llama.cpp server on port 8000 |
| | `./build_cuda/bin/llama-cli -m /models/mistral.gguf -p "Your prompt"` | Run model directly in container (provides performance metrics) |
| | `./build_cuda/bin/llama-server -m /models/mistral.gguf --port 8080` | Start HTTP server on port 8080 (OpenAI API compatible) |
| | `curl http://localhost:8080/completion -d '{"prompt":"Your prompt","n_predict":100}'` | Query server via HTTP API |
| | `http://localhost:8080` | Access basic web UI via browser |
| **🔧 System** | `sjsujetsontool set-hostname new-hostname` | Change hostname, regenerate system identity, write `/etc/device-id` (requires sudo) |
| | `sjsujetsontool mount-nfs <host> <remote_path> <mount_point>` | Mount NFS share for accessing centralized data |
| | `sjsujetsontool debug` | Run diagnostics to check system health and configuration |

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