# 🧠 SJSU Jetson Tool Cheatsheet

A quick reference guide for using the `sjsujetsontool` utility on NVIDIA Jetson devices. Full tutorial: [00_sjsujetsontool_guide.md](https://github.com/lkk688/edgeAI/blob/main/docs/curriculum/00_sjsujetsontool_guide.md)

## 📋 Command Reference Table

| Category | Command | Description |
|----------|---------|-------------|
| **📥 Installation** | `curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh \| bash` | One-line install (no sudo required) |
| | `sjsujetsontool update` | Update both container and script (may need to run twice) |
| **🔍 Basic** | `sjsujetsontool list` | Display all available commands |
| | `sjsujetsontool version` | Show script and container versions |
| | `sjsujetsontool status` | Show container state, GPU stats, and port status |
| **🐳 Container** | `sjsujetsontool shell` | Enter container shell |
| | `exit` | Exit container (container keeps running) |
| | `sjsujetsontool stop` | Stop the running Docker container |
| **🐍 Python** | `sjsujetsontool run /path/to/script.py` | Run a Python script inside the container |
| | `sjsujetsontool jupyter` | Launch JupyterLab on port 8888 |
| **🧠 Ollama** | `sjsujetsontool ollama serve` | Start Ollama server |
| | `sjsujetsontool ollama run mistral` | Run model in CLI mode |
| | `sjsujetsontool ollama list` | List installed models |
| | `sjsujetsontool ollama pull llama3` | Download a model |
| | `sjsujetsontool ollama delete mistral` | Remove a model from disk |
| | `sjsujetsontool ollama status` | Check if Ollama server is running |
| | `sjsujetsontool ollama ask "What is NVIDIA Jetson?"` | Ask a question (auto-pulls model) |
| | `sjsujetsontool ollama ask --model mistral "Explain transformers."` | Ask with specific model |
| **🔬 Llama.cpp** | `sjsujetsontool llama` | Start llama.cpp server on port 8000 |
| | `./build_cuda/bin/llama-cli -m /models/mistral.gguf -p "Your prompt"` | Run model directly (inside container) |
| | `./build_cuda/bin/llama-server -m /models/mistral.gguf --port 8080` | Start HTTP server on port 8080 |
| | `curl http://localhost:8080/completion -d '{"prompt":"Your prompt","n_predict":100}'` | Query server |
| **🔧 System** | `sjsujetsontool set-hostname new-hostname` | Change hostname (requires sudo) |
| | `sjsujetsontool mount-nfs <host> <remote_path> <mount_point>` | Mount NFS share |
| | `sjsujetsontool debug` | Run diagnostics |

## 📂 Important Paths

- Host directories mounted in container:
  - `/Developer` → `/Developer` in container
  - `/Developer/models` → `/models` in container

## ⚠️ Safety Tips

- Use a 5A USB-C adapter or official barrel jack for power stability
- Always stop containers with `sjsujetsontool stop` before unplugging
- Change hostname and machine-id after cloning to prevent network conflicts