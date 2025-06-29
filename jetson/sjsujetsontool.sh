#!/bin/bash

# === sjsujetsontool ===
# Custom dev CLI for Jetson Orin Nano

IMAGE_NAME="jetson-llm-v1"
WORKSPACE_DIR="$(pwd)/workspace"
DEV_DIR="/Developer"
MODELS_DIR="/Developer/models"
CONTAINER_NAME="jetson-dev"
CONTAINER_CMD="docker run --rm -it --runtime=nvidia --network host \
  -v $WORKSPACE_DIR:/workspace \
  -v $MODELS_DIR:/models \
  -v $DEV_DIR:/Developer \
  --name $CONTAINER_NAME $IMAGE_NAME"

show_help() {
  echo "Usage: jetson-devtool [option] [args]"
  echo "Options:"
  echo "  shell             - Open a shell inside the LLM container"
  echo "  jupyter           - Start JupyterLab inside the container"
  echo "  ollama            - Run Ollama server (port 11434)"
  echo "  llama             - Start llama.cpp server (port 8000)"
  echo "  fastapi           - Start a FastAPI app on port 8001"
  echo "  rag               - Launch LangChain-based RAG server"
  echo "  convert           - Convert HF model to GGUF (custom script)"
  echo "  run <file.py>     - Run a Python file inside the container"
  echo "  set-hostname <n>  - Change hostname (for cloned Jetsons)"
  echo "  setup-ssh <ghuser>- Add GitHub user's SSH key for login"
  echo "  update            - Update this script from GitHub"
  echo "  build             - Rebuild Docker image"
  echo "  status            - Show container and service status"
  echo "  mount-nfs [host] [remote_path] [local_path]  - Mount remote NFS share using .local name"
  echo "  list              - Show all available commands"
  echo "  stop              - Stop container"
  echo "  help              - Show this help message"
}

show_list() {
  echo "📦 Available Features in sjsujetsontool:"
  echo
  echo "  shell        → Open a shell inside the LLM container"
  echo "     ▶ sjsujetsontool shell"
  echo
  echo "  jupyter      → Start JupyterLab (port 8888)"
  echo "     ▶ sjsujetsontool jupyter"
  echo
  echo "  ollama       → Start Ollama API server (port 11434)"
  echo "     ▶ sjsujetsontool ollama"
  echo
  echo "  llama        → Start llama.cpp REST server (port 8000)"
  echo "     ▶ sjsujetsontool llama"
  echo
  echo "  fastapi      → Start a FastAPI app (port 8001)"
  echo "     ▶ sjsujetsontool fastapi"
  echo
  echo "  rag          → Run a LangChain-based RAG server"
  echo "     ▶ sjsujetsontool rag"
  echo
  echo "  convert      → Convert HF model to GGUF"
  echo "     ▶ sjsujetsontool convert"
  echo
  echo "  run          → Run any Python script inside container"
  echo "     ▶ sjsujetsontool run workspace/scripts/example.py"
  echo
  echo "  set-hostname → Change Jetson hostname (for clones)"
  echo "     ▶ sjsujetsontool set-hostname sjsujetson-02"
  echo
  echo "  setup-ssh    → Add GitHub SSH public key to Jetson"
  echo "     ▶ sjsujetsontool setup-ssh your_github_username"
  echo
  echo "  update       → Pull latest jetson-devtool script"
  echo "     ▶ sjsujetsontool update"
  echo
  echo "  build        → Rebuild the Docker image"
  echo "     ▶ sjsujetsontool build"
  echo
  echo "  mount-nfs    → Mount an NFS share from .local hostname"
  echo "     ▶ sjsujetsontool mount-nfs nfs-server.local /srv/nfs/shared /mnt/nfs/shared"
  echo
  echo "  status       → Show container and service status"
  echo "     ▶ sjsujetsontool status"
  echo
  echo "  stop         → Stop container"
  echo "     ▶ sjsujetsontool stop"
}

check_service() {
  local port=$1
  local name=$2
  if ss -tuln | grep -q ":$port "; then
    echo "✅ $name is running on port $port"
  else
    echo "❌ $name not running (port $port closed)"
  fi
}

case "$1" in
  shell)
    eval "$CONTAINER_CMD"
    ;;
  jupyter)
    eval "$CONTAINER_CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    ;;
  ollama)
    echo "🧠 Launching Ollama server inside container (port 11434)..."
    eval "$CONTAINER_CMD ollama serve"
    ;;
  llama)
    echo "🧠 Launching llama.cpp server inside container (port 8000)..."
    eval "$CONTAINER_CMD /Developer/llama.cpp/build_cuda/bin/llama-server -m /models/mistral.gguf --host 0.0.0.0 --port 8000"
    ;;
  fastapi)
    echo "🚀 Launching FastAPI server on port 8001..."
    eval "$CONTAINER_CMD uvicorn app.main:app --host 0.0.0.0 --port 8001"
    ;;
  rag)
    echo "📚 Starting LangChain RAG demo..."
    eval "$CONTAINER_CMD python3 rag_server.py"
    ;;
  convert)
    echo "🔁 Converting HF model to GGUF..."
    eval "$CONTAINER_CMD python3 /workspace/scripts/convert_hf_to_gguf.py"
    ;;
  run)
    shift
    if [ -z "$1" ]; then
      echo "❌ Missing script path. Usage: sjsujetsontool run <path/to/script.py>"
    else
      echo "🐍 Running Python script: $1"
      eval "$CONTAINER_CMD python3 $1"
    fi
    ;;
  mount-nfs)
    shift
    REMOTE_HOST=${1:-nfs-server.local}
    REMOTE_PATH=${2:-/srv/nfs/shared}
    MOUNT_POINT=${3:-/mnt/nfs/shared}

    echo "📡 Mounting NFS share from $REMOTE_HOST:$REMOTE_PATH to $MOUNT_POINT"

    sudo mkdir -p "$MOUNT_POINT"

    sudo mount -t nfs "$REMOTE_HOST:$REMOTE_PATH" "$MOUNT_POINT"

    if mountpoint -q "$MOUNT_POINT"; then
      echo "✅ Mounted successfully."
    else
      echo "❌ Failed to mount. Check NFS server or network."
    fi
    ;;
  build)
    echo "🛠️ Rebuilding Docker image..."
    docker build -t $IMAGE_NAME .
    ;;
  status)
    echo "📦 Docker Containers:"
    docker ps | grep "$CONTAINER_NAME" || echo "❌ Container '$CONTAINER_NAME' is not running"
    echo
    echo "📊 GPU Usage (tegrastats):"
    timeout 2s tegrastats || echo "⚠️ tegrastats not found"
    echo
    echo "🔍 Port Status:"
    check_service 8888 "JupyterLab"
    check_service 11434 "Ollama"
    check_service 8000 "llama.cpp"
    check_service 8001 "FastAPI"
    ;;
  list)
    show_list
    ;;
  update)
    echo "⬇️  Updating sjsujetsontool from GitHub..."
    SCRIPT_PATH=$(realpath "$0")
    BACKUP_PATH="${SCRIPT_PATH}.bak"

    echo "🔁 Backing up current script to $BACKUP_PATH"
    cp "$SCRIPT_PATH" "$BACKUP_PATH"

    curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/sjsujetsontool.sh -o "$SCRIPT_PATH"
    chmod +x "$SCRIPT_PATH"

    echo "✅ Update complete. Backup saved at $BACKUP_PATH"
    ;;
  set-hostname)
    shift
    if [ -z "$1" ]; then
      echo "❌ Missing hostname. Usage: sjsujetsontool set-hostname <new-hostname> [github_user]"
    else
      NEW_NAME="$1"
      GITHUB_USER="${2:-}"

      echo "🔧 Setting hostname to: $NEW_NAME"
      echo "$NEW_NAME" | sudo tee /etc/hostname > /dev/null

      echo "📝 Updating /etc/hosts..."
      sudo sed -i "s/127.0.1.1\s.*/127.0.1.1\t$NEW_NAME/" /etc/hosts

      echo "🔄 Resetting machine-id..."
      sudo truncate -s 0 /etc/machine-id
      sudo rm -f /var/lib/dbus/machine-id
      sudo ln -s /etc/machine-id /var/lib/dbus/machine-id

      echo "🆔 Writing device ID to /etc/device-id"
      echo "$NEW_NAME" | sudo tee /etc/device-id > /dev/null
      echo "🔁 Please reboot for changes to fully apply."
    fi
    ;;
  setup-ssh)
    shift
    if [ -z "$1" ]; then
      echo "❌ Usage: sjsujetsontool setup-ssh <github-username>"
      exit 1
    fi
    GH_USER="$1"
    echo "🔐 Setting up SSH access for GitHub user: $GH_USER"
    mkdir -p ~/.ssh
    curl -fsSL https://github.com/$GH_USER.keys >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    echo "✅ SSH key added. You can now login from any of $GH_USER's devices using SSH."
    ;;
  stop)
    echo "🛑 Stopping container..."
    docker stop $CONTAINER_NAME
    ;;
  help|*)
    show_help
    ;;
esac