#!/bin/bash

# === sjsujetsontool ===
# Custom dev CLI for Jetson Orin Nano
SCRIPT_VERSION="v0.9.0"

# ❗ Warn if run incorrectly via `bash sjsujetsontool version`
if [[ "$0" == "bash" && "$1" == "${BASH_SOURCE[0]}" ]]; then
  echo "⚠️  Please run this script directly, not via 'bash'."
  echo "✅ Correct: ./sjsujetsontool version"
  echo "❌ Wrong: bash sjsujetsontool version"
  exit 1
fi

# 📟 Detect Jetson hardware model (Orin, Xavier, Nano, etc.)
JETSON_MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
if [[ -n "$JETSON_MODEL" ]]; then
  echo "🧠 Detected Jetson Model: $JETSON_MODEL"
fi

# 🧰 Detect JetPack and CUDA version
JETPACK_VERSION=$(dpkg-query --show nvidia-jetpack 2>/dev/null | awk '{print $2}')
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep release | sed -E 's/.*release ([0-9.]+),.*/\1/')

if [[ -n "$JETPACK_VERSION" ]]; then
  echo "📦 JetPack Version: $JETPACK_VERSION"
fi
if [[ -n "$CUDA_VERSION" ]]; then
  echo "⚙️  CUDA Version: $CUDA_VERSION"
fi

# 🧠 Detect cuDNN version
CUDNN_VERSION=$(cat /usr/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2 | awk '{print $3}' | paste -sd.)
if [[ -n "$CUDNN_VERSION" ]]; then
  echo "🧬 cuDNN Version: $CUDNN_VERSION"
fi

# 🤖 Detect TensorRT version
TENSORRT_VERSION=$(dpkg-query --show libnvinfer8 2>/dev/null | awk '{print $2}')
if [[ -n "$TENSORRT_VERSION" ]]; then
  echo "🧠 TensorRT Version: $TENSORRT_VERSION"
fi

# Function to show a spinner during long-running operations
show_spinner() {
  local PID=$1
  local MESSAGE="$2"
  local CHARS="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
  
  while kill -0 $PID 2>/dev/null; do
    for (( i=0; i<${#CHARS}; i++ )); do
      echo -ne "\r${CHARS:$i:1} $MESSAGE"
      sleep 0.2
    done
  done
  
  # Clear the spinner line
  echo -ne "\r                                                  \r"
}

# Function to pull Docker image with progress indicator
pull_with_progress() {
  local IMAGE="$1"
  local MESSAGE="${2:-Downloading... Please wait}"
  
  # Start the pull in background
  docker pull $IMAGE &
  local PID=$!
  
  # Show spinner while pulling
  show_spinner $PID "$MESSAGE"
  
  # Check if pull was successful
  wait $PID
  if [ $? -eq 0 ]; then
    echo "✅ Image downloaded successfully."
    return 0
  else
    echo "❌ Failed to download image. Please check your network connection."
    return 1
  fi
}

#IMAGE_NAME="jetson-llm-v1"
DOCKERHUB_USER="cmpelkk"
IMAGE_NAME="jetson-llm"
IMAGE_TAG="v1"
LOCAL_IMAGE="$IMAGE_NAME:$IMAGE_TAG"
DEFAULT_REMOTE_TAG="latest"
#REMOTE_IMAGE="sjsujetson/jetson-llm:latest"
REMOTE_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:latest"

#WORKSPACE_DIR="$(pwd)/workspace"
#WORKSPACE_DIR="$(pwd)"
WORKSPACE_DIR="$(realpath .)"
DEV_DIR="/Developer"
MODELS_DIR="/Developer/models"
CONTAINER_NAME="jetson-dev"
# CONTAINER_CMD="docker run --rm -it --runtime=nvidia --network host \
#   -v $WORKSPACE_DIR:/workspace \
#   -v $MODELS_DIR:/models \
#   -v $DEV_DIR:/Developer \
#   --name $CONTAINER_NAME $IMAGE_NAME"

# CONTAINER_CMD="docker run --rm -it --runtime=nvidia --network host \
#   --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
#   -v $WORKSPACE_DIR:/workspace \
#   -v $MODELS_DIR:/models \
#   -v $DEV_DIR:/Developer \
#   --name $CONTAINER_NAME $IMAGE_NAME"
# Allow local container access to X11 display
xhost +local:docker >/dev/null 2>&1 || echo "Warning: xhost command failed. X11 forwarding may not work."

EXTRA_BINDS="-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev"
VOLUME_FLAGS="-v $WORKSPACE_DIR:/workspace -v $MODELS_DIR:/models -v $DEV_DIR:/Developer"
CREATE_CMD="docker create -it --runtime=nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e DISPLAY=$DISPLAY \
  --name $CONTAINER_NAME $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"
#EXEC_CMD is used after ensure_container_started() function
#Executes commands inside an already running container
EXEC_CMD="docker exec -it $CONTAINER_NAME" 
#Creates and starts a new container instance, starts fresh each time (stateless)
CONTAINER_CMD="docker run --rm -it --runtime=nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined \
  -e DISPLAY=$DISPLAY \
  $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"



ensure_container_started() {
  if ! docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "🛠️  Creating persistent container '$CONTAINER_NAME'..."
    
    # Check if image exists locally
    if ! docker image inspect $LOCAL_IMAGE >/dev/null 2>&1; then
      echo "📥 Image not found locally. Pulling $REMOTE_IMAGE..."
      
      # Pull with progress indicator
      if pull_with_progress $REMOTE_IMAGE "Downloading container image... Please wait"; then
        docker tag $REMOTE_IMAGE $LOCAL_IMAGE
      else
        exit 1
      fi
    fi
    
    echo "🔧 Creating container..."
    eval "$CREATE_CMD"
  fi
  docker start $CONTAINER_NAME >/dev/null
}

show_help() {
  echo "Usage: sjsujetsontool [option] [args]"
  echo "Options:"
  echo "  shell             - Open a shell inside the LLM container"
  echo "  jupyter           - Start JupyterLab inside the container"
  echo "  ollama <subcmd>    - Manage Ollama in container"
  echo "      serve                - Start Ollama REST API server"
  echo "      run <model>          - Run model interactively"
  echo "      list                 - List installed models"
  echo "      pull <model>         - Pull model from registry"
  echo "      delete <model>       - Delete model from disk"
  echo "      status               - Check if REST server is running"
  echo "      ask [--model xxx]    - Ask model with auto pull/cache"
  echo "  llama             - Start llama.cpp server (port 8000)"
  echo "  fastapi           - Start a FastAPI app on port 8001"
  echo "  rag               - Launch LangChain-based RAG server"
  echo "  convert           - Convert HF model to GGUF (custom script)"
  echo "  run <file.py>     - Run a Python file inside the container"
  echo "  set-hostname <n>  - Change hostname (for cloned Jetsons)"
  echo "  setup-ssh <ghuser>- Add GitHub user's SSH key for login"
  echo "  update            - Update both script and container image"
  echo "  update-container   - Update only the Docker container image"
  echo "  update-script      - Update only this script from GitHub"
  echo "  build             - Rebuild Docker image"
  echo "  status            - Show container and service status"
  echo "  mount-nfs [host] [remote_path] [local_path]  - Mount remote NFS share using .local name"
  echo "  list              - Show all available commands"
  echo "  juiceshop         - Run OWASP Juice Shop container"
  echo "  zaproxy           - Run OWASP ZAP Proxy container"
  echo "  stop              - Stop container"
  echo "  delete            - Delete container without saving"
  echo "  help              - Show this help message"
  echo "  version           - Show script version and image version"
  echo "  publish [--tag tag] - Push local image to Docker Hub"
  echo "  commit-and-publish  - Commit container and push as image"
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
  echo "  ollama       → Ollama model management CLI"
  echo "     ▶ sjsujetsontool ollama serve"
  echo "     ▶ sjsujetsontool ollama run mistral"
  echo "     ▶ sjsujetsontool ollama ask --model phi3 \"Explain LLMs\""
  echo "     ▶ sjsujetsontool ollama pull llama3"
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
  echo "  update       → Update both script and container image"
  echo "     ▶ sjsujetsontool update"
  echo
  echo "  update-container → Update only the Docker container image"
  echo "     ▶ sjsujetsontool update-container"
  echo
  echo "  update-script → Update only this script from GitHub"
  echo "     ▶ sjsujetsontool update-script"
  echo
  echo "  build        → Rebuild the Docker image"
  echo "     ▶ sjsujetsontool build"
  echo
  echo "  mount-nfs    → Mount an NFS share from .local hostname"
  echo "     ▶ sjsujetsontool mount-nfs nfs-server.local /srv/nfs/shared /mnt/nfs/shared"
  echo
  echo "  juiceshop    → Run OWASP Juice Shop container"
  echo "     ▶ sjsujetsontool juiceshop"
  echo
  echo "  zaproxy      → Run OWASP ZAP Proxy container"
  echo "     ▶ sjsujetsontool zaproxy"
  echo
  echo "  status       → Show container and service status"
  echo "     ▶ sjsujetsontool status"
  echo
  echo "  stop         → Stop container"
  echo "     ▶ sjsujetsontool stop"
  echo
  echo "  delete       → Delete container without saving"
  echo "     ▶ sjsujetsontool delete"
}

juiceshop() {
  CONTAINER_NAME="juice-shop"
  IMAGE_NAME="bkimminich/juice-shop"
  PORT=3000

  # Stop and remove any existing container (prevent conflict)
  if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "[INFO] Removing existing container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
  fi

  echo "[INFO] Pulling Juice Shop image..."
  
  # Pull with progress indicator
  if ! pull_with_progress $IMAGE_NAME "Downloading Juice Shop image... Please wait"; then
    exit 1
  fi

  echo "[INFO] Starting Juice Shop with --rm at http://localhost:$PORT"
  docker run --rm --name $CONTAINER_NAME --runtime=nvidia --network host \
    --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev \
    $IMAGE_NAME
}

# zaproxy() {
#   CONTAINER_NAME="zap-proxy"
#   IMAGE_NAME="zaproxy/zap-stable"
#   PORT=8080

#   # Stop and remove any existing container (prevent conflict)
#   if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
#     echo "[INFO] Removing existing container: $CONTAINER_NAME"
#     docker rm -f $CONTAINER_NAME
#   fi

#   echo "[INFO] Pulling ZAP Proxy image..."
  
#   # Pull with progress indicator
#   if ! pull_with_progress $IMAGE_NAME "Downloading ZAP Proxy image... Please wait"; then
#     exit 1
#   fi

#   # Setup X11 forwarding
#   echo "[INFO] Setting up X11 forwarding..."
  
#   # Allow X11 connections from localhost
#   xhost +local:docker 2>/dev/null || echo "[WARN] xhost not available, X11 forwarding may not work"
  
#   # Get current user info for proper X11 permissions
#   USER_ID=$(id -u)
#   GROUP_ID=$(id -g)
  
#   # Set DISPLAY if not already set
#   if [ -z "$DISPLAY" ]; then
#     export DISPLAY=:0
#     echo "[INFO] DISPLAY not set, using :0"
#   fi
  
#   echo "[INFO] Starting ZAP Proxy with X11 forwarding at http://localhost:$PORT"
#   echo "[INFO] DISPLAY=$DISPLAY, UID=$USER_ID, GID=$GROUP_ID"
  
#   docker run --rm --name $CONTAINER_NAME --runtime=nvidia --network host \
#     --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined \
#     -e DISPLAY=$DISPLAY \
#     -e XAUTHORITY=/tmp/.docker.xauth \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     -v "$HOME/.Xauthority:/tmp/.docker.xauth:ro" \
#     -v /dev:/dev \
#     --user "$USER_ID:$GROUP_ID" \
#     $IMAGE_NAME zap-x.sh
  
#   # Clean up X11 permissions
#   echo "[INFO] Cleaning up X11 permissions..."
#   xhost -local:docker 2>/dev/null || true
# }

zaproxy() {
  CONTAINER_NAME="zap-proxy"
  IMAGE_NAME="zaproxy/zap-stable"
  PORT=8080

  # Stop and remove any existing container (prevent conflict)
  if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "[INFO] Removing existing container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
  fi

  echo "[INFO] Pulling ZAP Proxy image..."
  
  # Pull with progress indicator
  if ! pull_with_progress $IMAGE_NAME "Downloading ZAP Proxy image... Please wait"; then
    exit 1
  fi

  echo "[INFO] Starting ZAP Proxy with --rm at http://localhost:$PORT"
  docker run --rm --name $CONTAINER_NAME --runtime=nvidia --network host \
    --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev \
    $IMAGE_NAME zap-x.sh
}

snapfix() {
  echo "[⚙️] Reverting Snap daemon to a stable version..."
  sudo snap revert snapd

  echo "[🚫] Holding Snap daemon to prevent future auto-refresh..."
  sudo snap refresh --hold snapd

  echo "[✅] Snap daemon reverted and held. This helps prevent browser launch issues on Jetson."
}

meshvpn() {
  LOCAL_PATH="/Developer/"
  SCRIPT_NAME="jetsonnebula.sh"
  GITHUB_RAW="https://raw.githubusercontent.com/lkk688/edgeAI/main/edgeInfra/${SCRIPT_NAME}"

  echo "[📥] Downloading ${SCRIPT_NAME} from GitHub..."
  sudo mkdir -p "$LOCAL_PATH"
  sudo curl -fsSL "$GITHUB_RAW" -o "$LOCAL_PATH/$SCRIPT_NAME"
  sudo chmod +x "$LOCAL_PATH/$SCRIPT_NAME"

  echo "[🔑] Setting token..."
  sudo "$LOCAL_PATH/$SCRIPT_NAME" set-token sjsucyberaijetsonsuper25

  echo "[⚙️] Installing Mesh VPN..."
  sudo "$LOCAL_PATH/$SCRIPT_NAME" install

  # echo "[📡] Checking Nebula status..."
  # sudo "$LOCAL_PATH/$SCRIPT_NAME" status
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
    ensure_container_started
    $EXEC_CMD bash
    ;;
  jupyter)
    ensure_container_started
    $EXEC_CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    ;;
  ollama)
    shift
    SUBCMD="$1"
    shift

    ensure_container_started

    case "$SUBCMD" in
      serve)
        echo "🧠 Starting Ollama server on port 11434..."
        $EXEC_CMD ollama serve
        ;;

      run)
        if [ -z "$1" ]; then
          echo "❌ Usage: sjsujetsontool ollama run <model>"
          exit 1
        fi
        MODEL="$1"
        echo "💬 Launching model '$MODEL' in CLI..."
        $EXEC_CMD ollama run "$MODEL"
        ;;

      list)
        echo "📃 Installed models:"
        $EXEC_CMD ollama list
        ;;

      pull)
        if [ -z "$1" ]; then
          echo "❌ Usage: sjsujetsontool ollama pull <model>"
          exit 1
        fi
        echo "⬇️ Pulling model: $1"
        $EXEC_CMD ollama pull "$1"
        ;;

      delete)
        if [ -z "$1" ]; then
          echo "❌ Usage: sjsujetsontool ollama delete <model>"
          exit 1
        fi
        echo "🗑️ Deleting model: $1"
        $EXEC_CMD ollama delete "$1"
        ;;

      status)
        echo "🔍 Checking Ollama port (11434)..."
        if docker exec "$CONTAINER_NAME" ss -tuln | grep -q ':11434'; then
          echo "✅ Ollama server is running."
        else
          echo "❌ Ollama server not running."
        fi
        ;;

      ask)
        CACHE_FILE="$WORKSPACE_DIR/.last_ollama_model"
        MODEL="phi3"

        if [[ "$1" == --model ]]; then
          shift
          MODEL="$1"
          shift
          echo "$MODEL" > "$CACHE_FILE"
        elif [ -f "$CACHE_FILE" ]; then
          MODEL=$(cat "$CACHE_FILE")
        fi

        PROMPT="$*"
        if [ -z "$PROMPT" ]; then
          echo "❌ Usage: sjsujetsontool ollama ask [--model model-name] \"your prompt\""
          exit 1
        fi

        echo "💬 Using model: $MODEL"
        echo "📦 Checking if model '$MODEL' is available..."
        if ! $EXEC_CMD ollama list | grep -q "^$MODEL "; then
          echo "⬇️ Pulling model '$MODEL'..."
          $EXEC_CMD ollama pull "$MODEL"
        fi

        echo "💬 Asking: $PROMPT"
        $EXEC_CMD curl -s http://localhost:11434/api/generate -d '{
          "model": "'"$MODEL"'",
          "prompt": "'"$PROMPT"'",
          "stream": false
        }' | jq -r .response
        ;;

      *)
        echo "❓ Unknown Ollama subcommand. Try: serve | run | list | pull | delete | ask"
        ;;
    esac
    ;;
  llama)
    echo "🧠 Launching llama.cpp server inside container (port 8000)..."
    eval "$CONTAINER_CMD llama-server -m /models/mistral.gguf --host 0.0.0.0 --port 8000"
    ;;
  gradio)
    ensure_container_started
    echo "🌐 Launching Gradio Ollama UI on port 7860..."
    $EXEC_CMD bash -c "ollama serve & sleep 2 && python3 /Developer/edgeAI/jetson/ollama_gradio_ui.py"
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
      exit 1
    fi
    ensure_container_started
    echo "🐍 Running Python script: $@"
    $EXEC_CMD python3 "$@"
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
    echo "ℹ️ The 'update' command has been split into two separate commands:"
    echo "  - 'update-container': Updates only the Docker container"
    echo "  - 'update-script': Updates only this script"
    echo "\nRunning both updates sequentially..."
    
    # First update container
    echo "\n🔄 Running container update..."
    $0 update-container
    
    # Then update script (will exit after completion)
    echo "\n🔄 Running script update..."
    $0 update-script
    exit 0
    ;;
    
  update-container)
    echo "🔍 Checking Docker image update..."
    LOCAL_ID=$(docker image inspect $LOCAL_IMAGE --format '{{.Id}}' 2>/dev/null)
    echo "⬇️ Pulling latest image..."
    
    # Pull with progress indicator
    if pull_with_progress $REMOTE_IMAGE "Downloading latest image... Please wait"; then
      REMOTE_ID=$(docker image inspect $REMOTE_IMAGE --format '{{.Id}}' 2>/dev/null)
      if [ "$LOCAL_ID" != "$REMOTE_ID" ]; then
        echo "📦 New version detected. Updating local image..."
        docker tag $REMOTE_IMAGE $LOCAL_IMAGE
        echo "✅ Local container updated from Docker Hub."
      else
        echo "✅ Local container is already up-to-date."
      fi
    else
      exit 1
    fi
    ;;
    
  update-script)
    echo "⬇️ Updating sjsujetsontool script..."
    SCRIPT_PATH=$(realpath "$0")
    BACKUP_PATH="${SCRIPT_PATH}.bak"
    TMP_PATH="${SCRIPT_PATH}.tmp"

    cp "$SCRIPT_PATH" "$BACKUP_PATH"
    echo "⬇️ Downloading latest script..."
    curl -f#L https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/sjsujetsontool.sh -o "$TMP_PATH"
    chmod +x "$TMP_PATH"

    echo "✅ Script downloaded. Replacing current script..."
    mv "$TMP_PATH" "$SCRIPT_PATH"

    echo "✅ Script updated."
    exit 0
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
  delete)
    echo "🗑️ Deleting container without saving..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo "✅ Container '$CONTAINER_NAME' deleted."
    ;;
  version)
    echo "🧾 sjsujetsontool Script Version: $SCRIPT_VERSION"
    echo "🧊 Docker Image: $LOCAL_IMAGE"
    IMAGE_ID=$(docker image inspect $LOCAL_IMAGE --format '{{.Id}}' 2>/dev/null)
    if [ -n "$IMAGE_ID" ]; then
      echo "🔍 Image ID: $IMAGE_ID"
    else
      echo "⚠️  Image not found locally."
    fi
    ;;
  juiceshop)
    juiceshop
    ;;
  zaproxy)
    zaproxy
    ;;
  snapfix)
    snapfix
    ;;
  meshvpn)
    meshvpn
    ;;
  publish)
    shift
    TAG="$DEFAULT_REMOTE_TAG"
    if [[ "$1" == "--tag" ]]; then
      shift
      TAG="$1"
    fi
    REMOTE_TAGGED="$DOCKERHUB_USER/$IMAGE_NAME:$TAG"
    echo "📤 Preparing to push local image '$LOCAL_IMAGE' as '$REMOTE_TAGGED'"
    if ! docker image inspect $LOCAL_IMAGE >/dev/null 2>&1; then
      echo "❌ Local image '$LOCAL_IMAGE' not found. Build it first."
      exit 1
    fi
    docker tag $LOCAL_IMAGE $REMOTE_TAGGED
    docker login || exit 1
    docker push $REMOTE_TAGGED
    echo "✅ Pushed image to Docker Hub: $REMOTE_TAGGED"
    ;;
  commit-and-publish)
    shift
    TAG="$DEFAULT_REMOTE_TAG"
    if [[ "$1" == "--tag" ]]; then
      shift
      TAG="$1"
    fi
    REMOTE_TAGGED="$DOCKERHUB_USER/$IMAGE_NAME:$TAG"
    echo "📝 Committing running container '$CONTAINER_NAME' to image '$LOCAL_IMAGE'..."
    docker commit "$CONTAINER_NAME" "$LOCAL_IMAGE"
    echo "🔖 Tagging image as '$REMOTE_TAGGED'..."
    docker tag "$LOCAL_IMAGE" "$REMOTE_TAGGED"
    docker login || exit 1
    echo "📤 Pushing to Docker Hub..."
    docker push "$REMOTE_TAGGED"
    echo "✅ Committed and pushed image: $REMOTE_TAGGED"
    ;;
  force_git_pull)
    # Set to your local repo path
    REPO_PATH="/Developer/edgeAI"
    # Confirm directory exists
    if [ ! -d "$REPO_PATH/.git" ]; then
        echo "Error: $REPO_PATH is not a valid Git repository."
        exit 1
    fi

    echo "Navigating to $REPO_PATH"
    cd "$REPO_PATH" || exit 1

    echo "Discarding all local changes and untracked files..."
    git reset --hard HEAD
    git clean -fd

    echo "Fetching latest changes from origin..."
    git fetch origin

    echo "Resetting local branch to match origin/main..."
    git reset --hard origin/main

    echo "Update complete. Local repository is now synced with origin/main."
    ;;
  help)
    show_help
    ;;
esac

#help|*)