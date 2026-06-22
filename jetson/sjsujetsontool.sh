#!/bin/bash

# === sjsujetsontool ===
# Custom dev CLI for Jetson Orin Nano
SCRIPT_VERSION="v1.0.0"

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

# 🧮 Total system RAM in MB. On Jetson this is UNIFIED CPU+GPU memory, so
# llama.cpp's batch/context buffers must fit inside it. 8GB-class devices OOM
# with the large default batch, so we scale it down below (see the llama arm).
TOTAL_RAM_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 8000)

# 🧰 Detect JetPack and CUDA version
JETPACK_VERSION=$(dpkg-query --show nvidia-jetpack 2>/dev/null | awk '{print $2}')
# Detect L4T BSP revision from /etc/nv_tegra_release (e.g. R36.4.7)
L4T_REVISION=$(head -1 /etc/nv_tegra_release 2>/dev/null | sed 's/# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/R\1.\2/')
_L4T_PKG=$(dpkg-query --show nvidia-l4t-core 2>/dev/null | awk '{print $2}')
if [[ -z "$L4T_REVISION" && -n "$_L4T_PKG" ]]; then
  L4T_REVISION="R$(echo "$_L4T_PKG" | cut -d- -f1 | cut -d: -f2)"
fi
if [[ -z "$JETPACK_VERSION" && -n "$L4T_REVISION" ]]; then
  _L4T_MAJOR=$(echo "$L4T_REVISION" | grep -oE '[0-9]+' | head -1)
  _L4T_MINOR=$(echo "$L4T_REVISION" | grep -oE '[0-9]+' | sed -n '2p')
  case "$_L4T_MAJOR" in
    32) JETPACK_VERSION="4.x (inferred)" ;;
    35) JETPACK_VERSION="5.x (inferred)" ;;
    36)
      if [[ "$_L4T_MINOR" -ge 4 ]]; then
        JETPACK_VERSION="6.1+ (inferred)"
      else
        JETPACK_VERSION="6.0 (inferred)"
      fi
      ;;
    37|38) JETPACK_VERSION="7.x (inferred)" ;;
  esac
fi

# Try nvcc first, then fall back to version file in CUDA install dir
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep release | sed -E 's/.*release ([0-9.]+),.*/\1/')
if [[ -z "$CUDA_VERSION" ]]; then
  CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
  if [[ -n "$CUDA_DIR" ]]; then
    CUDA_VERSION=$(basename "$CUDA_DIR" | sed 's/cuda-//')
  fi
fi

if [[ -n "$JETPACK_VERSION" ]]; then
  echo "📦 JetPack Version: $JETPACK_VERSION"
fi
if [[ -n "$L4T_REVISION" ]]; then
  echo "🏷️  L4T BSP Revision: $L4T_REVISION"
fi
if [[ -n "$CUDA_VERSION" ]]; then
  echo "⚙️  CUDA Version: $CUDA_VERSION"
fi

# 🧠 Detect cuDNN version (use sed, then fall back to dpkg-query)
CUDNN_VERSION=$(sed -n 's/^#define CUDNN_MAJOR \([0-9]*\)/\1/p; s/^#define CUDNN_MINOR \([0-9]*\)/\1/p; s/^#define CUDNN_PATCHLEVEL \([0-9]*\)/\1/p' /usr/include/cudnn_version.h 2>/dev/null | paste -sd.)
if [[ -z "$CUDNN_VERSION" ]]; then
  CUDNN_VERSION=$(dpkg-query -W -f='${Version}\n' 'libcudnn*' 2>/dev/null | grep -v '^$' | head -1 | cut -d- -f1)
fi
if [[ -n "$CUDNN_VERSION" ]]; then
  echo "🧬 cuDNN Version: $CUDNN_VERSION"
fi

# 🤖 Detect TensorRT version (try libnvinfer8, libnvinfer-bin, then tensorrt-libs fallback)
TENSORRT_VERSION=$(dpkg-query --show libnvinfer8 2>/dev/null | awk '{print $2}')
if [[ -z "$TENSORRT_VERSION" ]]; then
  TENSORRT_VERSION=$(dpkg-query --show libnvinfer-bin 2>/dev/null | awk '{print $2}')
fi
if [[ -z "$TENSORRT_VERSION" ]]; then
  TENSORRT_VERSION=$(dpkg-query -W -f='${Version}\n' 'tensorrt-libs' 'libnvinfer*' 2>/dev/null | grep -v '^$' | head -1 | cut -d- -f1)
fi
if [[ -n "$TENSORRT_VERSION" ]]; then
  echo "🤖 TensorRT Version: $TENSORRT_VERSION"
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

setup_check_internal() {
  echo "══════════════════════════════════════════════════"
  echo "⚙️  Checking /Developer folder and edgeAI git repository..."
  echo "══════════════════════════════════════════════════"

  # 1. Check if /Developer directory exists, if not, create it
  if [ ! -d "/Developer" ]; then
    echo "📂 Directory '/Developer' does not exist. Creating it..."
    if [ "$EUID" -ne 0 ]; then
      sudo mkdir -p /Developer
      sudo chmod 777 /Developer
    else
      mkdir -p /Developer
      chmod 777 /Developer
    fi
    echo "✅ Created '/Developer' with write permissions for all users."
  else
    echo "✅ Directory '/Developer' exists."
    # Check if writable. If not, fix it
    if [ ! -w "/Developer" ]; then
      echo "🔑 Directory '/Developer' is not writable by current user. Adjusting permissions..."
      if [ "$EUID" -ne 0 ]; then
        sudo chmod 777 /Developer
      else
        chmod 777 /Developer
      fi
      echo "✅ Adjusted '/Developer' permissions to 777."
    else
      echo "✅ Directory '/Developer' is writable."
    fi
  fi

  # Check if /Developer/models exists, if not, create it
  if [ ! -d "/Developer/models" ]; then
    echo "📂 Creating '/Developer/models' for local model storage..."
    mkdir -p /Developer/models
    chmod 777 /Developer/models
  fi

  # 2. Check if git is installed
  if ! command -v git &>/dev/null; then
    echo "📦 Git is not installed. Installing Git..."
    if [ "$EUID" -ne 0 ]; then
      sudo apt-get update && sudo apt-get install -y git
    else
      apt-get update && apt-get install -y git
    fi
  fi

  # 3. Check/clone edgeAI git repository
  REPO_DIR="/Developer/edgeAI"
  # Shared repo: it may be owned by another user (e.g. the repo was cloned by
  # 'sjsujetson' but 'student' runs the update). Git then refuses with
  # "fatal: detected dubious ownership". Add a per-user safe.directory exception
  # (idempotent) so pull/reset work regardless of who owns the .git tree.
  if command -v git &>/dev/null; then
    git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$REPO_DIR" \
      || git config --global --add safe.directory "$REPO_DIR"
  fi
  if [ ! -d "$REPO_DIR" ]; then
    echo "📥 Cloning edgeAI repository from GitHub into $REPO_DIR..."
    if git clone https://github.com/lkk688/edgeAI.git "$REPO_DIR"; then
      chmod -R 777 "$REPO_DIR" 2>/dev/null || sudo chmod -R 777 "$REPO_DIR"
      echo "✅ Successfully cloned edgeAI repository."
    else
      echo "❌ Failed to clone edgeAI repository. Check network or git settings."
      exit 1
    fi
  elif [ ! -d "$REPO_DIR/.git" ]; then
    echo "⚠️  $REPO_DIR exists but is not a valid git repository. Re-initializing..."
    sudo rm -rf "$REPO_DIR"
    if git clone https://github.com/lkk688/edgeAI.git "$REPO_DIR"; then
      chmod -R 777 "$REPO_DIR" 2>/dev/null || sudo chmod -R 777 "$REPO_DIR"
      echo "✅ Successfully re-cloned edgeAI repository."
    else
      echo "❌ Failed to clone edgeAI repository."
      exit 1
    fi
  else
    echo "✅ edgeAI repository already exists."
    echo "🔄 Pulling latest changes from origin..."
    if ( cd "$REPO_DIR" && git pull ); then
      chmod -R 777 "$REPO_DIR" 2>/dev/null || sudo chmod -R 777 "$REPO_DIR"
      echo "✅ Repository updated successfully."
    else
      echo "⚠️  git pull failed. Trying to force reset..."
      if ( cd "$REPO_DIR" && git fetch --all && git reset --hard origin/main ); then
        chmod -R 777 "$REPO_DIR" 2>/dev/null || sudo chmod -R 777 "$REPO_DIR"
        echo "✅ Repository force-reset to origin/main successfully."
      else
        echo "❌ Failed to pull or reset repository. Please check manually."
      fi
    fi
  fi
  echo "══════════════════════════════════════════════════"
}

#IMAGE_NAME="jetson-llm-v1"
DOCKERHUB_USER="cmpelkk"
IMAGE_NAME="jetson-llm"
IMAGE_TAG="v1"
LOCAL_IMAGE="$IMAGE_NAME:$IMAGE_TAG"
DEFAULT_REMOTE_TAG="latest"
#REMOTE_IMAGE="sjsujetson/jetson-llm:latest"
REMOTE_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:latest"

# 🌐 Headscale / Tailscale settings
HEADSCALE_LOGIN_SERVER="https://headscale.forgengi.org"
HEADSCALE_AUTHKEY="2566b0d9607d5e78bda28311963463d358352133c32d94ae"

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

# Detect TTY for non-interactive execution support
if [ -t 0 ]; then
  TTY_FLAGS="-it"
else
  TTY_FLAGS=""
fi

CREATE_CMD="docker create $TTY_FLAGS --runtime=nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e DISPLAY=$DISPLAY \
  --name $CONTAINER_NAME $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"
#EXEC_CMD is used after ensure_container_started() function
#Executes commands inside an already running container.
# Pass the host's ~/.env.local (API keys saved by `chat`/`setup-nvapi`) into the
# container so tools like the Next.js app / gradio UI see NVIDIA_API_KEY etc.
# (inside the container, $HOME is /root, so it can't read the host file directly).
ENVFILE_ARG=""
[ -f "$HOME/.env.local" ] && ENVFILE_ARG="--env-file $HOME/.env.local"
EXEC_CMD="docker exec $TTY_FLAGS $ENVFILE_ARG $CONTAINER_NAME"
#Creates and starts a new container instance, starts fresh each time (stateless)
CONTAINER_CMD="docker run --rm $TTY_FLAGS --runtime=nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined \
  -e DISPLAY=$DISPLAY \
  $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"

ensure_container_started() {
  # Ensure /Developer is set up and edgeAI is cloned before starting container
  if [ ! -d "/Developer/edgeAI/.git" ] || [ ! -w "/Developer" ]; then
    echo "⚠️  /Developer or edgeAI repository is not correctly configured. Running setup-check..."
    setup_check_internal
  fi

  if ! docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "🛠️  Creating persistent container '$CONTAINER_NAME'..."
    
    # Check if image exists locally
    if ! docker image inspect $LOCAL_IMAGE > /dev/null 2>&1; then
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

  # Fix TensorRT library stubs inside the container (needed on fresh Jetsons without JetPack host libs)
  fix_tensorrt_libs_internal
}

fix_tensorrt_libs_internal() {
  # On fresh Jetson nodes, the container may have 0-byte stub files for CUDA/TRT libs.
  # Fix them by symlinking to the real l4t-gpu-libs versions that exist inside the container.
  docker exec $CONTAINER_NAME bash -c '
    # Fix libcuda.so.1.1 stub -> real l4t-gpu-libs version
    REAL_CUDA="/opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1.1"
    STUB_CUDA="/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1.1"
    if [ -f "$REAL_CUDA" ] && { [ ! -s "$STUB_CUDA" ] || [ "$(stat -c%s "$STUB_CUDA" 2>/dev/null)" = "0" ]; }; then
      rm -f "$STUB_CUDA"
      ln -sf "$REAL_CUDA" "$STUB_CUDA"
      echo "  ✅ Fixed libcuda.so.1.1 -> l4t-gpu-libs"
    fi

    # Fix libnvdla_compiler.so stub if 0-byte (restore from /models cache if available)
    STUB_DLA="/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so"
    CACHED_DLA="/models/libnvdla_compiler.so"
    if [ -f "$STUB_DLA" ] && [ "$(stat -c%s "$STUB_DLA" 2>/dev/null)" = "0" ] && [ -s "$CACHED_DLA" ]; then
      cp "$CACHED_DLA" "$STUB_DLA"
      echo "  ✅ Restored libnvdla_compiler.so from /models cache"
    fi
  ' 2>/dev/null || true
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
  echo "  llama [model] [fg|bg] - Start llama-server (port 8080). No args = interactive menu"
  echo "                          (qwen2b default | qwen0.8b | qwen4b | gemma4 | custom HF repo:quant)"
  echo "  llama stop        - Stop a background llama-server"
  echo "  llama-cli [model] - Run llama-cli inference (same model choices; default qwen2b)"
  echo "  ollama-serve      - Start Ollama REST API server (port 11434)"
  echo "  ollama-run        - Run Ollama model interactively (defaults to gemma4)"
  echo "  vllm [model]      - Start vLLM serve engine (defaults to Qwen3-8B-speculator)"
  echo "  fastapi           - Start a FastAPI app on port 8001"
  echo "  convert           - Convert HF model to GGUF (custom script)"
  echo "  run <file.py>     - Run a Python file inside the container"
  echo "  set-hostname <n>  - Change hostname (for cloned Jetsons)"
  echo "  setup-student [user] [pass] - Create/repair non-sudo student account (docker + /Developer, display name 'Student')"
  echo "  setup-ssh <ghuser>- Add GitHub user's SSH key for login"
  echo "  setup-nvapi       - Setup NVIDIA NGC/Build API Key in local .env.local"
  echo "  chat              - Unified streaming chat; backend menu: local llama.cpp / NVIDIA / OpenAI / Anthropic / custom"
  echo "  curl              - Interactive curl builder (LLM API or plain URL); prints + sends the command"
  echo "  nv-chat           - Chat with NVIDIA Build API models interactively (legacy)"
  echo "  setup-check       - Check and configure host /Developer folder and edgeAI repository"
  echo "  update            - Update script, container image, and check /Developer setup"
  echo "  update-container   - Update only the Docker container image"
  echo "  update-script      - Update only this script from GitHub"
  echo "  healthcheck       - Deep system health check and diagnostics"
  echo "  sysupgrade        - Update apt packages (non-Jetson-critical only)"
  echo "  dockerfix         - Fix Docker on Jetson (iptables-legacy, containerd, daemon, 'docker' group access)"
  echo "  tailscale <sub>   - Manage Tailscale / Headscale VPN"
  echo "      install              - Install Tailscale if not present"
  echo "      up [--force]         - Join headscale network (checks conflicts)"
  echo "      status               - Show Tailscale connection status"
  echo "      down                 - Disconnect from Tailscale network"
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
  echo "  llama [model] [fg|bg] → Start llama-server (port 8080). No args = interactive menu"
  echo "     ▶ sjsujetsontool llama                  # menu: pick model + foreground/background"
  echo "     ▶ sjsujetsontool llama gemma4 bg        # Gemma 4 E2B in background"
  echo "     ▶ sjsujetsontool llama unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_S   # custom HF model"
  echo "     ▶ sjsujetsontool llama stop            # stop a background server"
  echo
  echo "  llama-cli [model] → Run llama-cli inference (same model choices)"
  echo "     ▶ sjsujetsontool llama-cli qwen4b -p \"Explain the Jetson Orin Nano\""
  echo
  echo "  ollama-serve → Start Ollama REST API server (port 11434)"
  echo "     ▶ sjsujetsontool ollama-serve"
  echo
  echo "  ollama-run   → Run Ollama model interactively"
  echo "     ▶ sjsujetsontool ollama-run gemma4"
  echo
  echo "  vllm         → Start vLLM serve engine (port 8000)"
  echo "     ▶ sjsujetsontool vllm RedHatAI/Qwen3-8B-speculator.eagle3"
  echo
  echo "  fastapi      → Start a FastAPI app (port 8001)"
  echo "     ▶ sjsujetsontool fastapi"
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
  echo "  setup-nvapi  → Setup NVIDIA NGC/Build API Key in local .env.local"
  echo "     ▶ sjsujetsontool setup-nvapi"
  echo
  echo "  chat         → Unified streaming chat (Rich UI) with a backend picker"
  echo "     ▶ sjsujetsontool chat        # menu: local llama.cpp / NVIDIA / OpenAI / Anthropic / custom"
  echo "     ▶ sjsujetsontool chat --local        # quick: local Jetson llama.cpp (:8080)"
  echo "       Cloud keys are saved to ~/.env.local (you're prompted on first use)."
  echo "       In-chat commands: /help /exit /server(switch backend) /save /reset /system /think"
  echo
  echo "  curl         → Interactive curl builder — answer a few prompts, it prints & sends the request"
  echo "     ▶ sjsujetsontool curl        # pick: LLM API (chat) or a plain URL"
  echo
  echo "  nv-chat      → Chat with NVIDIA Build API models interactively (legacy)"
  echo "     ▶ sjsujetsontool nv-chat"
  echo "     ▶ sjsujetsontool nv-chat -p \"Explain Orin Nano\" -m nvidia/llama-3.1-nemotron-nano-8b-v1"
  echo
  echo "  setup-check  → Configure host /Developer folder & clone/pull edgeAI git repo"
  echo "     ▶ sjsujetsontool setup-check"
  echo 
  echo "  update       → Run full update (script + setup-check + container image)"
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

dockerfix() {
  echo "════════════════════════════════════════════════════"
  echo "🐳 Jetson Docker Fix — iptables-legacy"
  echo "════════════════════════════════════════════════════"
  echo
  echo "📋 Problem: Docker on Jetson fails to start because the system"
  echo "   iptables is set to 'nf_tables' mode, but the Jetson L4T kernel"
  echo "   only supports 'iptables-legacy'. This causes:"
  echo "     iptables v1.8.7 (nf_tables): Couldn't load match 'addrtype'"
  echo

  # Self-heal: the docker engine package itself may be MISSING (e.g. removed by a
  # bad apt upgrade — dpkg leaves it in state 'rc'). Symptoms: "dockerd not present
  # or not executable", "docker: command not found". No iptables/systemd fix can
  # help that, so reinstall docker-ce first.
  if ! command -v dockerd >/dev/null 2>&1 || ! command -v docker >/dev/null 2>&1; then
    echo "⚠️  Docker engine binaries are MISSING (docker-ce not installed)."
    if [ -f /etc/apt/sources.list.d/docker.list ]; then
      echo "🔧 Reinstalling docker-ce + docker-ce-cli from the Docker apt repo..."
      sudo apt-get update
      if sudo apt-get install -y docker-ce docker-ce-cli; then
        echo "✅ docker-ce reinstalled: $(docker --version 2>/dev/null)"
      else
        echo "❌ Reinstall failed. Check network and /etc/apt/sources.list.d/docker.list"
        return 1
      fi
    else
      echo "❌ Docker apt repo not configured (/etc/apt/sources.list.d/docker.list missing)."
      echo "   Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/"
      return 1
    fi
    echo
  fi

  # Detect current iptables mode
  CURRENT_IPT=$(iptables --version 2>/dev/null)
  if echo "$CURRENT_IPT" | grep -q "legacy"; then
    echo "✅ iptables is already set to legacy mode: $CURRENT_IPT"
    echo
  else
    echo "⚠️  Current iptables mode: $CURRENT_IPT"
    echo "🔧 Switching to iptables-legacy..."
    sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
    sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
    echo "✅ Switched: $(iptables --version)"
    echo
  fi

  # Ensure containerd (docker depends on it) and the docker daemon are up & enabled.
  # Use `systemctl is-active` as the authoritative "is the daemon running" check —
  # `docker info` can fail for permission reasons even when the daemon is fine.
  sudo systemctl enable --now containerd >/dev/null 2>&1
  if systemctl is-active --quiet docker; then
    echo "✅ Docker daemon is already running."
  else
    echo "🔄 Docker daemon not active — unmasking, enabling, and starting..."
    sudo systemctl unmask docker >/dev/null 2>&1
    sudo systemctl enable docker >/dev/null 2>&1
    sudo systemctl restart docker
    sleep 3
    if systemctl is-active --quiet docker; then
      echo "✅ Docker daemon started successfully."
    else
      echo "❌ Docker daemon failed to start. Recent logs:"
      sudo journalctl -u docker -n 25 --no-pager | tail -25
      return 1
    fi
  fi

  # Socket access (docker group). The daemon can be UP while your user still sees
  # "Cannot connect to the Docker daemon" because the user is not in the 'docker'
  # group for THIS login session. This is the most common false "not running".
  if docker info >/dev/null 2>&1; then
    echo "✅ Docker is accessible by '$(whoami)'."
  else
    echo "⚠️  Docker is RUNNING, but '$(whoami)' cannot access the socket"
    echo "    — not in the 'docker' group for this login session."
    if id -nG "$(whoami)" | tr ' ' '\n' | grep -qx docker; then
      echo "ℹ️  You ARE in the 'docker' group, but this session predates it."
    else
      echo "🔧 Adding '$(whoami)' to the 'docker' group..."
      sudo usermod -aG docker "$(whoami)"
    fi
    if sg docker -c "docker info" >/dev/null 2>&1; then
      echo "✅ Group access verified — NEW shells get it automatically."
      echo "👉 To use it in THIS shell right now:  newgrp docker   (or log out and back in)"
    else
      echo "❌ Still no socket access. Inspect: ls -l /var/run/docker.sock ; getent group docker"
    fi
  fi

  echo
  if docker info >/dev/null 2>&1; then
    echo "🧪 Testing Docker + GPU access..."
    if docker run --rm --runtime=nvidia \
      -e NVIDIA_VISIBLE_DEVICES=all \
      nvcr.io/nvidia/l4t-base:r36.2.0 \
      nvidia-smi 2>/dev/null | grep -q "NVIDIA-SMI"; then
      echo "✅ Docker GPU test passed."
    else
      echo "⚠️  GPU test skipped (base image may not be cached — run 'sjsujetsontool update-container' first)."
      echo "   Quick test: docker run --rm hello-world"
      docker run --rm hello-world 2>&1 | tail -3
    fi
  else
    echo "⏭️  Skipping GPU test — apply the docker group first ('newgrp docker' or re-login), then re-run."
  fi

  echo
  echo "════════════════════════════════════════════════════"
  echo "✅ Docker fix complete. Run 'sjsujetsontool status' to verify."
  echo "════════════════════════════════════════════════════"
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

# Map a friendly model keyword -> Hugging Face GGUF spec for `llama`/`llama-cli`.
# Sets LLM_HF and LLM_NAME. Default selector is qwen2b. Returns 1 on unknown.
llama_model() {
  case "$1" in
    qwen|qwen2b|qwen-2b|qwen3.5|qwen3.5-2b) LLM_HF="unsloth/Qwen3.5-2B-MTP-GGUF:Q4_K_S";  LLM_NAME="Qwen3.5-2B (VLM)";;
    qwen0.8b|qwen-0.8b)                     LLM_HF="unsloth/Qwen3.5-0.8B-MTP-GGUF:Q4_K_S"; LLM_NAME="Qwen3.5-0.8B";;
    qwen4b|qwen-4b)                         LLM_HF="unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_S";   LLM_NAME="Qwen3.5-4B";;
    gemma|gemma4|gemma-4)                   LLM_HF="unsloth/gemma-4-E2B-it-GGUF:Q4_K_S";   LLM_NAME="Gemma-4-E2B (VLM)";;
    *) echo "❌ Unknown model '$1'. Options: qwen2b (default), qwen0.8b, qwen4b, gemma4"; return 1;;
  esac
  return 0
}

# Locate a llama.cpp binary inside the container, preferring the newest CUDA build.
# Sets LLAMA_BIN and LD_ENV. The build_cuda path holds the current (qwen35-capable) build.
llama_bin() {
  local want="$1" d
  LLAMA_BIN="$want"; LD_ENV=""
  for d in /opt/llama.cpp/build_cuda/bin /opt/llamacpp-new/build/bin /opt/llama.cpp/build/bin; do
    if $EXEC_CMD test -f "$d/$want"; then
      LLAMA_BIN="$d/$want"; LD_ENV="LD_LIBRARY_PATH=$d"; return 0
    fi
  done
  return 0
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
    shift
    # Subcommand: stop a background llama-server
    if [ "${1:-}" = "stop" ]; then
      ensure_container_started
      if $EXEC_CMD pkill -f "llama-server" >/dev/null 2>&1; then
        echo "🛑 Stopped the running llama-server."
      else
        echo "ℹ️  No llama-server was running."
      fi
      exit 0
    fi
    ensure_container_started

    # --- 1) pick a model -----------------------------------------------------
    # Usage:
    #   sjsujetsontool llama                      -> interactive menu (default Qwen3.5-2B)
    #   sjsujetsontool llama qwen2b|qwen0.8b|qwen4b|gemma4
    #   sjsujetsontool llama <hf-repo:quant>      -> any GGUF on Hugging Face
    #   sjsujetsontool llama <model> [fg|bg]      -> non-interactive
    MODEL_SEL="${1:-}"
    RUN_MODE="${2:-}"
    if [ -n "$MODEL_SEL" ]; then
      if [[ "$MODEL_SEL" == */* ]]; then          # looks like a Hugging Face repo path
        LLM_HF="$MODEL_SEL"; LLM_NAME="$MODEL_SEL"
      else
        llama_model "$MODEL_SEL" || exit 1
      fi
    elif [ -t 0 ]; then                            # interactive terminal -> show menu
      echo "🧠 Select a model to serve with llama.cpp:"
      echo "   1) Qwen3.5-2B    (default, vision-capable)"
      echo "   2) Qwen3.5-0.8B  (smallest / fastest)"
      echo "   3) Qwen3.5-4B    (most capable)"
      echo "   4) Gemma-4-E2B   (vision-capable)"
      echo "   5) Custom        (enter a Hugging Face GGUF, e.g. unsloth/Qwen3.5-2B-MTP-GGUF:Q4_K_S)"
      read -r -p "Choice [1-5], or press Enter for the default (Qwen3.5-2B): " _choice
      case "$_choice" in
        ""|1) llama_model qwen2b ;;
        2)    llama_model qwen0.8b ;;
        3)    llama_model qwen4b ;;
        4)    llama_model gemma4 ;;
        5)    read -r -p "Hugging Face model (repo:quant): " _hf
              if [ -n "$_hf" ]; then LLM_HF="$_hf"; LLM_NAME="$_hf"; else llama_model qwen2b; fi ;;
        *)    echo "⚠️  Invalid choice — using default."; llama_model qwen2b ;;
      esac
    else
      llama_model qwen2b                           # non-interactive default
    fi

    # --- 2) foreground or background ----------------------------------------
    if [ -z "$RUN_MODE" ]; then
      if [ -t 0 ]; then
        read -r -p "Run in [f]oreground or [b]ackground? (Enter = foreground): " _rm
        case "$_rm" in b|bg|background|B) RUN_MODE=bg ;; *) RUN_MODE=fg ;; esac
      else
        RUN_MODE=fg
      fi
    fi

    # --- 3) launch (b9743+ build supports qwen35/qwen3vl) -------------------
    llama_bin llama-server
    # Stop any previous llama-server first — two servers won't fit in a Jetson's
    # unified memory, and a leftover/leaked one is the #1 cause of cudaMalloc OOM.
    $EXEC_CMD pkill -f llama-server >/dev/null 2>&1 || true
    sleep 1
    # Memory-aware buffers: an 8GB-class Jetson OOMs with the big 2048 batch
    # (the compute buffer alone is ~1GB+). Scale batch/context to the device.
    if [ "${TOTAL_RAM_MB:-8000}" -lt 10000 ]; then
      LLAMA_BATCH=512;  LLAMA_CTX=4096    # 8GB Orin Nano
    else
      LLAMA_BATCH=2048; LLAMA_CTX=8192    # 16GB+ devices
    fi
    echo "🧮 Detected ${TOTAL_RAM_MB}MB RAM → batch=${LLAMA_BATCH}, ctx=${LLAMA_CTX}"
    # --jinja uses the model's chat template so requests can toggle thinking via
    # chat_template_kwargs:{enable_thinking:false} (much faster on Jetson).
    LLAMA_ARGS="-hf $LLM_HF --host 0.0.0.0 --port 8080 --jinja --ubatch-size $LLAMA_BATCH --batch-size $LLAMA_BATCH --ctx-size $LLAMA_CTX -ngl 99"
    if [ "$RUN_MODE" = "bg" ]; then
      echo "🧠 Starting $LLM_NAME llama.cpp server in the BACKGROUND (port 8080)  [$LLM_HF]"
      docker exec -d "$CONTAINER_NAME" bash -lc "env $LD_ENV $LLAMA_BIN $LLAMA_ARGS >/tmp/llama-server.log 2>&1"
      echo "✅ Server is starting (first run downloads the model)."
      echo "   • Web UI / API : http://localhost:8080"
      echo "   • View logs    : sjsujetsontool shell  ->  tail -f /tmp/llama-server.log"
      echo "   • Stop it      : sjsujetsontool llama stop"
    else
      echo "🧠 Launching $LLM_NAME llama.cpp server (foreground, port 8080)  [$LLM_HF]"
      echo "   (Ctrl+C to stop. Tip: re-run and choose 'background' to keep it serving.)"
      if [ -n "$LD_ENV" ]; then
        $EXEC_CMD env "$LD_ENV" "$LLAMA_BIN" $LLAMA_ARGS
      else
        $EXEC_CMD "$LLAMA_BIN" $LLAMA_ARGS
      fi
    fi
    ;;
  llama-cli)
    shift
    ensure_container_started
    # Optional first arg = model selector (qwen2b default); the rest pass to llama-cli.
    MODEL_SEL="qwen2b"
    case "${1:-}" in
      qwen|qwen2b|qwen-2b|qwen3.5|qwen3.5-2b|qwen0.8b|qwen-0.8b|qwen4b|qwen-4b|gemma|gemma4|gemma-4)
        MODEL_SEL="$1"; shift;;
    esac
    llama_model "$MODEL_SEL" || exit 1
    llama_bin llama-cli
    # Memory-aware batch (see the llama arm): keep 8GB Jetsons from OOMing.
    if [ "${TOTAL_RAM_MB:-8000}" -lt 10000 ]; then LLAMA_BATCH=512; else LLAMA_BATCH=2048; fi
    echo "🧠 Running $LLM_NAME llama.cpp CLI inside persistent container...  [$LLM_HF]  (batch=$LLAMA_BATCH)"
    if [ -n "$LD_ENV" ]; then
      $EXEC_CMD env "$LD_ENV" "$LLAMA_BIN" -hf "$LLM_HF" --ubatch-size $LLAMA_BATCH --batch-size $LLAMA_BATCH -ngl 99 "$@"
    else
      $EXEC_CMD "$LLAMA_BIN" -hf "$LLM_HF" --ubatch-size $LLAMA_BATCH --batch-size $LLAMA_BATCH -ngl 99 "$@"
    fi
    ;;
  ollama-serve)
    ensure_container_started
    echo "🧠 Starting Ollama server inside persistent container (port 11434)..."
    $EXEC_CMD ollama serve
    ;;
  ollama-run)
    shift
    ensure_container_started
    MODEL="${1:-gemma4}"
    echo "💬 Launching Ollama model '$MODEL' inside persistent container..."
    $EXEC_CMD ollama run "$MODEL"
    ;;
  vllm)
    shift
    MODEL="${1:-RedHatAI/Qwen3-8B-speculator.eagle3}"
    echo "🚀 Launching vLLM server inside vllm container (port 8000)..."
    echo "💬 Model: $MODEL"
    
    # Ensure Hugging Face cache directory exists on host
    mkdir -p "$HOME/.cache/huggingface"
    
    docker run $TTY_FLAGS --rm --runtime=nvidia --network host \
      -v $HOME/.cache/huggingface:/root/.cache/huggingface \
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
      vllm serve "$MODEL" \
      --gpu-memory-utilization 0.8 \
      "$@"
    ;;
  gradio)
    ensure_container_started
    echo "🌐 Launching Edge AI Chat UI (Gradio) on http://localhost:7860 ..."
    echo "   Pick a backend in the UI (local llama.cpp / NVIDIA / OpenAI / Anthropic)."
    echo "   Tip: start a local model first with  sjsujetsontool llama"
    $EXEC_CMD bash -lc "pip show gradio >/dev/null 2>&1 || pip install -q gradio requests; python3 /Developer/edgeAI/edgeLLM/gradio_chat_ui.py"
    ;;
  fastapi)
    echo "🚀 Launching FastAPI server on port 8001..."
    eval "$CONTAINER_CMD uvicorn app.main:app --host 0.0.0.0 --port 8001"
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
    echo "🔄 Running full update (script + sample code + container)..."
    echo "  Use 'update-script', 'force_git_pull', or 'update-container' to run individually."
    echo

    # 1) update this CLI script
    echo "📜 Step 1/3: Updating sjsujetsontool script from GitHub..."
    $0 update-script
    echo

    # 2) force-update the edgeAI sample code in /Developer/edgeAI
    echo "📥 Step 2/3: Updating edgeAI sample code (git pull --force)..."
    $0 force_git_pull || echo "⚠️  Skipped sample-code update (is /Developer/edgeAI a git repo?)."
    echo

    # 3) update the container image
    echo "🐳 Step 3/3: Updating container image..."
    $0 update-container
    exit 0
    ;;
    
  update-container)
    # --- Pre-check: ensure Docker is running AND accessible by this user ---
    if ! docker info &>/dev/null; then
      if systemctl is-active --quiet docker; then
        # Daemon is UP but this session can't reach the socket (docker group).
        # If the user is already in the group, re-exec with the group applied so
        # it "just works" without a re-login; otherwise point them to dockerfix.
        if [ -z "$SJ_DOCKER_SG_RETRY" ] && id -nG "$(whoami)" | tr ' ' '\n' | grep -qx docker; then
          echo "🔁 Docker is running; applying 'docker' group to this session and retrying..."
          exec sg docker -c "SJ_DOCKER_SG_RETRY=1 '$0' update-container"
        fi
        echo "❌ Docker is running, but '$(whoami)' cannot access it (not in the 'docker' group)."
        echo "   Fix it once with:  sjsujetsontool dockerfix"
        echo "   then:  newgrp docker   (or log out / back in), and re-run."
        exit 1
      fi
      echo "❌ Docker daemon is not running."
      _IPT=$(iptables --version 2>/dev/null)
      if echo "$_IPT" | grep -q "nf_tables"; then
        echo "🔍 Detected iptables in nf_tables mode (known Jetson issue)."
        echo "🔧 Auto-fixing: switching to iptables-legacy..."
        sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
        sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
        echo "✅ Switched to: $(iptables --version)"
      fi
      sudo systemctl restart docker
      sleep 3
      if ! systemctl is-active --quiet docker; then
        echo "❌ Docker still not running after fix attempt."
        echo "   Run: sjsujetsontool dockerfix"
        echo "   Or:  sudo journalctl -u docker -n 30 --no-pager"
        exit 1
      fi
      echo "✅ Docker is now running."
    fi
    echo "🔍 Checking Docker image update..."
    LOCAL_ID=$(docker image inspect $LOCAL_IMAGE --format '{{.Id}}' 2>/dev/null)
    echo "⬇️ Pulling latest image from Docker Hub..."
    
    # Pull with progress indicator
    if pull_with_progress $REMOTE_IMAGE "Downloading latest image... Please wait"; then
      REMOTE_ID=$(docker image inspect $REMOTE_IMAGE --format '{{.Id}}' 2>/dev/null)
      if [ "$LOCAL_ID" != "$REMOTE_ID" ]; then
        echo "📦 New version detected. Updating local image..."
        if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
          echo "🗑️  Removing old container '$CONTAINER_NAME' to force recreation from the new image..."
          docker rm -f $CONTAINER_NAME
        fi
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
    echo "⬇️ Updating sjsujetsontool script from GitHub..."
    SCRIPT_PATH=$(realpath "$0")
    BACKUP_PATH="${SCRIPT_PATH}.bak"
    TMP_PATH="${SCRIPT_PATH}.tmp"

    cp "$SCRIPT_PATH" "$BACKUP_PATH"
    echo "📂 Backup saved to: $BACKUP_PATH"
    echo "⬇️ Downloading latest script..."
    if curl -f#L https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/sjsujetsontool.sh -o "$TMP_PATH"; then
      chmod +x "$TMP_PATH"
      echo "✅ Script downloaded. Replacing current script..."
      mv "$TMP_PATH" "$SCRIPT_PATH"
      echo "✅ Script updated successfully. Re-run your command to use the new version."
      # Keep the companion chat client (chat.py) in sync.
      CHAT_PY="$(dirname "$SCRIPT_PATH")/sjsujetsontool-chat.py"
      if curl -fsSL -H "Cache-Control: no-cache" \
           https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/chat.py -o "${CHAT_PY}.tmp" 2>/dev/null; then
        mv "${CHAT_PY}.tmp" "$CHAT_PY"
        echo "✅ Chat client updated: $CHAT_PY"
      else
        rm -f "${CHAT_PY}.tmp"
        echo "⚠️  Could not update chat client (will be fetched on first 'sjsujetsontool chat')."
      fi
      # Agent mode (`chat --agent`) uses the edge_agent package shipped in the repo
      # (/Developer/edgeAI/edgeLLM/edge_agent) — chat.py finds it on sys.path, so no
      # extra files are fetched here.
    else
      echo "❌ Download failed. Restoring backup..."
      cp "$BACKUP_PATH" "$SCRIPT_PATH"
      exit 1
    fi
    exit 0
    ;;

  healthcheck)
    echo "════════════════════════════════════════════════════"
    echo "🔬 Jetson Deep System Health Check"
    echo "════════════════════════════════════════════════════"
    echo

    # --- Hardware & OS ---
    echo "📟 Hardware & OS"
    echo "  Model     : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo 'Unknown')"
    echo "  Kernel    : $(uname -r)"
    echo "  OS        : $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
    echo "  Arch      : $(uname -m)"
    echo

    # --- JetPack / L4T ---
    echo "📦 NVIDIA JetPack / L4T"
    _JP=$(dpkg-query --show nvidia-jetpack 2>/dev/null | awk '{print $2}')
    _L4T=$(head -1 /etc/nv_tegra_release 2>/dev/null | sed 's/# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/R\1.\2/')
    _L4T_PKG=$(dpkg-query --show nvidia-l4t-core 2>/dev/null | awk '{print $2}')
    if [[ -z "$_L4T" && -n "$_L4T_PKG" ]]; then
      _L4T="R$(echo "$_L4T_PKG" | cut -d- -f1 | cut -d: -f2)"
    fi
    if [[ -z "$_JP" && -n "$_L4T" ]]; then
      _L4T_MAJOR=$(echo "$_L4T" | grep -oE '[0-9]+' | head -1)
      _L4T_MINOR=$(echo "$_L4T" | grep -oE '[0-9]+' | sed -n '2p')
      case "$_L4T_MAJOR" in
        32) _JP="4.x (inferred from L4T $_L4T)" ;;
        35) _JP="5.x (inferred from L4T $_L4T)" ;;
        36)
          if [[ "$_L4T_MINOR" -ge 4 ]]; then
            _JP="6.1+ (inferred from L4T $_L4T)"
          else
            _JP="6.0 (inferred from L4T $_L4T)"
          fi
          ;;
        37|38) _JP="7.x (inferred from L4T $_L4T)" ;;
        *) _JP="Unknown (L4T $_L4T)" ;;
      esac
    fi
    echo "  JetPack   : ${_JP:-Not found}"
    echo "  L4T BSP   : ${_L4T:-Unknown} (pkg: ${_L4T_PKG:-N/A})"
    # Warn if JetPack is old
    _JP_MAJOR=$(echo "$_JP" | grep -oE '^[0-9]+')
    if [[ -n "$_JP_MAJOR" && "$_JP_MAJOR" -lt 6 ]]; then
      echo "  ⚠️  JetPack $_JP is below v6.x — consider upgrading via NVIDIA SDK Manager"
    fi
    echo

    # --- CUDA ---
    echo "⚙️  CUDA"
    _NVCC_VER=$(nvcc --version 2>/dev/null | grep release | sed -E 's/.*release ([0-9.]+),.*/\1/')
    if [[ -n "$_NVCC_VER" ]]; then
      echo "  CUDA      : $_NVCC_VER (via nvcc)"
    else
      _CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
      if [[ -n "$_CUDA_DIR" ]]; then
        echo "  CUDA      : $(basename $_CUDA_DIR | sed 's/cuda-//') (via $_CUDA_DIR, nvcc not in PATH)"
        echo "  Tip       : Add to ~/.bashrc: export PATH=$_CUDA_DIR/bin:\$PATH"
      else
        echo "  CUDA      : ❌ Not found"
      fi
    fi
    echo

    # --- cuDNN ---
    echo "🧬 cuDNN"
    _CUDNN=$(sed -n 's/^#define CUDNN_MAJOR \([0-9]*\)/\1/p; s/^#define CUDNN_MINOR \([0-9]*\)/\1/p; s/^#define CUDNN_PATCHLEVEL \([0-9]*\)/\1/p' /usr/include/cudnn_version.h 2>/dev/null | paste -sd.)
    if [[ -z "$_CUDNN" ]]; then
      _CUDNN=$(dpkg-query -W -f='${Version}\n' 'libcudnn*' 2>/dev/null | grep -v '^$' | head -1 | cut -d- -f1)
    fi
    echo "  cuDNN     : ${_CUDNN:-Not found}"
    echo

    # --- TensorRT ---
    echo "🤖 TensorRT"
    _TRT=$(dpkg-query --show libnvinfer8 2>/dev/null | awk '{print $2}')
    [[ -z "$_TRT" ]] && _TRT=$(dpkg-query --show libnvinfer-bin 2>/dev/null | awk '{print $2}')
    if [[ -z "$_TRT" ]]; then
      _TRT=$(dpkg-query -W -f='${Version}\n' 'tensorrt-libs' 'libnvinfer*' 2>/dev/null | grep -v '^$' | head -1 | cut -d- -f1)
    fi
    echo "  TensorRT  : ${_TRT:-Not found}"
    echo

    # --- Memory ---
    echo "💾 Memory"
    free -h | awk 'NR==1{print "  "$0} NR==2{print "  "$0} NR==3{print "  "$0}'
    echo

    # --- Disk ---
    echo "💿 Disk Usage"
    df -h --output=source,size,used,avail,pcent,target | awk 'NR==1{print "  "$0} /\/dev\//{ if ($6 != "/boot/efi") print "  "$0}'
    # Warn if root > 80%
    _DISK_PCT=$(df / | awk 'NR==2{print $5}' | tr -d '%')
    if [[ "$_DISK_PCT" -gt 80 ]]; then
      echo "  ⚠️  Root filesystem is ${_DISK_PCT}% full — consider cleanup"
    fi
    echo

    # --- CPU Temperature ---
    echo "🌡️  Temperatures"
    if command -v tegrastats &>/dev/null; then
      timeout 2s tegrastats 2>/dev/null | grep -oE '[a-z0-9_]+@[0-9.]+C' | while read t; do
        echo "  $t"
      done
    else
      for tz in /sys/class/thermal/thermal_zone*/; do
        _tname=$(cat "${tz}type" 2>/dev/null)
        _tval=$(cat "${tz}temp" 2>/dev/null)
        if [[ -n "$_tval" ]]; then
          printf "  %-20s %s°C\n" "$_tname" "$(echo "$_tval / 1000" | bc -l 2>/dev/null || echo "$_tval")"
        fi
      done
    fi
    echo

    # --- Power ---
    echo "⚡ Power"
    if [[ -f /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_label ]]; then
      for hwmon in /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/; do
        for ch in 1 2 3; do
          _lbl=$(cat "${hwmon}in${ch}_label" 2>/dev/null)
          _cur=$(cat "${hwmon}curr${ch}_input" 2>/dev/null)
          _volt=$(cat "${hwmon}in${ch}_input" 2>/dev/null)
          if [[ -n "$_lbl" && -n "$_cur" && -n "$_volt" ]]; then
            _mw=$(( _cur * _volt / 1000 ))
            printf "  %-20s %smA @ %smV = %smW\n" "$_lbl" "$_cur" "$_volt" "$_mw"
          fi
        done
      done
    else
      timeout 2s tegrastats 2>/dev/null | grep -oE 'VDD_[A-Z_]+ [0-9]+mW/[0-9]+mW' | while read p; do
        echo "  $p"
      done || echo "  (power info not available)"
    fi
    echo

    # --- Docker ---
    echo "🐳 Docker"
    _IPT_VER=$(iptables --version 2>/dev/null)
    echo "  iptables  : $_IPT_VER"
    if echo "$_IPT_VER" | grep -q "nf_tables"; then
      echo "  ⚠️  WARNING: iptables is in nf_tables mode. Docker will fail to start on Jetson!"
      echo "              Run 'sjsujetsontool dockerfix' to resolve this."
    fi
    if docker info &>/dev/null; then
      echo "  Status    : ✅ Running"
      echo "  Version   : $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
      echo "  Runtime   : $(docker info 2>/dev/null | grep Runtimes | sed 's/.*Runtimes: //')"
      printf "  Images    :"
      docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | head -6 | while IFS= read -r img; do
        printf " %s\n              " "$img"
      done
      echo
      _C_STATUS=$(docker ps --format '{{.Names}}: {{.Status}}' 2>/dev/null)
      if [[ -n "$_C_STATUS" ]]; then
        echo "  Running   : $_C_STATUS"
      else
        echo "  Running   : (no running containers)"
      fi
    else
      echo "  Status    : ❌ Docker not running"
      if echo "$_IPT_VER" | grep -q "nf_tables"; then
        echo "  Fix       : Run 'sjsujetsontool dockerfix' to switch to legacy iptables"
      else
        echo "  Fix       : sudo systemctl start docker"
      fi
    fi
    echo

    # --- Key Services / Ports ---
    echo "🔌 Key Services"
    check_service 8888 "JupyterLab"
    check_service 11434 "Ollama"
    check_service 8000 "llama.cpp / API"
    check_service 8001 "FastAPI"
    check_service 7860 "Gradio UI"
    echo

    # --- apt upgradable packages ---
    echo "📦 Apt Upgradable Packages"
    _UPG=$(apt list --upgradable 2>/dev/null | grep -v 'Listing' | wc -l)
    echo "  Available : ${_UPG} package(s) upgradable"
    if [[ "$_UPG" -gt 0 ]]; then
      echo "  Top pkgs  :"
      apt list --upgradable 2>/dev/null | grep -v 'Listing' | head -8 | awk '{print "              "$0}'
      echo "  Run       : sjsujetsontool sysupgrade   (to apply safe upgrades)"
    fi
    echo

    echo "════════════════════════════════════════════════════"
    echo "✅ Health check complete."
    echo "════════════════════════════════════════════════════"
    ;;

  sysupgrade)
    echo "🔄 Checking for apt upgrades..."
    echo "⚠️  Note: This upgrades generic Ubuntu packages only."
    echo "⚠️  DO NOT run 'apt upgrade' blindly on Jetson — it can break JetPack/L4T."
    echo
    # Refresh package index
    sudo apt-get update
    echo
    # Show what would be upgraded
    echo "📋 Packages to be upgraded:"
    apt list --upgradable 2>/dev/null | grep -v Listing
    echo
    read -r -p "❓ Proceed with upgrade? [y/N] " _CONFIRM
    if [[ "$_CONFIRM" =~ ^[Yy]$ ]]; then
      # Exclude Jetson/NVIDIA/L4T packages to avoid breaking BSP
      sudo apt-get upgrade --yes \
        -o 'APT::Get::List-Cleanup=false' \
        $(apt list --upgradable 2>/dev/null | grep -v Listing \
          | grep -v '^nvidia\|^libnv\|^cuda\|^libcuda\|^l4t\|^tensorrt\|^libnvinfer\|^libtensorrt' \
          | awk -F/ '{print $1}' | tr '\n' ' ')
      echo "✅ System packages upgraded."
    else
      echo "⏭️  Upgrade skipped."
    fi
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
  setup-student)
    shift
    # Create or repair a NON-sudo student account that can use /Developer and Docker.
    # Idempotent: if the account exists it is fixed up (display name, password,
    # groups, sudo removed); otherwise it is created.
    STUDENT_USER="${1:-student}"
    DEFAULT_STUDENT_PASS="Sjsujetson2026"
    STUDENT_PASS="${2:-$DEFAULT_STUDENT_PASS}"
    STUDENT_FULLNAME="Student"   # GUI/GDM display name (GECOS) — NOT "sjsujetson"

    echo "══════════════════════════════════════════════════"
    echo "🎓 Student account setup: '$STUDENT_USER' (display name: $STUDENT_FULLNAME)"
    echo "══════════════════════════════════════════════════"
    echo "🔐 This needs admin rights — you may be prompted for the sudo password."
    if ! sudo -v; then
      echo "❌ Could not obtain sudo. Run this as an admin user (e.g. sjsujetson)."
      exit 1
    fi

    if id "$STUDENT_USER" &>/dev/null; then
      echo "✅ User '$STUDENT_USER' exists — repairing settings."
      # Fix the GUI display name (the jetson-01 fix: GECOS was 'SJSUJetson')
      sudo usermod -c "$STUDENT_FULLNAME" "$STUDENT_USER"
    else
      echo "👤 Creating user '$STUDENT_USER'..."
      sudo useradd -m -s /bin/bash -c "$STUDENT_FULLNAME" "$STUDENT_USER"
    fi

    # Default password
    echo "$STUDENT_USER:$STUDENT_PASS" | sudo chpasswd && \
      echo "🔑 Password set to the class default."

    # Groups: containers (docker) + hardware/peripherals for AI+robotics labs.
    # Only add groups that actually exist on this box; never add an admin group.
    WANT_GROUPS="docker video audio render i2c gpio dialout plugdev input weston-launch gdm"
    ADD_GROUPS=""
    for g in $WANT_GROUPS; do
      getent group "$g" >/dev/null 2>&1 && ADD_GROUPS="${ADD_GROUPS:+$ADD_GROUPS,}$g"
    done
    if [ -n "$ADD_GROUPS" ]; then
      sudo usermod -aG "$ADD_GROUPS" "$STUDENT_USER"
      echo "👥 Added to groups: $ADD_GROUPS"
    fi

    # Guarantee NO admin rights (in case the account previously had them)
    for g in sudo adm admin wheel root; do
      sudo gpasswd -d "$STUDENT_USER" "$g" >/dev/null 2>&1 || true
    done
    echo "🚫 Ensured '$STUDENT_USER' is NOT in any sudo/admin group."

    # /Developer access: it's reachable when world-accessible OR group=docker+rwx.
    # If neither holds, grant the docker group access (student is in docker).
    if [ -d /Developer ]; then
      DEV_PERMS="$(stat -c '%A' /Developer)"
      DEV_GRP="$(stat -c '%G' /Developer)"
      if [[ "${DEV_PERMS:7:3}" != *w* ]] && [ "$DEV_GRP" != "docker" ]; then
        echo "📂 Granting docker group access to /Developer..."
        sudo chgrp docker /Developer && sudo chmod 2775 /Developer
      else
        echo "📂 /Developer already accessible ($DEV_PERMS $DEV_GRP)."
      fi
    fi

    # Refresh the login screen's cached display name (reads GECOS)
    sudo systemctl try-restart accounts-daemon >/dev/null 2>&1 || true

    echo "──────────────────────────────────────────────────"
    echo "✅ Student account ready:"
    echo "   • login    : $STUDENT_USER / $STUDENT_PASS"
    echo "   • fullname : $(getent passwd "$STUDENT_USER" | cut -d: -f5 | cut -d, -f1)"
    echo "   • groups   : $(id -nG "$STUDENT_USER")"
    echo "   • sudo     : $(id -nG "$STUDENT_USER" | grep -qw sudo && echo 'YES (unexpected!)' || echo 'no')"
    echo "💡 The GUI display name updates after the next login (or a reboot)."
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
  setup-nvapi)
    echo "══════════════════════════════════════════════════"
    echo "🔑 NVIDIA Build API Key Setup"
    echo "══════════════════════════════════════════════════"
    echo "To get your NVIDIA NGC API Key:"
    echo "  1. Visit https://build.nvidia.com"
    echo "  2. Sign in with a free NVIDIA developer account."
    echo "  3. Open a model card (e.g. Nemotron-3 Nano Omni) and click 'Get API Key'."
    echo "  4. Copy the API key starting with 'nvapi-'."
    echo
    echo "This key will be saved locally to the .env.local file in the Next.js project."
    echo "══════════════════════════════════════════════════"
    echo
    read -r -p "🔑 Paste your NVIDIA API Key (nvapi-...): " NV_KEY
    if [[ ! "$NV_KEY" =~ ^nvapi- ]]; then
      echo "❌ Invalid API Key. It must start with 'nvapi-'."
      exit 1
    fi

    # Find the Next.js app directory to write the key to .env.local
    TARGET_DIR=""
    if [ -d "/Developer/edgeAI/edgeLLM/nextjs-nemotron-app" ]; then
      TARGET_DIR="/Developer/edgeAI/edgeLLM/nextjs-nemotron-app"
    elif [ -d "./edgeLLM/nextjs-nemotron-app" ]; then
      TARGET_DIR="./edgeLLM/nextjs-nemotron-app"
    else
      TARGET_DIR="."
    fi
    ENV_PATH="${TARGET_DIR}/.env.local"
    
    echo "Writing to $ENV_PATH..."
    if [ -f "$ENV_PATH" ] && grep -q "NVIDIA_API_KEY=" "$ENV_PATH"; then
      # Update existing key
      if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i.bak "s|NVIDIA_API_KEY=.*|NVIDIA_API_KEY=${NV_KEY}|" "$ENV_PATH"
      else
        sed -i "s|NVIDIA_API_KEY=.*|NVIDIA_API_KEY=${NV_KEY}|" "$ENV_PATH"
      fi
      echo "✅ Updated existing NVIDIA_API_KEY in $ENV_PATH"
    else
      # Append new key
      echo "NVIDIA_API_KEY=${NV_KEY}" >> "$ENV_PATH"
      echo "✅ Saved NVIDIA_API_KEY to $ENV_PATH"
    fi

    echo
    echo "🧪 Testing connection to NVIDIA Build API using model 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning'..."
    TEST_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
      -H "Authorization: Bearer $NV_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "messages": [{"role": "user", "content": "Hello, write a 5-word greeting."}],
        "max_tokens": 50
      }')

    HTTP_CODE=$(echo "$TEST_RESPONSE" | tail -n 1)
    BODY=$(echo "$TEST_RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" -eq 200 ]; then
      echo "✅ API Test Succeeded (HTTP 200)!"
      # Parse response and thinking process
      python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    msg = data['choices'][0]['message']
    reasoning = msg.get('reasoning_content') or msg.get('reasoning')
    content = msg.get('content')
    if reasoning:
        print('🧠 [Thinking Process]:')
        print(reasoning.strip())
        print()
    if content:
        print('💬 [Final Response]:')
        print(content.strip())
    elif not reasoning:
        print(json.dumps(msg, indent=2))
except Exception as e:
    print('Failed to parse response JSON:', e)
" <<< "$BODY"
    else
      echo "❌ API Test Failed with HTTP status $HTTP_CODE!"
      echo "📜 Error details: $BODY"
      exit 1
    fi
    ;;
  chat)
    shift
    # Unified streaming chat client (Rich UI). The backend menu + per-backend API
    # key handling (saved to ~/.env.local) now live in chat.py, so the in-chat
    # /server command can switch backends mid-session too.
    CHAT_PY="$(dirname "$(realpath "$0")")/sjsujetsontool-chat.py"
    if [ ! -f "$CHAT_PY" ]; then
      echo "⬇️  Fetching chat client (chat.py)..."
      curl -fsSL -H "Cache-Control: no-cache" \
        https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/chat.py -o "$CHAT_PY" 2>/dev/null
    fi
    if [ ! -f "$CHAT_PY" ]; then
      echo "❌ Could not obtain chat client. Check network or run: sjsujetsontool update-script"
      exit 1
    fi
    # Quick shortcut to the local server; otherwise launch the backend chooser.
    if [ "${1:-}" = "--local" ]; then
      shift; exec python3 "$CHAT_PY" --url "http://localhost:8080/v1" "$@"
    elif [ "$#" -gt 0 ]; then
      exec python3 "$CHAT_PY" "$@"           # passthrough (flags / one-shot message)
    else
      exec python3 "$CHAT_PY" --backend      # interactive backend menu
    fi
    ;;
  curl)
    shift
    # Interactive curl builder: asks a few questions, prints the full curl command
    # (so you learn it), then offers to send it. Two modes: LLM API or plain URL.
    echo "══════════════════════════════════════════════════"
    echo "🌐 sjsujetsontool curl — interactive request builder"
    echo "══════════════════════════════════════════════════"
    echo "What do you want to call?"
    echo "  1) LLM API   (OpenAI-compatible chat — e.g. your 'sjsujetsontool llama' server)"
    echo "  2) Plain URL (any GET/POST)"
    read -r -p "Select [1-2] (Enter = 1): " _CMODE

    if [ "$_CMODE" = "2" ]; then
      # ---------- plain URL ----------
      read -r -p "URL: " _URL
      [ -z "$_URL" ] && { echo "❌ No URL given."; exit 1; }
      read -r -p "Method [GET/POST] (Enter = GET): " _M; _M="$(echo "${_M:-GET}" | tr a-z A-Z)"
      _ARGS=(curl -sS -X "$_M")
      read -r -p "Add a header? e.g. 'Accept: application/json' (blank to skip): " _H1
      [ -n "$_H1" ] && _ARGS+=(-H "$_H1")
      if [ "$_M" = "POST" ]; then
        read -r -p "Content-Type [application/json]: " _CT; _CT="${_CT:-application/json}"
        _ARGS+=(-H "Content-Type: $_CT")
        read -r -p "Request body (sent with -d), blank for none: " _DATA
        [ -n "$_DATA" ] && _ARGS+=(-d "$_DATA")
      fi
      _ARGS+=("$_URL")
      echo
      echo "📋 Generated command:"
      printf '   '; printf '%q ' "${_ARGS[@]}"; echo; echo
      echo "ℹ️  curl basics:  -X method · -H \"Header: value\" · -d 'body' (implies POST) · -s quiet · -L follow redirects"
      read -r -p "Send now? [Y/n]: " _GO
      [[ "$_GO" =~ ^[Nn]$ ]] && { echo "Not sent. (Copy the command above to run it later.)"; exit 0; }
      echo; "${_ARGS[@]}"; echo
    else
      # ---------- LLM API (OpenAI-compatible /v1/chat/completions) ----------
      read -r -p "Host/IP [localhost]: " _H; _H="${_H:-localhost}"
      read -r -p "Port [8080]: " _P; _P="${_P:-8080}"
      read -r -p "API key (optional, Enter to skip): " _K
      read -r -p "Your message [Explain Nvidia Jetson in 2 sentences.]: " _MSG
      _MSG="${_MSG:-Explain Nvidia Jetson in 2 sentences.}"
      read -r -p "Attach an image for a vision model? (path, blank = none): " _IMG
      if [ -n "$_IMG" ] && [ ! -f "$_IMG" ]; then echo "⚠️  Image not found: $_IMG (continuing text-only)"; _IMG=""; fi
      read -r -p "max_tokens [256]: " _MT; _MT="${_MT:-256}"
      read -r -p "Stream the output token-by-token? [y/N]: " _S
      read -r -p "Enable 'thinking' (slower, more reasoning)? [y/N]: " _T
      [[ "$_S" =~ ^[Yy]$ ]] && _STREAM=1 || _STREAM=0
      [[ "$_T" =~ ^[Yy]$ ]] && _THINK=1 || _THINK=0
      _URL="http://$_H:$_P/v1/chat/completions"
      # Build the JSON body safely. With an image, content becomes the OpenAI
      # multimodal array [text + image_url(base64 data URI)] for vision models.
      _BODY=$(MSG="$_MSG" IMG="$_IMG" MT="$_MT" STREAM="$_STREAM" THINK="$_THINK" python3 - <<'PY'
import json, os, base64, mimetypes
msg = os.environ["MSG"]; img = os.environ.get("IMG", "")
if img:
    data = open(img, "rb").read()
    mime = mimetypes.guess_type(img)[0] or "image/jpeg"
    uri = "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())
    content = [{"type": "text", "text": msg},
               {"type": "image_url", "image_url": {"url": uri}}]
else:
    content = msg
b = {"messages": [{"role": "user", "content": content}],
     "max_tokens": int(os.environ.get("MT") or 256),
     "stream": os.environ["STREAM"] == "1"}
if os.environ["THINK"] != "1":              # default: thinking OFF -> short & fast on Jetson
    b["chat_template_kwargs"] = {"enable_thinking": False}
print(json.dumps(b))
PY
)
      _ARGS=(curl)
      [ "$_STREAM" = "1" ] && _ARGS+=(-N) || _ARGS+=(-s)
      _ARGS+=("$_URL" -H "Content-Type: application/json")
      [ -n "$_K" ] && _ARGS+=(-H "Authorization: Bearer $_K")
      # Send the body from a temp file (-d @file). A base64 image is far larger than
      # the 128 KB per-argument limit, so it cannot go on the command line directly.
      _TMP="$(mktemp)"; printf '%s' "$_BODY" > "$_TMP"
      _ARGS+=(--data-binary "@$_TMP")
      echo
      echo "📋 Generated curl command:"
      echo "   curl $([ "$_STREAM" = 1 ] && echo '-N' || echo '-s') \"$_URL\" \\"
      echo "     -H \"Content-Type: application/json\" \\"
      [ -n "$_K" ] && echo "     -H \"Authorization: Bearer ****\" \\"
      if [ -n "$_IMG" ]; then
        echo "     -d '{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"$_MSG\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:<base64 of $_IMG>\"}}]}],\"max_tokens\":$_MT,...}'"
        echo "     (image base64 omitted above; the real request sends the body via a temp file)"
      else
        echo "     -d '$_BODY'"
      fi
      echo
      read -r -p "Send now? [Y/n]: " _GO
      [[ "$_GO" =~ ^[Nn]$ ]] && { rm -f "$_TMP"; echo "Not sent. (Tip: start a server first with 'sjsujetsontool llama'.)"; exit 0; }
      echo; "${_ARGS[@]}"; echo; rm -f "$_TMP"
    fi
    ;;
  nv-chat)
    shift
    # Find .env.local
    TARGET_DIR=""
    if [ -d "/Developer/edgeAI/edgeLLM/nextjs-nemotron-app" ]; then
      TARGET_DIR="/Developer/edgeAI/edgeLLM/nextjs-nemotron-app"
    elif [ -d "./edgeLLM/nextjs-nemotron-app" ]; then
      TARGET_DIR="./edgeLLM/nextjs-nemotron-app"
    else
      TARGET_DIR="."
    fi
    ENV_PATH="${TARGET_DIR}/.env.local"
    
    # Read NVIDIA_API_KEY from environment or .env.local
    NV_KEY="$NVIDIA_API_KEY"
    if [ -z "$NV_KEY" ] && [ -f "$ENV_PATH" ]; then
      NV_KEY=$(grep -E "^NVIDIA_API_KEY=" "$ENV_PATH" | cut -d= -f2- | tr -d '"' | tr -d "'")
    fi
    
    if [ -z "$NV_KEY" ]; then
      echo "❌ NVIDIA API Key is not set. Please run: sjsujetsontool setup-nvapi"
      exit 1
    fi

    # Parse arguments for model or prompt
    MODEL="nvidia/llama-3.1-nemotron-nano-8b-v1"
    PROMPT=""
    while [[ "$#" -gt 0 ]]; do
      case $1 in
        --model|-m)
          shift
          MODEL="$1"
          ;;
        -p|--prompt)
          shift
          PROMPT="$1"
          ;;
        *)
          # Any other argument is part of the prompt
          PROMPT="$*"
          break
          ;;
      esac
      shift
    done

    # If no prompt, start interactive chat session or show selection menu
    if [ -z "$PROMPT" ]; then
      echo "🤖 Select NVIDIA Build Model:"
      echo "  1) Llama 3.1 Nemotron Nano (8B) [Default]"
      echo "  2) Llama 3.3 Nemotron Super (49B)"
      echo "  3) Llama 3.1 Nemotron Ultra (253B)"
      echo "  4) Nemotron 3 Nano Omni (30B reasoning)"
      echo "  5) Nemotron 3 Ultra (550B reasoning)"
      echo "  6) Nemotron 3 Super (120B reasoning)"
      read -r -p "Select [1-6]: " MODEL_SEL
      case "$MODEL_SEL" in
        2) MODEL="nvidia/llama-3.3-nemotron-super-49b-v1" ;;
        3) MODEL="nvidia/llama-3.1-nemotron-ultra-253b-v1" ;;
        4) MODEL="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" ;;
        5) MODEL="nvidia/nemotron-3-ultra-550b-a55b" ;;
        6) MODEL="nvidia/nemotron-3-super-120b-a12b" ;;
        *) MODEL="nvidia/llama-3.1-nemotron-nano-8b-v1" ;;
      esac
      echo
      echo "💬 Starting interactive chat with $MODEL..."
      echo "   Type 'exit' to quit."
      echo "══════════════════════════════════════════════════"
      while true; do
        read -r -p "User > " USER_INPUT
        if [ "$USER_INPUT" = "exit" ] || [ "$USER_INPUT" = "quit" ]; then
          break
        fi
        if [ -n "$USER_INPUT" ]; then
          # Call Python script for single streaming query
          python3 - "$MODEL" "$USER_INPUT" "$NV_KEY" << 'EOF'
import sys
import time
import json
import urllib.request

model = sys.argv[1]
prompt = sys.argv[2]
api_key = sys.argv[3]

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}

payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "stream": True
}

req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

start_time = time.time()
first_token_time = None
total_tokens = 0
reasoning_tokens = 0
content_tokens = 0

try:
    with urllib.request.urlopen(req, timeout=15) as response:
        while True:
            line_bytes = response.readline()
            if not line_bytes:
                break
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choice = data['choices'][0]
                    delta = choice.get('delta', {})
                    
                    reasoning = delta.get('reasoning_content') or delta.get('reasoning')
                    if reasoning:
                        if first_token_time is None:
                            first_token_time = time.time()
                            print("🧠 [Thinking Process]: ", end="", flush=True)
                        print(reasoning, end="", flush=True)
                        reasoning_tokens += 1
                        total_tokens += 1
                        
                    content = delta.get('content')
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        if reasoning_tokens > 0 and content_tokens == 0:
                            print("\n\n💬 [Response]: ", end="", flush=True)
                        elif content_tokens == 0 and reasoning_tokens == 0:
                            print("💬 [Response]: ", end="", flush=True)
                        print(content, end="", flush=True)
                        content_tokens += 1
                        total_tokens += 1
                except Exception as e:
                    pass
        
        end_time = time.time()
        print()
        
        total_time = end_time - start_time
        if first_token_time:
            time_to_first = first_token_time - start_time
            gen_time = end_time - first_token_time
            gen_speed = total_tokens / gen_time if gen_time > 0 else 0
            print(f"\n⚡ [Performance]: Time-to-first-token: {time_to_first:.2f}s | Generation: {gen_speed:.1f} tokens/sec ({total_tokens} tokens generated in {gen_time:.2f}s)")
        else:
            print(f"\n⚡ [Performance]: Total time: {total_time:.2f}s")
            
except Exception as e:
    print(f"\n❌ Error during API call: {e}")
EOF
          echo
        fi
      done
    else
      # Single prompt run
      python3 - "$MODEL" "$PROMPT" "$NV_KEY" << 'EOF'
import sys
import time
import json
import urllib.request

model = sys.argv[1]
prompt = sys.argv[2]
api_key = sys.argv[3]

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}

payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "stream": True
}

req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

start_time = time.time()
first_token_time = None
total_tokens = 0
reasoning_tokens = 0
content_tokens = 0

try:
    with urllib.request.urlopen(req, timeout=15) as response:
        while True:
            line_bytes = response.readline()
            if not line_bytes:
                break
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choice = data['choices'][0]
                    delta = choice.get('delta', {})
                    
                    reasoning = delta.get('reasoning_content') or delta.get('reasoning')
                    if reasoning:
                        if first_token_time is None:
                            first_token_time = time.time()
                            print("🧠 [Thinking Process]: ", end="", flush=True)
                        print(reasoning, end="", flush=True)
                        reasoning_tokens += 1
                        total_tokens += 1
                        
                    content = delta.get('content')
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        if reasoning_tokens > 0 and content_tokens == 0:
                            print("\n\n💬 [Response]: ", end="", flush=True)
                        elif content_tokens == 0 and reasoning_tokens == 0:
                            print("💬 [Response]: ", end="", flush=True)
                        print(content, end="", flush=True)
                        content_tokens += 1
                        total_tokens += 1
                except Exception as e:
                    pass
        
        end_time = time.time()
        print()
        
        total_time = end_time - start_time
        if first_token_time:
            time_to_first = first_token_time - start_time
            gen_time = end_time - first_token_time
            gen_speed = total_tokens / gen_time if gen_time > 0 else 0
            print(f"\n⚡ [Performance]: Time-to-first-token: {time_to_first:.2f}s | Generation: {gen_speed:.1f} tokens/sec ({total_tokens} tokens generated in {gen_time:.2f}s)")
        else:
            print(f"\n⚡ [Performance]: Total time: {total_time:.2f}s")
            
except Exception as e:
    print(f"\n❌ Error during API call: {e}")
EOF
      echo
    fi
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
  dockerfix)
    dockerfix
    ;;
  meshvpn)
    meshvpn
    ;;
  tailscale)
    shift
    SUBCMD="$1"
    shift
    FORCE=false
    [[ "$1" == "--force" ]] && FORCE=true

    _TS_INSTALL_URL="https://tailscale.com/install.sh"

    # Helper: install tailscale if missing
    _ts_ensure_installed() {
      if command -v tailscale &>/dev/null; then
        echo "✅ Tailscale already installed: $(tailscale version | head -1)"
        return 0
      fi
      echo "📥 Tailscale not found. Installing..."
      if curl -fsSL "$_TS_INSTALL_URL" | sudo sh; then
        echo "✅ Tailscale installed: $(tailscale version | head -1)"
      else
        echo "❌ Tailscale installation failed. Check network and try again."
        return 1
      fi
    }

    # Helper: check headscale for hostname conflict via API
    _ts_check_hostname_conflict() {
      local hn="$1"
      # Query headscale machines API (unauthenticated listing, best-effort)
      local api_resp
      api_resp=$(curl -sf --max-time 8 \
        "${HEADSCALE_LOGIN_SERVER}/api/v1/machine" \
        -H "Authorization: Bearer ${HEADSCALE_AUTHKEY}" 2>/dev/null)
      if [[ -z "$api_resp" ]]; then
        echo "  ℹ️  Could not reach headscale API — skipping hostname conflict check."
        return 0
      fi
      # Check if hostname appears in the response
      if echo "$api_resp" | grep -qi "\"$hn\""; then
        echo "  ⚠️  Hostname conflict detected: '$hn' is already registered on the headscale server."
        echo "  💡 To avoid conflicts, rename this device first:"
        echo "       sjsujetsontool set-hostname <new-unique-name>"
        echo "     Or force re-registration with:"
        echo "       sjsujetsontool tailscale up --force"
        return 1
      else
        echo "  ✅ No hostname conflict: '$hn' is available on the headscale server."
        return 0
      fi
    }

    case "$SUBCMD" in
      install)
        _ts_ensure_installed
        ;;

      up)
        echo "══════════════════════════════════════════════════"
        echo "🌐 Joining Headscale Network"
        echo "══════════════════════════════════════════════════"
        _ts_ensure_installed || exit 1

        # --- Check if tailscaled is running ---
        if ! systemctl is-active --quiet tailscaled 2>/dev/null; then
          echo "🔄 Starting tailscaled service..."
          sudo systemctl enable --now tailscaled
        fi

        CURRENT_HN=$(hostname)
        echo "🖥️  This device hostname : $CURRENT_HN"
        echo "🌐 Headscale server     : $HEADSCALE_LOGIN_SERVER"
        echo

        # --- Check if already connected to any Tailscale/Headscale network ---
        BACKEND_STATE=$(tailscale status --json 2>/dev/null | python3 -c \
          "import sys,json; d=json.load(sys.stdin); print(d.get('BackendState',''))" 2>/dev/null)
        CURRENT_DNS=$(tailscale status --json 2>/dev/null | python3 -c \
          "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName',''))" 2>/dev/null)
        CURRENT_IPS=$(tailscale status --json 2>/dev/null | python3 -c \
          "import sys,json; d=json.load(sys.stdin); print(' '.join(d.get('TailscaleIPs',[])))" 2>/dev/null)

        if [[ "$BACKEND_STATE" == "Running" && "$FORCE" != "true" ]]; then
          echo "⚠️  This device is ALREADY connected to a Tailscale/Headscale network!"
          echo "   Backend State : $BACKEND_STATE"
          echo "   Tailscale IPs : $CURRENT_IPS"
          echo "   DNS Name      : $CURRENT_DNS"
          echo
          # Check if already on headscale server
          if echo "$CURRENT_DNS" | grep -qi "$(echo $HEADSCALE_LOGIN_SERVER | sed 's|https\?://||')"; then
            echo "✅ Already connected to the headscale server at $HEADSCALE_LOGIN_SERVER."
            echo "   Run 'sjsujetsontool tailscale status' for details."
            exit 0
          fi
          echo "   ⚠️  Connected to a DIFFERENT network (not this headscale server)."
          echo
          read -r -p "❓ Disconnect and re-join headscale? [y/N] " _TSCONFIRM
          if [[ ! "$_TSCONFIRM" =~ ^[Yy]$ ]]; then
            echo "⏭️  Aborted. No changes made."
            exit 0
          fi
          echo "🔌 Disconnecting from current network..."
          sudo tailscale down
        fi

        if [[ "$FORCE" == "true" && "$BACKEND_STATE" == "Running" ]]; then
          echo "⚡ --force specified. Disconnecting from current network..."
          sudo tailscale down
        fi

        # --- Hostname conflict check against headscale ---
        echo "🔍 Checking for hostname conflicts on headscale..."
        if ! _ts_check_hostname_conflict "$CURRENT_HN"; then
          if [[ "$FORCE" != "true" ]]; then
            exit 1
          fi
          echo "⚡ --force specified. Proceeding despite hostname conflict."
        fi
        echo

        # --- Join the headscale network ---
        echo "🚀 Joining headscale network..."
        if sudo tailscale up \
          --login-server "$HEADSCALE_LOGIN_SERVER" \
          --authkey "$HEADSCALE_AUTHKEY" \
          --hostname "$CURRENT_HN" \
          --accept-routes 2>&1; then
          echo
          echo "══════════════════════════════════════════════════"
          echo "✅ Successfully joined headscale network!"
          # Show resulting status
          NEW_IPS=$(tailscale status --json 2>/dev/null | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(' '.join(d.get('TailscaleIPs',[])))" 2>/dev/null)
          NEW_STATE=$(tailscale status --json 2>/dev/null | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('BackendState',''))" 2>/dev/null)
          echo "   Hostname      : $CURRENT_HN"
          echo "   Tailscale IPs : ${NEW_IPS:-'(pending — check again in a moment)'}"
          echo "   Backend State : $NEW_STATE"
          echo "   Server        : $HEADSCALE_LOGIN_SERVER"
          echo "══════════════════════════════════════════════════"
        else
          echo
          echo "══════════════════════════════════════════════════"
          echo "❌ Failed to join headscale network."
          echo "   Possible reasons:"
          echo "     • Invalid or expired authkey"
          echo "     • Headscale server unreachable: $HEADSCALE_LOGIN_SERVER"
          echo "     • Hostname already registered (try --force or rename)"
          echo "     • Network / firewall blocking UDP 41641"
          echo "   Run: journalctl -u tailscaled -n 30 --no-pager"
          echo "══════════════════════════════════════════════════"
          exit 1
        fi
        ;;

      status)
        echo "══════════════════════════════════════════════════"
        echo "🌐 Tailscale Status"
        echo "══════════════════════════════════════════════════"
        if ! command -v tailscale &>/dev/null; then
          echo "❌ Tailscale is not installed."
          echo "   Run: sjsujetsontool tailscale install"
          exit 1
        fi
        echo "📦 Version : $(tailscale version | head -1)"
        echo "🔧 Daemon  : $(systemctl is-active tailscaled 2>/dev/null || echo 'unknown')"
        echo
        # Parse JSON status
        _TS_JSON=$(tailscale status --json 2>/dev/null)
        if [[ -z "$_TS_JSON" ]]; then
          echo "❌ Could not get Tailscale status (daemon not running?)."
          exit 1
        fi
        _TS_STATE=$(echo "$_TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('BackendState',''))" 2>/dev/null)
        _TS_IPS=$(echo "$_TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(d.get('TailscaleIPs',[])))" 2>/dev/null)
        _TS_HN=$(echo "$_TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('HostName',''))" 2>/dev/null)
        _TS_DNS=$(echo "$_TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName',''))" 2>/dev/null)
        _TS_PEERS=$(echo "$_TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('Peer',{})))" 2>/dev/null)
        if [[ "$_TS_STATE" == "Running" ]]; then
          echo "✅ State   : $_TS_STATE"
        elif [[ "$_TS_STATE" == "NeedsLogin" ]]; then
          echo "🔐 State   : $_TS_STATE — run 'sjsujetsontool tailscale up'"
        else
          echo "❌ State   : $_TS_STATE"
        fi
        echo "   Hostname  : $_TS_HN"
        echo "   IPs       : $_TS_IPS"
        echo "   DNS Name  : $_TS_DNS"
        echo "   Peers     : $_TS_PEERS connected"
        echo
        # Show health warnings
        _TS_HEALTH=$(echo "$_TS_JSON" | python3 -c \
          "import sys,json; d=json.load(sys.stdin); [print('  ⚠️ ',w) for w in d.get('Health',[])]" 2>/dev/null)
        if [[ -n "$_TS_HEALTH" ]]; then
          echo "⚠️  Health Warnings:"
          echo "$_TS_HEALTH"
          echo
        fi
        echo "══════════════════════════════════════════════════"
        ;;

      down)
        echo "🔌 Disconnecting from Tailscale network..."
        if ! command -v tailscale &>/dev/null; then
          echo "❌ Tailscale is not installed."
          exit 1
        fi
        if sudo tailscale down 2>&1; then
          echo "✅ Tailscale disconnected successfully."
        else
          echo "❌ Failed to disconnect. Is tailscaled running?"
          echo "   Run: systemctl status tailscaled"
          exit 1
        fi
        ;;

      *)
        echo "❓ Usage: sjsujetsontool tailscale <subcommand>"
        echo "   install          - Install Tailscale if not already installed"
        echo "   up [--force]     - Join headscale network (checks hostname conflicts)"
        echo "   status           - Show Tailscale connection status"
        echo "   down             - Disconnect from Tailscale network"
        ;;
    esac
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
    docker push $REMOTE_TAGGED || {
      echo "🔑 Docker push failed. Attempting login..."
      docker login || exit 1
      docker push $REMOTE_TAGGED
    }
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
    echo "📤 Pushing to Docker Hub..."
    docker push "$REMOTE_TAGGED" || {
      echo "🔑 Docker push failed. Attempting login..."
      docker login || exit 1
      docker push "$REMOTE_TAGGED"
    }
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

    # Avoid "dubious ownership" when the repo is owned by another user (shared box)
    git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$REPO_PATH" \
      || git config --global --add safe.directory "$REPO_PATH"

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
  setup-check|setup_check)
    setup_check_internal
    ;;
  help)
    show_help
    ;;
esac

#help|*)