#!/bin/bash

# =============================================================================
# === sjsujetsontool v2 ===
# Custom dev CLI for NVIDIA Jetson Orin Nano & Jetson Thor
# Supports JetPack 6 & JetPack 7 switchable container images
# =============================================================================
SCRIPT_VERSION="v2.0.0"

# ❗ Warn if run incorrectly via `bash sjsujetsontoolv2.sh version`
if [[ "$0" == "bash" && "$1" == "${BASH_SOURCE[0]}" ]]; then
  echo "⚠️  Please run this script directly, not via 'bash'."
  echo "✅ Correct: ./sjsujetsontoolv2.sh version"
  echo "❌ Wrong: bash sjsujetsontoolv2.sh version"
  exit 1
fi

# 📟 Detect Jetson hardware model (Thor, Orin Nano, Xavier, etc.)
JETSON_MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
if [[ -z "$JETSON_MODEL" ]]; then
  JETSON_MODEL="Generic ARM64 Tegra"
fi
echo "🧠 Detected Jetson Model: $JETSON_MODEL"

# 🧮 Total system RAM in MB (Unified Memory)
TOTAL_RAM_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 8000)

# 🧰 Detect JetPack and CUDA version
JETPACK_VERSION=$(dpkg-query --show nvidia-jetpack 2>/dev/null | awk '{print $2}')
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
    37|38|39) JETPACK_VERSION="7.x (inferred)" ;;
  esac
fi

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

# =============================================================================
# Container Switcher Configuration (JP6 vs JP7)
# =============================================================================
CONFIG_FILE="$HOME/.sjsujetsontool_config"
DOCKERHUB_USER="cmpelkk"

# Images
JP6_IMAGE_NAME="cmpelkk/jetson-llm:v1"
JP7_IMAGE_NAME="cmpelkk/jetson-unified:jp7-thor"

# Determine default container based on hardware/L4T
_L4T_NUM=$(echo "$L4T_REVISION" | grep -oE '[0-9]+' | head -1)
if [[ "$JETSON_MODEL" == *"Thor"* ]] || [[ "$_L4T_NUM" -ge 38 ]]; then
  DEFAULT_CONTAINER="jp7"
else
  DEFAULT_CONTAINER="jp6"
fi

# Load user configuration if present
SELECTED_CONTAINER=""
if [ -f "$CONFIG_FILE" ]; then
  SELECTED_CONTAINER=$(grep -E '^CONTAINER=' "$CONFIG_FILE" 2>/dev/null | cut -d= -f2)
fi

# Parse CLI flags override (--jp6, --jp7)
CLI_CONTAINER_OVERRIDE=""
for arg in "$@"; do
  case "$arg" in
    --jp6|--container=jp6) CLI_CONTAINER_OVERRIDE="jp6" ;;
    --jp7|--container=jp7) CLI_CONTAINER_OVERRIDE="jp7" ;;
  esac
done

if [ -n "$CLI_CONTAINER_OVERRIDE" ]; then
  ACTIVE_CONTAINER="$CLI_CONTAINER_OVERRIDE"
elif [ -n "$SELECTED_CONTAINER" ]; then
  ACTIVE_CONTAINER="$SELECTED_CONTAINER"
else
  ACTIVE_CONTAINER="$DEFAULT_CONTAINER"
fi

if [ "$ACTIVE_CONTAINER" = "jp7" ]; then
  IMAGE_NAME="cmpelkk/jetson-unified"
  IMAGE_TAG="jp7-thor"
  LOCAL_IMAGE="$JP7_IMAGE_NAME"
  REMOTE_IMAGE="$JP7_IMAGE_NAME"
  CONTAINER_NAME="jetson-dev-jp7"
  echo "🚀 Active Container: JetPack 7 ($LOCAL_IMAGE)"
else
  IMAGE_NAME="jetson-llm"
  IMAGE_TAG="v1"
  LOCAL_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:$IMAGE_TAG"
  REMOTE_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:latest"
  CONTAINER_NAME="jetson-dev"
  echo "🚀 Active Container: JetPack 6 ($LOCAL_IMAGE)"
fi

# Paths
WORKSPACE_DIR="$(realpath .)"
DEV_DIR="/Developer"
MODELS_DIR="/Developer/models"

xhost +local:docker >/dev/null 2>&1 || echo "Warning: xhost command failed. X11 forwarding may not work."

EXTRA_BINDS="-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev"
VOLUME_FLAGS="-v $WORKSPACE_DIR:/workspace -v $MODELS_DIR:/models -v $DEV_DIR:/Developer"
DEVICE_FLAGS="--device-cgroup-rule='c 81:* rmw' --device-cgroup-rule='c 14:* rmw'"

if [ -t 0 ]; then
  TTY_FLAGS="-it"
else
  TTY_FLAGS=""
fi

CREATE_CMD="docker create $TTY_FLAGS --runtime=nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  $DEVICE_FLAGS \
  -e DISPLAY=$DISPLAY \
  --name $CONTAINER_NAME $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"

ENVFILE_ARG=""
[ -f "$HOME/.env.local" ] && ENVFILE_ARG="--env-file $HOME/.env.local"
EXEC_CMD="docker exec $TTY_FLAGS $ENVFILE_ARG $CONTAINER_NAME"

CONTAINER_CMD="docker run --rm $TTY_FLAGS --runtime=nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined \
  $DEVICE_FLAGS \
  -e DISPLAY=$DISPLAY \
  $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"

# Helper function to switch container preference
set_container_pref() {
  local target="$1"
  if [[ "$target" != "jp6" && "$target" != "jp7" ]]; then
    echo "❌ Invalid container choice: '$target'. Must be 'jp6' or 'jp7'."
    return 1
  fi
  echo "CONTAINER=$target" > "$CONFIG_FILE"
  echo "✅ Preference saved to $CONFIG_FILE: Active container set to $target"
}

validate_custom_hf_model() {
  local model_id="$1"
  local hf_token="$2"

  echo "🔎 Validating custom model '$model_id' on Hugging Face..."

  local auth_header=""
  [ -n "$hf_token" ] && auth_header="-H Authorization: Bearer $hf_token"

  local tmp_api_json="$(mktemp)"
  local http_code=$(curl -s -w "%{http_code}" -o "$tmp_api_json" $auth_header "https://huggingface.co/api/models/$model_id")

  if [ "$http_code" -eq 404 ]; then
    echo "❌ ERROR: Model '$model_id' was NOT found on Hugging Face (404)."
    echo "👉 Please verify the exact repo ID (e.g. Qwen/Qwen2.5-14B-Instruct, meta-llama/Llama-3.1-8B-Instruct)."
    rm -f "$tmp_api_json"
    return 1
  elif [ "$http_code" -eq 401 ]; then
    echo "🔑 ERROR: Model '$model_id' is gated or private (401 Unauthorized)."
    echo "👉 Ensure you accepted model access terms at https://huggingface.co/$model_id"
    echo "   and provide a valid HF_TOKEN."
    rm -f "$tmp_api_json"
    return 1
  fi

  local is_gated=$(jq -r '.gated // false' "$tmp_api_json" 2>/dev/null)
  if [ "$is_gated" = "true" ] && [ -z "$hf_token" ]; then
    echo "🔑 WARNING: '$model_id' is a GATED model. A valid HF_TOKEN will be required to download weights."
  fi

  # Check safetensors weight size (bytes)
  local total_bytes=$(jq -r '.safetensors.total // 0' "$tmp_api_json" 2>/dev/null)
  if [ "$total_bytes" -gt 0 ]; then
    local weights_gb=$(( total_bytes / 1073741824 ))
    echo "📊 Model Weight Size: ~${weights_gb} GB"

    local req_vram_gb=$(( weights_gb + 2 ))
    local total_ram_gb=$(( TOTAL_RAM_MB / 1024 ))

    echo "🧠 System Available Memory: ${total_ram_gb} GB | Estimated VRAM Required: ~${req_vram_gb} GB"

    if [ "$req_vram_gb" -gt "$total_ram_gb" ]; then
      echo "══════════════════════════════════════════════════"
      echo "⚠️  MEMORY WARNING: Device RAM (${total_ram_gb} GB) is smaller than estimated VRAM requirement (~${req_vram_gb} GB)."
      echo "   Launching '$model_id' may result in Out-Of-Memory (OOM) or KV cache allocation failure."
      echo "══════════════════════════════════════════════════"
      read -p "Do you still want to attempt launching '$model_id'? [y/N]: " confirm_oom
      if [[ ! "$confirm_oom" =~ ^[Yy]$ ]]; then
        echo "🛑 Launch canceled by user."
        rm -f "$tmp_api_json"
        return 1
      fi
    fi
  fi

  local tmp_cfg_json="$(mktemp)"
  curl -sL $auth_header "https://huggingface.co/$model_id/raw/main/config.json" -o "$tmp_cfg_json" 2>/dev/null
  local arch=$(jq -r '.architectures[0] // .model_type // "unknown"' "$tmp_cfg_json" 2>/dev/null)
  rm -f "$tmp_cfg_json" "$tmp_api_json"

  echo "✅ Model '$model_id' verified! Architecture: $arch"
  return 0
}

ensure_riva_scripts_present() {
  local target_dir="$HOME/.local/share/sjsujetsontool/riva"
  mkdir -p "$target_dir"

  if [ -f "/Developer/edgeAI/jetson/riva/setup_riva.sh" ]; then
    RIVA_SETUP_SCRIPT="/Developer/edgeAI/jetson/riva/setup_riva.sh"
    RIVA_START_SCRIPT="/Developer/edgeAI/jetson/riva/start_riva.sh"
    RIVA_STOP_SCRIPT="/Developer/edgeAI/jetson/riva/stop_riva.sh"
    return 0
  elif [ -f "./jetson/riva/setup_riva.sh" ]; then
    RIVA_SETUP_SCRIPT="$(realpath ./jetson/riva/setup_riva.sh)"
    RIVA_START_SCRIPT="$(realpath ./jetson/riva/start_riva.sh)"
    RIVA_STOP_SCRIPT="$(realpath ./jetson/riva/stop_riva.sh)"
    return 0
  elif [ -f "$target_dir/setup_riva.sh" ]; then
    RIVA_SETUP_SCRIPT="$target_dir/setup_riva.sh"
    RIVA_START_SCRIPT="$target_dir/start_riva.sh"
    RIVA_STOP_SCRIPT="$target_dir/stop_riva.sh"
    return 0
  fi

  echo "📥 Auto-downloading Riva setup and launch scripts from GitHub..."
  local base_url="https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/riva"
  
  if curl -fsSL "$base_url/setup_riva.sh" -o "$target_dir/setup_riva.sh" && \
     curl -fsSL "$base_url/start_riva.sh" -o "$target_dir/start_riva.sh" && \
     curl -fsSL "$base_url/stop_riva.sh" -o "$target_dir/stop_riva.sh"; then
    chmod +x "$target_dir"/*.sh
    RIVA_SETUP_SCRIPT="$target_dir/setup_riva.sh"
    RIVA_START_SCRIPT="$target_dir/start_riva.sh"
    RIVA_STOP_SCRIPT="$target_dir/stop_riva.sh"
    echo "✅ Auto-downloaded Riva scripts to $target_dir"
    return 0
  else
    echo "❌ Failed to download Riva setup scripts from GitHub."
    return 1
  fi
}

ensure_container_started() {
  if [ ! -d "/Developer/edgeAI/.git" ] || [ ! -w "/Developer" ]; then
    echo "⚠️  /Developer or edgeAI repository is not correctly configured. Running setup-check..."
    setup_check_internal
  fi

  if ! docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "🛠️  Creating persistent container '$CONTAINER_NAME' ($LOCAL_IMAGE)..."
    
    if ! docker image inspect $LOCAL_IMAGE > /dev/null 2>&1; then
      echo "📥 Image not found locally. Pulling $REMOTE_IMAGE..."
      docker pull $REMOTE_IMAGE || {
        echo "❌ Failed to pull $REMOTE_IMAGE."
        exit 1
      }
      docker tag $REMOTE_IMAGE $LOCAL_IMAGE 2>/dev/null || true
    fi
    
    echo "🔧 Creating container..."
    eval "$CREATE_CMD"
  fi
  docker start $CONTAINER_NAME >/dev/null
}

setup_check_internal() {
  echo "══════════════════════════════════════════════════"
  echo "⚙️  Checking /Developer folder and edgeAI git repository..."
  echo "══════════════════════════════════════════════════"

  if [ ! -d "/Developer" ]; then
    echo "📂 Directory '/Developer' does not exist. Creating it..."
    sudo mkdir -p /Developer && sudo chmod 777 /Developer
  fi
  if [ ! -d "/Developer/models" ]; then
    mkdir -p /Developer/models && chmod 777 /Developer/models
  fi

  REPO_DIR="/Developer/edgeAI"
  if command -v git &>/dev/null; then
    git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$REPO_DIR" \
      || git config --global --add safe.directory "$REPO_DIR"
  fi
  if [ ! -d "$REPO_DIR" ]; then
    echo "📥 Cloning edgeAI repository..."
    git clone https://github.com/lkk688/edgeAI.git "$REPO_DIR"
  fi
}

show_help() {
  echo "Usage: sjsujetsontoolv2 [command] [options]"
  echo "Script Version: $SCRIPT_VERSION"
  echo
  echo "Container Selection Options:"
  echo "  --jp6                     - Force JetPack 6 container (cmpelkk/jetson-llm:v1)"
  echo "  --jp7                     - Force JetPack 7 container (cmpelkk/jetson-unified:jp7-thor)"
  echo "  container status          - Show current active container selection"
  echo "  container set-jp6         - Set persistent preference to JP6 container"
  echo "  container set-jp7         - Set persistent preference to JP7 container"
  echo
  echo "vLLM Commands:"
  echo "  vllm start                - Interactive menu to start vLLM (Cosmos-Reason2 or Qwen3-8B)"
  echo "  vllm start --cosmos       - Start vLLM serving Cosmos-Reason2-8B (Port 8010)"
  echo "  vllm start --qwen         - Start vLLM serving Qwen3-8B / Qwen2.5-Coder (Port 8010)"
  echo "  vllm stop                 - Stop running vLLM containers"
  echo "  vllm test                 - Execute curl API tests against vLLM server"
  echo
  echo "Audio / Speech Commands:"
  echo "  audio start               - Interactive menu to start Audio Server (NVIDIA Riva or Whisper)"
  echo "  audio start --riva        - Start NVIDIA Riva Speech Server (Port 50051)"
  echo "  audio stop                - Stop running Audio / Speech servers"
  echo "  audio test                - Test NVIDIA Riva connection and host audio hardware"
  echo "  audio status              - Show audio server and hardware status"
  echo
  echo "Standard Commands:"
  echo "  shell                     - Open interactive shell in active container"
  echo "  update                    - Update edgeAI git repository"
  echo "  publish-jp7               - Instructions and commands to push JP7 container to Docker Hub"
  echo "  version                   - Display version and hardware information"
  echo "  help                      - Show this help message"
}

# Parse Subcommand
SUBCMD="$1"
shift 2>/dev/null || true

case "$SUBCMD" in
  audio)
    ACTION="$1"
    shift 2>/dev/null || true
    case "$ACTION" in
      setup)
        echo "🛠️ Initializing NVIDIA Riva Speech Server setup..."
        if ensure_riva_scripts_present; then
          bash "$RIVA_SETUP_SCRIPT"
        else
          echo "❌ Unable to resolve or download Riva setup script."
          exit 1
        fi
        ;;

      start)
        AUDIO_SERVER_CHOICE="$1"
        if [ "$AUDIO_SERVER_CHOICE" = "--riva" ] || [ "$AUDIO_SERVER_CHOICE" = "riva" ]; then
          SEL="1"
        else
          echo "=================================================="
          echo "🎙️ Select Audio / Speech Server to start:"
          echo "1) NVIDIA Riva Speech Server (ASR + TTS, gRPC Port 50051)"
          echo "2) Whisper / Faster-Whisper ASR Server"
          echo "=================================================="
          read -p "Choice [1-2, default: 1]: " SEL
          SEL="${SEL:-1}"
        fi

        if [ "$SEL" = "1" ]; then
          echo "1️⃣ Checking NVIDIA Riva Speech Server (Port 50051)..."
          if python3 -c "import socket; s = socket.socket(); s.settimeout(2); exit(0 if s.connect_ex(('localhost', 50051)) == 0 else 1)" 2>/dev/null; then
            echo "  ✅ NVIDIA Riva Speech Server is ALREADY RUNNING on port 50051."
          else
            ensure_riva_scripts_present
            if [ -d "/Developer/models/riva" ] && [ -n "$(ls -A /Developer/models/riva 2>/dev/null)" ]; then
              bash "$RIVA_START_SCRIPT"
            else
              echo "  ℹ️  Riva models directory (/Developer/models/riva) is not initialized."
              echo "  🛠️ Automatically executing setup script ($RIVA_SETUP_SCRIPT)..."
              bash "$RIVA_SETUP_SCRIPT"
              bash "$RIVA_START_SCRIPT"
            fi
          fi
        else
          echo "🎙️ Starting Whisper / Faster-Whisper ASR container..."
          docker run -d --rm --name whisper-server --runtime=nvidia --network host ghcr.io/fedirz/faster-whisper-server:latest 2>/dev/null || true
          echo "  ✅ Whisper server started."
        fi
        ;;

      stop)
        echo "🛑 Stopping Audio / Speech Servers..."
        ensure_riva_scripts_present 2>/dev/null || true
        if [ -n "$RIVA_STOP_SCRIPT" ] && [ -f "$RIVA_STOP_SCRIPT" ]; then
          bash "$RIVA_STOP_SCRIPT"
        else
          docker stop riva-speech 2>/dev/null || true
        fi
        docker stop whisper-server 2>/dev/null || true
        echo "✅ Stopped Audio / Speech servers."
        ;;

      test|status)
        echo "=================================================="
        echo "🎙️ Audio & Speech Server Status / Test"
        echo "=================================================="
        echo "📋 1. Riva gRPC Server Check (Port 50051):"
        if python3 -c "import socket; s = socket.socket(); s.settimeout(2); exit(0 if s.connect_ex(('localhost', 50051)) == 0 else 1)" 2>/dev/null; then
          echo "  ✅ NVIDIA Riva Speech Server is ONLINE on port 50051 (ASR + TTS Ready)"
        else
          echo "  ❌ NVIDIA Riva Speech Server is OFFLINE on port 50051."
        fi

        echo "📋 2. Host Audio Hardware Devices:"
        if command -v arecord &>/dev/null; then
          echo "  🎤 Microphones (arecord -l):"
          arecord -l 2>/dev/null | grep -E '^card' || echo "     No recording cards found."
        fi
        if command -v aplay &>/dev/null; then
          echo "  🔊 Speakers (aplay -l):"
          aplay -l 2>/dev/null | grep -E '^card' || echo "     No playback cards found."
        fi
        echo "=================================================="
        ;;

      *)
        echo "Usage: sjsujetsontool audio [start|stop|test|status|setup]"
        ;;
    esac
    ;;
  container)
    ACTION="$1"
    case "$ACTION" in
      status)
        echo "=================================================="
        echo "Active Container Selection: $ACTIVE_CONTAINER"
        echo "Image: $LOCAL_IMAGE"
        echo "Container Name: $CONTAINER_NAME"
        echo "Default for this hardware: $DEFAULT_CONTAINER"
        echo "=================================================="
        ;;
      set-jp6|jp6)
        set_container_pref "jp6"
        ;;
      set-jp7|jp7)
        set_container_pref "jp7"
        ;;
      select|*)
        echo "Select default container image:"
        echo "1) JetPack 6 (cmpelkk/jetson-llm:v1 - Orin Nano)"
        echo "2) JetPack 7 (cmpelkk/jetson-unified:jp7-thor - Jetson Thor / Orin)"
        read -p "Choice [1-2]: " choice
        case "$choice" in
          1) set_container_pref "jp6" ;;
          2) set_container_pref "jp7" ;;
          *) echo "Invalid choice." ;;
        esac
        ;;
    esac
    ;;

  vllm)
    ACTION="$1"
    shift 2>/dev/null || true
    case "$ACTION" in
      start)
        MODEL_CHOICE="$1"
        if [ "$MODEL_CHOICE" = "--cosmos" ] || [ "$MODEL_CHOICE" = "cosmos" ]; then
          SEL="1"
        elif [ "$MODEL_CHOICE" = "--qwen" ] || [ "$MODEL_CHOICE" = "qwen" ]; then
          SEL="2"
        elif [ "$MODEL_CHOICE" = "--model" ] || [ "$MODEL_CHOICE" = "-m" ]; then
          SEL="3"
          CUSTOM_HF_MODEL="$2"
        else
          echo "=================================================="
          echo "🚀 Select vLLM model to serve on Jetson:"
          echo "1) Cosmos-Reason2-8B (nvidia/Cosmos-Reason2-8B, VLM + Reasoning, Port 8010)"
          echo "2) Qwen2.5-Coder-7B (Qwen/Qwen2.5-Coder-7B-Instruct, Text LLM, Port 8010)"
          echo "3) Custom HuggingFace Model (Input any HF model ID supported by vLLM & device)"
          echo "=================================================="
          read -p "Choice [1-3, default: 1]: " SEL
          SEL="${SEL:-1}"
        fi

        # 1. Resolve HF_TOKEN from environment, ~/.env.local, or prompt user
        ACTIVE_HF_TOKEN="$HF_TOKEN"
        [ -z "$ACTIVE_HF_TOKEN" ] && ACTIVE_HF_TOKEN="$HUGGINGFACE_TOKEN"
        if [ -z "$ACTIVE_HF_TOKEN" ] && [ -f "$HOME/.env.local" ]; then
          ACTIVE_HF_TOKEN=$(grep -E '^(HF_TOKEN|HUGGINGFACE_TOKEN)=' "$HOME/.env.local" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'")
        fi
        if [ -z "$ACTIVE_HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
          ACTIVE_HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null | tr -d '\n')
        fi

        MODEL_VOL_FLAG=""
        IS_COSMOS=0

        if [ "$SEL" = "1" ]; then
          IS_COSMOS=1
          MODEL_NAME="nvidia/Cosmos-Reason2-8B"
          LOCAL_MODEL_PATH="/Developer/models/cosmos-reason2-8b"
          if [ -d "$LOCAL_MODEL_PATH" ]; then
            echo "📂 Found local model directory at $LOCAL_MODEL_PATH"
            MODEL_NAME="/models/cosmos-reason2-8b"
            MODEL_VOL_FLAG="-v $LOCAL_MODEL_PATH:/models/cosmos-reason2-8b:ro"
          else
            echo "🌐 Target model repo: 'nvidia/Cosmos-Reason2-8B' on Hugging Face"
            if [ -z "$ACTIVE_HF_TOKEN" ]; then
              echo "🔑 HuggingFace Token required to access/download 'nvidia/Cosmos-Reason2-8B'."
              read -sp "Enter your HF_TOKEN: " USER_HF_TOKEN
              echo
              if [ -n "$USER_HF_TOKEN" ]; then
                ACTIVE_HF_TOKEN="$USER_HF_TOKEN"
                echo "HF_TOKEN=\"$ACTIVE_HF_TOKEN\"" >> "$HOME/.env.local"
                echo "💾 Saved token to $HOME/.env.local"
              fi
            fi
          fi
          echo "🚀 Preparing vLLM with Cosmos-Reason2-8B on port 8010..."
        elif [ "$SEL" = "2" ]; then
          MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
          echo "🚀 Preparing vLLM with Qwen2.5-Coder-7B on port 8010..."
        else
          if [ -z "$CUSTOM_HF_MODEL" ]; then
            read -p "Enter HuggingFace Model ID (e.g. Qwen/Qwen2.5-14B-Instruct): " CUSTOM_HF_MODEL
          fi

          if [ -z "$CUSTOM_HF_MODEL" ]; then
            echo "❌ No model ID provided. Aborting."
            exit 1
          fi

          # Validate Custom Model
          if ! validate_custom_hf_model "$CUSTOM_HF_MODEL" "$ACTIVE_HF_TOKEN"; then
            exit 1
          fi
          MODEL_NAME="$CUSTOM_HF_MODEL"
          echo "🚀 Preparing vLLM with custom model '$MODEL_NAME' on port 8010..."
        fi

        HF_ENV_FLAG=""
        if [ -n "$ACTIVE_HF_TOKEN" ]; then
          HF_ENV_FLAG="-e HF_TOKEN=$ACTIVE_HF_TOKEN -e HUGGINGFACE_TOKEN=$ACTIVE_HF_TOKEN"
        fi

        # Optional: try to drop page cache non-interactively without prompting for sudo password
        if [ "$EUID" -eq 0 ]; then
          sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
        else
          sudo -n sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
        fi

        # Clean up any previously running container
        docker stop vllm-server 2>/dev/null || true
        docker rm -f vllm-server 2>/dev/null || true

        echo "🐳 Launching vLLM container 'vllm-server'..."
        if [ "$IS_COSMOS" -eq 1 ]; then
          docker run -d \
            --name vllm-server \
            --runtime=nvidia \
            --network host \
            $HF_ENV_FLAG \
            $MODEL_VOL_FLAG \
            -v "${HOME}/.cache/vllm":/root/.cache/vllm \
            ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
            vllm serve "$MODEL_NAME" \
              --served-model-name nvidia/cosmos-reason2-8b-fp8 \
              --max-model-len 8192 \
              --gpu-memory-utilization 0.7 \
              --reasoning-parser qwen3 \
              --media-io-kwargs '{"video": {"num_frames": -1}}' \
              --enable-prefix-caching \
              --port 8010
        else
          docker run -d \
            --name vllm-server \
            --runtime=nvidia \
            --network host \
            $HF_ENV_FLAG \
            -v "${HOME}/.cache/vllm":/root/.cache/vllm \
            ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
            vllm serve "$MODEL_NAME" \
              --served-model-name "$MODEL_NAME" \
              --max-model-len 4096 \
              --gpu-memory-utilization 0.7 \
              --port 8010
        fi

        echo "⏳ Monitoring container startup and checking model loading progress..."
        SERVER_READY=0
        CONTAINER_FAILED=0

        for i in {1..30}; do
          # Check if container is still running
          if ! docker ps --format '{{.Names}}' | grep -q "^vllm-server$"; then
            CONTAINER_FAILED=1
            break
          fi

          # Check if API endpoint is responding
          if curl -s http://localhost:8010/v1/models >/dev/null 2>&1; then
            SERVER_READY=1
            break
          fi

          echo -n "."
          sleep 2
        done
        echo

        if [ "$CONTAINER_FAILED" -eq 1 ]; then
          echo "══════════════════════════════════════════════════"
          echo "❌ CRITICAL ERROR: vLLM container exited unexpectedly!"
          echo "══════════════════════════════════════════════════"
          echo "📋 Container Startup Log Output:"
          docker logs --tail 30 vllm-server 2>&1
          echo "══════════════════════════════════════════════════"
          echo "💡 Diagnostic Actions:"
          echo " 1. Gated Model Access: If you see '401 Unauthorized' or '404 Repository Not Found', visit:"
          echo "    👉 https://huggingface.co/nvidia/Cosmos-Reason2-8B"
          echo "    Log in, accept the license agreement, and generate a Read token at https://huggingface.co/settings/tokens"
          echo " 2. Pass Token: Export your token prior to running sjsujetsontool:"
          echo "    export HF_TOKEN=\"hf_your_token_here\""
          echo "    sjsujetsontool vllm start"
          echo "══════════════════════════════════════════════════"
          exit 1
        elif [ "$SERVER_READY" -eq 1 ]; then
          echo "✅ vLLM server is READY and responding on http://localhost:8010/v1 !"
          echo "👉 Test server APIs with: sjsujetsontool vllm test"
        else
          echo "⏳ vLLM container is still running and loading model weights into GPU memory."
          echo "👉 Run 'sjsujetsontool vllm test' in a few seconds or check logs with: docker logs -f vllm-server"
        fi
        ;;

      stop)
        echo "🛑 Stopping vLLM container 'vllm-server'..."
        docker stop vllm-server 2>/dev/null || true
        docker rm -f vllm-server 2>/dev/null || true
        echo "✅ Stopped vLLM server."
        ;;

      test)
        echo "🧪 Testing vLLM Server API on http://localhost:8010/v1 ..."
        if ! docker ps --format '{{.Names}}' | grep -q "^vllm-server$"; then
          echo "❌ Container 'vllm-server' is NOT running."
          echo "📋 Checking logs of last container instance:"
          docker logs --tail 25 vllm-server 2>&1 || true
          echo "👉 Start it first using: sjsujetsontool vllm start"
          exit 1
        fi

        echo "⏳ Checking if vLLM server is responsive..."
        SERVER_READY=0
        for i in {1..20}; do
          if curl -s http://localhost:8010/v1/models >/dev/null 2>&1; then
            SERVER_READY=1
            break
          fi
          echo -n "."
          sleep 2
        done
        echo

        if [ "$SERVER_READY" -eq 0 ]; then
          echo "⚠️  vLLM server is still loading model weights into GPU memory or experienced a startup error."
          echo "📋 Container Log Output:"
          docker logs --tail 25 vllm-server 2>&1
          exit 1
        fi

        echo "📋 1. Querying GET /v1/models:"
        curl -s http://localhost:8010/v1/models | jq . 2>/dev/null || curl -s http://localhost:8010/v1/models
        echo
        echo "💬 2. Querying POST /v1/chat/completions:"
        DETECTED_MODEL=$(curl -s http://localhost:8010/v1/models | jq -r '.data[0].id' 2>/dev/null || echo "nvidia/cosmos-reason2-8b-fp8")
        curl -s http://localhost:8010/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d "{
            \"model\": \"$DETECTED_MODEL\",
            \"messages\": [
              {\"role\": \"user\", \"content\": \"Hello Jetson Thor! Respond in 5 words.\"}
            ],
            \"max_tokens\": 40
          }" | jq . 2>/dev/null || curl -s http://localhost:8010/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d "{
            \"model\": \"$DETECTED_MODEL\",
            \"messages\": [
              {\"role\": \"user\", \"content\": \"Hello Jetson Thor! Respond in 5 words.\"}
            ],
            \"max_tokens\": 40
          }"
        echo
        ;;

      *)
        echo "Usage: sjsujetsontoolv2 vllm [start|stop|test]"
        ;;
    esac
    ;;

  shell)
    ensure_container_started
    echo "🐚 Opening shell inside container '$CONTAINER_NAME'..."
    docker exec $TTY_FLAGS $CONTAINER_NAME bash
    ;;

  update)
    echo "🔄 Updating edgeAI repository..."
    setup_check_internal
    ( cd /Developer/edgeAI && git pull )
    echo "✅ Updated edgeAI repository."
    ;;

  publish-jp7|publish)
    echo "=================================================="
    echo "📤 Instructions to Push JetPack 7 Container to Docker Hub"
    echo "=================================================="
    echo "1. Login to Docker Hub:"
    echo "   docker login -u cmpelkk"
    echo
    echo "2. Build local JetPack 7 image on Jetson Thor:"
    echo "   cd /Developer/edgeAI"
    echo "   docker build -f jetson/Dockerfile.jp7 \\"
    echo "     --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 \\"
    echo "     --build-arg REBUILD_OPENCV=0 \\"
    echo "     --build-arg INSTALL_ISAAC_ROS=1 \\"
    echo "     -t cmpelkk/jetson-unified:jp7-thor ."
    echo
    echo "3. Tag & Push to Docker Hub:"
    echo "   docker tag cmpelkk/jetson-unified:jp7-thor cmpelkk/jetson-unified:jp7"
    echo "   docker push cmpelkk/jetson-unified:jp7-thor"
    echo "   docker push cmpelkk/jetson-unified:jp7"
    echo "=================================================="
    ;;

  version)
    echo "sjsujetsontool $SCRIPT_VERSION"
    echo "Active Container: $LOCAL_IMAGE"
    echo "Jetson Model: $JETSON_MODEL"
    echo "L4T Revision: $L4T_REVISION"
    echo "CUDA Version: $CUDA_VERSION"
    ;;

  help|*)
    show_help
    ;;
esac
