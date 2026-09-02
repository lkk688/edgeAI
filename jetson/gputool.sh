#!/bin/bash

# === gputool ===
# Custom Non-Sudo / User-space CLI for Edge AI & GPU Devices
SCRIPT_VERSION="v1.0.0"

# Colors for premium UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Paths
GPUTOOL_DIR="$HOME/.gputool"
TAILSCALE_DIR="$GPUTOOL_DIR/tailscale"
TS_SOCKET="$GPUTOOL_DIR/tailscaled.sock"
TS_STATE="$GPUTOOL_DIR/tailscaled.state"
TS_LOG="$GPUTOOL_DIR/tailscaled.log"
TS_PID_FILE="$GPUTOOL_DIR/tailscaled.pid"
SCRIPT_PATH="$HOME/.local/bin/gputool"

# Constants
HEADSCALE_LOGIN_SERVER="https://headscale.forgengi.org"
HEADSCALE_AUTHKEY="2566b0d9607d5e78bda28311963463d358352133c32d94ae"
TAILSCALE_DEFAULT_VERSION="1.68.1"
SCRIPT_URL="https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/gputool.sh"
CHAT_PY_URL="https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/chat.py"
CHAT_PY_PATH="$GPUTOOL_DIR/chat.py"

# Helper print functions
info() { echo -e "${BLUE}[⚙️]${NC} $*"; }
success() { echo -e "${GREEN}[✅]${NC} $*"; }
warn() { echo -e "${YELLOW}[⚠️]${NC} $*"; }
error() { echo -e "${RED}[❌]${NC} $*"; }

# === GPU / CUDA detection helpers ===
# These probe the host for an NVIDIA GPU and a CUDA toolkit so that other
# commands can pick a compatible PyTorch build and compile llama.cpp.

# Locate the nvcc binary: PATH first, then common CUDA install locations.
detect_nvcc_bin() {
  if command -v nvcc &>/dev/null; then command -v nvcc; return 0; fi
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then echo "/usr/local/cuda/bin/nvcc"; return 0; fi
  local candidate
  for candidate in /usr/local/cuda-*/bin/nvcc; do
    [[ -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

# nvcc toolkit version as MAJOR.MINOR (e.g. "13.0"); empty if nvcc is absent.
detect_nvcc_version() {
  local nb; nb=$(detect_nvcc_bin) || return 1
  "$nb" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -n1
}

# Max CUDA runtime version supported by the installed driver (via nvidia-smi).
detect_driver_cuda_version() {
  command -v nvidia-smi &>/dev/null || return 1
  nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' | head -n1
}

# GPU compute capability as MAJOR.MINOR (e.g. "12.0" for Blackwell, "6.1" for Pascal).
detect_gpu_compute_cap() {
  command -v nvidia-smi &>/dev/null || return 1
  local cc
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
  [[ -n "$cc" && "$cc" != "[N/A]" ]] && { echo "$cc"; return 0; }
  return 1
}

# Friendly GPU name (e.g. "NVIDIA GeForce RTX 5080").
detect_gpu_name() {
  command -v nvidia-smi &>/dev/null || return 1
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1
}

# Convert a MAJOR.MINOR version into an integer*10 for easy comparison (12.0 -> 120, 6.1 -> 61).
_ver_to_int() { awk -v v="${1:-0}" 'BEGIN{printf "%d", v*10}'; }

# Decide which PyTorch CUDA wheel index to use based on GPU arch + CUDA toolkit/driver.
# Echoes a single token on stdout (cu128 | cu126 | cu121 | cu118 | cpu) and logs reasoning to stderr.
#   - Modern GPUs (compute capability >= 7.0, incl. Blackwell sm_120) -> cu128
#   - Older GPUs (compute capability  < 7.0, e.g. Pascal/Maxwell)     -> cu118
#   - No GPU but a toolkit present -> map from the toolkit/driver CUDA version
#   - Nothing CUDA-related found   -> default to cu128 (CUDA 12.8)
select_torch_cuda_tag() {
  local cc nvcc_ver drv_ver tag reason cc_num
  cc=$(detect_gpu_compute_cap 2>/dev/null || true)
  nvcc_ver=$(detect_nvcc_version 2>/dev/null || true)
  drv_ver=$(detect_driver_cuda_version 2>/dev/null || true)
  cc_num=$(_ver_to_int "${cc:-0}")

  if [[ -z "$cc" && -z "$nvcc_ver" && -z "$drv_ver" ]]; then
    tag="cu128"; reason="No GPU or CUDA toolkit detected; defaulting to CUDA 12.8 wheels."
  elif [[ -n "$cc" ]]; then
    if (( cc_num < 70 )); then
      tag="cu118"; reason="GPU compute capability $cc is older than 7.0; using CUDA 11.8 wheels for compatibility."
    else
      tag="cu128"; reason="GPU compute capability $cc supports modern wheels; using CUDA 12.8 (Blackwell-ready)."
      local drv_num=0
      [[ -n "$drv_ver" ]] && drv_num=$(_ver_to_int "$drv_ver")
      # Blackwell (sm_120) on a CUDA 13 driver: cu130 is the closest match and
      # measurably quicker here — 21.7 vs 20.2 TFLOPS fp32 on an RTX 5080.
      # Both carry sm_120 kernels; cu124 and older stop at sm_90 and fail every
      # launch with "no kernel image is available for execution on the device".
      if (( cc_num >= 120 && drv_num >= 130 )); then
        tag="cu130"; reason="Blackwell (sm_$cc) on a CUDA $drv_ver driver; using CUDA 13.0 wheels."
      elif (( drv_num < 121 && drv_num >= 118 )); then
        # Driver too old for the 12.8 runtime: step down to a build it can run.
        tag="cu118"; reason="GPU is modern but the driver supports only up to CUDA $drv_ver; capping to CUDA 11.8 wheels."
      fi
    fi
  else
    # No GPU capability info, but a toolkit/driver exists: choose from its CUDA version.
    local ref ref_num
    ref="${nvcc_ver:-$drv_ver}"
    ref_num=$(_ver_to_int "$ref")
    if   (( ref_num >= 130 )); then tag="cu130"; reason="CUDA toolkit $ref detected; using CUDA 13.0 wheels."
    elif (( ref_num >= 124 )); then tag="cu128"; reason="CUDA toolkit $ref detected; using CUDA 12.8 wheels."
    elif (( ref_num >= 121 )); then tag="cu121"; reason="CUDA toolkit $ref detected; using CUDA 12.1 wheels."
    else                            tag="cu118"; reason="CUDA toolkit $ref detected; using CUDA 11.8 wheels."
    fi
  fi
  echo "$reason" >&2
  echo "$tag"
}

# Detect GPU/CUDA and install the best-matching PyTorch build into the given conda env.
# Falls back to the chosen index -> plain PyPI on failure. Returns non-zero only if all attempts fail.
install_pytorch_auto() {
  local env_name="$1"
  local gpu_name cc nvcc_ver drv_ver tag
  gpu_name=$(detect_gpu_name 2>/dev/null || true)
  cc=$(detect_gpu_compute_cap 2>/dev/null || true)
  nvcc_ver=$(detect_nvcc_version 2>/dev/null || true)
  drv_ver=$(detect_driver_cuda_version 2>/dev/null || true)

  echo "   • GPU Detected      : ${gpu_name:-None}"
  echo "   • Compute Capability: ${cc:-Unknown}"
  echo "   • nvcc Toolkit      : ${nvcc_ver:-Not found}"
  echo "   • Driver CUDA Max   : ${drv_ver:-Unknown}"

  tag=$(select_torch_cuda_tag)
  info "Selected PyTorch build: $tag"
  warn "⏳ Downloading PyTorch wheels (~800MB+). This can take several minutes"
  warn "   depending on your network connection speed. Please do not close the terminal..."

  if conda run -n "$env_name" pip install torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/$tag"; then
    return 0
  fi
  warn "Failed to install via the '$tag' wheel index. Retrying with default PyPI torch..."
  conda run -n "$env_name" pip install torch torchvision torchaudio
}

# Ensure a build tool is available inside the conda env; install via conda-forge (pip fallback).
# Usage: ensure_conda_tool <env_name> <command> <conda_pkg> [pip_pkg]
ensure_conda_tool() {
  local env_name="$1" cmd="$2" conda_pkg="$3" pip_pkg="${4:-}"
  if conda run -n "$env_name" bash -lc "command -v $cmd" &>/dev/null; then
    success "Found '$cmd' in conda env '$env_name'."
    return 0
  fi
  info "'$cmd' is missing in env '$env_name'. Installing '$conda_pkg' from conda-forge..."
  if conda install -y -n "$env_name" -c conda-forge "$conda_pkg" &>/dev/null; then
    success "Installed '$conda_pkg'."
    return 0
  fi
  if [[ -n "$pip_pkg" ]]; then
    warn "Conda install of '$conda_pkg' failed. Trying pip install '$pip_pkg'..."
    if conda run -n "$env_name" pip install "$pip_pkg" &>/dev/null; then
      success "Installed '$pip_pkg' via pip."
      return 0
    fi
  fi
  error "Failed to install '$cmd' (tried conda pkg '$conda_pkg'${pip_pkg:+ and pip pkg '$pip_pkg'})."
  return 1
}

# Helper download functions (with Python 3 fallback if curl/wget are missing)
# Sends no-cache headers so we never get a stale copy from the GitHub raw CDN
# (raw.githubusercontent.com caches aggressively and ignores query strings).
download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl &>/dev/null; then
    curl -fsSL -H "Cache-Control: no-cache" -H "Pragma: no-cache" "$url" -o "$dest"
  elif command -v wget &>/dev/null; then
    wget -qO "$dest" --header="Cache-Control: no-cache" --header="Pragma: no-cache" "$url"
  elif command -v python3 &>/dev/null; then
    python3 -c "
import urllib.request
req = urllib.request.Request('$url', headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})
data = urllib.request.urlopen(req, timeout=30).read()
open('$dest', 'wb').write(data)
" &>/dev/null
  else
    return 1
  fi
}

# Download the chat client (chat.py) from GitHub into ~/.gputool/.
# Non-fatal: chat.py is also fetched on demand by `gputool chat` if missing.
download_chat_py() {
  mkdir -p "$GPUTOOL_DIR"
  if download_file "$CHAT_PY_URL" "$CHAT_PY_PATH"; then
    chmod +x "$CHAT_PY_PATH" 2>/dev/null
    success "Chat client installed: $CHAT_PY_PATH"
    return 0
  fi
  warn "Could not download chat client (chat.py). 'gputool chat' will retry the download when first run."
  return 1
}

http_get_auth() {
  local url="$1"
  local token="$2"
  if command -v curl &>/dev/null; then
    curl -sf --max-time 8 "$url" -H "Authorization: Bearer $token"
  elif command -v wget &>/dev/null; then
    wget -qO- --timeout=8 --header="Authorization: Bearer $token" "$url" 2>/dev/null
  elif command -v python3 &>/dev/null; then
    python3 -c "
import urllib.request
try:
    req = urllib.request.Request('$url', headers={'Authorization': 'Bearer $token'})
    print(urllib.request.urlopen(req, timeout=8).read().decode())
except Exception:
    pass
" 2>/dev/null
  else
    return 1
  fi
}

# ❗ Warn if run incorrectly via `bash gputool version`
if [[ "$0" == "bash" && "$1" == "${BASH_SOURCE[0]}" ]]; then
  warn "Please run this script directly, not via 'bash'."
  echo "✅ Correct: ./gputool version"
  echo "❌ Wrong: bash gputool version"
  exit 1
fi

show_help() {
  echo -e "${BOLD}🚀 gputool — Non-Sudo Edge AI & GPU Device Utility Tool${NC} (${SCRIPT_VERSION})"
  echo "Usage: gputool <command> [arguments]"
  echo
  echo "Core Commands:"
  echo "  help                     - Show this help message"
  echo "  version                  - Show script version"
  echo "  device [--online]        - Full device report: GPU, driver health, disk, conda, gputool"
  echo "  profile                  - Show the serving profile this GPU gets (VRAM, NVFP4/FP8, tuned defaults)"
  echo "  container <action>       - Persistent PyTorch GPU container: status|pull|start|shell|run|test|stop|set"
  echo "  agent <start|stop|status|test> - Agent sidecar (:8002); .test. runs the full tool suite"
  echo "  install-ai [extras]      - pip install gputool-ai (chat + agent) — no repo checkout needed"
  echo ""
  echo "Shared HuggingFace Cache (no sudo, campus network):"
  echo "  hf-cache setup           - Install rclone, create a key, configure the shared-cache remote"
  echo "  hf-cache mount           - Mount the shared cache at ~/hf-shared"
  echo "  hf-cache enable          - Point HF_HOME at it for every new shell (auto-mounts on login)"
  echo "  hf-cache status          - Show remote, mount, local cache size and HF_HOME"
  echo "  hf-cache disable         - Undo enable; hf-cache unmount - detach the mount"
  echo "  install                  - Install gputool script to ~/.local/bin/ and setup PATH"
  echo "  update-script            - Pull the latest gputool script from GitHub"
  echo
  echo "AI & Machine Learning Commands:"
  echo "  install-conda [path]     - Download and silently install Miniconda (default: ~/miniconda3)"
  echo "  setup-lerobot [env_name] - Create Conda env and install PyTorch (RTX 5080), LeRobot, and HF"
  echo "  setup-env [env_name] [python_ver] - Create Conda env with custom python, PyTorch (RTX 5080) & HF"
  echo "  check [env_name]         - Run a complete diagnostic check of GPU, PyTorch, HF, LeRobot & Tailscale"
  echo
  echo "Llama.cpp & LLM Commands (RTX GPU Offloading):"
  echo "  setup-vllm [env]         - Install vLLM into a Conda env (default: py312)"
  echo "  vllm <action> [model] [port]             - alias of serve-vllm (same name as sjsujetsontool)"
  echo "  llama <action> [model] [port]            - alias of serve-llamacpp (same name as sjsujetsontool)"
  echo "  serve-vllm <action> [model] [port] [--env E] [--max-len N] [--gpu-mem F] [--api-key K] [-f|-d]"
  echo "      start [model] [port]                 - Serve an OpenAI-compatible API (default: Qwen/Qwen3.5-4B on 8000)"
  echo "      stop | status                        - Stop the server / show process, API and GPU memory"
  echo ""
  echo "  setup-llamacpp [env_name] - Compile llama.cpp with CUDA support inside Conda env"
  echo "  download-model [repo] [file] [env] - Download a GGUF model from Hugging Face"
  echo "  serve-llamacpp <action> [model] [port] [--foreground|--background] - Manage llama-server"
  echo "      start [model] [port] [-d|--background]  - Serve in background (default); detached daemon"
  echo "      start [model] [port] [-f|--foreground]  - Serve in foreground (attached, Ctrl+C to stop)"
  echo "      start ... [--host <addr>]                - Bind address (default 0.0.0.0 = LAN-accessible; 127.0.0.1 = local-only)"
  echo "      start ... [--api-key <token>]            - Require 'Authorization: Bearer <token>' on all requests"
  echo "      start ... [--mmproj <file>|--no-mmproj]  - Vision: auto-detects mmproj*.gguf in models dir (image input)"
  echo "      stop                                     - Stop tracked + any stray llama-server services"
  echo "      status                                   - Show running server and API health"
  echo "  chat [message] [--host <ip>] [--port <p>] [--api-key <token>] [--system <txt>] [--think] - Terminal chat client (streaming)"
  echo
  echo "Tailscale Commands (🔒 Userspace VPN, NO ROOT/SUDO Required):"
  echo "  tailscale <sub>          - Manage userspace Tailscale client"
  echo "      setup                - Download and configure Tailscale static binaries"
  echo "      up [--force]         - Connect to Headscale network in user-space mode"
  echo "      status               - Check connection, Tailscale IPs, and proxy variables"
  echo "      down                 - Disconnect from network and stop background daemon"
  echo "      restart              - Restart the userspace tailscaled daemon"
  echo
  echo "Examples:"
  echo "  gputool tailscale setup"
  echo "  gputool tailscale up"
  echo "  gputool install-conda"
  echo "  gputool setup-lerobot my_env"
  echo "  gputool setup-env py312 3.12"
  echo "  gputool check my_env"
  echo "  gputool setup-llamacpp my_env"
  echo "  gputool download-model unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q6_K_XL.gguf my_env"
  echo "  gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080            # background daemon (default)"
  echo "  gputool serve-llamacpp start Qwen3.5-9B-UD-Q6_K_XL.gguf 8080 --foreground # attached, Ctrl+C to stop"
  echo "  gputool serve-llamacpp stop                                             # stop all llama-server services"
  echo "  gputool chat                                                            # interactive chat with local server"
  echo "  gputool chat \"What is CUDA?\" --host 10.31.96.155 --api-key sjsugputool   # one-shot query to a remote peer"
}

# Spinner helper
show_spinner() {
  local PID=$1
  local MESSAGE="$2"
  local CHARS="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
  while kill -0 $PID 2>/dev/null; do
    for (( i=0; i<${#CHARS}; i++ )); do
      echo -ne "\r${CHARS:$i:1} $MESSAGE"
      sleep 0.15
    done
  done
  echo -ne "\r                                                  \r"
}

# Auto-install/setup PATH
install_gputool() {
  info "Installing gputool to $SCRIPT_PATH ..."
  mkdir -p "$(dirname "$SCRIPT_PATH")"
  
  # Copy this file to target path
  cp "$0" "$SCRIPT_PATH"
  chmod +x "$SCRIPT_PATH"
  success "gputool script copied and made executable."

  # Check if target is in PATH
  if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    info "Adding ~/.local/bin to your PATH..."
    local SHELL_RC=""
    if [ -n "$ZSH_VERSION" ]; then
      SHELL_RC="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
      SHELL_RC="$HOME/.bashrc"
    else
      SHELL_RC="$HOME/.profile"
    fi
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    success "Added PATH to $SHELL_RC"
    echo "👉 Please run: source $SHELL_RC"
  else
    success "gputool is already in your PATH."
  fi

  # Fetch the companion chat client so 'gputool chat' works out of the box.
  download_chat_py

  success "Installation complete! You can now run: gputool help"
}

# Update script from GitHub
update_script() {
  info "Updating gputool script from GitHub..."
  local TEMP_FILE
  TEMP_FILE=$(mktemp)
  if download_file "$SCRIPT_URL" "$TEMP_FILE"; then
    chmod +x "$TEMP_FILE"
    mv "$TEMP_FILE" "$SCRIPT_PATH"
    success "gputool has been updated successfully to latest version."
    # Keep the companion chat client in sync with the updated script.
    download_chat_py
  else
    error "Failed to download update from GitHub."
    rm -f "$TEMP_FILE"
    exit 1
  fi
}

# Helper: check for running tailscaled daemon
is_daemon_running() {
  if [[ -f "$TS_PID_FILE" ]]; then
    local PID
    PID=$(cat "$TS_PID_FILE" 2>/dev/null)
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      return 0
    fi
  fi
  # Fallback: check via pgrep
  local FG_PID
  FG_PID=$(pgrep -f "tailscaled.*userspace-networking.*${TS_SOCKET}" | head -1)
  if [[ -n "$FG_PID" ]]; then
    echo "$FG_PID" > "$TS_PID_FILE"
    return 0
  fi
  return 1
}

# Check headscale for hostname conflict via API
check_hostname_conflict() {
  local hn="$1"
  local api_resp
  api_resp=$(http_get_auth "${HEADSCALE_LOGIN_SERVER}/api/v1/machine" "${HEADSCALE_AUTHKEY}")
  if [[ -z "$api_resp" ]]; then
    info "Could not reach headscale API — skipping hostname conflict check."
    return 0
  fi
  if echo "$api_resp" | grep -qi "\"$hn\""; then
    warn "Hostname conflict detected: '$hn' is already registered on the headscale server."
    echo "   💡 To avoid conflicts, rename your local machine hostname or specify --force"
    return 1
  else
    success "No hostname conflict: '$hn' is available on the headscale server."
    return 0
  fi
}

# Download & Setup Tailscale
setup_tailscale() {
  echo "══════════════════════════════════════════════════"
  echo "📦 Setting up Userspace Tailscale"
  echo "══════════════════════════════════════════════════"
  mkdir -p "$GPUTOOL_DIR"

  # Detect CPU architecture
  local ARCH
  ARCH=$(uname -m)
  local TS_ARCH=""
  case "$ARCH" in
    x86_64|amd64)   TS_ARCH="amd64" ;;
    aarch64|arm64)  TS_ARCH="arm64" ;;
    armv7l)         TS_ARCH="arm" ;;
    *)              error "Unsupported architecture: $ARCH"; exit 1 ;;
  esac

  info "Detected architecture: $ARCH ($TS_ARCH)"
  local DOWNLOAD_URL="https://pkgs.tailscale.com/stable/tailscale_${TAILSCALE_DEFAULT_VERSION}_${TS_ARCH}.tgz"
  local TMP_TGZ="/tmp/tailscale_${TAILSCALE_DEFAULT_VERSION}.tgz"

  info "Downloading static package from Tailscale..."
  echo "   URL: $DOWNLOAD_URL"
  if ! download_file "$DOWNLOAD_URL" "$TMP_TGZ"; then
    error "Download failed. Please check internet connection."
    exit 1
  fi

  info "Extracting files..."
  tar -xzf "$TMP_TGZ" -C "$GPUTOOL_DIR"
  local EXTRACTED_DIR
  EXTRACTED_DIR=$(find "$GPUTOOL_DIR" -maxdepth 1 -type d -name "tailscale_*" | head -1)
  if [[ -z "$EXTRACTED_DIR" ]]; then
    error "Failed to locate extracted folder."
    rm -f "$TMP_TGZ"
    exit 1
  fi

  rm -rf "$TAILSCALE_DIR"
  mv "$EXTRACTED_DIR" "$TAILSCALE_DIR"
  rm -f "$TMP_TGZ"

  success "Tailscale binaries configured successfully in:"
  echo "   $TAILSCALE_DIR"
  echo
  echo "✨ Next, run: gputool tailscale up"
}

# Start daemon and bring interface up
up_tailscale() {
  local FORCE=${1:-""}

  echo "══════════════════════════════════════════════════"
  echo "🌐 Starting Userspace Tailscale VPN"
  echo "══════════════════════════════════════════════════"

  # Verification
  if [[ ! -f "$TAILSCALE_DIR/tailscale" || ! -f "$TAILSCALE_DIR/tailscaled" ]]; then
    error "Tailscale binaries not found. Please run 'gputool tailscale setup' first."
    exit 1
  fi

  local CURRENT_HN
  CURRENT_HN=$(hostname)

  # Check if daemon is active
  if is_daemon_running; then
    success "tailscaled background daemon is already running (PID: $(cat "$TS_PID_FILE"))."
  else
    info "Starting userspace-networking tailscaled daemon..."
    rm -f "$TS_SOCKET"
    nohup "$TAILSCALE_DIR/tailscaled" \
      --tun=userspace-networking \
      --socks5-server=localhost:1055 \
      --outbound-http-proxy-listen=localhost:1055 \
      --socket="$TS_SOCKET" \
      --state="$TS_STATE" \
      > "$TS_LOG" 2>&1 &
    
    local DAEMON_PID=$!
    echo "$DAEMON_PID" > "$TS_PID_FILE"
    
    # Wait for socket to become ready
    local TIMEOUT=10
    while [[ ! -S "$TS_SOCKET" && $TIMEOUT -gt 0 ]]; do
      sleep 0.5
      ((TIMEOUT--))
    done

    if [[ -S "$TS_SOCKET" ]]; then
      success "tailscaled daemon started successfully (PID: $DAEMON_PID)."
    else
      error "tailscaled failed to start. Logs:"
      tail -n 15 "$TS_LOG"
      exit 1
    fi
  fi

  # Conflict check
  info "Checking for hostname conflicts on headscale server..."
  if ! check_hostname_conflict "$CURRENT_HN"; then
    if [[ "$FORCE" != "--force" ]]; then
      error "Hostname conflict detected. Change hostname or use: gputool tailscale up --force"
      exit 1
    fi
    warn "--force specified. Proceeding despite conflict."
  fi

  # Up command
  info "Connecting to Headscale at $HEADSCALE_LOGIN_SERVER ..."
  if "$TAILSCALE_DIR/tailscale" --socket="$TS_SOCKET" up \
    --login-server "$HEADSCALE_LOGIN_SERVER" \
    --authkey "$HEADSCALE_AUTHKEY" \
    --hostname "$CURRENT_HN" \
    --accept-routes; then
    
    echo
    echo "══════════════════════════════════════════════════"
    success "Successfully connected to Headscale!"
    
    local TS_STATUS_JSON
    TS_STATUS_JSON=$("$TAILSCALE_DIR/tailscale" --socket="$TS_SOCKET" status --json 2>/dev/null)
    local TS_IPS
    TS_IPS=$(echo "$TS_STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(d.get('TailscaleIPs',[])))" 2>/dev/null)
    
    echo "   Hostname       : $CURRENT_HN"
    echo "   Tailscale IPs  : $TS_IPS"
    echo "   SOCKS5 Proxy   : localhost:1055"
    echo "   HTTP Proxy     : localhost:1055"
    echo "══════════════════════════════════════════════════"
    echo "💡 Note: Since this runs in userspace, you must use"
    echo "   proxies for outgoing traffic to other nodes."
    echo "   Example: curl -x http://localhost:1055 http://<peer-ip>"
  else
    error "Failed to bring up Tailscale interface. Check daemon log: $TS_LOG"
    exit 1
  fi
}

# Check Tailscale status
status_tailscale() {
  echo "══════════════════════════════════════════════════"
  echo "🌐 Userspace Tailscale VPN Status"
  echo "══════════════════════════════════════════════════"

  if [[ ! -f "$TAILSCALE_DIR/tailscale" ]]; then
    error "Tailscale is not set up. Run 'gputool tailscale setup'."
    exit 1
  fi

  if ! is_daemon_running; then
    error "tailscaled background daemon is not running."
    echo "   👉 Start it via: gputool tailscale up"
    exit 1
  fi

  echo "📦 Tailscale Version : $($TAILSCALE_DIR/tailscale version | head -1)"
  echo "🔧 Daemon PID        : $(cat "$TS_PID_FILE")"
  echo "🔌 Socket Path       : $TS_SOCKET"
  echo

  local TS_JSON
  TS_JSON=$("$TAILSCALE_DIR/tailscale" --socket="$TS_SOCKET" status --json 2>/dev/null)
  if [[ -z "$TS_JSON" ]]; then
    error "Could not retrieve JSON status from tailscaled."
    exit 1
  fi

  local TS_STATE
  TS_STATE=$(echo "$TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('BackendState',''))" 2>/dev/null)
  local TS_IPS
  TS_IPS=$(echo "$TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(d.get('TailscaleIPs',[])))" 2>/dev/null)
  local TS_HN
  TS_HN=$(echo "$TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('HostName',''))" 2>/dev/null)
  local TS_PEERS
  TS_PEERS=$(echo "$TS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('Peer',{})))" 2>/dev/null)

  if [[ "$TS_STATE" == "Running" ]]; then
    echo -e "🟢 Connection State  : ${GREEN}${TS_STATE}${NC}"
  else
    echo -e "🔴 Connection State  : ${RED}${TS_STATE}${NC}"
  fi
  echo "   Device Hostname   : $TS_HN"
  echo "   Tailscale IPs     : $TS_IPS"
  echo "   Connected Peers   : $TS_PEERS"
  echo

  echo "🛡️  User-Space Proxy Configuration:"
  echo "   • SOCKS5 Proxy    : localhost:1055"
  echo "   • HTTP Proxy      : localhost:1055"
  echo
  echo "💡 How to access other nodes from this machine:"
  echo "   - Web requests:"
  echo "     export http_proxy=http://localhost:1055"
  echo "     export https_proxy=http://localhost:1055"
  echo "     curl http://<peer-ip-or-dns>"
  echo "   - SSH tunnel command:"
  echo "     ssh -o ProxyCommand=\"nc -X 5 -x localhost:1055 %h %p\" user@<peer-ip>"
  echo "══════════════════════════════════════════════════"
}

# Stop client and stop background process
down_tailscale() {
  echo "══════════════════════════════════════════════════"
  echo "🔌 Stopping Userspace Tailscale VPN"
  echo "══════════════════════════════════════════════════"

  if [[ ! -f "$TAILSCALE_DIR/tailscale" ]]; then
    error "Tailscale is not set up."
    exit 1
  fi

  if is_daemon_running; then
    info "Sending disconnect signal to tailscaled..."
    "$TAILSCALE_DIR/tailscale" --socket="$TS_SOCKET" down 2>/dev/null
    
    local PID
    PID=$(cat "$TS_PID_FILE" 2>/dev/null)
    info "Stopping background daemon (PID: $PID)..."
    kill "$PID" 2>/dev/null
    
    # Wait for process to exit
    local TIMEOUT=10
    while kill -0 "$PID" 2>/dev/null && [ $TIMEOUT -gt 0 ]; do
      sleep 0.5
      ((TIMEOUT--))
    done
    
    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
      warn "Daemon did not exit cleanly. Force killing..."
      kill -9 "$PID" 2>/dev/null
    fi
    
    rm -f "$TS_PID_FILE"
    rm -f "$TS_SOCKET"
    success "Tailscale daemon stopped."
  else
    warn "tailscaled daemon is not currently running."
  fi
  success "Tailscale disconnected."
}

# Download and install Miniconda silently
install_conda() {
  local target_path="${1:-$HOME/miniconda3}"
  
  echo "══════════════════════════════════════════════════"
  echo "📦 Installing Miniconda"
  echo "══════════════════════════════════════════════════"
  
  # Check if already installed
  if [[ -f "$target_path/bin/conda" ]]; then
    success "Miniconda is already installed at $target_path"
    "$target_path/bin/conda" init bash &>/dev/null
    return 0
  fi

  local miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  local temp_installer="/tmp/Miniconda3-latest-Linux-x86_64.sh"

  info "Downloading Miniconda installer..."
  echo "   URL: $miniconda_url"
  if ! download_file "$miniconda_url" "$temp_installer"; then
    error "Failed to download Miniconda installer. Please check network connection."
    exit 1
  fi

  info "Running Miniconda silent installer..."
  echo "   Destination: $target_path"
  if ! bash "$temp_installer" -b -u -p "$target_path"; then
    error "Miniconda silent installation failed."
    rm -f "$temp_installer"
    exit 1
  fi

  rm -f "$temp_installer"
  success "Miniconda installed successfully at $target_path."

  # Initialize conda for the current shell context & config files
  info "Initializing Conda for your shell profiles..."
  "$target_path/bin/conda" init bash &>/dev/null
  if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
    "$target_path/bin/conda" init zsh &>/dev/null
  fi

  # Auto-accept Anaconda Terms of Service to prevent CondaToSNonInteractiveError
  info "Accepting Anaconda Terms of Service for default channels..."
  "$target_path/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &>/dev/null
  "$target_path/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r &>/dev/null

  success "Conda initialization completed."
  echo "👉 To configure your active shell, please run: source ~/.bashrc"
  echo "══════════════════════════════════════════════════"
}

# Setup general Python ML environment with PyTorch (CUDA 12.8+ / Blackwell) & Hugging Face
setup_ml_env() {
  local env_name="${1:-py312}"
  local python_ver="${2:-3.12}"
  
  echo "══════════════════════════════════════════════════"
  echo "🐍 Setting up Python ML Environment ($env_name, Python $python_ver)"
  echo "══════════════════════════════════════════════════"
  
  # Find Conda
  local CONDA_SH=""
  for path in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/home/010796032@SJSUAD/miniconda3/etc/profile.d/conda.sh" \
    "/home/$USER/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$path" ]]; then
      CONDA_SH="$path"
      break
    fi
  done

  if [[ -n "$CONDA_SH" ]]; then
    info "Found conda at $CONDA_SH. Activating conda..."
    source "$CONDA_SH"
  elif command -v conda &>/dev/null; then
    info "Conda is already in PATH."
  else
    warn "Conda not found. Automatically triggering Miniconda installation..."
    install_conda "$HOME/miniconda3"
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
      error "Failed to locate Conda profile script after auto-installation."
      exit 1
    fi
  fi

  # Create conda environment
  info "Creating conda environment '$env_name' with Python $python_ver..."
  if ! conda env list | grep -q "^$env_name "; then
    if ! conda create -y -n "$env_name" python="$python_ver"; then
      error "Failed to create conda environment '$env_name'."
      exit 1
    fi
    success "Conda environment '$env_name' created."
  else
    warn "Conda environment '$env_name' already exists. Reusing it."
  fi

  # Install PyTorch (auto-detect GPU/CUDA to choose the matching wheel)
  echo
  info "Detecting GPU and CUDA toolkit to choose the best PyTorch build..."
  if ! install_pytorch_auto "$env_name"; then
    error "PyTorch installation failed."
    exit 1
  fi
  success "PyTorch installed."

  # Install Hugging Face Hub
  info "Installing Hugging Face Hub..."
  if ! conda run -n "$env_name" pip install huggingface_hub; then
    error "Hugging Face Hub installation failed."
    exit 1
  fi
  success "Hugging Face Hub installed."

  # Verification
  info "Running verification script..."
  echo
  conda run -n "$env_name" python3 -c "
import torch
import huggingface_hub

print('==================================================')
print('🧬 PyTorch Version    :', torch.__version__)
print('🟢 CUDA Available      :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🖥️  GPU Device Name    :', torch.cuda.get_device_name(0))
    print('⚙️  CUDA Device Arch   :', torch.cuda.get_arch_list())
print('🤗 HF Hub Version     :', huggingface_hub.__version__)
print('==================================================')
"

  echo
  success "Environment setup complete!"
  echo "👉 To activate this environment, run:"
  echo "   conda activate $env_name"
  echo "══════════════════════════════════════════════════"
}

# Setup Conda Env and install PyTorch + LeRobot + Hugging Face
setup_lerobot_env() {
  local env_name="${1:-lerobot}"
  
  echo "══════════════════════════════════════════════════"
  echo "🐍 Setting up LeRobot & PyTorch Environment"
  echo "══════════════════════════════════════════════════"
  
  # --- Find Conda initialization script ---
  local CONDA_SH=""
  for path in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/home/010796032@SJSUAD/miniconda3/etc/profile.d/conda.sh" \
    "/home/$USER/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$path" ]]; then
      CONDA_SH="$path"
      break
    fi
  done

  if [[ -n "$CONDA_SH" ]]; then
    info "Found conda at $CONDA_SH. Activating conda..."
    source "$CONDA_SH"
  elif command -v conda &>/dev/null; then
    info "Conda is already in PATH."
  else
    warn "Conda not found. Automatically triggering Miniconda installation..."
    install_conda "$HOME/miniconda3"
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
      error "Failed to locate Conda profile script after auto-installation."
      exit 1
    fi
  fi

  # --- Create conda environment ---
  info "Creating conda environment '$env_name' with Python 3.10..."
  if ! conda env list | grep -q "^$env_name "; then
    if ! conda create -y -n "$env_name" python=3.10; then
      error "Failed to create conda environment '$env_name'."
      exit 1
    fi
    success "Conda environment '$env_name' created."
  else
    warn "Conda environment '$env_name' already exists. Reusing it."
  fi

  # --- Install CMake ---
  info "Installing CMake < 4 via Conda (required for compiling LeRobot simulation libraries)..."
  if ! conda install -y -n "$env_name" -c conda-forge "cmake<4"; then
    warn "Conda installation of CMake failed. Trying fallback pip installation..."
    conda run -n "$env_name" pip install "cmake<4"
  fi

  # --- Install PyTorch (auto-detect GPU/CUDA to choose the matching wheel) ---
  echo
  info "Detecting GPU and CUDA toolkit to choose the best PyTorch build..."
  if ! install_pytorch_auto "$env_name"; then
    error "PyTorch installation failed."
    exit 1
  fi
  success "PyTorch installed."

  # --- Install LeRobot and Hugging Face Hub ---
  info "Installing LeRobot (with extra dependencies) and Hugging Face Hub..."
  warn "⏳ Downloading LeRobot simulation libraries. This may also take a few minutes..."
  # 'lerobot[all]' installs standard aloha, pusht and other simulation/robotics dependencies
  if ! conda run -n "$env_name" pip install "lerobot[all]" huggingface_hub; then
    warn "Installing 'lerobot[all]' failed. Trying base 'lerobot' and manual simulation libraries (mujoco, h5py)..."
    if ! conda run -n "$env_name" pip install lerobot huggingface_hub; then
      error "LeRobot installation failed."
      exit 1
    fi
    info "Attempting to install standard simulation libraries (mujoco, h5py) separately..."
    conda run -n "$env_name" pip install mujoco h5py
  fi
  success "LeRobot and Hugging Face packages installed."

  # --- Verification ---
  info "Running verification script..."
  echo
  conda run -n "$env_name" python3 -c "
import torch
import lerobot
import huggingface_hub

print('==================================================')
print('🧬 PyTorch Version    :', torch.__version__)
print('🟢 CUDA Available      :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🖥️  GPU Device Name    :', torch.cuda.get_device_name(0))
    print('⚙️  CUDA Device Arch   :', torch.cuda.get_arch_list())
print('🤗 HF Hub Version     :', huggingface_hub.__version__)
print('🤖 LeRobot Version    :', lerobot.__version__)
print('==================================================')
"

  echo
  success "Environment setup complete!"
  echo "👉 To activate this environment, run:"
  echo "   conda activate $env_name"
  echo "══════════════════════════════════════════════════"
}

# ── Print one aligned "label : value" row with a status glyph ──────────────
# Usage: _dev_row <glyph> <label> <value>
_dev_row() {
  local glyph="$1" label="$2" value="$3"
  printf "   %b %-18s : %s\n" "$glyph" "$label" "$value"
}
_dev_ok()   { _dev_row "${GREEN}●${NC}"  "$1" "$2"; }
_dev_warn() { _dev_row "${YELLOW}●${NC}" "$1" "$2"; }
_dev_bad()  { _dev_row "${RED}●${NC}"    "$1" "$2"; }
_dev_dim()  { _dev_row "${BLUE}○${NC}"   "$1" "$2"; }

_dev_section() {
  echo
  echo -e "${BOLD}$1${NC}"
}

# ── GPU profile ───────────────────────────────────────────────────────────
# One place that decides what this machine can serve. gputool runs on Jetson
# iGPUs and on desktop cards from Ampere to Blackwell, and the sensible serving
# defaults differ by a lot between them. Everything below is derived from
# measurements on the bench rather than guessed:
#
#   RTX 5080  16 GB  sm_120  Qwen3.5-4B bf16 needs max-len 8192 / 0.92 / 16 seqs
#                            (32768 OOMs during CUDA-graph profiling)
#   RTX 4090  24 GB  sm_89   comfortably twice that
#   RTX 3090  24 GB  sm_86   same capacity, but no FP8 and no NVFP4 kernels
#   Jetson    shared memory  keep well clear of the system RAM budget
#
# Sets: GPU_PROFILE, GPU_VRAM_MB, GPU_CC, GPU_IS_JETSON,
#       PROF_MAX_LEN, PROF_GPU_MEM, PROF_MAX_SEQS, PROF_CTX
detect_gpu_profile() {
  GPU_IS_JETSON=0
  GPU_VRAM_MB=0
  GPU_CC=""
  GPU_PROFILE="unknown"

  # Jetson exposes a model string in the device tree and shares memory with the
  # CPU, so "free VRAM" is really "free system RAM".
  if [[ -r /proc/device-tree/model ]] && grep -qi jetson /proc/device-tree/model 2>/dev/null; then
    GPU_IS_JETSON=1
  elif [[ -f /etc/nv_tegra_release ]]; then
    GPU_IS_JETSON=1
  fi

  if command -v nvidia-smi &>/dev/null; then
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')
    GPU_CC=$(detect_gpu_compute_cap 2>/dev/null)
  fi
  [[ -z "$GPU_VRAM_MB" ]] && GPU_VRAM_MB=0

  if (( GPU_IS_JETSON == 1 )); then
    # Unified memory: leave headroom for the OS or the whole board swaps.
    GPU_PROFILE="jetson"
    PROF_MAX_LEN=4096;  PROF_GPU_MEM=0.75; PROF_MAX_SEQS=4;  PROF_CTX=4096
  elif (( GPU_VRAM_MB >= 40000 )); then
    GPU_PROFILE="large"          # A100/H100/6000-class
    PROF_MAX_LEN=32768; PROF_GPU_MEM=0.90; PROF_MAX_SEQS=64; PROF_CTX=32768
  elif (( GPU_VRAM_MB >= 22000 )); then
    GPU_PROFILE="24gb"           # 3090 / 4090
    PROF_MAX_LEN=16384; PROF_GPU_MEM=0.90; PROF_MAX_SEQS=32; PROF_CTX=16384
  elif (( GPU_VRAM_MB >= 14000 )); then
    GPU_PROFILE="16gb"           # 5080 / 4080 — measured defaults
    PROF_MAX_LEN=8192;  PROF_GPU_MEM=0.92; PROF_MAX_SEQS=16; PROF_CTX=8192
  elif (( GPU_VRAM_MB > 0 )); then
    GPU_PROFILE="small"
    PROF_MAX_LEN=4096;  PROF_GPU_MEM=0.90; PROF_MAX_SEQS=8;  PROF_CTX=4096
  else
    GPU_PROFILE="cpu"
    PROF_MAX_LEN=4096;  PROF_GPU_MEM=0.90; PROF_MAX_SEQS=4;  PROF_CTX=4096
  fi
  return 0
}

# True when the GPU can run NVFP4 (Blackwell, sm_120+).
gpu_supports_nvfp4() {
  local cc; cc=$(detect_gpu_compute_cap 2>/dev/null | tr -d '.')
  [[ -n "$cc" ]] && (( cc >= 120 ))
}
# True when the GPU has hardware FP8 (Ada sm_89+).
gpu_supports_fp8() {
  local cc; cc=$(detect_gpu_compute_cap 2>/dev/null | tr -d '.')
  [[ -n "$cc" ]] && (( cc >= 89 ))
}

# Print the tuned profile and what it implies for model choice.
show_gpu_profile() {
  detect_gpu_profile
  echo "══════════════════════════════════════════════════"
  echo -e "${BOLD}🎛️  Serving profile${NC}"
  echo "══════════════════════════════════════════════════"
  _dev_dim "GPU" "$(detect_gpu_name 2>/dev/null || echo unknown)"
  _dev_dim "VRAM" "${GPU_VRAM_MB} MiB"
  _dev_dim "Compute" "${GPU_CC:-unknown}$( (( GPU_IS_JETSON == 1 )) && echo "  (Jetson, unified memory)")"
  _dev_dim "Profile" "$GPU_PROFILE"
  echo
  _dev_dim "vLLM max-model-len" "$PROF_MAX_LEN"
  _dev_dim "vLLM gpu-mem-util" "$PROF_GPU_MEM"
  _dev_dim "vLLM max-num-seqs" "$PROF_MAX_SEQS"
  _dev_dim "llama.cpp ctx-size" "$PROF_CTX"
  echo
  gpu_supports_nvfp4 && _dev_ok "NVFP4" "supported — a 12B fits where a 4B bf16 would" \
                     || _dev_dim "NVFP4" "not supported on this GPU (needs Blackwell)"
  gpu_supports_fp8  && _dev_ok "FP8" "supported" \
                    || _dev_dim "FP8" "not supported (needs Ada or newer)"
  echo "══════════════════════════════════════════════════"
}

# ── PyTorch container ─────────────────────────────────────────────────────
# A persistent GPU container, in the shape sjsujetsontool uses on Jetson: one
# long-lived container you exec into, rather than a fresh `docker run` per
# command, so pip installs and background servers survive between steps.
#
# Image choice is not cosmetic. A cu12.4 image enumerates a Blackwell card
# perfectly and then fails every kernel launch with "no kernel image is
# available for execution on the device", which reads like a driver fault and
# is not one. Measured on this bench:
#
#   pytorch:2.9.1-cuda13.0   sm_100/120        5080: 21.1 TFLOPS fp32   OK
#   pytorch:2.11.0-cuda12.8  sm_75..90,100,120 5080: 20.2   4090: 35.4  OK
#   pytorch:2.5.1-cuda12.4   sm_50..90         5080: FAILS to launch
CONTAINER_IMAGE_CU130="pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime"
CONTAINER_IMAGE_CU128="pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
CONTAINER_NAME="${GPUTOOL_CONTAINER_NAME:-gputool-dev}"
CONTAINER_PREF_FILE="$GPUTOOL_DIR/container-image"
CONTAINER_MOUNT="/workspace"

# Best image for this GPU: CUDA 13 for Blackwell on a CUDA 13 driver, else 12.8
# (which also covers Ada's sm_89 through its sm_86 binaries).
default_container_image() {
  local cc drv
  cc=$(detect_gpu_compute_cap 2>/dev/null | tr -d '.')
  drv=$(detect_driver_cuda_version 2>/dev/null | tr -d '.')
  if [[ -n "$cc" && -n "$drv" ]] && (( cc >= 120 && drv >= 130 )); then
    echo "$CONTAINER_IMAGE_CU130"
  else
    echo "$CONTAINER_IMAGE_CU128"
  fi
}

container_image() {
  if [[ -n "${GPUTOOL_CONTAINER_IMAGE:-}" ]]; then echo "$GPUTOOL_CONTAINER_IMAGE"; return; fi
  if [[ -s "$CONTAINER_PREF_FILE" ]]; then head -n1 "$CONTAINER_PREF_FILE"; return; fi
  default_container_image
}

_container_running() { docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true; }
_container_exists()  { docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; }

_require_docker() {
  if ! command -v docker &>/dev/null; then
    error "docker is not installed on this node."
    echo "   The bench nodes need it installed by an admin — see the rebuild guide."
    return 1
  fi
  if ! docker ps >/dev/null 2>&1; then
    error "docker is installed but not usable by $(whoami)."
    echo "   An admin must run:  sudo usermod -aG docker $(whoami)   (then log out and back in)"
    return 1
  fi
  return 0
}

manage_container() {
  local action="${1:-status}"; shift 2>/dev/null || true
  local image; image=$(container_image)
  local work="${GPUTOOL_CONTAINER_WORKSPACE:-$HOME}"

  case "$action" in
    status)
      echo "══════════════════════════════════════════════════"
      echo -e "${BOLD}📦 PyTorch container${NC}"
      echo "══════════════════════════════════════════════════"
      _dev_dim "Image" "$image"
      _dev_dim "Best for this GPU" "$(default_container_image)"
      _dev_dim "Name" "$CONTAINER_NAME"
      _dev_dim "Workspace" "$work -> $CONTAINER_MOUNT"
      if ! command -v docker &>/dev/null; then
        _dev_bad "Docker" "not installed"
      elif ! docker ps >/dev/null 2>&1; then
        _dev_bad "Docker" "installed but not usable by $(whoami) (needs the docker group)"
      elif _container_running; then
        _dev_ok "State" "running — gputool container shell"
      elif _container_exists; then
        _dev_warn "State" "stopped — gputool container start"
      else
        _dev_warn "State" "not created — gputool container start"
      fi
      if docker image inspect "$image" >/dev/null 2>&1; then
        _dev_ok "Image pulled" "$(docker image inspect -f '{{.Size}}' "$image" 2>/dev/null | awk '{printf "%.1f GB", $1/1073741824}')"
      else
        _dev_warn "Image pulled" "no — first start will pull it"
      fi
      echo "══════════════════════════════════════════════════"
      ;;

    pull)
      _require_docker || return 1
      info "Pulling $image (several GB on first use)..."
      docker pull "$image" && success "Pulled." || { error "Pull failed."; return 1; }
      ;;

    start)
      _require_docker || return 1
      if _container_running; then success "Already running: $CONTAINER_NAME"; return 0; fi
      _container_exists && docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
      info "Image     : $image"
      info "Workspace : $work -> $CONTAINER_MOUNT"
      # --gpus all needs the NVIDIA container toolkit; say so plainly if absent.
      if ! docker run -d --name "$CONTAINER_NAME" --gpus all \
             -v "$work:$CONTAINER_MOUNT" -w "$CONTAINER_MOUNT" \
             --shm-size=8g --ipc=host \
             "$image" sleep infinity >/dev/null 2>/tmp/gputool-container.err; then
        error "Could not start the container:"
        sed 's/^/   /' /tmp/gputool-container.err | head -4
        grep -qi nvidia /tmp/gputool-container.err && \
          echo "   Looks like the NVIDIA container toolkit is missing — an admin must install it."
        return 1
      fi
      sleep 1
      success "Started $CONTAINER_NAME"
      echo "   Shell in : gputool container shell"
      echo "   One-shot : gputool container run nvidia-smi"
      ;;

    shell)
      _require_docker || return 1
      _container_running || { error "Not running. Start it: gputool container start"; return 1; }
      exec docker exec -it "$CONTAINER_NAME" bash
      ;;

    run|exec)
      _require_docker || return 1
      _container_running || { error "Not running. Start it: gputool container start"; return 1; }
      [[ $# -eq 0 ]] && { error "Nothing to run. Usage: gputool container run <command>"; return 1; }
      docker exec -w "$CONTAINER_MOUNT" "$CONTAINER_NAME" bash -lc "$*"
      ;;

    stop)
      _require_docker || return 1
      if _container_exists; then
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
        success "Stopped and removed $CONTAINER_NAME"
      else
        info "No container named $CONTAINER_NAME."
      fi
      ;;

    set)
      local choice="${1:-}"
      case "$choice" in
        cu130|13|cuda13) echo "$CONTAINER_IMAGE_CU130" > "$CONTAINER_PREF_FILE" ;;
        cu128|12|cuda12) echo "$CONTAINER_IMAGE_CU128" > "$CONTAINER_PREF_FILE" ;;
        auto|"")         rm -f "$CONTAINER_PREF_FILE" ;;
        *)               echo "$choice" > "$CONTAINER_PREF_FILE" ;;
      esac
      success "Image is now: $(container_image)"
      echo "   Recreate the container to pick it up: gputool container stop && gputool container start"
      ;;

    test)
      # Enumerating the GPU is not the test; launching a kernel is.
      _require_docker || return 1
      _container_running || { error "Not running. Start it: gputool container start"; return 1; }
      info "Running a real matmul inside the container..."
      docker exec "$CONTAINER_NAME" python -c "
import torch, time
print('  torch', torch.__version__, '| archs:', torch.cuda.get_arch_list()[-3:])
print('  device: sm_%d%d' % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
d = torch.device('cuda')
a = torch.randn(4096, 4096, device=d); b = torch.randn(4096, 4096, device=d)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): c = a @ b
torch.cuda.synchronize(); dt = (time.time() - t) / 20
print('  matmul OK - %.1f TFLOPS fp32' % (2*4096**3/dt/1e12))
"
      ;;

    *)
      error "Unknown container action: $action"
      echo "Usage: gputool container <status|pull|start|shell|run|test|stop|set>"
      return 1 ;;
  esac
}

# ── gputool-ai package ────────────────────────────────────────────────────
# The chat client and the agent sidecar used to be delivered by fetching a
# single chat.py from GitHub and by pointing at a repo checkout. That meant a
# node needed the whole edgeAI tree for agent mode, and the sidecar looked for
# an absolute /Developer path that exists only on the Jetson images.
#
# They are now one pip-installable package, so a bare node needs neither.
GPUTOOL_AI_SPEC="${GPUTOOL_AI_SPEC:-git+https://github.com/lkk688/edgeAI@main#subdirectory=packages/gputool-ai}"

# Echo the python that should run user-facing tools: the conda env if one is
# active or present, otherwise the system interpreter.
_ai_python() {
  local p
  for p in "$HOME/miniconda3/envs/${VLLM_DEFAULT_ENV:-py312}/bin/python" \
           "$HOME/miniconda/envs/${VLLM_DEFAULT_ENV:-py312}/bin/python"; do
    [[ -x "$p" ]] && { echo "$p"; return 0; }
  done
  command -v python3 2>/dev/null || command -v python 2>/dev/null
}

# Locate an installed console script, checking the env's bin as well as PATH.
_ai_script() {
  local name="$1" py bin
  command -v "$name" &>/dev/null && { command -v "$name"; return 0; }
  py=$(_ai_python); bin="$(dirname "$py")/$name"
  [[ -x "$bin" ]] && { echo "$bin"; return 0; }
  return 1
}

install_ai() {
  local extras="${1:-all}"
  echo "══════════════════════════════════════════════════"
  echo -e "${BOLD}📦 Installing gputool-ai (chat + agent)${NC}"
  echo "══════════════════════════════════════════════════"
  local py; py=$(_ai_python)
  [[ -z "$py" ]] && { error "No python interpreter found."; return 1; }
  info "Interpreter : $py"
  info "Extras      : $extras   (base | rich | agent | all)"
  info "Source      : $GPUTOOL_AI_SPEC"
  echo

  local spec="gputool-ai @ $GPUTOOL_AI_SPEC"
  [[ "$extras" != "base" ]] && spec="gputool-ai[$extras] @ $GPUTOOL_AI_SPEC"

  if ! "$py" -m pip install --upgrade "$spec"; then
    error "Install failed."
    echo "   Needs git and network access. For an offline node, copy the"
    echo "   packages/gputool-ai directory over and run:  pip install ./gputool-ai"
    return 1
  fi
  echo
  local c a
  c=$(_ai_script gputool-chat || echo "")
  a=$(_ai_script gputool-agent || echo "")
  [[ -n "$c" ]] && success "gputool-chat  -> $c"  || warn "gputool-chat not on PATH"
  [[ -n "$a" ]] && success "gputool-agent -> $a" || warn "gputool-agent not on PATH (install the 'agent' extra)"
  echo "   gputool chat / gputool agent now use these automatically."
}

# ── agent: FastAPI sidecar for the Agent Lab ─────────────────────────────
# Ported from sjsujetsontool, with the Jetson-specific paths made overridable so
# the same command works on a desktop checkout. Runs on the host (not a
# container) because it imports the edge_agent package and reads ~/.env.local.
AGENT_PORT="${AGENT_SIDECAR_PORT:-8002}"
AGENT_LOG="$GPUTOOL_DIR/agent.log"
AGENT_PIDFILE="$GPUTOOL_DIR/agent.pid"

_agent_dirs() {
  # Desktop checkouts live wherever the user cloned edgeAI; Jetson images put it
  # under /Developer. Try the env override, then a repo-relative guess, then the
  # Jetson path.
  local guess_root
  guess_root=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}" 2>/dev/null || echo .)")/.." 2>/dev/null && pwd)
  AGENT_APP_DIR="${GPUTOOL_AGENT_DIR:-}"
  AGENT_PKG_DIR="${GPUTOOL_EDGE_AGENT_DIR:-}"
  local c
  if [[ -z "$AGENT_APP_DIR" ]]; then
    for c in "$HOME/edgeAI/edgeLLM/nextjs-nemotron-app/agent_sidecar" \
             "$guess_root/edgeLLM/nextjs-nemotron-app/agent_sidecar" \
             "/Developer/edgeAI/edgeLLM/nextjs-nemotron-app/agent_sidecar"; do
      [[ -d "$c" ]] && { AGENT_APP_DIR="$c"; break; }
    done
  fi
  if [[ -z "$AGENT_PKG_DIR" ]]; then
    for c in "$HOME/edgeAI/edgeLLM/edge_agent" \
             "$guess_root/edgeLLM/edge_agent" \
             "/Developer/edgeAI/edgeLLM/edge_agent"; do
      [[ -d "$c" ]] && { AGENT_PKG_DIR="$c"; break; }
    done
  fi
}

agent_backend() {
  local action="${1:-status}"
  mkdir -p "$GPUTOOL_DIR"
  _agent_dirs

  case "$action" in
    status)
      echo "══════════════════════════════════════════════════"
      echo -e "${BOLD}🤖 Agent backend${NC}"
      echo "══════════════════════════════════════════════════"
      local ai_agent; ai_agent=$(_ai_script gputool-agent 2>/dev/null || echo "")
      if [[ -n "$ai_agent" ]]; then
        _dev_ok "Source" "installed package ($ai_agent)"
        _dev_dim "Workspace" "${GPUTOOL_WORKSPACE:-$(pwd)}"
      else
        _dev_warn "Source" "not installed — run: gputool install-ai"
        _dev_dim "Repo fallback" "${AGENT_APP_DIR:-<none found>}"
      fi
      _dev_dim "Port" "$AGENT_PORT"
      if wget -qO- --timeout=3 "http://localhost:$AGENT_PORT/health" >/dev/null 2>&1 \
         || curl -fs --max-time 3 "http://localhost:$AGENT_PORT/health" >/dev/null 2>&1; then
        _dev_ok "State" "running"
      else
        _dev_warn "State" "not running — start with: gputool agent start"
      fi
      echo "══════════════════════════════════════════════════"
      ;;
    start|bg|fg)
      # Prefer the pip-installed entry point: no repo checkout needed.
      local ai_agent; ai_agent=$(_ai_script gputool-agent 2>/dev/null || echo "")
      if [[ -n "$ai_agent" ]]; then
        info "Using installed gputool-agent ($ai_agent)"
        info "Workspace: ${GPUTOOL_WORKSPACE:-$(pwd)}"
        if [[ "$action" == "fg" ]]; then exec "$ai_agent"; fi
        nohup "$ai_agent" > "$AGENT_LOG" 2>&1 & echo $! > "$AGENT_PIDFILE"
        sleep 3
        if kill -0 "$(cat "$AGENT_PIDFILE" 2>/dev/null)" 2>/dev/null; then
          success "Agent backend started (PID $(cat "$AGENT_PIDFILE"))."
          echo "   🔗 http://localhost:$AGENT_PORT   📜 $AGENT_LOG"
        else
          error "Agent backend failed to start. Last lines:"
          tail -n 12 "$AGENT_LOG" 2>/dev/null | sed "s/^/   /"
          return 1
        fi
        return 0
      fi
      if [[ -z "$AGENT_APP_DIR" ]]; then
        error "Agent sidecar not found."
        echo "   Point gputool at it:  export GPUTOOL_AGENT_DIR=/path/to/agent_sidecar"
        return 1
      fi
      local py; py=$(command -v python3 || command -v python)
      [[ -z "$py" ]] && { error "python3 not found."; return 1; }
      # edge_agent is imported from the repo, so it has to be on PYTHONPATH.
      [[ -n "$AGENT_PKG_DIR" ]] && export PYTHONPATH="$(dirname "$AGENT_PKG_DIR"):${PYTHONPATH:-}"
      info "Sidecar : $AGENT_APP_DIR"
      info "Port    : $AGENT_PORT"
      if [[ "$action" == "fg" ]]; then
        cd "$AGENT_APP_DIR" || return 1
        exec "$py" -m uvicorn main:app --host 0.0.0.0 --port "$AGENT_PORT"
      fi
      ( cd "$AGENT_APP_DIR" && nohup "$py" -m uvicorn main:app --host 0.0.0.0 --port "$AGENT_PORT" \
          > "$AGENT_LOG" 2>&1 & echo $! > "$AGENT_PIDFILE" )
      sleep 3
      if kill -0 "$(cat "$AGENT_PIDFILE" 2>/dev/null)" 2>/dev/null; then
        success "Agent backend started (PID $(cat "$AGENT_PIDFILE"))."
        echo "   🔗 http://localhost:$AGENT_PORT   📜 $AGENT_LOG"
      else
        error "Agent backend failed to start. Last lines:"
        tail -n 12 "$AGENT_LOG" 2>/dev/null | sed 's/^/   /'
        return 1
      fi
      ;;
    test)
      # Drive every agent tool against a real model and check the filesystem,
      # not the model.s prose. Non-zero exit on any failure, so it fits CI.
      shift 2>/dev/null || true
      local tester; tester=$(_ai_script gputool-agent-test 2>/dev/null || echo "")
      if [[ -z "$tester" ]]; then
        error "gputool-agent-test not installed. Run: gputool install-ai"
        return 1
      fi
      "$tester" --agent-url "http://127.0.0.1:$AGENT_PORT" "$@"
      return $?
      ;;
    stop)
      local pid
      pid=$(cat "$AGENT_PIDFILE" 2>/dev/null)
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
        rm -f "$AGENT_PIDFILE"
        success "Agent backend stopped."
      else
        info "Agent backend was not running."
        rm -f "$AGENT_PIDFILE"
      fi
      ;;
    *)
      error "Unknown agent action: $action"
      echo "Usage: gputool agent <start|stop|status|fg|test>"
      return 1 ;;
  esac
}

# ── Shared HuggingFace cache over the campus network ──────────────────────
# The bench shares one large HF cache hosted on the RTX 4090 box. It is mounted
# per-user over SFTP with rclone + FUSE, so nothing here needs sudo. Re-reads of
# already-fetched blobs come from a size-capped local cache at full speed; only
# the first touch of a blob crosses the network.
HF_REMOTE_HOST="${HF_REMOTE_HOST:-10.31.81.235}"
HF_REMOTE_USER="${HF_REMOTE_USER:-lkk}"
HF_REMOTE_PATH="${HF_REMOTE_PATH:-/DATA10T/huggingface}"
HF_MOUNT="${HF_MOUNT:-$HOME/hf-shared}"
HF_VFS_MAX="${HF_VFS_MAX:-20G}"
RCLONE_BIN="$HOME/.local/bin/rclone"
RCLONE_CONF="$HOME/.config/rclone/rclone.conf"
HF_RC_BEGIN="# >>> gputool hf-cache >>>"
HF_RC_END="# <<< gputool hf-cache <<<"

_hf_mounted() { mount 2>/dev/null | grep -q " $HF_MOUNT "; }

# Install the rclone static binary into ~/.local/bin (no sudo, no package manager).
_hf_install_rclone() {
  if [[ -x "$RCLONE_BIN" ]]; then
    info "rclone already present ($("$RCLONE_BIN" version | head -n1))"
    return 0
  fi
  info "Downloading rclone (static binary, no sudo required)..."
  local tmp url
  tmp=$(mktemp -d)
  url="https://downloads.rclone.org/rclone-current-linux-amd64.zip"
  if command -v curl &>/dev/null; then
    curl -fsSL "$url" -o "$tmp/rclone.zip"
  elif command -v wget &>/dev/null; then
    wget -qO "$tmp/rclone.zip" "$url"
  else
    error "Need curl or wget to download rclone."; rm -rf "$tmp"; return 1
  fi
  if ! ( cd "$tmp" && unzip -qo rclone.zip ); then
    error "unzip failed (is the 'unzip' command available?)."; rm -rf "$tmp"; return 1
  fi
  mkdir -p "$HOME/.local/bin"
  cp "$tmp"/rclone-*-linux-amd64/rclone "$RCLONE_BIN" && chmod +x "$RCLONE_BIN"
  rm -rf "$tmp"
  success "Installed rclone $("$RCLONE_BIN" version | head -n1 | awk '{print $2}')"
}

# One-time preparation: FUSE check, rclone, an SSH key, and the remote definition.
_hf_setup() {
  echo "══════════════════════════════════════════════════"
  echo -e "${BOLD}🤗 Shared HF cache — setup${NC}"
  echo "══════════════════════════════════════════════════"

  # FUSE must be usable without root; fusermount is the setuid helper that allows it.
  if [[ ! -c /dev/fuse ]]; then
    error "/dev/fuse is missing — this host cannot mount FUSE filesystems."
    return 1
  fi
  if ! command -v fusermount3 &>/dev/null && ! command -v fusermount &>/dev/null; then
    error "fusermount is not installed — ask an admin for the 'fuse3' package."
    return 1
  fi
  success "FUSE is usable without sudo."

  _hf_install_rclone || return 1

  if [[ ! -f "$HOME/.ssh/id_ed25519" ]]; then
    info "Generating an SSH key for this host..."
    ssh-keygen -t ed25519 -N "" -C "$(whoami)@$(hostname)" -f "$HOME/.ssh/id_ed25519" >/dev/null 2>&1
    success "Key created at ~/.ssh/id_ed25519"
  fi

  # Rewrite only our [hf] stanza, leaving any other rclone remotes intact.
  mkdir -p "$(dirname "$RCLONE_CONF")"
  if [[ -f "$RCLONE_CONF" ]] && grep -q "^\[hf\]" "$RCLONE_CONF"; then
    awk 'BEGIN{skip=0} /^\[hf\]$/{skip=1; next} /^\[/{skip=0} skip==0{print}' \
      "$RCLONE_CONF" > "$RCLONE_CONF.tmp" && mv "$RCLONE_CONF.tmp" "$RCLONE_CONF"
  fi
  {
    echo "[hf]"
    echo "type = sftp"
    echo "host = $HF_REMOTE_HOST"
    echo "user = $HF_REMOTE_USER"
    echo "key_file = ~/.ssh/id_ed25519"
    echo "shell_type = unix"
    echo "known_hosts_file = none"
    echo "md5sum_command = md5sum"
    echo "sha1sum_command = sha1sum"
  } >> "$RCLONE_CONF"
  success "rclone remote 'hf' configured for $HF_REMOTE_USER@$HF_REMOTE_HOST"

  echo
  info "Testing access to the cache host..."
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        "$HF_REMOTE_USER@$HF_REMOTE_HOST" "test -d '$HF_REMOTE_PATH'" 2>/dev/null; then
    success "Reachable. Next: gputool hf-cache mount"
  else
    warn "Cannot log in to $HF_REMOTE_USER@$HF_REMOTE_HOST yet."
    echo
    echo "  Authorise this host once, from a machine that can already reach it:"
    echo
    echo -e "${CYAN}  ssh $HF_REMOTE_USER@$HF_REMOTE_HOST \"echo '$(cat "$HOME/.ssh/id_ed25519.pub")' >> ~/.ssh/authorized_keys\"${NC}"
    echo
    echo "  Then re-run: gputool hf-cache setup"
    return 1
  fi
}

_hf_mount() {
  if [[ ! -x "$RCLONE_BIN" ]]; then
    error "rclone not installed. Run: gputool hf-cache setup"
    return 1
  fi
  if _hf_mounted; then
    success "Already mounted at $HF_MOUNT"
    return 0
  fi
  mkdir -p "$HF_MOUNT"
  info "Mounting $HF_REMOTE_PATH from $HF_REMOTE_HOST ..."
  # --vfs-cache-mode full keeps re-reads local; the cap bounds local disk use.
  "$RCLONE_BIN" mount "hf:$HF_REMOTE_PATH" "$HF_MOUNT" \
    --vfs-cache-mode full --vfs-cache-max-size "$HF_VFS_MAX" \
    --dir-cache-time 24h --attr-timeout 1h \
    --daemon --daemon-wait 30s 2>/dev/null
  sleep 2
  if _hf_mounted; then
    success "Mounted at $HF_MOUNT"
    echo "   This shell only    : export HF_HOME=$HF_MOUNT"
    echo "   Every future shell : gputool hf-cache enable"
  else
    error "Mount failed. Verify setup with: gputool hf-cache setup"
    return 1
  fi
}

_hf_unmount() {
  if ! _hf_mounted; then
    info "Not mounted."
    return 0
  fi
  fusermount3 -u "$HF_MOUNT" 2>/dev/null || fusermount -u "$HF_MOUNT" 2>/dev/null
  sleep 1
  if _hf_mounted; then
    error "Could not unmount — a process is still using $HF_MOUNT"
    return 1
  fi
  success "Unmounted $HF_MOUNT"
}

# Persist HF_HOME and a login auto-mount into .bashrc, inside removable markers.
# Persist HF_HOME and a login auto-mount.
#
# The block is PREPENDED, not appended: Ubuntu's stock .bashrc returns early for
# non-interactive shells, and batch jobs arrive as `ssh node 'python train.py'`,
# which is non-interactive. Appending would leave HF_HOME unset for exactly the
# case that matters most.
_hf_enable() {
  local rc="$HOME/.bashrc"
  if grep -qF "$HF_RC_BEGIN" "$rc" 2>/dev/null; then
    info "Already enabled in $rc"
  else
    local tmp; tmp=$(mktemp)
    {
      echo "$HF_RC_BEGIN"
      echo "export HF_HOME=\"$HF_MOUNT\""
      echo "# Mount on demand. The lock keeps parallel logins from racing."
      echo "if [ -x \"\$HOME/.local/bin/gputool\" ] && ! mount 2>/dev/null | grep -q \" $HF_MOUNT \"; then"
      echo "  mkdir -p \"\$HOME/.gputool\" 2>/dev/null"
      echo "  if mkdir \"\$HOME/.gputool/hfmount.lock\" 2>/dev/null; then"
      echo "    ( \"\$HOME/.local/bin/gputool\" hf-cache mount >/dev/null 2>&1"
      echo "      rmdir \"\$HOME/.gputool/hfmount.lock\" 2>/dev/null ) &"
      echo "  fi"
      echo "fi"
      echo "$HF_RC_END"
      echo ""
      [[ -f "$rc" ]] && cat "$rc"
    } > "$tmp"
    mv "$tmp" "$rc"
    success "Enabled — HF_HOME=$HF_MOUNT in every shell, interactive or not."
  fi
  echo "   For the current shell: export HF_HOME=$HF_MOUNT"
}

_hf_disable() {
  local rc="$HOME/.bashrc"
  if ! grep -qF "$HF_RC_BEGIN" "$rc" 2>/dev/null; then
    info "Not enabled."
    return 0
  fi
  sed -i "\|$HF_RC_BEGIN|,\|$HF_RC_END|d" "$rc"
  success "Removed the hf-cache block from $rc"
  echo "   The mount itself is untouched — remove it with: gputool hf-cache unmount"
}

_hf_status() {
  echo "══════════════════════════════════════════════════"
  echo -e "${BOLD}🤗 Shared HF cache — status${NC}"
  echo "══════════════════════════════════════════════════"

  if [[ -x "$RCLONE_BIN" ]]; then
    _dev_ok "rclone" "$("$RCLONE_BIN" version | head -n1)"
  else
    _dev_bad "rclone" "not installed — run: gputool hf-cache setup"
  fi
  _dev_dim "Remote" "$HF_REMOTE_USER@$HF_REMOTE_HOST:$HF_REMOTE_PATH"

  if ssh -o BatchMode=yes -o ConnectTimeout=6 "$HF_REMOTE_USER@$HF_REMOTE_HOST" true 2>/dev/null; then
    _dev_ok "Reachable" "yes"
  else
    _dev_bad "Reachable" "no — run: gputool hf-cache setup"
  fi

  if _hf_mounted; then
    _dev_ok  "Mount" "$HF_MOUNT"
    _dev_dim "Capacity" "$(df -h "$HF_MOUNT" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')"
    _dev_dim "Shared models" "$(ls "$HF_MOUNT/hub" 2>/dev/null | grep -c '^models--') repos"
  else
    _dev_warn "Mount" "not mounted — run: gputool hf-cache mount"
  fi

  _dev_dim "Local VFS cache" "$(du -shx "$HOME/.cache/rclone" 2>/dev/null | cut -f1 || echo 0) used, cap $HF_VFS_MAX"

  if [[ "${HF_HOME:-}" == "$HF_MOUNT" ]]; then
    _dev_ok "HF_HOME" "$HF_HOME"
  else
    _dev_warn "HF_HOME" "${HF_HOME:-unset} — this shell is not using the shared cache"
  fi

  if grep -qF "$HF_RC_BEGIN" "$HOME/.bashrc" 2>/dev/null; then
    _dev_ok "Persisted" "yes, via ~/.bashrc"
  else
    _dev_dim "Persisted" "no — enable with: gputool hf-cache enable"
  fi
  echo "══════════════════════════════════════════════════"
}

hf_cache() {
  case "${1:-status}" in
    setup)          _hf_setup ;;
    mount)          _hf_mount ;;
    unmount|umount) _hf_unmount ;;
    enable)         _hf_enable ;;
    disable)        _hf_disable ;;
    status)         _hf_status ;;
    *)
      error "Unknown hf-cache subcommand: ${1:-}"
      echo "Valid subcommands: setup, mount, unmount, enable, disable, status"
      return 1
      ;;
  esac
}

# Show a complete device overview: GPU, driver health, disk, conda, gputool.
# Read-only, no sudo, no network unless --online is passed. This is the first
# command to run on an unfamiliar machine and the first check when a GPU job
# fails for no obvious reason.
device_check() {
  local check_online=0
  [[ "${1:-}" == "--online" || "${1:-}" == "-o" ]] && check_online=1

  local issues=0

  echo "══════════════════════════════════════════════════"
  echo -e "${BOLD}🖥️  Device Overview${NC}  ($(hostname))"
  echo "══════════════════════════════════════════════════"

  # ── Host ────────────────────────────────────────────────────────────────
  _dev_section "Host"
  local os_pretty="unknown"
  [[ -r /etc/os-release ]] && os_pretty=$(. /etc/os-release; echo "$PRETTY_NAME")
  _dev_dim "OS"        "$os_pretty"
  _dev_dim "Kernel"    "$(uname -r)"
  _dev_dim "Address"   "$(hostname -I 2>/dev/null | awk '{print $1}')"
  _dev_dim "Uptime"    "$(uptime -p 2>/dev/null | sed 's/^up //')"
  _dev_dim "CPU / RAM" "$(nproc) cores / $(free -g 2>/dev/null | awk '/^Mem:/{print $2}') GB"

  # ── GPU ─────────────────────────────────────────────────────────────────
  _dev_section "GPU"
  if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    local gname gdrv gcuda gmem gused gtemp gutil
    gname=$(detect_gpu_name 2>/dev/null)
    gdrv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
    gcuda=$(detect_driver_cuda_version 2>/dev/null)
    gmem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1)
    gused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -n1)
    gtemp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -n1)
    gutil=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -n1)
    _dev_ok  "Adapter"      "$gname"
    _dev_ok  "Driver"       "$gdrv  (CUDA $gcuda)"
    _dev_dim "Memory"       "$gused used of $gmem"
    _dev_dim "Load"         "${gutil:-n/a}  @  ${gtemp:-?}°C"
    local capability; capability=$(detect_gpu_compute_cap 2>/dev/null)
    [[ -n "$capability" ]] && _dev_dim "Compute cap" "sm_${capability//./}"

    # Who is holding the GPU right now
    local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    if [[ -n "$apps" ]]; then
      _dev_warn "In use by" "$(echo "$apps" | wc -l) process(es) — run 'nvidia-smi' for detail"
    else
      _dev_dim "In use by" "nothing — GPU is idle"
    fi
  else
    _dev_bad "Adapter" "nvidia-smi cannot reach the driver"
    issues=$((issues+1))
  fi

  # ── Driver health ───────────────────────────────────────────────────────
  # These four rows are what actually predict whether the GPU survives the
  # next kernel upgrade. A node can look fine today and lose the driver on
  # the next reboot if the module is prebuilt-only or gcc-12 is missing.
  _dev_section "Driver health"
  # WSL2 borrows the driver from the Windows host: there is no nvidia kernel
  # module and no DKMS tree, so those two checks would be false alarms there.
  if grep -qi microsoft /proc/version 2>/dev/null; then
    _dev_dim "Kernel module" "n/a — WSL2 uses the Windows host driver"
    _dev_dim "DKMS" "n/a — managed on the Windows side"
  else
    local nmods; nmods=$(lsmod 2>/dev/null | grep -c '^nvidia')
    if (( nmods > 0 )); then
      _dev_ok "Kernel module" "$nmods loaded"
    else
      _dev_bad "Kernel module" "not loaded — try: sudo modprobe nvidia"
      issues=$((issues+1))
    fi

    local dkms_line; dkms_line=$(dkms status 2>/dev/null | grep -i nvidia | head -n1)
    if [[ -n "$dkms_line" ]]; then
      _dev_ok "DKMS" "$(echo "$dkms_line" | cut -c1-52)"
    else
      _dev_warn "DKMS" "no NVIDIA module registered — driver is prebuilt-only"
      issues=$((issues+1))
    fi
  fi

  local kgcc; kgcc=$(sed -nE 's/.*gcc-([0-9]+).*/\1/p' /proc/version 2>/dev/null | head -n1)
  if [[ -n "$kgcc" ]]; then
    if [[ -x "/usr/bin/gcc-$kgcc" ]]; then
      _dev_ok "Build toolchain" "gcc-$kgcc present (kernel was built with it)"
    else
      _dev_bad "Build toolchain" "gcc-$kgcc MISSING — DKMS rebuilds will fail"
      issues=$((issues+1))
    fi
  fi

  local ncc; ncc=$(detect_nvcc_version 2>/dev/null)
  [[ -n "$ncc" ]] && _dev_dim "nvcc toolkit" "$ncc" || _dev_dim "nvcc toolkit" "not installed (not required for PyTorch)"

  # ── Storage ─────────────────────────────────────────────────────────────
  _dev_section "Storage"
  local avail_g total_s used_pct
  avail_g=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc '0-9')
  total_s=$(df -h / 2>/dev/null | awk 'NR==2{print $2}')
  used_pct=$(df -h / 2>/dev/null | awk 'NR==2{print $5}')
  if   (( avail_g < 15 )); then _dev_bad  "Root volume" "${avail_g}G free of $total_s ($used_pct used) — apt will fail"; issues=$((issues+1))
  elif (( avail_g < 50 )); then _dev_warn "Root volume" "${avail_g}G free of $total_s ($used_pct used) — running low"
  else                          _dev_ok   "Root volume" "${avail_g}G free of $total_s ($used_pct used)"
  fi
  _dev_dim "Home"  "$(du -shx "$HOME" 2>/dev/null | cut -f1) in $HOME"
  _dev_dim "Caches" "$(du -shx "$HOME/.cache" 2>/dev/null | cut -f1 || echo 0) reclaimable in ~/.cache"

  # ── Conda ───────────────────────────────────────────────────────────────
  _dev_section "Conda"
  local conda_root=""
  for p in "$HOME/miniconda3" "$HOME/miniconda" "$HOME/anaconda3" "/opt/conda"; do
    [[ -x "$p/bin/conda" ]] && { conda_root="$p"; break; }
  done
  if [[ -n "$conda_root" ]]; then
    _dev_ok  "Install" "$("$conda_root/bin/conda" --version 2>/dev/null) at $conda_root"
    local envs; envs=$(ls -1 "$conda_root/envs" 2>/dev/null)
    if [[ -n "$envs" ]]; then
      _dev_dim "Environments" "$(echo "$envs" | paste -sd, | sed 's/,/, /g')"
    else
      _dev_warn "Environments" "none created yet — run: gputool setup-env py312 3.12"
    fi
    [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && _dev_dim "Active" "$CONDA_DEFAULT_ENV"
  else
    _dev_warn "Install" "not found — run: gputool install-conda"
    issues=$((issues+1))
  fi

  # ── gputool ─────────────────────────────────────────────────────────────
  _dev_section "gputool"
  _dev_ok "Version" "$SCRIPT_VERSION"
  if [[ -x "$SCRIPT_PATH" ]]; then
    _dev_dim "Installed at" "$SCRIPT_PATH"
  else
    _dev_warn "Installed at" "not in ~/.local/bin — run: gputool install"
  fi
  if (( check_online )); then
    local remote_ver
    remote_ver=$(curl -fsSL --max-time 8 "$SCRIPT_URL" 2>/dev/null | sed -nE 's/^SCRIPT_VERSION="(.*)"/\1/p' | head -n1)
    if [[ -z "$remote_ver" ]]; then
      _dev_warn "Latest" "could not reach GitHub"
    elif [[ "$remote_ver" == "$SCRIPT_VERSION" ]]; then
      _dev_ok "Latest" "$remote_ver — up to date"
    else
      _dev_warn "Latest" "$remote_ver available — run: gputool update-script"
    fi
  fi

  # ── Verdict ─────────────────────────────────────────────────────────────
  echo
  echo "══════════════════════════════════════════════════"
  if (( issues == 0 )); then
    success "Device is healthy — GPU, driver, storage and Conda all check out."
  else
    warn "$issues item(s) need attention — see the red and yellow rows above."
  fi
  echo "══════════════════════════════════════════════════"
  return 0
}

# Run system diagnostic checks (GPU, Conda, PyTorch, Hugging Face, LeRobot, Tailscale)
system_check() {
  local env_name="${1:-lerobot}"

  echo "══════════════════════════════════════════════════"
  echo "🖥️  System Hardware Check"
  echo "══════════════════════════════════════════════════"
  if command -v nvidia-smi &>/dev/null; then
    success "NVIDIA Driver found via nvidia-smi."
    local nv_info
    nv_info=$(nvidia-smi 2>/dev/null)
    if [[ -n "$nv_info" ]]; then
      local gpu_name
      gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
      local drv_ver
      drv_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
      local cuda_ver
      cuda_ver=$(echo "$nv_info" | grep -o "CUDA Version: [0-9.]*" | head -n 1 | awk '{print $3}')
      echo "   • GPU Name       : $gpu_name"
      echo "   • Driver Version : $drv_ver"
      echo "   • CUDA Version   : $cuda_ver"
    else
      warn "NVIDIA driver is present but nvidia-smi query failed."
    fi
  else
    warn "nvidia-smi not found. GPU driver might not be installed or in PATH."
  fi

  echo
  echo "══════════════════════════════════════════════════"
  echo "🐍 Conda Environment Check"
  echo "══════════════════════════════════════════════════"

  # Find Conda
  local CONDA_SH=""
  for path in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/home/010796032@SJSUAD/miniconda3/etc/profile.d/conda.sh" \
    "/home/$USER/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$path" ]]; then
      CONDA_SH="$path"
      break
    fi
  done

  if [[ -n "$CONDA_SH" ]]; then
    source "$CONDA_SH"
  fi

  if ! command -v conda &>/dev/null; then
    error "Conda command not found. Environment cannot be checked."
    exit 1
  fi

  success "Conda is installed."
  echo "   • Conda Path    : $(which conda)"
  echo "   • Conda Version : $(conda --version | awk '{print $2}')"

  if ! conda env list | grep -q "^$env_name "; then
    error "Conda environment '$env_name' does not exist."
    echo "   👉 You can set it up using: gputool setup-lerobot $env_name"
    exit 1
  fi
  success "Conda environment '$env_name' exists."

  info "Running Python diagnostic checks in Conda env '$env_name'..."

  # Run Python verification script via temporary file because conda run doesn't handle stdin redirection well
  local py_check_file="$GPUTOOL_DIR/syscheck.py"
  mkdir -p "$GPUTOOL_DIR"
  cat << 'EOF' > "$py_check_file"
import sys

def print_section(title):
    print(f"\n\033[1;35m════ {title} ════\033[0m")

def print_row(label, value, success=True):
    color = "\033[0;32m" if success else "\033[0;31m"
    icon = "✅" if success else "❌"
    print(f"   • {label:<22} : {color}{value:<30}\033[0m {icon}")

# 1. Check PyTorch & CUDA
print_section("PyTorch & CUDA Diagnostic")
try:
    import torch
    torch_ok = True
    torch_version = torch.__version__
    cuda_ok = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_ok else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    gpu_capability = str(torch.cuda.get_device_capability(0)) if cuda_ok else "N/A"
except ImportError:
    torch_ok = False
    torch_version = "Not Installed"
    cuda_ok = False
    cuda_version = "N/A"
    gpu_name = "N/A"
    gpu_capability = "N/A"

print_row("PyTorch Installed", torch_version, torch_ok)
print_row("CUDA Available", str(cuda_ok), cuda_ok)
if cuda_ok:
    print_row("CUDA Backend Ver", cuda_version, True)
    print_row("GPU Device Name", gpu_name, True)
    print_row("Compute Capability", gpu_capability, True)

# 2. Check Hugging Face Hub
print_section("Hugging Face Hub Diagnostic")
try:
    import huggingface_hub
    hf_ok = True
    hf_version = huggingface_hub.__version__
    try:
        token = huggingface_hub.get_token()
        hf_logged_in = "Logged In" if token else "Not Logged In"
    except Exception:
        hf_logged_in = "Not Logged In"
    
    # Check connectivity to HF
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
        hf_conn = "Connected"
        hf_conn_ok = True
    except Exception:
        hf_conn = "Offline / Connection Failed"
        hf_conn_ok = False
except ImportError:
    hf_ok = False
    hf_version = "Not Installed"
    hf_logged_in = "N/A"
    hf_conn = "N/A"
    hf_conn_ok = False

print_row("HF Hub Installed", hf_version, hf_ok)
print_row("HF Auth Status", hf_logged_in, hf_ok)
print_row("HF Hub Connectivity", hf_conn, hf_conn_ok if hf_ok else False)

# 3. Check LeRobot
print_section("LeRobot Diagnostic")
try:
    import lerobot
    lerobot_ok = True
    lerobot_version = lerobot.__version__
    
    # Check simulator imports
    sims = []
    for sim_pkg in ['gymnasium', 'mujoco', 'h5py']:
        try:
            __import__(sim_pkg)
            sims.append(f"{sim_pkg}(OK)")
        except ImportError:
            sims.append(f"{sim_pkg}(Missing)")
    sim_status = ", ".join(sims)
except ImportError:
    lerobot_ok = False
    lerobot_version = "Not Installed"
    sim_status = "N/A"

print_row("LeRobot Installed", lerobot_version, lerobot_ok)
if lerobot_ok:
    print_row("Simulation Packages", sim_status, "Missing" not in sim_status)
EOF

  conda run -n "$env_name" python3 "$py_check_file"
  rm -f "$py_check_file"
  echo
  echo "══════════════════════════════════════════════════"
  echo "🌐 Userspace Tailscale VPN & Proxy Check"
  echo "══════════════════════════════════════════════════"
  if is_daemon_running; then
    success "tailscaled background daemon is running."
    local TS_PID
    TS_PID=$(cat "$TS_PID_FILE" 2>/dev/null)
    echo "   • Daemon PID      : $TS_PID"
    
    # Check if proxy port 1055 is active
    local proxy_listening=false
    if command -v ss &>/dev/null; then
      if ss -tuln | grep -q ":1055 "; then
        proxy_listening=true
      fi
    elif command -v netstat &>/dev/null; then
      if netstat -tuln | grep -q ":1055 "; then
        proxy_listening=true
      fi
    fi
    # Python fallback check for port 1055
    if [[ "$proxy_listening" == "false" ]]; then
      if python3 -c "import socket; s = socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 1055))" &>/dev/null; then
        proxy_listening=true
      fi
    fi

    if [[ "$proxy_listening" == "true" ]]; then
      success "Proxy port 1055 is listening."
    else
      warn "Proxy port 1055 is NOT listening."
    fi
    
    # Get tailscale IP
    if [[ -f "$TAILSCALE_DIR/tailscale" ]]; then
      local TS_IPS
      TS_IPS=$("$TAILSCALE_DIR/tailscale" --socket="$TS_SOCKET" status --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(d.get('TailscaleIPs',[])))" 2>/dev/null)
      if [[ -n "$TS_IPS" ]]; then
        echo "   • Tailscale IPs   : $TS_IPS"
      else
        warn "Could not retrieve Tailscale IPs (disconnected or starting)."
      fi
    fi
  else
    warn "tailscaled daemon is not running."
    echo "   💡 Start with: gputool tailscale up"
  fi
  echo "══════════════════════════════════════════════════"
}

# Compile and set up llama.cpp with CUDA support
setup_llamacpp() {
  local env_name="${1:-lerobot}"
  
  echo "══════════════════════════════════════════════════"
  echo "🔧 Compiling & Setting up llama.cpp"
  echo "══════════════════════════════════════════════════"
  
  # Find Conda
  local CONDA_SH=""
  for path in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/home/010796032@SJSUAD/miniconda3/etc/profile.d/conda.sh" \
    "/home/$USER/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$path" ]]; then
      CONDA_SH="$path"
      break
    fi
  done

  if [[ -n "$CONDA_SH" ]]; then
    source "$CONDA_SH"
  fi

  if ! command -v conda &>/dev/null; then
    error "Conda command not found. Cannot configure build environment."
    exit 1
  fi

  if ! conda env list | grep -q "^$env_name "; then
    error "Conda environment '$env_name' does not exist."
    echo "   💡 Please create it or specify a valid env name."
    exit 1
  fi

  # Clone llama.cpp
  mkdir -p "$GPUTOOL_DIR"
  local src_dir="$GPUTOOL_DIR/llamacpp-src"
  if [[ ! -d "$src_dir" ]]; then
    info "Cloning llama.cpp repository..."
    if ! git clone --depth=1 https://github.com/ggerganov/llama.cpp.git "$src_dir"; then
      error "Failed to clone llama.cpp repository."
      exit 1
    fi
  else
    info "llama.cpp source directory already exists at $src_dir. Updating..."
    cd "$src_dir" && git pull && cd - &>/dev/null
  fi

  # Setup CUDA path for compiler config
  export PATH="/usr/local/cuda/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
  export CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

  # Find nvcc
  local nvcc_bin=""
  if command -v nvcc &>/dev/null; then
    nvcc_bin=$(command -v nvcc)
  elif [[ -f "/usr/local/cuda/bin/nvcc" ]]; then
    nvcc_bin="/usr/local/cuda/bin/nvcc"
  fi

  if [[ -z "$nvcc_bin" ]]; then
    error "nvcc compiler not found in PATH or /usr/local/cuda/bin."
    echo "   👉 Please ensure NVIDIA CUDA Toolkit is installed."
    exit 1
  fi
  local nvcc_ver gpu_name gpu_cc
  nvcc_ver=$(detect_nvcc_version 2>/dev/null || true)
  gpu_name=$(detect_gpu_name 2>/dev/null || true)
  gpu_cc=$(detect_gpu_compute_cap 2>/dev/null || true)
  success "Found CUDA compiler: $nvcc_bin (CUDA ${nvcc_ver:-unknown})"
  echo "   • GPU Detected      : ${gpu_name:-None}"
  echo "   • Compute Capability: ${gpu_cc:-Unknown}"

  # --- Ensure required build tools are present inside the conda env ---
  info "Checking build dependencies (cmake, ninja) in env '$env_name'..."
  # Resolve a cmake binary. Prefer the system one; only fall back to the conda
  # env's copy, and then call it by absolute path so conda's compilers never
  # get onto PATH ahead of the system toolchain (see the host-compiler note below).
  local cmake_bin=""
  [[ -x /usr/bin/cmake ]] && cmake_bin=/usr/bin/cmake
  if [[ -z "$cmake_bin" ]]; then
    if ! ensure_conda_tool "$env_name" cmake "cmake" "cmake"; then
      error "Cannot continue without cmake."
      exit 1
    fi
    cmake_bin=$(conda run -n "$env_name" bash -lc 'command -v cmake' 2>/dev/null | tr -d '\r' | tail -n1)
    [[ -x "$cmake_bin" ]] || { error "cmake installed but could not be located."; exit 1; }
  fi
  info "cmake          : $cmake_bin"
  # Ninja is optional but greatly speeds up the build; a failure is non-fatal.
  # Because we now invoke cmake directly rather than through `conda run`, a ninja
  # that lives only inside the conda env is not on PATH — so hand cmake its
  # absolute path, or fall back to Make rather than failing to configure.
  local use_ninja=0 ninja_bin=""
  if [[ -x /usr/bin/ninja ]]; then
    ninja_bin=/usr/bin/ninja; use_ninja=1
  elif ensure_conda_tool "$env_name" ninja "ninja"; then
    ninja_bin=$(conda run -n "$env_name" bash -lc 'command -v ninja' 2>/dev/null | tr -d '\r' | tail -n1)
    [[ -x "$ninja_bin" ]] && use_ninja=1 || ninja_bin=""
  fi
  if (( use_ninja == 0 )); then
    warn "Proceeding without Ninja; will use the default Make generator."
  fi

  local build_dir="$src_dir/build"
  local generator_args=()
  if (( use_ninja == 1 )); then
    generator_args=(-G Ninja -DCMAKE_MAKE_PROGRAM="$ninja_bin")
  fi

  # A pre-existing build dir created with a different generator makes cmake abort.
  # Wipe it if the cached generator no longer matches what we are about to use.
  if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    local cached_gen want_gen
    cached_gen=$(grep -E '^CMAKE_GENERATOR:' "$build_dir/CMakeCache.txt" 2>/dev/null | cut -d= -f2-)
    want_gen=$([[ "$use_ninja" -eq 1 ]] && echo "Ninja" || echo "Unix Makefiles")
    if [[ -n "$cached_gen" && "$cached_gen" != "$want_gen" ]]; then
      warn "Build generator changed ('$cached_gen' -> '$want_gen'). Clearing stale build directory..."
      rm -rf "$build_dir"
    fi
  fi

  # Pick host compilers explicitly.
  #
  # Conda ships its own gcc (11.2, built against an older glibc). Letting it lead
  # the PATH works on Ubuntu 22.04 but fails on 24.04 with
  #   undefined reference to `__libc_csu_fini'
  # because that symbol was removed in glibc 2.34. So build with the SYSTEM
  # compiler and pass it to cmake by absolute path. nvcc is also picky about host
  # gcc versions, so prefer a known-good pair over whatever `cc` happens to be.
  local host_cc="" host_cxx=""
  local v
  for v in 13 12 11; do
    if [[ -x "/usr/bin/gcc-$v" && -x "/usr/bin/g++-$v" ]]; then
      host_cc="/usr/bin/gcc-$v"; host_cxx="/usr/bin/g++-$v"; break
    fi
  done
  if [[ -z "$host_cc" ]] && [[ -x /usr/bin/gcc && -x /usr/bin/g++ ]]; then
    host_cc=/usr/bin/gcc; host_cxx=/usr/bin/g++
  fi
  local compiler_args=()
  if [[ -n "$host_cc" ]]; then
    compiler_args=(
      -DCMAKE_C_COMPILER="$host_cc"
      -DCMAKE_CXX_COMPILER="$host_cxx"
      -DCMAKE_CUDA_HOST_COMPILER="$host_cc"
    )
    info "Host compiler  : $host_cc (system toolchain, not conda's)"
  else
    warn "No system gcc/g++ pair found; falling back to whatever cmake picks."
  fi

  # Build only for this machine's GPU. Compiling every architecture is slow and
  # is what makes an unattended build look hung.
  local cc_arch arch_args=()
  cc_arch=$(detect_gpu_compute_cap 2>/dev/null | tr -d '.')
  if [[ -n "$cc_arch" ]]; then
    arch_args=(-DCMAKE_CUDA_ARCHITECTURES="$cc_arch")
    info "CUDA arch      : sm_$cc_arch (this GPU only)"
  fi

  info "Configuring build with CUDA support enabled${use_ninja:+ (Ninja generator)}..."
  if ! "$cmake_bin" -S "$src_dir" -B "$build_dir" "${generator_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_COMPILER="$nvcc_bin" \
    "${compiler_args[@]}" "${arch_args[@]}"; then
    warn "CMake configuration failed. Wiping build directory and retrying once..."
    rm -rf "$build_dir"
    if ! "$cmake_bin" -S "$src_dir" -B "$build_dir" "${generator_args[@]}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DLLAMA_CURL=OFF \
      -DCMAKE_CUDA_COMPILER="$nvcc_bin" \
      "${compiler_args[@]}" "${arch_args[@]}"; then
      error "CMake configuration failed."
      exit 1
    fi
  fi

  # Compile release target.
  #
  # Parallelism is capped by AVAILABLE RAM, not core count. Each CUDA translation
  # unit in ggml-cuda can take 2+ GB in nvcc; -j20 on a 30 GB node exhausts memory
  # and the kernel OOM-killer picks sshd, taking the machine off the network with
  # no way back short of a power cycle. Budget ~4 GB of headroom per job.
  local num_jobs cores avail_gb mem_jobs
  cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  avail_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')
  [[ -z "$avail_gb" || "$avail_gb" -lt 1 ]] && avail_gb=4
  mem_jobs=$(( avail_gb / 4 ))
  (( mem_jobs < 2 )) && mem_jobs=2
  if (( cores < mem_jobs )); then num_jobs=$cores; else num_jobs=$mem_jobs; fi
  info "Compiling llama.cpp with -j$num_jobs (${cores} cores, ${avail_gb} GB RAM free)"
  # Keep a full log and surface the real compiler error on failure. Piping to tee
  # makes $? the exit status of tee, so the build result has to come from
  # PIPESTATUS or a failed build reads as a successful one.
  local build_log="$build_dir/gputool-build.log"
  "$cmake_bin" --build "$build_dir" --config Release -j"$num_jobs" 2>&1 | tee "$build_log"
  if (( ${PIPESTATUS[0]} != 0 )); then
    error "llama.cpp compilation failed. First errors from the build:"
    grep -iE "error:|undefined reference|No such file or directory|unsupported" "$build_log" \
      | head -n 12 | cut -c1-160 | sed 's/^/   /'
    echo "   Full log: $build_log"
    exit 1
  fi

  # Install binaries AND their shared libraries so they run standalone
  # (the build tree bakes in absolute RPATHs; copying the .so files next to the
  #  binaries plus exporting LD_LIBRARY_PATH at serve time keeps them portable).
  mkdir -p "$GPUTOOL_DIR/bin"
  if [[ -f "$build_dir/bin/llama-cli" && -f "$build_dir/bin/llama-server" ]]; then
    cp "$build_dir/bin/llama-cli" "$build_dir/bin/llama-server" "$GPUTOOL_DIR/bin/"
    # Copy any shared libraries produced by the build (libllama, libggml*, etc.).
    local lib_count=0
    shopt -s nullglob
    for so in "$build_dir/bin/"*.so*; do
      cp -P "$so" "$GPUTOOL_DIR/bin/" && ((lib_count++))
    done
    shopt -u nullglob
    chmod +x "$GPUTOOL_DIR/bin/llama-cli" "$GPUTOOL_DIR/bin/llama-server"
    success "llama.cpp compiled successfully!"
    echo "   Installed binaries:"
    echo "   • $GPUTOOL_DIR/bin/llama-cli"
    echo "   • $GPUTOOL_DIR/bin/llama-server"
    echo "   • Shared libraries copied: $lib_count"
    # Quick sanity check: confirm the CUDA backend can enumerate the GPU.
    info "Verifying CUDA backend (listing devices)..."
    if LD_LIBRARY_PATH="$GPUTOOL_DIR/bin:${LD_LIBRARY_PATH:-}" \
        "$GPUTOOL_DIR/bin/llama-cli" --list-devices 2>/dev/null | grep -qiE 'CUDA[0-9]'; then
      success "CUDA backend active — GPU is visible to llama.cpp."
    else
      warn "Could not confirm a CUDA device via --list-devices. The binary built, but verify GPU drivers."
    fi
  else
    error "Compiled binaries not found where expected in build output."
    exit 1
  fi
  echo "══════════════════════════════════════════════════"
}

# Download GGUF models using huggingface_hub inside conda env
download_model() {
  local repo_id="${1:-unsloth/Qwen3.5-9B-GGUF}"
  local filename="${2:-Qwen3.5-9B-UD-Q6_K_XL.gguf}"
  local env_name="${3:-lerobot}"
  
  echo "══════════════════════════════════════════════════"
  echo "📥 Downloading GGUF Model from Hugging Face"
  echo "══════════════════════════════════════════════════"
  echo "   Repo ID    : $repo_id"
  echo "   Filename   : $filename"
  echo "   Target Dir : $GPUTOOL_DIR/models"
  echo
  
  # Find Conda
  local CONDA_SH=""
  for path in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/home/010796032@SJSUAD/miniconda3/etc/profile.d/conda.sh" \
    "/home/$USER/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$path" ]]; then
      CONDA_SH="$path"
      break
    fi
  done

  if [[ -n "$CONDA_SH" ]]; then
    source "$CONDA_SH"
  fi

  if ! command -v conda &>/dev/null; then
    error "Conda command not found. Cannot run huggingface downloader."
    exit 1
  fi

  # Check if env exists
  if ! conda env list | grep -q "^$env_name "; then
    error "Conda environment '$env_name' does not exist."
    exit 1
  fi

  # Ensure huggingface_hub is installed in the conda environment
  if ! conda run -n "$env_name" python3 -c "import huggingface_hub" &>/dev/null; then
    info "Installing huggingface_hub in Conda env '$env_name' first..."
    conda run -n "$env_name" pip install huggingface_hub
  fi

  # Create models directory
  local models_dir="$GPUTOOL_DIR/models"
  mkdir -p "$models_dir"

  info "Starting download via huggingface_hub API (with symlink resolution)..."
  warn "⏳ This model is large. Download speed depends on the network interface."
  warn "   Please do not interrupt or close the terminal."
  
  local download_py="$GPUTOOL_DIR/hf_download.py"
  cat << EOF > "$download_py"
import sys
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id="${repo_id}",
        filename="${filename}",
        local_dir="${models_dir}",
        local_dir_use_symlinks=False
    )
    print("SUCCESS_PATH:" + path)
except Exception as e:
    print("ERROR:" + str(e), file=sys.stderr)
    sys.exit(1)
EOF

  local download_out
  if download_out=$(conda run -n "$env_name" python3 "$download_py" 2>&1); then
    rm -f "$download_py"
    local actual_path
    actual_path=$(echo "$download_out" | grep "SUCCESS_PATH:" | cut -d':' -f2-)
    success "Download complete!"
    echo "   Model saved at: $actual_path"
  else
    rm -f "$download_py"
    error "Download failed."
    echo "$download_out"
    exit 1
  fi
  echo "══════════════════════════════════════════════════"
}

# Start, stop, or check status of llama-server
serve_llamacpp() {
  local action="${1:-status}"
  shift 2>/dev/null || true

  # Parse remaining args: flags control run mode / bind host / auth / vision; positionals are [model] [port].
  local run_mode="background"
  local host="0.0.0.0"   # bind all interfaces by default so peers on the LAN can reach it
  local api_key="${GPUTOOL_LLAMA_API_KEY:-}"   # optional bearer token; env var provides a default
  local mmproj=""        # multimodal projector path; "" = auto-detect, "none" = disable
  detect_gpu_profile
  local ctx_size="$PROF_CTX"   # from the GPU profile; override with --ctx-size
  # Multi-token prediction. An MTP-enabled GGUF drafts several tokens per step and
  # verifies them in one pass, worth ~1.6x on decode. llama.cpp does not support
  # MTP together with parallel slots, so --mtp forces -np 1: fast for one user,
  # useless for a shared node. Off unless asked for.
  local spec_type="none"
  local spec_n="6"
  local positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--foreground) run_mode="foreground" ;;
      -d|--background) run_mode="background" ;;
      --host) host="${2:-0.0.0.0}"; shift ;;
      --host=*) host="${1#*=}" ;;
      --api-key) api_key="${2:-}"; shift ;;
      --api-key=*) api_key="${1#*=}" ;;
      --mmproj) mmproj="${2:-}"; shift ;;
      --mmproj=*) mmproj="${1#*=}" ;;
      --no-mmproj) mmproj="none" ;;
      --ctx-size|-c) ctx_size="${2:-32768}"; shift ;;
      --ctx-size=*) ctx_size="${1#*=}" ;;
      --mtp) spec_type="draft-mtp" ;;
      --spec-type) spec_type="${2:-none}"; shift ;;
      --spec-type=*) spec_type="${1#*=}" ;;
      --spec-n) spec_n="${2:-6}"; shift ;;
      --spec-n=*) spec_n="${1#*=}" ;;
      *) positional+=("$1") ;;
    esac
    shift
  done
  local model_param="${positional[0]:-Qwen3.5-9B-UD-Q6_K_XL.gguf}"
  local port="${positional[1]:-8080}"

  local pid_file="$GPUTOOL_DIR/llama-server.pid"
  local log_file="$GPUTOOL_DIR/llama-server.log"
  local server_bin="$GPUTOOL_DIR/bin/llama-server"

  case "$action" in
    start)
      echo "══════════════════════════════════════════════════"
      echo "🚀 Starting llama-server"
      echo "══════════════════════════════════════════════════"
      
      if [[ ! -f "$server_bin" ]]; then
        error "llama-server binary not found. Please run: gputool setup-llamacpp"
        exit 1
      fi

      # Check if already running
      if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid=$(cat "$pid_file" 2>/dev/null)
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
          warn "llama-server is already running (PID: $old_pid)."
          echo "   If you want to restart, run: gputool serve-llamacpp stop"
          exit 0
        fi
      fi

      # Resolve model path
      local model_path=""
      if [[ -f "$model_param" ]]; then
        model_path="$model_param"
      elif [[ -f "$GPUTOOL_DIR/models/$model_param" ]]; then
        model_path="$GPUTOOL_DIR/models/$model_param"
      else
        error "Model file not found: '$model_param'"
        echo "   Looked locally and in: $GPUTOOL_DIR/models/"
        echo "   👉 You can download the default model using: gputool download-model"
        exit 1
      fi

      # Resolve the multimodal projector (vision). Auto-detect an mmproj*.gguf next
      # to the model unless the user passed --mmproj <path> or --no-mmproj.
      # A model + matching mmproj turns llama-server into a vision (image) server.
      local mmproj_path=""
      if [[ "$mmproj" == "none" ]]; then
        :
      elif [[ -n "$mmproj" ]]; then
        if [[ -f "$mmproj" ]]; then
          mmproj_path="$mmproj"
        elif [[ -f "$GPUTOOL_DIR/models/$mmproj" ]]; then
          mmproj_path="$GPUTOOL_DIR/models/$mmproj"
        else
          warn "Specified mmproj not found: '$mmproj' (serving text-only)."
        fi
      else
        # Auto-detect: prefer F16, then any mmproj in the model's directory.
        local model_dir; model_dir=$(dirname "$model_path")
        local cand
        for cand in "$model_dir"/*mmproj*F16*.gguf "$model_dir"/*mmproj*.gguf; do
          [[ -f "$cand" ]] && { mmproj_path="$cand"; break; }
        done
      fi

      info "Serving model: $model_path"
      info "Bind host     : $host"
      info "Port          : $port"
      info "Run mode      : $run_mode"
      if [[ -n "$mmproj_path" ]]; then
        info "Vision (mmproj): enabled — $(basename "$mmproj_path")"
      else
        info "Vision (mmproj): disabled (text-only)"
      fi
      if [[ -n "$api_key" ]]; then
        info "API key auth  : enabled (clients must send 'Authorization: Bearer <key>')"
      else
        info "API key auth  : disabled (open access)"
      fi

      # Optional bearer-token auth: only added when a key is provided.
      local auth_args=()
      [[ -n "$api_key" ]] && auth_args=(--api-key "$api_key")
      # Optional multimodal projector: only added when resolved.
      local mmproj_args=()
      [[ -n "$mmproj_path" ]] && mmproj_args=(--mmproj "$mmproj_path")

      # Performance flags tuned for an RTX 5080 (16 GB) serving Qwen3.5-9B:
      #  - flash-attn on    : faster + required for quantized KV cache
      #  - q8_0 KV cache    : ~half the memory of f16, so 32k context fits in 16 GB
      #  - batch/ubatch     : larger batches improve prompt-processing throughput
      # (On smaller GPUs, lower --ctx-size, e.g. --ctx-size 8192.)
      local perf_args=(
        --ctx-size "$ctx_size"
        --batch-size 4096
        --ubatch-size 2048
        --flash-attn on
        --cache-type-k q8_0
        --cache-type-v q8_0
      )
      info "Context size  : $ctx_size  (flash-attn on, KV cache q8_0)"

      # MTP cannot coexist with parallel slots, so pin -np 1 when it is enabled.
      local spec_args=()
      if [[ "$spec_type" != "none" ]]; then
        spec_args=(--spec-type "$spec_type" --spec-draft-n-max "$spec_n" -np 1)
        info "Speculative   : $spec_type, draft n_max=$spec_n (-np 1 forced; no batching)"
      fi

      # Resolve a friendly URL host for display (0.0.0.0 isn't dialable directly).
      local url_host="$host"
      if [[ "$host" == "0.0.0.0" ]]; then
        url_host=$(hostname -I 2>/dev/null | awk '{print $1}')
        [[ -z "$url_host" ]] && url_host="localhost"
      fi

      # Export LD_LIBRARY_PATH so the server finds the shared libraries installed
      # alongside it in ~/.gputool/bin (libllama, libggml-cuda, etc.).
      export LD_LIBRARY_PATH="$GPUTOOL_DIR/bin:${LD_LIBRARY_PATH:-}"

      if [[ "$run_mode" == "foreground" ]]; then
        # Attached mode: blocks the terminal and streams logs live (Ctrl+C to stop).
        info "Starting in FOREGROUND (press Ctrl+C to stop)."
        echo "   🔗 API Base URL: http://$url_host:$port/v1"
        echo "══════════════════════════════════════════════════"
        exec "$server_bin" \
          --model "$model_path" \
          --host "$host" \
          --port "$port" \
          -ngl 99 \
          "${perf_args[@]}" \
          "${spec_args[@]}" \
          "${mmproj_args[@]}" \
          "${auth_args[@]}"
      fi

      # Background (daemon) mode: detach via nohup, log to file (-ngl 99 offloads all layers).
      info "Logging to    : $log_file"
      nohup "$server_bin" \
        --model "$model_path" \
        --host "$host" \
        --port "$port" \
        -ngl 99 \
        "${perf_args[@]}" \
        "${spec_args[@]}" \
        "${mmproj_args[@]}" \
        "${auth_args[@]}" \
        > "$log_file" 2>&1 &

      local server_pid=$!
      echo "$server_pid" > "$pid_file"
      sleep 2

      if kill -0 "$server_pid" 2>/dev/null; then
        success "llama-server started in background (PID: $server_pid)."
        echo "   🔗 API Base URL: http://$url_host:$port/v1"
        [[ "$host" == "0.0.0.0" ]] && echo "   🌐 Reachable from LAN peers at the address above (bound to all interfaces)."
        echo "   🛑 Stop it with : gputool serve-llamacpp stop"
        echo "   💡 Try querying model completions via curl:"
        echo "      curl http://$url_host:$port/v1/chat/completions \\"
        echo "        -H \"Content-Type: application/json\" \\"
        [[ -n "$api_key" ]] && echo "        -H \"Authorization: Bearer $api_key\" \\"
        echo "        -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
      else
        error "llama-server failed to start immediately. Check logs:"
        tail -n 20 "$log_file"
        rm -f "$pid_file"
        exit 1
      fi
      echo "══════════════════════════════════════════════════"
      ;;
      
    stop)
      echo "══════════════════════════════════════════════════"
      echo "🔌 Stopping llama-server"
      echo "══════════════════════════════════════════════════"

      # Gather candidate PIDs from the tracked PID file AND any stray background
      # instances (matched by our binary path and by process name), so this stops
      # both the daemon we started and any orphaned llama-server services.
      local candidates=()
      if [[ -f "$pid_file" ]]; then
        local fpid
        fpid=$(cat "$pid_file" 2>/dev/null)
        [[ -n "$fpid" ]] && candidates+=("$fpid")
      fi
      local p
      for p in $(pgrep -f "$server_bin" 2>/dev/null) $(pgrep -x llama-server 2>/dev/null); do
        candidates+=("$p")
      done

      # Deduplicate, drop this script (and its parent), and keep only live PIDs.
      local uniq_pids=()
      declare -A _seen=()
      for p in "${candidates[@]}"; do
        [[ -z "$p" || -n "${_seen[$p]:-}" ]] && continue
        _seen[$p]=1
        [[ "$p" == "$$" || "$p" == "$PPID" ]] && continue
        kill -0 "$p" 2>/dev/null && uniq_pids+=("$p")
      done

      if [[ ${#uniq_pids[@]} -eq 0 ]]; then
        warn "No running llama-server processes found."
        rm -f "$pid_file"
        echo "══════════════════════════════════════════════════"
        return 0 2>/dev/null || exit 0
      fi

      info "Found ${#uniq_pids[@]} llama-server process(es) to stop: ${uniq_pids[*]}"
      for p in "${uniq_pids[@]}"; do
        info "Stopping llama-server (PID: $p)..."
        kill "$p" 2>/dev/null

        local timeout=10
        while kill -0 "$p" 2>/dev/null && [[ $timeout -gt 0 ]]; do
          sleep 0.5
          ((timeout--))
        done

        if kill -0 "$p" 2>/dev/null; then
          warn "PID $p did not exit cleanly. Force killing..."
          kill -9 "$p" 2>/dev/null
        fi
        success "Stopped PID $p."
      done
      rm -f "$pid_file"
      echo "══════════════════════════════════════════════════"
      ;;
      
    status)
      echo "══════════════════════════════════════════════════"
      echo "📊 llama-server Status Check"
      echo "══════════════════════════════════════════════════"
      if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
          success "llama-server is running (PID: $pid)."
          
          # Check health endpoint
          if command -v curl &>/dev/null; then
            local health_resp
            health_resp=$(curl -s --connect-timeout 2 "http://localhost:$port/health")
            if [[ "$health_resp" == *'"status":'* || "$health_resp" == *"ok"* ]]; then
              success "Server API health check: Responsive (OK)"
            else
              warn "Server process active but health check returned: $health_resp"
            fi
          fi
          
          echo "   • Log Location : $log_file"
          echo "   • CLI Command  : ps -p $pid -o command"
          ps -p "$pid" -o command 2>/dev/null | tail -n 1
        else
          error "llama-server process is NOT running, but PID file exists."
          rm -f "$pid_file"
        fi
      else
        # No tracked PID file — check for any stray/foreground instances.
        local stray
        stray=$(pgrep -f "$server_bin" 2>/dev/null | tr '\n' ' ')
        if [[ -n "$stray" ]]; then
          warn "No tracked PID file, but found running llama-server process(es): $stray"
          echo "   🛑 Stop them with: gputool serve-llamacpp stop"
        else
          warn "llama-server is NOT running."
        fi
      fi
      echo "══════════════════════════════════════════════════"
      ;;
      
    *)
      error "Unknown serve-llamacpp action: '$action'"
      echo "Usage: gputool serve-llamacpp <start|stop|status> [model_name_or_path] [port] [-f|--foreground|-d|--background]"
      exit 1
      ;;
  esac
}

# Interactive / one-shot terminal chat client for any OpenAI-compatible endpoint
# (defaults to the locally served llama.cpp server). Streams the response with a
# colored terminal UI. The actual client lives in chat.py (fetched to
# ~/.gputool/chat.py by `gputool install` / `gputool update`); if it is missing
# we download it on demand so the command is self-healing.
# ── vLLM serving ──────────────────────────────────────────────────────────
# vLLM serves an OpenAI-compatible API with continuous batching, which is what
# you want when several students hit one GPU at once. llama.cpp (above) is the
# better choice for a quantized model on a small card; vLLM is the better choice
# for full-precision weights and concurrent requests.
VLLM_DEFAULT_MODEL="${VLLM_DEFAULT_MODEL:-Qwen/Qwen3.5-4B}"
VLLM_DEFAULT_PORT="${VLLM_DEFAULT_PORT:-8000}"
VLLM_DEFAULT_ENV="${VLLM_DEFAULT_ENV:-py312}"

# Resolve the python interpreter of a conda env, echoing its path.
_vllm_python() {
  local env_name="${1:-$VLLM_DEFAULT_ENV}" root
  for root in "$HOME/miniconda3" "$HOME/miniconda" "$HOME/anaconda3" "/opt/conda"; do
    if [[ -x "$root/envs/$env_name/bin/python" ]]; then
      echo "$root/envs/$env_name/bin/python"; return 0
    fi
  done
  return 1
}

# Install vLLM into a conda env. vLLM pins its own torch build, so let pip
# resolve it rather than forcing the wheel index used elsewhere.
setup_vllm() {
  local env_name="${1:-$VLLM_DEFAULT_ENV}"
  echo "══════════════════════════════════════════════════"
  echo -e "${BOLD}⚡ vLLM setup — env '$env_name'${NC}"
  echo "══════════════════════════════════════════════════"

  local py
  if ! py=$(_vllm_python "$env_name"); then
    error "Conda env '$env_name' not found. Create it first:"
    echo "   gputool setup-env $env_name 3.12"
    return 1
  fi
  info "Using $py"

  local have
  have=$("$py" -c "import vllm; print(vllm.__version__)" 2>/dev/null)
  if [[ -n "$have" ]]; then
    success "vLLM $have already installed."

    "$py" -m pip install -q ninja >/dev/null 2>&1
  else
    info "Installing vLLM (large download; pins its own torch build)..."
    "$py" -m pip install --upgrade pip >/dev/null 2>&1
    # flashinfer JIT-compiles attention kernels at runtime and shells out to ninja.

    if "$py" -m pip install vllm ninja; then
      have=$("$py" -c "import vllm; print(vllm.__version__)" 2>/dev/null)
      success "Installed vLLM $have"
    else
      error "vLLM install failed. See the pip output above."
      return 1
    fi
  fi

  # A Blackwell card needs a CUDA 12.8+ torch build; warn early rather than at load.
  "$py" - <<'PYCHK'
import torch
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
print(f"   torch {torch.__version__} | cuda={torch.cuda.is_available()} | sm_{cc[0]}{cc[1]}" if cc else "   no GPU visible")
PYCHK
  echo
  success "Ready. Serve with: gputool serve-vllm start"
}

# Put the conda env's bin and the CUDA toolkit on PATH for the server process.
# vLLM's flashinfer backend JIT-compiles attention kernels on first run: it shells
# out to `ninja` (installed into the env's bin, not the caller's PATH) and to
# `nvcc` (under /usr/local/cuda, which is not on PATH on these hosts). Without
# both, the engine dies at startup with FileNotFoundError and no useful message.
_vllm_prepare_env() {
  local py="$1"
  local env_bin; env_bin="$(dirname "$py")"
  local cuda_home=""
  for c in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-12; do
    [[ -x "$c/bin/nvcc" ]] && { cuda_home="$c"; break; }
  done
  export PATH="$env_bin${cuda_home:+:$cuda_home/bin}:${PATH:-}"
  # flashinfer JIT-compiles one job per core by default. On a 20-core node with
  # 30 GB RAM that peaks near 27 GB during a first load and can OOM-kill sshd,
  # taking the whole machine off the network. Cap it unless the caller overrode it.
  export MAX_JOBS="${MAX_JOBS:-4}"
  export NVCC_THREADS="${NVCC_THREADS:-1}"
  [[ -n "$cuda_home" ]] && export CUDA_HOME="$cuda_home"
  if ! command -v ninja &>/dev/null; then
    warn "ninja not found — flashinfer cannot build kernels. Run: gputool setup-vllm"
  fi
  [[ -z "$cuda_home" ]] && warn "No CUDA toolkit found; flashinfer JIT may fail."
  return 0
}

serve_vllm() {
  local action="${1:-status}"
  shift 2>/dev/null || true

  local run_mode="background"
  local host="0.0.0.0"
  local api_key="${GPUTOOL_VLLM_API_KEY:-}"
  local env_name="$VLLM_DEFAULT_ENV"
  # Measured on an RTX 5080 (16 GB) with Qwen3.5-4B: at these values the engine
  # reports 3.68 GiB of KV cache, 98,304 tokens, 12x concurrency. Raising max_len
  # or max_seqs OOMs during CUDA-graph profiling, because this model is multimodal
  # and the vision tower plus mm-encoder cache eat into the same budget.
  # Defaults come from the GPU profile, so a 5080, a 4090 and a Jetson each get
  # values that actually fit. Explicit flags still win.
  detect_gpu_profile
  local max_len="$PROF_MAX_LEN"
  local gpu_mem="$PROF_GPU_MEM"
  local max_seqs="$PROF_MAX_SEQS"
  local extra=()
  local positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--foreground) run_mode="foreground" ;;
      -d|--background) run_mode="background" ;;
      --host) host="${2:-0.0.0.0}"; shift ;;
      --host=*) host="${1#*=}" ;;
      --api-key) api_key="${2:-}"; shift ;;
      --api-key=*) api_key="${1#*=}" ;;
      --env) env_name="${2:-$VLLM_DEFAULT_ENV}"; shift ;;
      --env=*) env_name="${1#*=}" ;;
      --max-len) max_len="${2:-8192}"; shift ;;
      --max-len=*) max_len="${1#*=}" ;;
      --gpu-mem) gpu_mem="${2:-0.92}"; shift ;;
      --gpu-mem=*) gpu_mem="${1#*=}" ;;
      --max-seqs) max_seqs="${2:-16}"; shift ;;
      --max-seqs=*) max_seqs="${1#*=}" ;;
      --) shift; extra+=("$@"); break ;;
      *) positional+=("$1") ;;
    esac
    shift
  done
  local model="${positional[0]:-$VLLM_DEFAULT_MODEL}"
  local port="${positional[1]:-$VLLM_DEFAULT_PORT}"

  local pid_file="$GPUTOOL_DIR/vllm-server.pid"
  local log_file="$GPUTOOL_DIR/vllm-server.log"
  mkdir -p "$GPUTOOL_DIR"

  local url_host="$host"
  if [[ "$host" == "0.0.0.0" ]]; then
    url_host=$(hostname -I 2>/dev/null | awk '{print $1}')
    [[ -z "$url_host" ]] && url_host="localhost"
  fi

  case "$action" in
    start)
      echo "══════════════════════════════════════════════════"
      echo -e "${BOLD}⚡ Starting vLLM server${NC}"
      echo "══════════════════════════════════════════════════"

      if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file" 2>/dev/null)" 2>/dev/null; then
        warn "A vLLM server is already running (PID $(cat "$pid_file"))."
        echo "   Stop it first: gputool serve-vllm stop"
        return 1
      fi

      local py
      if ! py=$(_vllm_python "$env_name"); then
        error "Conda env '$env_name' not found. Run: gputool setup-vllm $env_name"
        return 1
      fi
      if ! "$py" -c "import vllm" 2>/dev/null; then
        error "vLLM is not installed in '$env_name'. Run: gputool setup-vllm $env_name"
        return 1
      fi

      _vllm_prepare_env "$py"


      info "Model         : $model"
      info "Env           : $env_name"
      info "Bind host     : $host"
      info "Port          : $port"
      info "Max model len : $max_len"
      info "GPU mem util  : $gpu_mem"
      info "Max sequences : $max_seqs"
      if [[ -n "${HF_HOME:-}" ]]; then
        info "HF_HOME       : $HF_HOME"
      fi
      if [[ -n "$api_key" ]]; then
        info "API key auth  : enabled"
      else
        info "API key auth  : disabled (open access)"
      fi

      local auth_args=()
      [[ -n "$api_key" ]] && auth_args=(--api-key "$api_key")

      local serve_args=(
        -m vllm.entrypoints.openai.api_server
        --model "$model"
        --host "$host"
        --port "$port"
        --max-model-len "$max_len"
        --gpu-memory-utilization "$gpu_mem"
        --max-num-seqs "$max_seqs"
      )

      if [[ "$run_mode" == "foreground" ]]; then
        info "Starting in FOREGROUND (Ctrl+C to stop)."
        echo "   🔗 API Base URL: http://$url_host:$port/v1"
        echo "══════════════════════════════════════════════════"
        exec "$py" "${serve_args[@]}" "${auth_args[@]}" "${extra[@]}"
      fi

      info "Logging to    : $log_file"
      nohup "$py" "${serve_args[@]}" "${auth_args[@]}" "${extra[@]}" > "$log_file" 2>&1 &
      local server_pid=$!
      echo "$server_pid" > "$pid_file"

      # First start downloads weights, so allow a generous window before giving up.
      info "Loading weights — first run downloads the model, which can take a while."
      local waited=0 limit=900
      while (( waited < limit )); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
          error "Server exited during startup. Last lines of $log_file:"
          tail -n 15 "$log_file"
          rm -f "$pid_file"
          return 1
        fi
        if grep -qiE "Application startup complete|Uvicorn running on" "$log_file" 2>/dev/null; then
          echo
          success "vLLM is serving (PID $server_pid) after ${waited}s."
          echo "   🔗 API Base URL : http://$url_host:$port/v1"
          echo "   📋 Models       : curl http://$url_host:$port/v1/models"
          echo "   📜 Logs         : tail -f $log_file"
          echo "   🛑 Stop         : gputool serve-vllm stop"
          return 0
        fi
        sleep 5
        waited=$((waited+5))
        printf "."
      done
      echo
      warn "Still not ready after ${limit}s — it may still be downloading."
      echo "   Watch progress: tail -f $log_file"
      return 0
      ;;

    stop)
      echo "══════════════════════════════════════════════════"
      echo -e "${BOLD}🛑 Stopping vLLM server${NC}"
      echo "══════════════════════════════════════════════════"
      local stopped=0
      if [[ -f "$pid_file" ]]; then
        local pid; pid=$(cat "$pid_file" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
          kill "$pid" 2>/dev/null
          local w=0
          while kill -0 "$pid" 2>/dev/null && (( w < 30 )); do sleep 1; w=$((w+1)); done
          kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
          success "Stopped tracked server (PID $pid)."
          stopped=1
        fi
        rm -f "$pid_file"
      fi
      # vLLM spawns worker processes; sweep any strays so the GPU is really freed.
      local strays
      # vLLM spawns worker processes named VLLM::EngineCore which do NOT match the
      # entrypoint pattern. Missing them leaves the GPU fully allocated after a
      # "successful" stop, and the next start then fails on memory. Sweep both.
      strays=$(pgrep -u "$(whoami)" -f "vllm\.entrypoints\.openai\.api_server|VLLM::" 2>/dev/null | tr '\n' ' ')
      if [[ -n "${strays// /}" ]]; then
        info "Cleaning up stray vLLM processes: $strays"
        # shellcheck disable=SC2086
        kill $strays 2>/dev/null; sleep 3
        # shellcheck disable=SC2086
        kill -9 $strays 2>/dev/null
        stopped=1
      fi
      (( stopped == 1 )) && success "vLLM stopped." || info "No vLLM server was running."
      ;;

    status)
      echo "══════════════════════════════════════════════════"
      echo -e "${BOLD}⚡ vLLM server status${NC}"
      echo "══════════════════════════════════════════════════"
      local running=0 pid=""
      if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file" 2>/dev/null)
        kill -0 "$pid" 2>/dev/null && running=1
      fi
      if (( running == 1 )); then
        _dev_ok "Process" "running (PID $pid)"
      else
        _dev_warn "Process" "not running"
        [[ -f "$pid_file" ]] && rm -f "$pid_file"
      fi
      _dev_dim "Endpoint" "http://$url_host:$port/v1"

      # Probe the API. curl is absent on some of these hosts, so fall back to wget.
      local models=""
      if command -v curl &>/dev/null; then
        models=$(curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null)
      elif command -v wget &>/dev/null; then
        models=$(wget -qO- --timeout=5 "http://127.0.0.1:$port/v1/models" 2>/dev/null)
      fi
      if [[ -n "$models" ]]; then
        local served
        served=$(echo "$models" | tr ',' '\n' | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' | head -n1 | cut -d'"' -f4)
        _dev_ok "API" "responding — serving ${served:-unknown}"
      else
        _dev_warn "API" "no response on port $port"
      fi

      if command -v nvidia-smi &>/dev/null; then
        _dev_dim "GPU memory" "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n1)"
      fi
      [[ -f "$log_file" ]] && _dev_dim "Log" "$log_file ($(wc -l < "$log_file") lines)"
      echo "══════════════════════════════════════════════════"
      ;;

    *)
      error "Unknown serve-vllm action: $action"
      echo "Usage: gputool serve-vllm <start|stop|status> [model] [port] [flags]"
      return 1
      ;;
  esac
}

chat_llamacpp() {
  # Prefer the pip-installed console script when present; the downloaded
  # single-file client stays as the fallback for nodes without the package.
  local ai_chat; ai_chat=$(_ai_script gputool-chat 2>/dev/null || echo "")
  if [[ -n "$ai_chat" ]]; then
    exec "$ai_chat" "$@"
  fi
  if [[ ! -f "$CHAT_PY_PATH" ]]; then
    info "Chat client not found locally; downloading it now..."
    if ! download_chat_py; then
      error "Could not obtain the chat client (chat.py)."
      echo "   👉 Check your network, or run: gputool update"
      exit 1
    fi
  fi

  local py
  py=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
  if [[ -z "$py" ]]; then
    error "Python 3 is required for 'gputool chat' but was not found in PATH."
    exit 1
  fi
  "$py" "$CHAT_PY_PATH" "$@"
}

# Main command dispatcher
# Only run when executed directly, not when sourced (allows reusing the helper functions).
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  return 0 2>/dev/null || true
fi

CMD="${1:-help}"
case "$CMD" in
  help)
    show_help
    ;;
  version)
    echo "gputool version $SCRIPT_VERSION"
    ;;
  install)
    install_gputool
    ;;
  update-script)
    update_script
    ;;
  setup-lerobot)
    shift
    setup_lerobot_env "${1:-}"
    ;;
  setup-env)
    shift
    setup_ml_env "${1:-}" "${2:-}"
    ;;
  install-conda)
    shift
    install_conda "${1:-}"
    ;;
  hf-cache|hf)
    shift
    hf_cache "${1:-status}"
    ;;
  device|info)
    shift
    device_check "${1:-}"
    ;;
  container)
    shift
    manage_container "$@"
    ;;
  profile)
    show_gpu_profile
    ;;
  install-ai)
    shift
    install_ai "${1:-all}"
    ;;
  agent)
    shift
    agent_backend "${1:-status}" "${@:2}"
    ;;
  check|system-check)
    shift
    system_check "${1:-}"
    ;;
  setup-llamacpp)
    shift
    setup_llamacpp "${1:-}"
    ;;
  download-model)
    shift
    download_model "${1:-}" "${2:-}" "${3:-}"
    ;;
  setup-vllm)
    shift
    setup_vllm "${1:-}"
    ;;
  vllm|serve-vllm)
    shift
    serve_vllm "$@"
    ;;
  llama|serve-llamacpp)
    shift
    serve_llamacpp "$@"
    ;;
  chat)
    shift
    chat_llamacpp "$@"
    ;;
  tailscale)
    shift
    SUBCMD="${1:-}"
    case "$SUBCMD" in
      setup)
        setup_tailscale
        ;;
      up)
        shift
        up_tailscale "${1:-}"
        ;;
      status)
        status_tailscale
        ;;
      down)
        down_tailscale
        ;;
      restart)
        down_tailscale
        up_tailscale
        ;;
      *)
        error "Unknown tailscale subcommand: $SUBCMD"
        echo "Valid subcommands: setup, up, status, down, restart"
        exit 1
        ;;
    esac
    ;;
  *)
    error "Unknown command: $CMD"
    show_help
    exit 1
    ;;
esac
