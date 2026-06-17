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
      # If the driver is too old for the 12.8 runtime, step down to a build it can run.
      if [[ -n "$drv_ver" ]]; then
        local drv_num; drv_num=$(_ver_to_int "$drv_ver")
        if (( drv_num < 121 && drv_num >= 118 )); then
          tag="cu118"; reason="GPU is modern but the driver supports only up to CUDA $drv_ver; capping to CUDA 11.8 wheels."
        fi
      fi
    fi
  else
    # No GPU capability info, but a toolkit/driver exists: choose from its CUDA version.
    local ref ref_num
    ref="${nvcc_ver:-$drv_ver}"
    ref_num=$(_ver_to_int "$ref")
    if   (( ref_num >= 124 )); then tag="cu128"; reason="CUDA toolkit $ref detected; using CUDA 12.8 wheels."
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
  if ! ensure_conda_tool "$env_name" cmake "cmake" "cmake"; then
    error "Cannot continue without cmake."
    exit 1
  fi
  # Ninja is optional but greatly speeds up the build; a failure is non-fatal.
  local use_ninja=0
  if ensure_conda_tool "$env_name" ninja "ninja"; then
    use_ninja=1
  else
    warn "Proceeding without Ninja; will use the default Make generator."
  fi

  # Configure build using cmake inside conda env
  local build_dir="$src_dir/build"
  local generator_args=()
  [[ "$use_ninja" -eq 1 ]] && generator_args=(-G Ninja)

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

  info "Configuring build with CUDA support enabled${use_ninja:+ (Ninja generator)}..."
  if ! conda run -n "$env_name" cmake -S "$src_dir" -B "$build_dir" "${generator_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_COMPILER="$nvcc_bin"; then
    warn "CMake configuration failed. Wiping build directory and retrying once..."
    rm -rf "$build_dir"
    if ! conda run -n "$env_name" cmake -S "$src_dir" -B "$build_dir" "${generator_args[@]}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DLLAMA_CURL=OFF \
      -DCMAKE_CUDA_COMPILER="$nvcc_bin"; then
      error "CMake configuration failed."
      exit 1
    fi
  fi

  # Compile release target
  info "Compiling llama.cpp Release binaries using all CPU cores..."
  local num_jobs
  num_jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  if ! conda run -n "$env_name" cmake --build "$build_dir" --config Release -j"$num_jobs"; then
    error "llama.cpp compilation failed."
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
  local ctx_size="32768" # context window (tuned for RTX 5080 16GB; override with --ctx-size)
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
chat_llamacpp() {
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
  serve-llamacpp)
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
