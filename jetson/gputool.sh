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

# Helper print functions
info() { echo -e "${BLUE}[⚙️]${NC} $*"; }
success() { echo -e "${GREEN}[✅]${NC} $*"; }
warn() { echo -e "${YELLOW}[⚠️]${NC} $*"; }
error() { echo -e "${RED}[❌]${NC} $*"; }

# Helper download functions (with Python 3 fallback if curl/wget are missing)
download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl &>/dev/null; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget &>/dev/null; then
    wget -qO "$dest" "$url"
  elif command -v python3 &>/dev/null; then
    python3 -c "import urllib.request; urllib.request.urlretrieve('$url', '$dest')" &>/dev/null
  else
    return 1
  fi
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
  echo "  setup-lerobot [env_name] - Create Conda env and install PyTorch (RTX 5080), LeRobot, and HF"
  echo "  check [env_name]         - Run a complete diagnostic check of GPU, PyTorch, HF, LeRobot & Tailscale"
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
  echo "  gputool setup-lerobot my_env"
  echo "  gputool check my_env"
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
    error "Conda not found. Please make sure Miniconda/Anaconda is installed."
    echo "   Suggested installation: https://docs.google.com/miniconda/"
    exit 1
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

  # --- Install PyTorch ---
  info "Installing PyTorch (with CUDA 12.8 / Blackwell support)..."
  warn "⏳ Downloading PyTorch wheels (~800MB+). This can take several minutes"
  warn "   depending on your network connection speed. Please do not close the terminal..."
  # Blackwell RTX 5080 requires CUDA 12.8+; try PyTorch's cu128 wheel index first
  if ! conda run -n "$env_name" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; then
    warn "Failed to install via cu128 wheel index. Retrying with default PyPI torch..."
    if ! conda run -n "$env_name" pip install torch torchvision torchaudio; then
      error "PyTorch installation failed."
      exit 1
    fi
  fi
  success "PyTorch installed."

  # --- Install LeRobot and Hugging Face Hub ---
  info "Installing LeRobot (with extra dependencies) and Hugging Face Hub..."
  warn "⏳ Downloading LeRobot simulation libraries. This may also take a few minutes..."
  # 'lerobot[all]' installs standard aloha, pusht and other simulation/robotics dependencies
  if ! conda run -n "$env_name" pip install "lerobot[all]" huggingface_hub; then
    warn "Installing 'lerobot[all]' failed. Trying base 'lerobot'..."
    if ! conda run -n "$env_name" pip install lerobot huggingface_hub; then
      error "LeRobot installation failed."
      exit 1
    fi
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

# Main command dispatcher
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
  check|system-check)
    shift
    system_check "${1:-}"
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
