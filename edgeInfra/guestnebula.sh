#!/bin/bash
set -e

# Debug function
debug_paths() {
  echo "[DEBUG] Current working directory: $(pwd)"
  echo "[DEBUG] Script directory: $(dirname "$0")"
  echo "[DEBUG] Absolute script path: $(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  echo "[DEBUG] NEBULA_DIR: $NEBULA_DIR"
  echo "[DEBUG] NEBULA_BIN: $NEBULA_BIN"
  echo "[DEBUG] TOKEN_FILE: $TOKEN_FILE"
}

# Default client ID
DEFAULT_CLIENT_ID="guest01"

# Use command line argument if provided, otherwise use default
CLIENT_ID="${2:-$DEFAULT_CLIENT_ID}"

API_SERVER="http://lkk688.duckdns.org:8000"
NEBULA_DIR="./nebula-config" # Local config directory
NEBULA_BIN="/usr/local/bin/nebula" # System-wide binary location

# Default token and platform settings
DEFAULT_TOKEN="guest-secure-token-2025"
DEFAULT_PLATFORM="linux-arm64"

# Try to load client-specific token if available
TOKEN_FILE="./nebula-config/token.txt"
if [ -f "$TOKEN_FILE" ]; then
  TOKEN=$(cat "$TOKEN_FILE")
else
  TOKEN="$DEFAULT_TOKEN"
fi

PLATFORM="$DEFAULT_PLATFORM"

# === Function: download config ===
download_config() {
  echo "[INFO] Downloading bundle for $CLIENT_ID from $API_SERVER..."
  
  # Get absolute path for NEBULA_DIR
  ABSOLUTE_NEBULA_DIR="$(pwd)/$NEBULA_DIR"
  
  # Check if NEBULA_DIR is writable or can be created
  if [ -d "$ABSOLUTE_NEBULA_DIR" ] && [ ! -w "$ABSOLUTE_NEBULA_DIR" ]; then
    echo "[ERROR] $ABSOLUTE_NEBULA_DIR exists but is not writable. Please run with sudo or fix permissions."
    return 1
  elif [ ! -d "$ABSOLUTE_NEBULA_DIR" ] && [ ! -w "$(dirname "$ABSOLUTE_NEBULA_DIR")" ]; then
    echo "[ERROR] Cannot create $ABSOLUTE_NEBULA_DIR. Parent directory is not writable."
    return 1
  fi
  
  # Create a temporary directory for downloads
  TMP_DIR=$(mktemp -d)
  cd "$TMP_DIR"
  
  # Download the client bundle
  echo "[INFO] Downloading client bundle..."
  if ! curl -v -f -o "$CLIENT_ID.zip" -G -d "token=$TOKEN" -d "platform=$PLATFORM" "$API_SERVER/download/$CLIENT_ID"; then
    echo "[ERROR] Failed to download client bundle. Check your token and network connection."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  echo "[DEBUG] Downloaded file size: $(ls -lh "$CLIENT_ID.zip" | awk '{print $5}')"
  
  # Download CA certificate
  echo "[INFO] Downloading CA certificate..."
  if ! curl -v -f -o "ca.zip" -G -d "token=jetsonsupertoken" "$API_SERVER/public/downloads/ca.zip"; then
    echo "[WARNING] Failed to download CA certificate. Will try to use existing one if available."
  else
    echo "[DEBUG] CA certificate file size: $(ls -lh "ca.zip" | awk '{print $5}')"
  fi
  
  # Extract the downloaded files
  echo "[INFO] Extracting files..."
  if [ ! -f "$CLIENT_ID.zip" ]; then
    echo "[ERROR] Downloaded file not found: $CLIENT_ID.zip"
    ls -la
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  if ! unzip -o "$CLIENT_ID.zip"; then
    echo "[ERROR] Failed to extract client bundle."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Extract CA certificate if downloaded successfully
  if [ -f "ca.zip" ]; then
    if ! unzip -o "ca.zip" -d ca_temp; then
      echo "[WARNING] Failed to extract CA certificate."
    fi
  fi
  
  # Save current directory and get the original working directory
  CURRENT_DIR=$(pwd)
  ORIGINAL_DIR=$(cd - > /dev/null && pwd)
  
  # Create Nebula directory with absolute path
  ABSOLUTE_NEBULA_DIR="$ORIGINAL_DIR/$NEBULA_DIR"
  echo "[DEBUG] Creating directory at: $ABSOLUTE_NEBULA_DIR"
  mkdir -p "$ABSOLUTE_NEBULA_DIR"
  
  # Verify directory was created
  if [ ! -d "$ABSOLUTE_NEBULA_DIR" ]; then
    echo "[ERROR] Failed to create directory: $ABSOLUTE_NEBULA_DIR"
    ls -la "$ORIGINAL_DIR"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Go back to temp directory
  cd "$CURRENT_DIR"
  
  # Copy client files
  if [ -f "$CLIENT_ID.crt" ] && [ -f "$CLIENT_ID.key" ] && [ -f "config.yml" ]; then
    cp "$CLIENT_ID.crt" "$CLIENT_ID.key" "config.yml" "$ABSOLUTE_NEBULA_DIR/"
  else
    echo "[ERROR] Required client files not found in the downloaded bundle."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Copy CA certificate if available
  if [ -d "ca_temp" ] && [ -f "ca_temp/ca.crt" ]; then
    cp "ca_temp/ca.crt" "$ABSOLUTE_NEBULA_DIR/"
  elif [ ! -f "$ABSOLUTE_NEBULA_DIR/ca.crt" ]; then
    echo "[ERROR] CA certificate not found and not available in $ABSOLUTE_NEBULA_DIR."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  else
    echo "[INFO] Using existing CA certificate in $ABSOLUTE_NEBULA_DIR."
  fi
  
  # Clean up
  cd - > /dev/null
  rm -rf "$TMP_DIR"
  
  echo "[INFO] Configuration files successfully installed to $ABSOLUTE_NEBULA_DIR."
  echo "[DEBUG] To verify, run: ls -la $ABSOLUTE_NEBULA_DIR"
}

# === Function: install nebula ===
install_nebula() {
  echo "[INFO] Installing Nebula locally..."
  
  # Check if current directory is writable
  if [ ! -w "$(pwd)" ]; then
    echo "[ERROR] Current directory is not writable. Please run from a directory where you have write permissions."
    return 1
  fi
  
  # Create a temporary directory for downloads
  TMP_DIR=$(mktemp -d)
  cd "$TMP_DIR"
  
  # Download the client bundle which includes the nebula binary
  echo "[INFO] Downloading client bundle with Nebula binary..."
  if ! curl -v -f -o "$CLIENT_ID.zip" -G -d "token=$TOKEN" -d "platform=$PLATFORM" "$API_SERVER/download/$CLIENT_ID"; then
    echo "[ERROR] Failed to download client bundle. Check your token and network connection."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  echo "[DEBUG] Downloaded file size: $(ls -lh "$CLIENT_ID.zip" | awk '{print $5}')"
  
  # Extract the downloaded files
  if [ ! -f "$CLIENT_ID.zip" ]; then
    echo "[ERROR] Downloaded file not found: $CLIENT_ID.zip"
    ls -la
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  if ! unzip -o "$CLIENT_ID.zip"; then
    echo "[ERROR] Failed to extract client bundle."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Check if nebula binary exists in the bundle
  if [ ! -f "nebula" ]; then
    echo "[ERROR] Nebula binary not found in the downloaded bundle."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Use absolute path for nebula binary
  ABSOLUTE_NEBULA_BIN="$NEBULA_BIN"
  
  echo "[DEBUG] Installing nebula binary to system location: $ABSOLUTE_NEBULA_BIN"
  
  # Copy the nebula binary to the system location (requires sudo)
  if ! sudo cp "nebula" "$ABSOLUTE_NEBULA_BIN"; then
    echo "[ERROR] Failed to copy nebula binary to: $ABSOLUTE_NEBULA_BIN"
    echo "[ERROR] This operation requires sudo privileges to write to $ABSOLUTE_NEBULA_BIN"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  cd - > /dev/null
  sudo chmod +x "$ABSOLUTE_NEBULA_BIN"
  
  # Verify the binary works
  if ! "$ABSOLUTE_NEBULA_BIN" -version >/dev/null 2>&1; then
    echo "[WARNING] Nebula binary installed but may not be working correctly."
  else
    echo "[INFO] Nebula binary installed successfully: $($ABSOLUTE_NEBULA_BIN -version 2>&1 | head -n 1)"
  fi
  
  # Clean up
  rm -rf "$TMP_DIR"
  return 0
}

# === Function: run nebula ===
run_nebula() {
  echo "[INFO] Running Nebula VPN client..."
  
  # Get absolute paths - NEBULA_BIN is already an absolute path, don't prepend pwd
  ABSOLUTE_NEBULA_DIR="$(pwd)/$NEBULA_DIR"
  ABSOLUTE_NEBULA_BIN="$NEBULA_BIN"
  
  echo "[DEBUG] Using config directory: $ABSOLUTE_NEBULA_DIR"
  echo "[DEBUG] Using nebula binary: $ABSOLUTE_NEBULA_BIN"
  
  # Check if required files exist
  if [ ! -f "$ABSOLUTE_NEBULA_BIN" ]; then
    echo "[ERROR] Nebula binary not found at $ABSOLUTE_NEBULA_BIN. Please install it first."
    return 1
  fi
  
  if [ ! -f "$ABSOLUTE_NEBULA_DIR/config.yml" ]; then
    echo "[ERROR] Configuration file not found at $ABSOLUTE_NEBULA_DIR/config.yml. Please download it first."
    echo "[DEBUG] To verify directory contents, run: ls -la $ABSOLUTE_NEBULA_DIR"
    return 1
  fi
  
  echo "[INFO] Starting Nebula VPN in foreground mode..."
  echo "[INFO] Press Ctrl+C to stop the VPN connection"
  echo "[INFO] Note: Nebula requires root privileges to create network interfaces"
  echo "-------------------------------------------"
  
  # Change to the nebula-config directory
  echo "[DEBUG] Changing to directory: $(dirname "$0")/nebula-config"
  cd "$(dirname "$0")/nebula-config"
  
  # Check if we're running as root
  if [ "$(id -u)" -ne 0 ]; then
    echo "[WARNING] Not running as root. Using sudo to run Nebula..."
    sudo "$ABSOLUTE_NEBULA_BIN" -config config.yml
  else
    # Run nebula in foreground
    "$ABSOLUTE_NEBULA_BIN" -config config.yml
  fi
  
  return $?
}

# === Function: check nebula status ===
check_status() {
  echo "[INFO] Checking nebula status..."
  
  # Use absolute path for nebula binary
  ABSOLUTE_NEBULA_BIN="$NEBULA_BIN"
  
  echo "[DEBUG] Using nebula binary: $ABSOLUTE_NEBULA_BIN"
  
  # Check if nebula interface exists
  if ip addr show nebula1 >/dev/null 2>&1; then
    NEBULA_IP=$(ip -4 addr show nebula1 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    echo "[STATUS] Nebula interface is UP with IP: $NEBULA_IP ✅"
  else
    echo "[STATUS] Nebula interface is NOT UP ❌"
  fi
  
  # Check if nebula process is running
  if pgrep -f "$ABSOLUTE_NEBULA_BIN" >/dev/null; then
    echo "[STATUS] Nebula process is running ✅"
  else
    echo "[STATUS] Nebula process is NOT running ❌"
  fi
  
  # Try to ping the lighthouse if interface is up
  if ip addr show nebula1 >/dev/null 2>&1; then
    echo "\n[CONNECTIVITY] Testing connection to lighthouse (192.168.100.1):"
    if ping -c 1 -W 2 192.168.100.1 >/dev/null 2>&1; then
      echo "[STATUS] Lighthouse is reachable ✅"
    else
      echo "[STATUS] Lighthouse is NOT reachable ❌"
    fi
  fi
}

# === Function: set token ===
set_token() {
  if [ -z "$1" ]; then
    echo "[ERROR] Please provide a token value"
    return 1
  fi
  
  # Get absolute path for token file
  ABSOLUTE_TOKEN_FILE="$(pwd)/$TOKEN_FILE"
  echo "[DEBUG] Using token file: $ABSOLUTE_TOKEN_FILE"
  
  echo "[INFO] Setting client token to: $1"
  mkdir -p "$(dirname "$ABSOLUTE_TOKEN_FILE")"
  echo "$1" > "$ABSOLUTE_TOKEN_FILE"
  echo "[INFO] Token saved to $ABSOLUTE_TOKEN_FILE"
}

# === Main CLI logic ===
# Print debug information
debug_paths

case "$1" in
  download)
    echo "[INFO] Downloading configuration for guest user ($CLIENT_ID)..."
    if download_config; then
      echo "[SUCCESS] Nebula VPN configuration downloaded successfully!"
      echo "[DEBUG] To verify, run: ls -la $(pwd)/$NEBULA_DIR"
    else
      echo "[ERROR] Nebula VPN configuration download failed. Please check the errors above."
      exit 1
    fi
    ;;
  install)
    echo "[INFO] Installing Nebula binary for guest user ($CLIENT_ID)..."
    if install_nebula; then
      echo "[SUCCESS] Nebula binary installed successfully!"
    else
      echo "[ERROR] Nebula binary installation failed. Please check the errors above."
      exit 1
    fi
    ;;
  run)
    echo "[INFO] Running Nebula VPN for guest user ($CLIENT_ID)..."
    run_nebula
    ;;
  status)
    check_status
    ;;
  set-token)
    set_token "$2"
    ;;
  debug)
    echo "[DEBUG] Running diagnostics..."
    debug_paths
    echo "[DEBUG] Checking for nebula-config directory..."
    ls -la "$(dirname "$0")/nebula-config" 2>/dev/null || echo "[ERROR] Directory not found: $(dirname "$0")/nebula-config"
    echo "[DEBUG] Checking for nebula binary..."
    # Use absolute path for nebula binary without prepending current directory
    ls -la "$NEBULA_BIN" 2>/dev/null || echo "[ERROR] Binary not found: $NEBULA_BIN"
    echo "[DEBUG] Checking for token file..."
    ls -la "$(dirname "$0")/nebula-config/token.txt" 2>/dev/null || echo "[WARNING] Token file not found: $(dirname "$0")/nebula-config/token.txt"
    ;;

  *)
    echo "Usage: guestnebula download [client_id]  # download configuration files"
    echo "       guestnebula install [client_id]   # install nebula binary"
    echo "       guestnebula run [client_id]       # run nebula in foreground"
    echo "       guestnebula status [client_id]    # check status"
    echo "       guestnebula set-token [client_id] <token>  # set client-specific token"
    echo "       guestnebula debug                 # run diagnostics to troubleshoot issues"
    echo ""
    echo "Notes:"
    echo "  - Configuration files are stored in ./nebula-config (relative to script directory)"
    echo "  - Nebula binary is installed to $NEBULA_BIN (system-wide location)"
    echo "  - The script will change to the nebula-config directory before running the binary"
    echo "  - Installation requires sudo privileges to write to $NEBULA_BIN"
    echo ""
    echo "Note: If [client_id] is not provided, 'guest01' will be used as default"
    echo ""
    echo "This version of the script:"
    echo "- Uses local directory for configuration (./nebula-config)"
    echo "- Installs Nebula binary to system location ($NEBULA_BIN)"
    echo "- Requires sudo for installation and for creating TUN/TAP interfaces"
    echo "- Runs Nebula in foreground mode (Ctrl+C to stop)"
    ;;
esac