#!/bin/bash
set -e

# === Parse hostname and determine identity ===
hostname=$(hostname)
suffix=$(echo "$hostname" | grep -oE '[0-9]{2}$')
nodename="jetson${suffix}"

API_SERVER="http://lkk688.duckdns.org:8000"
NEBULA_DIR="/etc/nebula"
NEBULA_BIN="/usr/local/bin/nebula"
SERVICE_FILE="/etc/systemd/system/nebula.service"

# Store original directory for absolute path calculations
ORIGINAL_DIR=$(pwd)
ABSOLUTE_NEBULA_DIR="$NEBULA_DIR"
ABSOLUTE_NEBULA_BIN="$NEBULA_BIN"

# Default token and platform settings
DEFAULT_TOKEN="jetsonsupertoken"
DEFAULT_PLATFORM="linux-arm64"

# Try to load client-specific token if available
TOKEN_FILE="/etc/nebula/token.txt"
# Set absolute token file path after TOKEN_FILE is defined
ABSOLUTE_TOKEN_FILE="$TOKEN_FILE"

if [ -f "$TOKEN_FILE" ]; then
  TOKEN=$(cat "$TOKEN_FILE")
else
  TOKEN="$DEFAULT_TOKEN"
fi

PLATFORM="$DEFAULT_PLATFORM"

# === Function: debug paths ===
debug_paths() {
  echo "[DEBUG] Current working directory: $(pwd)"
  echo "[DEBUG] Script directory: $(dirname "$0")"
  echo "[DEBUG] Absolute script path: $(realpath "$0")"
  echo "[DEBUG] NEBULA_DIR: $NEBULA_DIR"
  echo "[DEBUG] NEBULA_BIN: $NEBULA_BIN"
  echo "[DEBUG] TOKEN_FILE: $TOKEN_FILE"
  echo "[DEBUG] ABSOLUTE_NEBULA_DIR: $ABSOLUTE_NEBULA_DIR"
  echo "[DEBUG] ABSOLUTE_NEBULA_BIN: $ABSOLUTE_NEBULA_BIN"
  echo "[DEBUG] ABSOLUTE_TOKEN_FILE: $ABSOLUTE_TOKEN_FILE"
}

# === Function: download config ===
download_config() {
  echo "[INFO] Downloading bundle for $nodename from $API_SERVER..."
  
  # Create a temporary directory for downloads
  TMP_DIR=$(mktemp -d)
  cd "$TMP_DIR"
  
  # Download the client bundle
  echo "[INFO] Downloading client bundle..."
  if ! curl -f -O -G -d "token=$TOKEN" -d "platform=$PLATFORM" "$API_SERVER/download/$nodename"; then
    echo "[ERROR] Failed to download client bundle. Check your token and network connection."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Download CA certificate if needed
  echo "[INFO] Downloading CA certificate..."
  if ! curl -f -O -G -d "token=$TOKEN" "$API_SERVER/public/downloads/ca.zip"; then
    echo "[WARNING] Failed to download CA certificate. Will try to use existing one if available."
  fi
  
  # Extract the downloaded files
  echo "[INFO] Extracting files..."
  if ! unzip -o "$nodename.zip"; then
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
  
  # Create Nebula directory and copy files
  echo "[DEBUG] Creating Nebula directory: $ABSOLUTE_NEBULA_DIR"
  if ! sudo mkdir -p "$ABSOLUTE_NEBULA_DIR"; then
    echo "[ERROR] Failed to create directory: $ABSOLUTE_NEBULA_DIR"
    echo "[DEBUG] Listing current directory contents:"
    ls -la "$ORIGINAL_DIR"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Verify directory was created
  if [ ! -d "$ABSOLUTE_NEBULA_DIR" ]; then
    echo "[ERROR] Directory $ABSOLUTE_NEBULA_DIR was not created successfully"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Copy client files
  if [ -f "$nodename.crt" ] && [ -f "$nodename.key" ] && [ -f "config.yml" ]; then
    echo "[DEBUG] Copying client files to: $ABSOLUTE_NEBULA_DIR"
    sudo cp "$nodename.crt" "$nodename.key" "config.yml" "$ABSOLUTE_NEBULA_DIR/"
  else
    echo "[ERROR] Required client files not found in the downloaded bundle."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Copy CA certificate if available
  if [ -d "ca_temp" ] && [ -f "ca_temp/ca.crt" ]; then
    echo "[DEBUG] Copying CA certificate to: $ABSOLUTE_NEBULA_DIR"
    sudo cp "ca_temp/ca.crt" "$ABSOLUTE_NEBULA_DIR/"
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
  
  echo "[DEBUG] Verifying downloaded configuration:"
  sudo ls -la "$ABSOLUTE_NEBULA_DIR"
  
  echo "[INFO] Configuration files successfully installed to $ABSOLUTE_NEBULA_DIR."
}

# === Function: install nebula ===
install_nebula() {
  if command -v nebula >/dev/null 2>&1; then
    echo "[INFO] Nebula already installed at: $(command -v nebula)"
    return 0
  fi
  
  echo "[INFO] Installing Nebula..."
  # Create a temporary directory for downloads
  TMP_DIR=$(mktemp -d)
  cd "$TMP_DIR"
  
  # Download the client bundle which includes the nebula binary
  echo "[INFO] Downloading client bundle with Nebula binary..."
  if ! curl -f -O -G -d "token=$TOKEN" -d "platform=$PLATFORM" "$API_SERVER/download/$nodename"; then
    echo "[ERROR] Failed to download client bundle. Check your token and network connection."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Extract the downloaded files
  if ! unzip -o "$nodename.zip"; then
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
  
  # Move the nebula binary to the correct location
  echo "[DEBUG] Copying nebula binary to: $ABSOLUTE_NEBULA_BIN"
  if ! sudo cp "nebula" "$ABSOLUTE_NEBULA_BIN"; then
    echo "[ERROR] Failed to copy nebula binary to $ABSOLUTE_NEBULA_BIN."
    echo "[DEBUG] Listing current directory contents:"
    ls -la "$ORIGINAL_DIR"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  sudo chmod +x "$ABSOLUTE_NEBULA_BIN"
  
  # Verify the binary works
  if ! sudo "$ABSOLUTE_NEBULA_BIN" -version >/dev/null 2>&1; then
    echo "[WARNING] Nebula binary installed but may not be working correctly."
  else
    echo "[INFO] Nebula binary installed successfully: $(sudo $ABSOLUTE_NEBULA_BIN -version 2>&1 | head -n 1)"
  fi
  
  # Clean up
  cd - > /dev/null
  rm -rf "$TMP_DIR"
  return 0
}

# === Function: create systemd service ===
setup_systemd() {
  echo "[INFO] Creating systemd service for Nebula..."
  
  # Check if required files exist
  echo "[DEBUG] Checking for nebula binary at: $ABSOLUTE_NEBULA_BIN"
  if [ ! -f "$ABSOLUTE_NEBULA_BIN" ]; then
    echo "[ERROR] Nebula binary not found at $ABSOLUTE_NEBULA_BIN. Please install it first."
    return 1
  fi
  
  echo "[DEBUG] Checking for config file at: $ABSOLUTE_NEBULA_DIR/config.yml"
  if [ ! -f "$ABSOLUTE_NEBULA_DIR/config.yml" ]; then
    echo "[ERROR] Configuration file not found at $ABSOLUTE_NEBULA_DIR/config.yml. Please download it first."
    return 1
  fi
  
  # Create systemd service file
  echo "[DEBUG] Creating systemd service file with absolute paths"
  if ! sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Nebula VPN
After=network.target

[Service]
ExecStart=$ABSOLUTE_NEBULA_BIN -config $ABSOLUTE_NEBULA_DIR/config.yml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
  then
    echo "[ERROR] Failed to create systemd service file."
    return 1
  fi
  
  echo "[INFO] Reloading systemd daemon..."
  if ! sudo systemctl daemon-reexec; then
    echo "[WARNING] Failed to reload systemd daemon. Continuing anyway..."
  fi
  
  echo "[INFO] Enabling and starting Nebula service..."
  if ! sudo systemctl enable nebula; then
    echo "[ERROR] Failed to enable Nebula service."
    return 1
  fi
  
  if ! sudo systemctl restart nebula; then
    echo "[ERROR] Failed to start Nebula service."
    return 1
  fi
  
  echo "[INFO] Nebula service successfully configured and started."
  return 0
}

# === Function: check nebula status ===
check_status() {
  echo "[INFO] Checking nebula service status..."
  echo "[DEBUG] Using nebula binary at: $ABSOLUTE_NEBULA_BIN"
  echo "[DEBUG] Using config directory at: $ABSOLUTE_NEBULA_DIR"
  
  # Check if service is active
  if sudo systemctl is-active nebula >/dev/null 2>&1; then
    echo "[STATUS] Nebula service is ACTIVE ✅"
  else
    echo "[STATUS] Nebula service is NOT ACTIVE ❌"
  fi
  
  # Check if service is enabled
  if sudo systemctl is-enabled nebula >/dev/null 2>&1; then
    echo "[STATUS] Nebula service is ENABLED at boot ✅"
  else
    echo "[STATUS] Nebula service is NOT ENABLED at boot ❌"
  fi
  
  # Check if nebula interface exists
  if ip addr show nebula1 >/dev/null 2>&1; then
    NEBULA_IP=$(ip -4 addr show nebula1 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    echo "[STATUS] Nebula interface is UP with IP: $NEBULA_IP ✅"
  else
    echo "[STATUS] Nebula interface is NOT UP ❌"
  fi
  
  # Show detailed service status
  echo "\n[DETAILS] Detailed service status:"
  sudo systemctl status nebula --no-pager
  
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
  
  echo "[INFO] Setting client token to: $1"
  echo "[DEBUG] Using token file at: $ABSOLUTE_TOKEN_FILE"
  
  # Ensure token file path is valid
  if [ -z "$ABSOLUTE_TOKEN_FILE" ]; then
    echo "[ERROR] Token file path is empty. Using default path: /etc/nebula/token.txt"
    ABSOLUTE_TOKEN_FILE="/etc/nebula/token.txt"
  fi
  
  # Create directory if it doesn't exist
  sudo mkdir -p "$(dirname "$ABSOLUTE_TOKEN_FILE")"
  
  # Save token to file
  echo "$1" | sudo tee "$ABSOLUTE_TOKEN_FILE" > /dev/null
  
  # Update TOKEN variable for immediate use
  TOKEN="$1"
  
  echo "[INFO] Token saved to $ABSOLUTE_TOKEN_FILE"
}

# === Main CLI logic ===

# Call debug_paths for all commands to help with troubleshooting
debug_paths

case "$1" in
  install)
    echo "[INFO] Starting Nebula VPN installation for $nodename..."
    if download_config && install_nebula && setup_systemd; then
      echo "[SUCCESS] Nebula VPN installed and configured successfully!"
      check_status
    else
      echo "[ERROR] Nebula VPN installation failed. Please check the errors above."
      exit 1
    fi
    ;;
  status)
    check_status
    ;;
  update)
    echo "[INFO] Updating Nebula VPN configuration for $nodename..."
    if download_config && setup_systemd; then
      echo "[SUCCESS] Nebula VPN configuration updated successfully!"
      check_status
    else
      echo "[ERROR] Nebula VPN update failed. Please check the errors above."
      exit 1
    fi
    ;;
  set-token)
    set_token "$2"
    ;;
  restart)
    echo "[INFO] Restarting Nebula VPN service..."
    if sudo systemctl restart nebula; then
      echo "[SUCCESS] Nebula VPN service restarted successfully!"
      check_status
    else
      echo "[ERROR] Failed to restart Nebula VPN service."
      exit 1
    fi
    ;;
  debug)
    echo "[INFO] Running diagnostics..."
    echo "[DEBUG] Checking for nebula config directory:"
    if [ -d "$ABSOLUTE_NEBULA_DIR" ]; then
      echo "[STATUS] Config directory exists: $ABSOLUTE_NEBULA_DIR ✅"
      sudo ls -la "$ABSOLUTE_NEBULA_DIR"
    else
      echo "[STATUS] Config directory NOT found: $ABSOLUTE_NEBULA_DIR ❌"
    fi
    
    echo "[DEBUG] Checking for nebula binary:"
    if [ -f "$ABSOLUTE_NEBULA_BIN" ]; then
      echo "[STATUS] Nebula binary exists: $ABSOLUTE_NEBULA_BIN ✅"
      sudo ls -la "$ABSOLUTE_NEBULA_BIN"
    else
      echo "[STATUS] Nebula binary NOT found: $ABSOLUTE_NEBULA_BIN ❌"
    fi
    
    echo "[DEBUG] Checking for token file:"
    if [ -f "$ABSOLUTE_TOKEN_FILE" ]; then
      echo "[STATUS] Token file exists: $ABSOLUTE_TOKEN_FILE ✅"
      sudo ls -la "$ABSOLUTE_TOKEN_FILE"
    else
      echo "[STATUS] Token file NOT found: $ABSOLUTE_TOKEN_FILE ❌"
    fi
    ;;
  *)
    echo "Usage: jetsonnebula install              # install/start Nebula"
    echo "       jetsonnebula status               # check status"
    echo "       jetsonnebula update               # update config only"
    echo "       jetsonnebula set-token <token>    # set client-specific token"
    echo "       jetsonnebula restart              # restart Nebula service"
    echo "       jetsonnebula debug                # run diagnostics"
    echo ""
    echo "Notes:"
    echo "- Configuration files are stored in $NEBULA_DIR"
    echo "- Nebula binary is installed to $NEBULA_BIN"
    echo "- All paths are resolved to absolute paths at runtime"
    echo "- This script uses systemd for service management"
    ;;
esac

# Set token Get your specific token from the system administrator
#sudo ./jetsonnebula.sh set-token "your-secure-token-here"
#Second option:
# Create the nebula directory if it doesn't exist
# sudo mkdir -p /etc/nebula

# # Create the token file
# sudo bash -c 'echo "your-secure-token-here" > /etc/nebula/token.txt'

# # Set proper permissions
# sudo chmod 600 /etc/nebula/token.txt

# - Install Nebula: sudo ./jetsonnebula.sh install
# - Verify it's working: sudo ./jetsonnebula.sh status