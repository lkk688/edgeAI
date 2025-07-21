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

# Default token and platform settings
DEFAULT_TOKEN="jetsonsupertoken"
DEFAULT_PLATFORM="linux-arm64"

# Try to load client-specific token if available
TOKEN_FILE="/etc/nebula/token.txt"
if [ -f "$TOKEN_FILE" ]; then
  TOKEN=$(cat "$TOKEN_FILE")
else
  TOKEN="$DEFAULT_TOKEN"
fi

PLATFORM="$DEFAULT_PLATFORM"

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
  sudo mkdir -p "$NEBULA_DIR"
  
  # Copy client files
  if [ -f "$nodename.crt" ] && [ -f "$nodename.key" ] && [ -f "config.yml" ]; then
    sudo cp "$nodename.crt" "$nodename.key" "config.yml" "$NEBULA_DIR/"
  else
    echo "[ERROR] Required client files not found in the downloaded bundle."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Copy CA certificate if available
  if [ -d "ca_temp" ] && [ -f "ca_temp/ca.crt" ]; then
    sudo cp "ca_temp/ca.crt" "$NEBULA_DIR/"
  elif [ ! -f "$NEBULA_DIR/ca.crt" ]; then
    echo "[ERROR] CA certificate not found and not available in $NEBULA_DIR."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  else
    echo "[INFO] Using existing CA certificate in $NEBULA_DIR."
  fi
  
  # Clean up
  cd - > /dev/null
  rm -rf "$TMP_DIR"
  
  echo "[INFO] Configuration files successfully installed to $NEBULA_DIR."
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
  if ! sudo cp "nebula" "$NEBULA_BIN"; then
    echo "[ERROR] Failed to copy nebula binary to $NEBULA_BIN."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  sudo chmod +x "$NEBULA_BIN"
  
  # Verify the binary works
  if ! sudo "$NEBULA_BIN" -version >/dev/null 2>&1; then
    echo "[WARNING] Nebula binary installed but may not be working correctly."
  else
    echo "[INFO] Nebula binary installed successfully: $($NEBULA_BIN -version 2>&1 | head -n 1)"
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
  if [ ! -f "$NEBULA_BIN" ]; then
    echo "[ERROR] Nebula binary not found at $NEBULA_BIN. Please install it first."
    return 1
  fi
  
  if [ ! -f "$NEBULA_DIR/config.yml" ]; then
    echo "[ERROR] Configuration file not found at $NEBULA_DIR/config.yml. Please download it first."
    return 1
  fi
  
  # Create systemd service file
  if ! sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Nebula VPN
After=network.target

[Service]
ExecStart=$NEBULA_BIN -config $NEBULA_DIR/config.yml
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
  sudo mkdir -p "$(dirname "$TOKEN_FILE")"
  echo "$1" | sudo tee "$TOKEN_FILE" > /dev/null
  echo "[INFO] Token saved to $TOKEN_FILE"
}

# === Main CLI logic ===
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
  *)
    echo "Usage: jetsonnebula install              # install/start Nebula"
    echo "       jetsonnebula status               # check status"
    echo "       jetsonnebula update               # update config only"
    echo "       jetsonnebula set-token <token>    # set client-specific token"
    echo "       jetsonnebula restart              # restart Nebula service"
    ;;
esac