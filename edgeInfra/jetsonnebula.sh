#!/bin/bash
set -e

# === Parse hostname and determine identity ===
hostname=$(hostname)
suffix=$(echo "$hostname" | grep -oE '[0-9]{2}$')
nodename="jetson${suffix}"

CONFIG_SERVER="http://your-fastapi-server:8000/nebula"
NEBULA_DIR="/etc/nebula"
NEBULA_BIN="/usr/local/bin/nebula"
SERVICE_FILE="/etc/systemd/system/nebula.service"

# === Function: download config ===
download_config() {
  echo "[INFO] Downloading config for $nodename from $CONFIG_SERVER..."
  sudo mkdir -p "$NEBULA_DIR"
  for f in "$nodename.crt" "$nodename.key" "ca.crt" "config.yml"; do
    curl -sSf "$CONFIG_SERVER/$nodename/$f" -o "$NEBULA_DIR/$f"
  done
}

# === Function: install nebula ===
install_nebula() {
  if command -v nebula >/dev/null 2>&1; then
    echo "[INFO] Nebula already installed at: $(command -v nebula)"
  else
    echo "[INFO] Installing Nebula..."
    wget -q https://github.com/slackhq/nebula/releases/latest/download/nebula-linux-arm64.tar.gz
    tar -xzf nebula-linux-arm64.tar.gz
    sudo mv nebula "$NEBULA_BIN"
    rm -f nebula-linux-arm64.tar.gz
  fi
}

# === Function: create systemd service ===
setup_systemd() {
  echo "[INFO] Creating systemd service for Nebula..."
  sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Nebula VPN
After=network.target

[Service]
ExecStart=$NEBULA_BIN -config $NEBULA_DIR/config.yml
Restart=always

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reexec
  sudo systemctl enable --now nebula
}

# === Function: check nebula status ===
check_status() {
  echo "[INFO] Checking nebula service status..."
  sudo systemctl status nebula --no-pager
}


# === Main CLI logic ===
case "$1" in
  install)
    install_nebula
    download_config
    setup_systemd
    check_status
    ;;
  status)
    check_status
    ;;
  *)
    echo "Usage: jetsonnebula install     # install/start Nebula"
    echo "       jetsonnebula status     # check status"
    ;;
esac