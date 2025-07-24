#!/bin/bash

# Script to update Nebula configuration with hostname-based certificate names
# Usage: sudo ./update_nebula_config.sh

set -e  # Exit on any error

CONFIG_DIR="/etc/nebula"
CONFIG_FILE="$CONFIG_DIR/config.yml"
BACKUP_FILE="$CONFIG_DIR/config.yml.backup.$(date +%Y%m%d_%H%M%S)"
TEMP_CONFIG="/tmp/nebula_config_temp.yml"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

echo "[INFO] Starting Nebula configuration update..."

# Step 1: Backup existing config
if [[ -f "$CONFIG_FILE" ]]; then
    echo "[INFO] Backing up existing config to: $BACKUP_FILE"
    cp "$CONFIG_FILE" "$BACKUP_FILE"
else
    echo "[WARN] No existing config file found at $CONFIG_FILE"
fi

# Step 2: Get hostname and extract last two digits
HOSTNAME=$(hostname)
echo "[INFO] Current hostname: $HOSTNAME"

# Extract last two digits from hostname (e.g., sjsujetson-01 -> 01)
LAST_TWO_DIGITS=$(echo "$HOSTNAME" | grep -o '[0-9]\{2\}$' || echo "00")
if [[ -z "$LAST_TWO_DIGITS" ]]; then
    echo "[WARN] Could not extract two digits from hostname, using '00'"
    LAST_TWO_DIGITS="00"
fi

echo "[INFO] Extracted digits: $LAST_TWO_DIGITS"

# Step 3: Create new config based on template
echo "[INFO] Creating new config with jetson$LAST_TWO_DIGITS certificate names"

# Use the provided template and modify cert/key paths
cat > "$TEMP_CONFIG" << EOF
#/etc/nebula/config.yml
pki:
  ca: /etc/nebula/ca.crt
  cert: /etc/nebula/jetson$LAST_TWO_DIGITS.crt
  key: /etc/nebula/jetson$LAST_TWO_DIGITS.key

lighthouse:
  am_lighthouse: false
  interval: 60
  hosts:
    - "192.168.100.1"
    - "192.168.100.2"

static_host_map:
  "192.168.100.1": ["lkk688.duckdns.org:8883"]
  "192.168.100.2": ["edgeai.duckdns.org:4242"]

relay:
  relays:
    - 192.168.100.2
  am_relay: false
  use_relays: true

punchy:
  punch: true

listen:
  host: 0.0.0.0
  port: 0

tun:
  dev: nebula1

firewall:
  inbound:
    - port: any
      proto: any
      host: any
  outbound:
    - port: any
      proto: any
      host: any
EOF

# Step 4: Validate the certificate files exist
CERT_FILE="$CONFIG_DIR/jetson$LAST_TWO_DIGITS.crt"
KEY_FILE="$CONFIG_DIR/jetson$LAST_TWO_DIGITS.key"

if [[ ! -f "$CERT_FILE" ]]; then
    echo "[WARN] Certificate file not found: $CERT_FILE"
    echo "[INFO] Please ensure the certificate file exists before restarting Nebula"
fi

if [[ ! -f "$KEY_FILE" ]]; then
    echo "[WARN] Key file not found: $KEY_FILE"
    echo "[INFO] Please ensure the key file exists before restarting Nebula"
fi

# Step 5: Move new config to final location
echo "[INFO] Installing new config file"
mv "$TEMP_CONFIG" "$CONFIG_FILE"
chown root:root "$CONFIG_FILE"
chmod 644 "$CONFIG_FILE"

echo "[INFO] New config file created with:"
echo "  - Certificate: $CERT_FILE"
echo "  - Key: $KEY_FILE"

# Step 6: Restart Nebula service
echo "[INFO] Restarting Nebula service..."
systemctl restart nebula

# Step 7: Wait a moment and check status
echo "[INFO] Waiting 3 seconds for service to start..."
sleep 3

echo "[INFO] Checking Nebula service status..."
systemctl status nebula --no-pager -l

echo "[INFO] Nebula configuration update completed!"
echo "[INFO] Backup saved to: $BACKUP_FILE"
echo "[INFO] Active config: $CONFIG_FILE"