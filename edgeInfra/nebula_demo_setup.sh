#!/bin/bash
# Nebula Overlay Network Demo Setup Script
# This script demonstrates how to set up a basic Nebula overlay network
# with one lighthouse and one client node

set -e

# Colors for output
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

echo -e "${BLUE}=== Nebula Overlay Network Demo Setup ===${NC}"
echo -e "This script will set up a basic Nebula overlay network with:"
echo -e "  - One lighthouse node (this machine)"
echo -e "  - One client node (can be another machine)"

# Create working directory
WORK_DIR="./nebula-demo"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Check if nebula and nebula-cert are installed
if ! command -v nebula &> /dev/null || ! command -v nebula-cert &> /dev/null; then
    echo -e "${YELLOW}Nebula binaries not found. Downloading...${NC}"
    
    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    if [[ "$ARCH" == "x86_64" ]]; then
        ARCH="amd64"
    elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        ARCH="arm64"
    else
        echo -e "${RED}Unsupported architecture: $ARCH${NC}"
        exit 1
    fi
    
    # Download latest release
    DOWNLOAD_URL="https://github.com/slackhq/nebula/releases/latest/download/nebula-$OS-$ARCH.tar.gz"
    echo -e "${BLUE}Downloading from: $DOWNLOAD_URL${NC}"
    
    curl -L "$DOWNLOAD_URL" -o nebula.tar.gz
    tar -xzf nebula.tar.gz
    
    # Make binaries executable and move to a directory in PATH
    chmod +x nebula nebula-cert
    
    echo -e "${GREEN}Nebula binaries downloaded and extracted${NC}"
    
    # Set path to local binaries
    NEBULA_BIN="./nebula"
    NEBULA_CERT_BIN="./nebula-cert"
else
    NEBULA_BIN="nebula"
    NEBULA_CERT_BIN="nebula-cert"
    echo -e "${GREEN}Nebula binaries found in PATH${NC}"
fi

# Create directories for lighthouse and client
mkdir -p lighthouse client

# Step 1: Create CA certificate
echo -e "\n${BLUE}Step 1: Creating Certificate Authority${NC}"
cd "$WORK_DIR"
$NEBULA_CERT_BIN ca -name "Nebula Demo Network"

echo -e "${GREEN}CA certificate created:${NC}"
ls -la ca.crt ca.key

# Step 2: Create lighthouse certificate
echo -e "\n${BLUE}Step 2: Creating lighthouse certificate${NC}"
$NEBULA_CERT_BIN sign -name "lighthouse" -ip "192.168.100.1/24" -ca-crt ca.crt -ca-key ca.key

# Move lighthouse certificate to its directory
mv lighthouse.crt lighthouse.key lighthouse/
cp ca.crt lighthouse/

# Step 3: Create client certificate
echo -e "\n${BLUE}Step 3: Creating client certificate${NC}"
$NEBULA_CERT_BIN sign -name "client1" -ip "192.168.100.2/24" -ca-crt ca.crt -ca-key ca.key

# Move client certificate to its directory
mv client1.crt client1.key client/
cp ca.crt client/

# Step 4: Create lighthouse config
echo -e "\n${BLUE}Step 4: Creating lighthouse configuration${NC}"
cat > lighthouse/config.yml << EOF
pki:
  ca: ./ca.crt
  cert: ./lighthouse.crt
  key: ./lighthouse.key

static_host_map:
  "192.168.100.1": ["localhost:4242"]

lighthouse:
  am_lighthouse: true
  serve_dns: false
  interval: 60

listen:
  host: 0.0.0.0
  port: 4242

punchy:
  punch: true
  respond: true

tun:
  disabled: false
  dev: nebula1
  drop_local_broadcast: false
  drop_multicast: false
  tx_queue: 500
  mtu: 1300

logging:
  level: info
  format: text

firewall:
  outbound:
    - port: any
      proto: any
      host: any

  inbound:
    - port: any
      proto: icmp
      host: any
    - port: 22
      proto: tcp
      host: any
    - port: 80,443
      proto: tcp
      host: any
EOF

# Step 5: Create client config
echo -e "\n${BLUE}Step 5: Creating client configuration${NC}"
cat > client/config.yml << EOF
pki:
  ca: ./ca.crt
  cert: ./client1.crt
  key: ./client1.key

static_host_map:
  "192.168.100.1": ["YOUR_PUBLIC_IP:4242"]

lighthouse:
  am_lighthouse: false
  interval: 60
  hosts:
    - "192.168.100.1"

listen:
  host: 0.0.0.0
  port: 0

punchy:
  punch: true
  respond: true

tun:
  disabled: false
  dev: nebula1
  drop_local_broadcast: false
  drop_multicast: false
  tx_queue: 500
  mtu: 1300

logging:
  level: info
  format: text

firewall:
  outbound:
    - port: any
      proto: any
      host: any

  inbound:
    - port: any
      proto: icmp
      host: any
    - port: 22
      proto: tcp
      host: any
EOF

# Step 6: Test configurations
echo -e "\n${BLUE}Step 6: Testing configurations${NC}"
echo -e "${YELLOW}Testing lighthouse config...${NC}"
cd "$WORK_DIR/lighthouse"
$NEBULA_BIN -config config.yml -test

echo -e "\n${YELLOW}Testing client config...${NC}"
cd "$WORK_DIR/client"
$NEBULA_BIN -config config.yml -test

# Step 7: Create client bundle
echo -e "\n${BLUE}Step 7: Creating client bundle for distribution${NC}"
cd "$WORK_DIR"
zip -r client-bundle.zip client/

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\n${YELLOW}To start the lighthouse:${NC}"
echo -e "  cd $WORK_DIR/lighthouse"
echo -e "  sudo $NEBULA_BIN -config config.yml"

echo -e "\n${YELLOW}To distribute to client:${NC}"
echo -e "  1. Copy $WORK_DIR/client-bundle.zip to the client machine"
echo -e "  2. Edit client/config.yml to set YOUR_PUBLIC_IP to the public IP of this lighthouse machine"
echo -e "  3. On the client machine, run: sudo nebula -config config.yml"

echo -e "\n${YELLOW}To test connectivity:${NC}"
echo -e "  From lighthouse: ping 192.168.100.2"
echo -e "  From client: ping 192.168.100.1"

echo -e "\n${BLUE}For more information, see the Nebula Overlay Network Guide in the documentation.${NC}"