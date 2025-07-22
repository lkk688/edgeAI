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

# Function to validate token format
validate_token() {
  local token="$1"
  # Check if token is empty or only whitespace
  if [ -z "${token// /}" ]; then
    echo "[WARNING] Token is empty or contains only whitespace"
    return 1
  fi
  # Check if token contains any special characters that might need URL encoding
  if [[ "$token" =~ [^a-zA-Z0-9] ]]; then
    echo "[WARNING] Token contains special characters that might need URL encoding"
  fi
  return 0
}

# Function to URL encode a string
urlencode() {
  local string="$1"
  local length="${#string}"
  local encoded=""
  local pos c o
  
  for (( pos=0; pos<length; pos++ )); do
    c="${string:$pos:1}"
    case "$c" in
      [a-zA-Z0-9.~_-]) o="$c" ;;
      *) printf -v o '%%%02X' "'$c" ;;
    esac
    encoded+="$o"
  done
  echo "$encoded"
}

if [ -f "$TOKEN_FILE" ]; then
  TOKEN=$(cat "$TOKEN_FILE")
  echo "[DEBUG] Loaded token from file: $TOKEN_FILE"
  if ! validate_token "$TOKEN"; then
    echo "[WARNING] Using default token instead of invalid token from file"
    TOKEN="$DEFAULT_TOKEN"
  fi
else
  echo "[DEBUG] Token file not found. Using default token."
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
  
  # Test API server connectivity and authentication
  echo "[DEBUG] Testing API server connectivity..."
  if ! curl -s --head "$API_SERVER" > /dev/null; then
    echo "[ERROR] Cannot connect to API server: $API_SERVER"
    echo "[DEBUG] Please check your network connection and server status."
    return 1
  fi
  echo "[DEBUG] API server is reachable."
  
  # Test authentication with token
  echo "[DEBUG] Testing API authentication with token..."
  local encoded_token=$(urlencode "$TOKEN")
  local auth_test_result=$(curl -s -o /dev/null -w "%{http_code}" -G -d "token=$encoded_token" "$API_SERVER/auth/test")
  
  if [ "$auth_test_result" = "200" ] || [ "$auth_test_result" = "204" ]; then
    echo "[DEBUG] Authentication successful with provided token."
  elif [ "$auth_test_result" = "401" ] || [ "$auth_test_result" = "403" ]; then
    echo "[WARNING] Authentication failed with provided token (HTTP $auth_test_result)."
    echo "[DEBUG] Token value (first 3 chars): ${TOKEN:0:3}..."
    echo "[DEBUG] You may need to set a valid token using the 'set-token' command."
    echo "[DEBUG] Continuing with installation, but downloads may fail."
  else
    echo "[WARNING] Unexpected response from authentication test: HTTP $auth_test_result"
    echo "[DEBUG] The server may not support the auth test endpoint or may be misconfigured."
  fi
  
  # Create a temporary directory for downloads
  TMP_DIR=$(mktemp -d)
  cd "$TMP_DIR"
  
  # Download the client bundle
  echo "[INFO] Downloading client bundle..."
  echo "[DEBUG] Using token: $TOKEN"
  echo "[DEBUG] Platform: $PLATFORM"
  echo "[DEBUG] Node name: $nodename"
  
  # URL encode the token
  local encoded_token=$(urlencode "$TOKEN")
  echo "[DEBUG] Using URL encoded token for API requests"
  
  # Try primary endpoint first
  PRIMARY_URL="$API_SERVER/download/$nodename"
  echo "[DEBUG] Trying primary client bundle URL: $PRIMARY_URL"
  
  if curl -v -f -o "${nodename}.zip" -G -d "token=$encoded_token" -d "platform=$PLATFORM" "$PRIMARY_URL"; then
    echo "[INFO] Successfully downloaded client bundle from primary URL."
  else
    # Try alternative endpoint if primary fails
    ALT_URL="$API_SERVER/api/download/$nodename"
    echo "[DEBUG] Primary download failed. Trying alternative URL: $ALT_URL"
    
    if curl -v -f -o "${nodename}.zip" -G -d "token=$encoded_token" -d "platform=$PLATFORM" "$ALT_URL"; then
      echo "[INFO] Successfully downloaded client bundle from alternative URL."
    else
      # Try direct URL with token in query string as a last resort
      DIRECT_URL="$API_SERVER/download/${nodename}?token=$encoded_token&platform=$PLATFORM"
      echo "[DEBUG] Alternative download failed. Trying direct URL (token redacted)"
      
      if curl -v -f -o "${nodename}.zip" "$DIRECT_URL"; then
        echo "[INFO] Successfully downloaded client bundle using direct URL."
      else
        echo "[ERROR] Failed to download client bundle from all endpoints. Check your token and network connection."
        echo "[DEBUG] Curl exit code: $?"
        echo "[DEBUG] Token value (first 3 chars): ${TOKEN:0:3}..."
        cd - > /dev/null
        rm -rf "$TMP_DIR"
        return 1
      fi
    fi
  fi
  
  # Debug: List downloaded files
  echo "[DEBUG] Downloaded files in temporary directory:"
  ls -la
  
  # Download CA certificate - Using approach from guestnebula.sh which works fine
  echo "[INFO] Downloading CA certificate..."
  
  # Use a specific token for CA download that is known to work
  if ! curl -v -f -o "ca.zip" -G -d "token=jetsonsupertoken" "$API_SERVER/public/downloads/ca.zip"; then
    echo "[ERROR] Failed to download CA certificate. This is required for Nebula to function."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  else
    echo "[DEBUG] CA certificate file size: $(ls -lh "ca.zip" | awk '{print $5}')"
  fi
  
  # Function to check if a file is a valid zip archive
  check_zip_file() {
    local zip_file="$1"
    echo "[DEBUG] Checking zip file integrity: $zip_file"
    
    # Check if file exists and is not empty
    if [ ! -f "$zip_file" ] || [ ! -s "$zip_file" ]; then
      echo "[ERROR] File does not exist or is empty: $zip_file"
      return 1
    fi
    
    # Check file type
    local file_type=$(file -b "$zip_file")
    echo "[DEBUG] File type: $file_type"
    
    # Check if it's a zip file
    if [[ "$file_type" != *"Zip archive"* && "$file_type" != *"ZIP archive"* ]]; then
      echo "[ERROR] Not a valid zip archive: $zip_file"
      # If it's an HTML or text file, show the first few lines to help diagnose the issue
      if [[ "$file_type" == *"HTML"* || "$file_type" == *"text"* ]]; then
        echo "[DEBUG] File appears to be HTML or text. First 5 lines:"
        head -n 5 "$zip_file"
      fi
      return 1
    fi
    
    # Try to list contents without extracting
    if ! unzip -l "$zip_file" > /dev/null 2>&1; then
      echo "[ERROR] File is corrupt or not a valid zip archive: $zip_file"
      return 1
    fi
    
    return 0
  }
  
  # List files again before extraction
  echo "[DEBUG] Files before extraction:"
  ls -la
  
  # Extract the downloaded files
  echo "[INFO] Extracting files..."
  if [ -f "$nodename.zip" ]; then
    # Check if the zip file is valid
    if check_zip_file "$nodename.zip"; then
      echo "[DEBUG] Zip file is valid, proceeding with extraction"
      if ! unzip -o "$nodename.zip"; then
        echo "[ERROR] Failed to extract client bundle despite valid zip file."
        cd - > /dev/null
        rm -rf "$TMP_DIR"
        return 1
      fi
    else
      echo "[ERROR] Client bundle is not a valid zip file."
      cd - > /dev/null
      rm -rf "$TMP_DIR"
      return 1
    fi
  else
    echo "[ERROR] Client bundle zip file not found: $nodename.zip"
    echo "[DEBUG] Current directory contents:"
    ls -la
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Extract CA certificate if downloaded successfully
  if [ -f "ca.zip" ]; then
    echo "[DEBUG] Checking CA certificate zip file"
    if check_zip_file "ca.zip"; then
      echo "[DEBUG] CA zip file is valid, proceeding with extraction"
      if ! unzip -o "ca.zip" -d ca_temp; then
        echo "[WARNING] Failed to extract CA certificate despite valid zip file."
      fi
    else
      echo "[WARNING] CA certificate is not a valid zip file. Will try to use existing one if available."
    fi
  else
    echo "[DEBUG] No CA certificate zip file found"
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
  
  # Function to try downloading individual files directly
  download_individual_files() {
    echo "[INFO] Attempting to download individual client files directly..."
    local encoded_token=$(urlencode "$TOKEN")
    local success=true
    
    # Try to download certificate
    echo "[DEBUG] Downloading client certificate..."
    if ! curl -f -o "$nodename.crt" -G -d "token=$encoded_token" "$API_SERVER/download/$nodename/cert"; then
      echo "[ERROR] Failed to download client certificate."
      success=false
    fi
    
    # Try to download key
    echo "[DEBUG] Downloading client key..."
    if ! curl -f -o "$nodename.key" -G -d "token=$encoded_token" "$API_SERVER/download/$nodename/key"; then
      echo "[ERROR] Failed to download client key."
      success=false
    fi
    
    # Try to download config
    echo "[DEBUG] Downloading client config..."
    if ! curl -f -o "config.yml" -G -d "token=$encoded_token" "$API_SERVER/download/$nodename/config"; then
      echo "[ERROR] Failed to download client config."
      success=false
    fi
    
    if [ "$success" = true ]; then
      echo "[INFO] Successfully downloaded individual client files."
      return 0
    else
      echo "[ERROR] Failed to download one or more individual client files."
      return 1
    fi
  }
  
  # Function to generate default configuration file if needed
  generate_default_config() {
    echo "[INFO] Generating default configuration file..."
    local success=true
    
    # We no longer generate self-signed certificates as they won't work with the CA
    # Instead, we require proper certificates to be downloaded
    
    # Generate a default config.yml if it doesn't exist
    if [ ! -f "config.yml" ]; then
      echo "[DEBUG] Generating default config.yml..."
      cat > "config.yml" << EOF
# This is a default Nebula configuration file generated by jetsonnebula.sh
# You should replace this with a proper configuration when possible

pki:
  ca: /etc/nebula/ca.crt
  cert: /etc/nebula/$nodename.crt
  key: /etc/nebula/$nodename.key

static_host_map:
  "10.42.42.1": ["lkk688.duckdns.org:4242"]

lighthouse:
  am_lighthouse: false
  interval: 60
  hosts:
    - "10.42.42.1"

listen:
  host: 0.0.0.0
  port: 4242

tun:
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
      proto: any
      host: any
EOF
      
      if [ ! -f "config.yml" ] || [ ! -s "config.yml" ]; then
        echo "[ERROR] Failed to generate default config.yml."
        success=false
      else
        echo "[INFO] Generated default config.yml."
      fi
    fi
    
    if [ "$success" = true ]; then
      echo "[INFO] Successfully generated default configuration files."
      echo "[WARNING] These are TEMPORARY files and should be replaced with proper ones when possible."
      return 0
    else
      echo "[ERROR] Failed to generate one or more default configuration files."
      return 1
    fi
  }
  
  # Copy client files
  if [ -f "$nodename.crt" ] && [ -f "$nodename.key" ] && [ -f "config.yml" ]; then
    echo "[DEBUG] Copying client files to: $ABSOLUTE_NEBULA_DIR"
    sudo cp "$nodename.crt" "$nodename.key" "config.yml" "$ABSOLUTE_NEBULA_DIR/"
  else
    echo "[WARNING] Required client files not found in the downloaded bundle."
    echo "[INFO] Trying to download individual files directly..."
    
    if download_individual_files; then
      echo "[DEBUG] Copying individually downloaded files to: $ABSOLUTE_NEBULA_DIR"
      sudo cp "$nodename.crt" "$nodename.key" "config.yml" "$ABSOLUTE_NEBULA_DIR/"
    else
      echo "[ERROR] Could not download required client files. Cannot proceed without valid certificates."
      echo "[INFO] Please obtain a valid token and run 'set-token' followed by 'install' again."
      cd - > /dev/null
      rm -rf "$TMP_DIR"
      return 1
    fi
  fi
  
  # Copy CA certificate if available
  if [ -d "ca_temp" ] && [ -f "ca_temp/ca.crt" ]; then
    echo "[DEBUG] Copying CA certificate from ca_temp directory to: $ABSOLUTE_NEBULA_DIR"
    sudo cp "ca_temp/ca.crt" "$ABSOLUTE_NEBULA_DIR/"
  elif [ -f "ca.crt" ]; then
    echo "[DEBUG] Copying CA certificate from current directory to: $ABSOLUTE_NEBULA_DIR"
    sudo cp "ca.crt" "$ABSOLUTE_NEBULA_DIR/"
  elif [ -f "$ABSOLUTE_NEBULA_DIR/ca.crt" ]; then
    echo "[INFO] Using existing CA certificate in $ABSOLUTE_NEBULA_DIR."
  else
    echo "[ERROR] CA certificate not found anywhere. Cannot proceed without a valid CA certificate."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  # Clean up
  cd - > /dev/null
  rm -rf "$TMP_DIR"
  
  echo "[DEBUG] Verifying downloaded configuration:"
  sudo ls -la "$ABSOLUTE_NEBULA_DIR"
  
  # Verify that all required files are present
  if [ ! -f "$ABSOLUTE_NEBULA_DIR/$nodename.crt" ] || [ ! -f "$ABSOLUTE_NEBULA_DIR/$nodename.key" ] || [ ! -f "$ABSOLUTE_NEBULA_DIR/config.yml" ] || [ ! -f "$ABSOLUTE_NEBULA_DIR/ca.crt" ]; then
    echo "[ERROR] One or more required files are missing from $ABSOLUTE_NEBULA_DIR."
    echo "[DEBUG] Required files:"
    echo "  - $ABSOLUTE_NEBULA_DIR/$nodename.crt: $([ -f "$ABSOLUTE_NEBULA_DIR/$nodename.crt" ] && echo "Present" || echo "Missing")"
    echo "  - $ABSOLUTE_NEBULA_DIR/$nodename.key: $([ -f "$ABSOLUTE_NEBULA_DIR/$nodename.key" ] && echo "Present" || echo "Missing")"
    echo "  - $ABSOLUTE_NEBULA_DIR/config.yml: $([ -f "$ABSOLUTE_NEBULA_DIR/config.yml" ] && echo "Present" || echo "Missing")"
    echo "  - $ABSOLUTE_NEBULA_DIR/ca.crt: $([ -f "$ABSOLUTE_NEBULA_DIR/ca.crt" ] && echo "Present" || echo "Missing")"
    return 1
  fi
  
  echo "[INFO] Configuration files successfully installed to $ABSOLUTE_NEBULA_DIR."
  echo "[INFO] All required files are present and ready for use."
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
  if ! curl -f -o "$nodename.zip" -G -d "token=$TOKEN" -d "platform=$PLATFORM" "$API_SERVER/download/$nodename"; then
    echo "[ERROR] Failed to download client bundle. Check your token and network connection."
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    return 1
  fi
  
  echo "[DEBUG] Downloaded client bundle file size: $(ls -lh "$nodename.zip" | awk '{print $5}')"
  
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
  
  local new_token="$1"
  echo "[INFO] Setting client token to: $new_token"
  
  # Validate token format
  if ! validate_token "$new_token"; then
    echo "[ERROR] Invalid token format. Please provide a valid token."
    return 1
  fi
  
  echo "[DEBUG] Using token file at: $ABSOLUTE_TOKEN_FILE"
  
  # Ensure token file path is valid
  if [ -z "$ABSOLUTE_TOKEN_FILE" ]; then
    echo "[ERROR] Token file path is empty. Using default path: /etc/nebula/token.txt"
    ABSOLUTE_TOKEN_FILE="/etc/nebula/token.txt"
  fi
  
  # Create directory if it doesn't exist
  sudo mkdir -p "$(dirname "$ABSOLUTE_TOKEN_FILE")"
  
  # Save token to file (trim any whitespace)
  echo "$new_token" | tr -d '[:space:]' | sudo tee "$ABSOLUTE_TOKEN_FILE" > /dev/null
  
  # Verify token was saved correctly
  if [ ! -f "$ABSOLUTE_TOKEN_FILE" ]; then
    echo "[ERROR] Failed to save token to file: $ABSOLUTE_TOKEN_FILE"
    return 1
  fi
  
  # Read back the token to ensure it was saved correctly
  local saved_token=$(sudo cat "$ABSOLUTE_TOKEN_FILE")
  if [ -z "$saved_token" ]; then
    echo "[ERROR] Token was saved but appears to be empty"
    return 1
  fi
  
  # Update TOKEN variable for immediate use
  TOKEN="$saved_token"
  
  echo "[INFO] Token saved to $ABSOLUTE_TOKEN_FILE"
  echo "[DEBUG] Token value: $TOKEN"
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
  test-token)
    echo "[INFO] Testing token authentication with API server..."
    
    # Check if token file exists
    if [ ! -f "$ABSOLUTE_TOKEN_FILE" ]; then
      echo "[ERROR] Token file not found: $ABSOLUTE_TOKEN_FILE"
      echo "[INFO] Please set a token first using: $0 set-token <your-token>"
      exit 1
    fi
    
    # Load token
    TOKEN=$(sudo cat "$ABSOLUTE_TOKEN_FILE")
    if [ -z "$TOKEN" ]; then
      echo "[ERROR] Token file exists but is empty"
      echo "[INFO] Please set a token using: $0 set-token <your-token>"
      exit 1
    fi
    
    # Validate token
    if ! validate_token "$TOKEN"; then
      echo "[ERROR] Token is invalid"
      echo "[INFO] Please set a valid token using: $0 set-token <your-token>"
      exit 1
    fi
    
    echo "[DEBUG] Token value (first 3 chars): ${TOKEN:0:3}..."
    echo "[DEBUG] API server: $API_SERVER"
    
    # Test API server connectivity
    echo "[DEBUG] Testing API server connectivity..."
    if ! curl -s --head "$API_SERVER" > /dev/null; then
      echo "[ERROR] Cannot connect to API server: $API_SERVER"
      echo "[DEBUG] Please check your network connection and server status."
      exit 1
    fi
    
    echo "[STATUS] API server is reachable: $API_SERVER ✅"
    
    # Test authentication with token
    echo "[DEBUG] Testing API authentication with token..."
    local encoded_token=$(urlencode "$TOKEN")
    
    # Try different authentication endpoints
    local endpoints=(
      "/auth/test"
      "/api/auth/test"
      "/download/$nodename"
      "/api/download/$nodename"
    )
    
    local success=false
    
    for endpoint in "${endpoints[@]}"; do
      echo "[DEBUG] Testing endpoint: $API_SERVER$endpoint"
      local auth_test_result=$(curl -s -o /dev/null -w "%{http_code}" -G -d "token=$encoded_token" "$API_SERVER$endpoint")
      
      if [ "$auth_test_result" = "200" ] || [ "$auth_test_result" = "204" ]; then
        echo "[SUCCESS] Authentication successful with provided token ✅"
        echo "[DEBUG] Endpoint $endpoint returned HTTP $auth_test_result"
        success=true
        break
      elif [ "$auth_test_result" = "401" ] || [ "$auth_test_result" = "403" ]; then
        echo "[ERROR] Authentication failed with provided token (HTTP $auth_test_result) ❌"
        echo "[DEBUG] Endpoint $endpoint returned HTTP $auth_test_result"
      else
        echo "[WARNING] Unexpected response from endpoint $endpoint: HTTP $auth_test_result ⚠️"
      fi
    done
    
    if [ "$success" = "true" ]; then
      echo "[SUCCESS] Token is valid and working with the API server"
      exit 0
    else
      echo "[ERROR] Token authentication failed with all endpoints"
      echo "[INFO] Please check your token or contact your administrator"
      exit 1
    fi
    ;;
  test-download)
    echo "[INFO] Testing client bundle download..."
    
    # Check if token file exists and load token
    if [ ! -f "$ABSOLUTE_TOKEN_FILE" ]; then
      echo "[ERROR] Token file not found: $ABSOLUTE_TOKEN_FILE"
      echo "[INFO] Please set a token first using: $0 set-token <your-token>"
      exit 1
    fi
    
    # Load token
    TOKEN=$(sudo cat "$ABSOLUTE_TOKEN_FILE")
    if [ -z "$TOKEN" ]; then
      echo "[ERROR] Token file exists but is empty"
      echo "[INFO] Please set a token using: $0 set-token <your-token>"
      exit 1
    fi
    
    # Validate token
    if ! validate_token "$TOKEN"; then
      echo "[ERROR] Token is invalid"
      echo "[INFO] Please set a valid token using: $0 set-token <your-token>"
      exit 1
    fi
    
    echo "[DEBUG] Token value (first 3 chars): ${TOKEN:0:3}..."
    echo "[DEBUG] API server: $API_SERVER"
    echo "[DEBUG] Platform: $platform"
    echo "[DEBUG] Node name: $nodename"
    
    # Test API server connectivity
    echo "[DEBUG] Testing API server connectivity..."
    if ! curl -s --head "$API_SERVER" > /dev/null; then
      echo "[ERROR] Cannot connect to API server: $API_SERVER"
      echo "[DEBUG] Please check your network connection and server status."
      exit 1
    fi
    
    echo "[STATUS] API server is reachable: $API_SERVER ✅"
    
    # Create a temporary directory for testing downloads
    local test_dir="/tmp/nebula_test_download"
    rm -rf "$test_dir" # Clean up any previous test
    mkdir -p "$test_dir"
    echo "[DEBUG] Created temporary test directory: $test_dir"
    
    # URL encode the token
    local encoded_token=$(urlencode "$TOKEN")
    
    # Test client bundle download
    echo "[INFO] Testing client bundle download..."
    
    # Try different download methods
    local bundle_file="$test_dir/client.zip"
    local download_success=false
    local download_methods=(
      "curl -s -o '$bundle_file' -d 'token=$encoded_token' '$API_SERVER/download/$nodename'"
      "curl -s -o '$bundle_file' -d 'token=$encoded_token' '$API_SERVER/api/download/$nodename'"
      "curl -s -o '$bundle_file' '$API_SERVER/download/$nodename?token=$encoded_token'"
      "curl -s -o '$bundle_file' '$API_SERVER/api/download/$nodename?token=$encoded_token'"
    )
    
    for method in "${download_methods[@]}"; do
      echo "[DEBUG] Trying download method: $method"
      eval $method
      local curl_exit=$?
      
      echo "[DEBUG] curl exit code: $curl_exit"
      
      if [ $curl_exit -eq 0 ] && [ -f "$bundle_file" ] && [ -s "$bundle_file" ]; then
        # Check if it's a valid zip file
        if check_zip_file "$bundle_file"; then
          echo "[SUCCESS] Client bundle downloaded successfully ✅"
          download_success=true
          break
        else
          echo "[ERROR] Downloaded file is not a valid zip archive ❌"
          rm -f "$bundle_file"
        fi
      else
        echo "[ERROR] Failed to download client bundle ❌"
        rm -f "$bundle_file"
      fi
    done
    
    if [ "$download_success" = "true" ]; then
      echo "[DEBUG] Testing bundle extraction..."
      mkdir -p "$test_dir/extract"
      if unzip -q "$bundle_file" -d "$test_dir/extract"; then
        echo "[SUCCESS] Bundle extraction successful ✅"
        echo "[DEBUG] Extracted files:"
        ls -la "$test_dir/extract"
        
        # Check for required files
        local required_files=("$nodename.crt" "$nodename.key" "config.yml")
        local missing_files=false
        
        for file in "${required_files[@]}"; do
          if [ ! -f "$test_dir/extract/$file" ]; then
            echo "[ERROR] Required file missing from bundle: $file ❌"
            missing_files=true
          else
            echo "[STATUS] Required file found in bundle: $file ✅"
          fi
        done
        
        if [ "$missing_files" = "false" ]; then
          echo "[SUCCESS] All required files found in client bundle ✅"
        else
          echo "[ERROR] Some required files are missing from the client bundle ❌"
        fi
      else
        echo "[ERROR] Failed to extract client bundle ❌"
      fi
    fi
    
    # Test CA certificate download
    echo "[INFO] Testing CA certificate download..."
    local ca_file="$test_dir/ca.crt"
    local ca_success=false
    local ca_methods=(
      "curl -s -o '$ca_file' -d 'token=$encoded_token' '$API_SERVER/ca'"
      "curl -s -o '$ca_file' -d 'token=$encoded_token' '$API_SERVER/api/ca'"
      "curl -s -o '$ca_file' '$API_SERVER/ca?token=$encoded_token'"
      "curl -s -o '$ca_file' '$API_SERVER/api/ca?token=$encoded_token'"
    )
    
    for method in "${ca_methods[@]}"; do
      echo "[DEBUG] Trying CA download method: $method"
      eval $method
      local curl_exit=$?
      
      echo "[DEBUG] curl exit code: $curl_exit"
      
      if [ $curl_exit -eq 0 ] && [ -f "$ca_file" ] && [ -s "$ca_file" ]; then
        # Check if it looks like a certificate
        if grep -q "BEGIN CERTIFICATE" "$ca_file"; then
          echo "[SUCCESS] CA certificate downloaded successfully ✅"
          ca_success=true
          break
        else
          echo "[ERROR] Downloaded file is not a valid certificate ❌"
          rm -f "$ca_file"
        fi
      else
        echo "[ERROR] Failed to download CA certificate ❌"
        rm -f "$ca_file"
      fi
    done
    
    # Summary
    echo "\n[SUMMARY] Download Test Results:"
    if [ "$download_success" = "true" ]; then
      echo "[✅] Client bundle download: SUCCESS"
    else
      echo "[❌] Client bundle download: FAILED"
    fi
    
    if [ "$ca_success" = "true" ]; then
      echo "[✅] CA certificate download: SUCCESS"
    else
      echo "[❌] CA certificate download: FAILED"
    fi
    
    # Clean up
    echo "[DEBUG] Cleaning up test directory..."
    rm -rf "$test_dir"
    
    if [ "$download_success" = "true" ] && [ "$ca_success" = "true" ]; then
      echo "[SUCCESS] All download tests passed successfully! ✅"
      echo "[INFO] You can now proceed with installation: $0 install"
      exit 0
    else
      echo "[ERROR] Some download tests failed. Please check the errors above. ❌"
      exit 1
    fi
    ;;
  test-config)
    echo "[INFO] Testing Nebula configuration files..."
    
    # Check if Nebula directory exists
    if [ ! -d "$ABSOLUTE_NEBULA_DIR" ]; then
      echo "[ERROR] Nebula configuration directory not found: $ABSOLUTE_NEBULA_DIR"
      echo "[INFO] Please run installation first: $0 install"
      exit 1
    fi
    
    echo "[DEBUG] Checking configuration files in: $ABSOLUTE_NEBULA_DIR"
    
    # Check for required files
    local required_files=("$nodename.crt" "$nodename.key" "config.yml" "ca.crt")
    local missing_files=false
    
    for file in "${required_files[@]}"; do
      if [ ! -f "$ABSOLUTE_NEBULA_DIR/$file" ]; then
        echo "[ERROR] Required file missing: $ABSOLUTE_NEBULA_DIR/$file ❌"
        missing_files=true
      else
        echo "[STATUS] Required file found: $ABSOLUTE_NEBULA_DIR/$file ✅"
      fi
    done
    
    if [ "$missing_files" = "true" ]; then
      echo "[ERROR] Some required configuration files are missing ❌"
      echo "[INFO] You may need to run: $0 update"
      exit 1
    fi
    
    # Validate certificate files
    echo "[DEBUG] Validating certificate files..."
    
    # Check client certificate
    echo "[DEBUG] Checking client certificate: $ABSOLUTE_NEBULA_DIR/$nodename.crt"
    if ! openssl x509 -in "$ABSOLUTE_NEBULA_DIR/$nodename.crt" -noout 2>/dev/null; then
      echo "[ERROR] Invalid client certificate: $ABSOLUTE_NEBULA_DIR/$nodename.crt ❌"
      exit 1
    else
      echo "[STATUS] Client certificate is valid ✅"
      echo "[DEBUG] Certificate details:"
      openssl x509 -in "$ABSOLUTE_NEBULA_DIR/$nodename.crt" -noout -text | grep -E 'Subject:|Issuer:|Not Before:|Not After :'
    fi
    
    # Check CA certificate
    echo "[DEBUG] Checking CA certificate: $ABSOLUTE_NEBULA_DIR/ca.crt"
    if ! openssl x509 -in "$ABSOLUTE_NEBULA_DIR/ca.crt" -noout 2>/dev/null; then
      echo "[ERROR] Invalid CA certificate: $ABSOLUTE_NEBULA_DIR/ca.crt ❌"
      exit 1
    else
      echo "[STATUS] CA certificate is valid ✅"
      echo "[DEBUG] CA certificate details:"
      openssl x509 -in "$ABSOLUTE_NEBULA_DIR/ca.crt" -noout -text | grep -E 'Subject:|Issuer:|Not Before:|Not After :'
    fi
    
    # Check private key
    echo "[DEBUG] Checking private key: $ABSOLUTE_NEBULA_DIR/$nodename.key"
    if ! openssl rsa -in "$ABSOLUTE_NEBULA_DIR/$nodename.key" -check -noout 2>/dev/null; then
      echo "[ERROR] Invalid private key: $ABSOLUTE_NEBULA_DIR/$nodename.key ❌"
      exit 1
    else
      echo "[STATUS] Private key is valid ✅"
    fi
    
    # Verify certificate and key match
    echo "[DEBUG] Verifying certificate and key match..."
    local cert_modulus=$(openssl x509 -in "$ABSOLUTE_NEBULA_DIR/$nodename.crt" -noout -modulus 2>/dev/null | openssl md5 2>/dev/null)
    local key_modulus=$(openssl rsa -in "$ABSOLUTE_NEBULA_DIR/$nodename.key" -noout -modulus 2>/dev/null | openssl md5 2>/dev/null)
    
    if [ "$cert_modulus" = "$key_modulus" ]; then
      echo "[STATUS] Certificate and key match ✅"
    else
      echo "[ERROR] Certificate and key do not match ❌"
      echo "[DEBUG] Certificate modulus: $cert_modulus"
      echo "[DEBUG] Key modulus: $key_modulus"
      exit 1
    fi
    
    # Verify certificate is signed by CA
    echo "[DEBUG] Verifying certificate is signed by CA..."
    if openssl verify -CAfile "$ABSOLUTE_NEBULA_DIR/ca.crt" "$ABSOLUTE_NEBULA_DIR/$nodename.crt" > /dev/null 2>&1; then
      echo "[STATUS] Certificate is properly signed by CA ✅"
    else
      echo "[ERROR] Certificate is NOT signed by the provided CA ❌"
      echo "[DEBUG] Verification output:"
      openssl verify -CAfile "$ABSOLUTE_NEBULA_DIR/ca.crt" "$ABSOLUTE_NEBULA_DIR/$nodename.crt"
      exit 1
    fi
    
    # Check config.yml
    echo "[DEBUG] Checking config.yml..."
    if [ ! -s "$ABSOLUTE_NEBULA_DIR/config.yml" ]; then
      echo "[ERROR] config.yml is empty ❌"
      exit 1
    fi
    
    # Check if config.yml has required sections
    local required_sections=("pki:" "static_host_map:" "lighthouse:" "listen:" "punchy:" "tun:")
    local missing_sections=false
    
    for section in "${required_sections[@]}"; do
      if ! grep -q "$section" "$ABSOLUTE_NEBULA_DIR/config.yml"; then
        echo "[ERROR] Required section missing in config.yml: $section ❌"
        missing_sections=true
      else
        echo "[STATUS] Required section found in config.yml: $section ✅"
      fi
    done
    
    if [ "$missing_sections" = "true" ]; then
      echo "[ERROR] Some required sections are missing in config.yml ❌"
      exit 1
    fi
    
    # Check if Nebula binary exists and is executable
    if [ ! -x "$ABSOLUTE_NEBULA_BIN" ]; then
      echo "[ERROR] Nebula binary not found or not executable: $ABSOLUTE_NEBULA_BIN ❌"
      exit 1
    else
      echo "[STATUS] Nebula binary is available and executable ✅"
      echo "[DEBUG] Nebula version:"
      sudo "$ABSOLUTE_NEBULA_BIN" -version
    fi
    
    # Test config with Nebula binary
    echo "[DEBUG] Testing configuration with Nebula binary..."
    if sudo "$ABSOLUTE_NEBULA_BIN" -config "$ABSOLUTE_NEBULA_DIR/config.yml" -test; then
      echo "[SUCCESS] Nebula configuration test passed ✅"
    else
      echo "[ERROR] Nebula configuration test failed ❌"
      exit 1
    fi
    
    echo "[SUCCESS] All configuration tests passed! ✅"
    echo "[INFO] Your Nebula configuration appears to be valid and complete."
    echo "[INFO] You can start/restart the service with: $0 restart"
    exit 0
    ;;
  test-connectivity)
    echo "[INFO] Testing Nebula VPN connectivity..."
    
    # Check if Nebula is running
    if ! sudo systemctl is-active --quiet nebula; then
      echo "[ERROR] Nebula service is not running ❌"
      echo "[INFO] Please start Nebula first: $0 restart"
      exit 1
    fi
    
    echo "[STATUS] Nebula service is running ✅"
    
    # Get Nebula interface name and IP
    echo "[DEBUG] Checking Nebula network interface..."
    local nebula_interface=$(ip -o link show | grep -i nebula | awk -F': ' '{print $2}' | cut -d '@' -f 1)
    
    if [ -z "$nebula_interface" ]; then
      echo "[ERROR] Nebula network interface not found ❌"
      echo "[DEBUG] Available network interfaces:"
      ip -o link show
      exit 1
    fi
    
    echo "[STATUS] Nebula interface found: $nebula_interface ✅"
    
    # Get Nebula IP address
    local nebula_ip=$(ip -o addr show dev "$nebula_interface" | grep -oP 'inet \K[\d.]+')
    
    if [ -z "$nebula_ip" ]; then
      echo "[ERROR] Nebula IP address not found ❌"
      echo "[DEBUG] Interface details:"
      ip addr show dev "$nebula_interface"
      exit 1
    fi
    
    echo "[STATUS] Nebula IP address: $nebula_ip ✅"
    
    # Extract lighthouse IPs from config
    echo "[DEBUG] Checking lighthouse configuration..."
    local lighthouse_ips=$(grep -A 10 'lighthouse:' "$ABSOLUTE_NEBULA_DIR/config.yml" | grep -oP '\d+\.\d+\.\d+\.\d+')
    
    if [ -z "$lighthouse_ips" ]; then
      echo "[ERROR] No lighthouse IPs found in config ❌"
      echo "[DEBUG] Lighthouse section in config:"
      grep -A 10 'lighthouse:' "$ABSOLUTE_NEBULA_DIR/config.yml"
      exit 1
    fi
    
    echo "[STATUS] Lighthouse IPs found in config ✅"
    
    # Test connectivity to lighthouse nodes
    echo "[INFO] Testing connectivity to lighthouse nodes..."
    local ping_success=false
    
    for ip in $lighthouse_ips; do
      echo "[DEBUG] Testing connectivity to lighthouse: $ip"
      if ping -c 3 -W 2 -I "$nebula_interface" "$ip" > /dev/null 2>&1; then
        echo "[STATUS] Successfully pinged lighthouse: $ip ✅"
        ping_success=true
      else
        echo "[ERROR] Failed to ping lighthouse: $ip ❌"
      fi
    done
    
    # Check for other hosts in the network
    echo "[INFO] Checking for other hosts in the Nebula network..."
    
    # Extract network CIDR from config
    local network_cidr=$(grep -A 10 'tun:' "$ABSOLUTE_NEBULA_DIR/config.yml" | grep -oP 'unsafe_routes.*\d+\.\d+\.\d+\.\d+/\d+')
    
    if [ -z "$network_cidr" ]; then
      # Try alternative method to get network
      network_cidr="${nebula_ip%.*}.0/24"
      echo "[WARNING] Could not determine network CIDR from config, using: $network_cidr ⚠️"
    else
      network_cidr=$(echo "$network_cidr" | grep -oP '\d+\.\d+\.\d+\.\d+/\d+')
      echo "[STATUS] Network CIDR from config: $network_cidr ✅"
    fi
    
    # Scan for hosts (limited to 10 for performance)
    echo "[DEBUG] Scanning for active hosts in the Nebula network (limited scan)..."
    local active_hosts=0
    
    # Extract first 3 octets of IP
    local base_ip="${nebula_ip%.*}"
    
    # Scan a limited range (1-20) to avoid long waits
    for i in {1..20}; do
      local target_ip="$base_ip.$i"
      
      # Skip our own IP
      if [ "$target_ip" = "$nebula_ip" ]; then
        continue
      fi
      
      if ping -c 1 -W 1 -I "$nebula_interface" "$target_ip" > /dev/null 2>&1; then
        echo "[STATUS] Found active host: $target_ip ✅"
        active_hosts=$((active_hosts + 1))
      fi
    done
    
    # Check Nebula status using the binary
    echo "[DEBUG] Checking Nebula status..."
    if [ -x "$ABSOLUTE_NEBULA_BIN" ]; then
      echo "[DEBUG] Nebula status output:"
      sudo "$ABSOLUTE_NEBULA_BIN" -config "$ABSOLUTE_NEBULA_DIR/config.yml" -status
    else
      echo "[WARNING] Nebula binary not found or not executable, skipping status check ⚠️"
    fi
    
    # Summary
    echo "\n[SUMMARY] Connectivity Test Results:"
    echo "[INFO] Nebula interface: $nebula_interface"
    echo "[INFO] Nebula IP address: $nebula_ip"
    
    if [ "$ping_success" = "true" ]; then
      echo "[✅] Lighthouse connectivity: SUCCESS"
    else
      echo "[❌] Lighthouse connectivity: FAILED"
    fi
    
    if [ "$active_hosts" -gt 0 ]; then
      echo "[✅] Other hosts found in network: $active_hosts"
    else
      echo "[⚠️] No other hosts found in network (this may be normal if you're the only node)"
    fi
    
    # Check UDP ports
    echo "[DEBUG] Checking if Nebula UDP port is open..."
    local nebula_port=$(grep -A 5 'listen:' "$ABSOLUTE_NEBULA_DIR/config.yml" | grep -oP ':\s*\K\d+')
    
    if [ -z "$nebula_port" ]; then
      echo "[WARNING] Could not determine Nebula port from config ⚠️"
      nebula_port=4242 # Default port
    fi
    
    echo "[DEBUG] Checking UDP port $nebula_port..."
    if sudo lsof -i :"$nebula_port" -P -n | grep UDP > /dev/null; then
      echo "[STATUS] Nebula UDP port $nebula_port is open ✅"
    else
      echo "[ERROR] Nebula UDP port $nebula_port is not open ❌"
      echo "[DEBUG] This may indicate a firewall issue or that Nebula is not running correctly"
    fi
    
    # Final assessment
    if [ "$ping_success" = "true" ]; then
      echo "[SUCCESS] Nebula VPN appears to be working correctly! ✅"
      exit 0
    else
      echo "[WARNING] Nebula VPN may have connectivity issues ⚠️"
      echo "[INFO] Check your firewall settings and ensure UDP port $nebula_port is open"
      exit 1
    fi
    ;;
  test-all)
    echo "[INFO] Running comprehensive Nebula VPN diagnostics..."
    echo "[INFO] This will run all test commands in sequence to provide a complete report."
    echo "\n==================================================="
    echo "🔍 STEP 1: Testing Token Authentication"
    echo "==================================================="
    
    # Run token test but capture exit code to continue tests
    "$0" test-token
    local token_result=$?
    
    echo "\n==================================================="
    echo "🔍 STEP 2: Testing Configuration Download"
    echo "==================================================="
    
    # Run download test but capture exit code to continue tests
    "$0" test-download
    local download_result=$?
    
    echo "\n==================================================="
    echo "🔍 STEP 3: Testing Configuration Files"
    echo "==================================================="
    
    # Run config test but capture exit code to continue tests
    "$0" test-config
    local config_result=$?
    
    echo "\n==================================================="
    echo "🔍 STEP 4: Testing Network Connectivity"
    echo "==================================================="
    
    # Run connectivity test but capture exit code to continue tests
    "$0" test-connectivity
    local connectivity_result=$?
    
    # Print summary report
    echo "\n==================================================="
    echo "📋 DIAGNOSTIC SUMMARY REPORT"
    echo "==================================================="
    echo "Token Authentication:  $([ $token_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
    echo "Configuration Download: $([ $download_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
    echo "Configuration Files:   $([ $config_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
    echo "Network Connectivity:  $([ $connectivity_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
    echo "==================================================="
    
    # Determine overall status
    if [ $token_result -eq 0 ] && [ $download_result -eq 0 ] && [ $config_result -eq 0 ] && [ $connectivity_result -eq 0 ]; then
      echo "\n[SUCCESS] All tests passed! Your Nebula VPN is properly configured and working. ✅"
      exit 0
    else
      echo "\n[WARNING] Some tests failed. Please review the detailed output above for troubleshooting. ⚠️"
      
      # Provide specific recommendations based on which tests failed
      if [ $token_result -ne 0 ]; then
        echo "[RECOMMENDATION] Token issues detected. Try setting a new token: $0 set-token YOUR_TOKEN"
      fi
      
      if [ $download_result -ne 0 ]; then
        echo "[RECOMMENDATION] Download issues detected. Check your API server and token configuration."
      fi
      
      if [ $config_result -ne 0 ]; then
        echo "[RECOMMENDATION] Configuration issues detected. Try running: $0 update"
      fi
      
      if [ $connectivity_result -ne 0 ]; then
        echo "[RECOMMENDATION] Connectivity issues detected. Check your network and firewall settings."
      fi
      
      exit 1
    fi
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
      echo "[DEBUG] Nebula version:"
      sudo "$ABSOLUTE_NEBULA_BIN" -version
    else
      echo "[STATUS] Nebula binary NOT found: $ABSOLUTE_NEBULA_BIN ❌"
    fi
    
    echo "[DEBUG] Checking for token file:"
    if [ -f "$ABSOLUTE_TOKEN_FILE" ]; then
      echo "[STATUS] Token file exists: $ABSOLUTE_TOKEN_FILE ✅"
      sudo ls -la "$ABSOLUTE_TOKEN_FILE"
      echo "[DEBUG] Token value (first 3 chars): $(sudo head -c 3 "$ABSOLUTE_TOKEN_FILE")..."
    else
      echo "[STATUS] Token file NOT found: $ABSOLUTE_TOKEN_FILE ❌"
    fi
    
    echo "[DEBUG] Testing API server connectivity..."
    if curl -s --head "$API_SERVER" > /dev/null; then
      echo "[STATUS] API server is reachable: $API_SERVER ✅"
      
      # Test authentication with token
      if [ -f "$ABSOLUTE_TOKEN_FILE" ]; then
        echo "[DEBUG] Testing API authentication with token..."
        local encoded_token=$(urlencode "$TOKEN")
        local auth_test_result=$(curl -s -o /dev/null -w "%{http_code}" -G -d "token=$encoded_token" "$API_SERVER/auth/test")
        
        if [ "$auth_test_result" = "200" ] || [ "$auth_test_result" = "204" ]; then
          echo "[STATUS] Authentication successful with provided token ✅"
        elif [ "$auth_test_result" = "401" ] || [ "$auth_test_result" = "403" ]; then
          echo "[STATUS] Authentication failed with provided token (HTTP $auth_test_result) ❌"
          echo "[DEBUG] You may need to set a valid token using the 'set-token' command."
        else
          echo "[STATUS] Unexpected response from authentication test: HTTP $auth_test_result ⚠️"
          echo "[DEBUG] The server may not support the auth test endpoint or may be misconfigured."
        fi
      else
        echo "[STATUS] Cannot test authentication: No token file found ❌"
      fi
    else
      echo "[STATUS] Cannot connect to API server: $API_SERVER ❌"
      echo "[DEBUG] Please check your network connection and server status."
    fi
    
    echo "[DEBUG] Checking systemd service status:"
    if sudo systemctl is-active --quiet nebula; then
      echo "[STATUS] Nebula service is running ✅"
      sudo systemctl status nebula --no-pager
    else
      echo "[STATUS] Nebula service is NOT running ❌"
      sudo systemctl status nebula --no-pager || true
    fi
    ;;
  delete)
    echo "[INFO] Stopping Nebula VPN service and removing all files..."
    
    # Stop and disable the service
    echo "[INFO] Stopping and disabling Nebula service..."
    if sudo systemctl is-active --quiet nebula; then
      if sudo systemctl stop nebula; then
        echo "[STATUS] Nebula service stopped successfully ✅"
      else
        echo "[ERROR] Failed to stop Nebula service ❌"
      fi
    else
      echo "[STATUS] Nebula service is already stopped ✅"
    fi
    
    if sudo systemctl is-enabled --quiet nebula 2>/dev/null; then
      if sudo systemctl disable nebula; then
        echo "[STATUS] Nebula service disabled successfully ✅"
      else
        echo "[ERROR] Failed to disable Nebula service ❌"
      fi
    else
      echo "[STATUS] Nebula service is already disabled ✅"
    fi
    
    # Remove systemd service file
    echo "[INFO] Removing systemd service file..."
    if [ -f "$SERVICE_FILE" ]; then
      if sudo rm -f "$SERVICE_FILE"; then
        echo "[STATUS] Systemd service file removed successfully ✅"
      else
        echo "[ERROR] Failed to remove systemd service file ❌"
      fi
    else
      echo "[STATUS] Systemd service file not found, skipping ✅"
    fi
    
    # Reload systemd daemon
    echo "[INFO] Reloading systemd daemon..."
    sudo systemctl daemon-reload
    
    # Remove Nebula binary
    echo "[INFO] Removing Nebula binary..."
    if [ -f "$ABSOLUTE_NEBULA_BIN" ]; then
      if sudo rm -f "$ABSOLUTE_NEBULA_BIN"; then
        echo "[STATUS] Nebula binary removed successfully ✅"
      else
        echo "[ERROR] Failed to remove Nebula binary ❌"
      fi
    else
      echo "[STATUS] Nebula binary not found, skipping ✅"
    fi
    
    # Remove Nebula configuration directory
    echo "[INFO] Removing Nebula configuration directory..."
    if [ -d "$ABSOLUTE_NEBULA_DIR" ]; then
      if sudo rm -rf "$ABSOLUTE_NEBULA_DIR"; then
        echo "[STATUS] Nebula configuration directory removed successfully ✅"
      else
        echo "[ERROR] Failed to remove Nebula configuration directory ❌"
      fi
    else
      echo "[STATUS] Nebula configuration directory not found, skipping ✅"
    fi
    
    # Remove token file if it exists in a different location
    if [ -f "$ABSOLUTE_TOKEN_FILE" ] && [ "$ABSOLUTE_TOKEN_FILE" != "$ABSOLUTE_NEBULA_DIR/token.txt" ]; then
      echo "[INFO] Removing token file..."
      if sudo rm -f "$ABSOLUTE_TOKEN_FILE"; then
        echo "[STATUS] Token file removed successfully ✅"
      else
        echo "[ERROR] Failed to remove token file ❌"
      fi
    fi
    
    echo "[SUCCESS] Nebula VPN has been completely removed from the system! ✅"
    ;;
  *)
    echo "Usage: jetsonnebula install              # install/start Nebula"
    echo "       jetsonnebula status               # check status"
    echo "       jetsonnebula update               # update config only"
    echo "       jetsonnebula set-token <token>    # set authentication token"
    echo "       jetsonnebula test-token           # test if token works with API server"
    echo "       jetsonnebula test-download        # test client bundle download"
    echo "       jetsonnebula test-config          # test Nebula configuration files"
    echo "       jetsonnebula test-connectivity    # test Nebula VPN connectivity"
    echo "       jetsonnebula test-all             # run all tests and provide a diagnostic report"
    echo "       jetsonnebula debug                # run diagnostics"
    echo "       jetsonnebula restart              # restart Nebula service"
    echo "       jetsonnebula delete               # delete everything and stop services"
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