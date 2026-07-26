#!/usr/bin/env bash
# =============================================================================
# NVIDIA Riva Quickstart Automated Setup Script for Jetson (Thor / Orin)
# =============================================================================
set -e

RIVA_VERSION="2.14.0"
RIVA_DIR="/Developer/riva_quickstart_v${RIVA_VERSION}"
RIVA_MODELS_DIR="/Developer/models/riva"

echo "================================================================="
echo " 🛠️ Setting up NVIDIA Riva Quickstart v${RIVA_VERSION} on Jetson"
echo "================================================================="

mkdir -p /Developer/models
mkdir -p "$RIVA_DIR"

if [ -f "$RIVA_DIR/config.sh" ]; then
  echo "📂 Riva Quickstart package found at $RIVA_DIR"
else
  echo "📥 Attempting to download Riva Quickstart ARM64 package..."
  
  # Function to download riva quickstart with NGC CLI & handle auth
  download_with_ngc() {
    if ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION}" --dest "$RIVA_DIR" 2>/dev/null; then
      return 0
    fi

    # Download failed (likely 403 Forbidden due to missing API key)
    echo "🔑 NGC Authentication required to download Riva resources."
    
    local active_key="$NGC_API_KEY"
    if [ -z "$active_key" ] && [ -f "$HOME/.ngc/config" ]; then
      active_key=$(grep -E '^apikey' "$HOME/.ngc/config" | head -1 | cut -d= -f2 | tr -d ' ')
    fi

    if [ -z "$active_key" ]; then
      echo "══════════════════════════════════════════════════"
      echo "🔑 NGC API Key Required for NVIDIA Riva"
      echo "══════════════════════════════════════════════════"
      echo "NVIDIA NGC requires a free API key to download Riva Speech Server."
      echo "👉 Generate your free NGC API Key in 1 minute at:"
      echo "   https://org.ngc.nvidia.com/setup/api-key"
      echo "══════════════════════════════════════════════════"
      read -sp "Enter your NGC API Key: " USER_NGC_KEY
      echo
      if [ -n "$USER_NGC_KEY" ]; then
        active_key="$USER_NGC_KEY"
        mkdir -p "$HOME/.ngc"
        cat <<EOF > "$HOME/.ngc/config"
[GLOBAL]
apikey = $active_key
format_type = ascii
EOF
    if [ -n "$active_key" ]; then
      mkdir -p "$HOME/.ngc"
      cat <<EOF > "$HOME/.ngc/config"
[GLOBAL]
apikey = $active_key
format_type = ascii
EOF
      echo "💾 NGC API Key configured in $HOME/.ngc/config"
      echo "🚀 Retrying download with NGC CLI..."
      if ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION}" --dest "$RIVA_DIR"; then
        return 0
      fi

      echo "══════════════════════════════════════════════════"
      echo "🔒 NGC Catalog Resource Access Authorization Needed"
      echo "══════════════════════════════════════════════════"
      echo "Your NGC API key is saved, but NGC requires 1-click resource access approval."
      echo
      echo "👉 Open this link in your browser, log into NGC, and click 'Get Resource':"
      echo "   https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64"
      echo "══════════════════════════════════════════════════"
      read -p "Press Enter after clicking 'Get Resource' on NGC to retry... "
      ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION}" --dest "$RIVA_DIR"
    fi

  if command -v ngc &>/dev/null; then
    echo "  → Using NGC CLI to download riva_quickstart_arm64:${RIVA_VERSION}..."
    download_with_ngc
  else
    echo "📥 Installing ARM64 NGC CLI..."
    TMP_NGC="$(mktemp -d)"
    if curl -sL "https://ngc.nvidia.com/downloads/ngccli_arm64.zip" -o "$TMP_NGC/ngccli_arm64.zip" 2>/dev/null && [ -s "$TMP_NGC/ngccli_arm64.zip" ]; then
      unzip -q -o "$TMP_NGC/ngccli_arm64.zip" -d "$TMP_NGC"
      mkdir -p "$HOME/.local/bin"
      cp -r "$TMP_NGC/ngc-cli"* "$HOME/.local/bin/"
      ln -sf "$HOME/.local/bin/ngc-cli/ngc" "$HOME/.local/bin/ngc"
      chmod +x "$HOME/.local/bin/ngc"
      export PATH="$HOME/.local/bin:$PATH"
      rm -rf "$TMP_NGC"
      echo "✅ Installed ARM64 NGC CLI to $HOME/.local/bin/ngc"
    fi

    if command -v ngc &>/dev/null; then
      echo "  → Using NGC CLI to download riva_quickstart_arm64:${RIVA_VERSION}..."
      download_with_ngc
    else
      echo "❌ Failed to install NGC CLI. Please install ngc manually."
      exit 1
    fi
  fi
fi

cd "$RIVA_DIR"

# Configure config.sh for Jetson Tegra platform
if [ -f "config.sh" ]; then
  echo "⚙️ Configuring Riva Quickstart config.sh..."
  sed -i "s|^riva_target_gpu_family=.*|riva_target_gpu_family=\"tegra\"|" config.sh
  sed -i "s|^riva_tegra_platform=.*|riva_tegra_platform=\"orin\"|" config.sh
  sed -i "s|^riva_model_loc=.*|riva_model_loc=\"$RIVA_MODELS_DIR\"|" config.sh
fi

echo "🚀 Initializing Riva models into $RIVA_MODELS_DIR..."
bash riva_init.sh

echo "================================================================="
echo " ✅ NVIDIA Riva Quickstart Setup Complete!"
echo " 👉 Start Riva server with: sjsujetsontool audio start --riva"
echo "================================================================="
