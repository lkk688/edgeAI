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
  
  # Auto-install ARM64 NGC CLI if missing
  if ! command -v ngc &>/dev/null && [ -f "$HOME/.local/bin/ngc" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if command -v ngc &>/dev/null; then
    echo "  → Using NGC CLI to download riva_quickstart_arm64:${RIVA_VERSION}..."
    ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION}" --dest "$RIVA_DIR"
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
      ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION}" --dest "$RIVA_DIR"
    else
      echo "══════════════════════════════════════════════════"
      echo "🔑 NGC Authentication Required for Riva"
      echo "══════════════════════════════════════════════════"
      echo "NVIDIA Riva Quickstart for ARM64 requires an NGC API Key."
      echo
      echo "1️⃣ Generate your free NGC API Key:"
      echo "   👉 Visit: https://org.ngc.nvidia.com/setup/api-key"
      echo
      echo "2️⃣ Configure NGC API key on Jetson:"
      echo "   ngc config set"
      echo
      echo "3️⃣ Download Riva Quickstart ARM64:"
      echo "   ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION} --dest $RIVA_DIR"
      echo "══════════════════════════════════════════════════"
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
