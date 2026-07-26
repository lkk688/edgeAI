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
  
  if command -v ngc &>/dev/null; then
    echo "  → Using NGC CLI to download riva_quickstart_arm64:${RIVA_VERSION}..."
    ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION}" --dest "$RIVA_DIR"
  else
    TMP_TAR="$(mktemp)"
    curl -sL "https://catalog.ngc.nvidia.com/api/v1/resources/nvidia/riva/riva_quickstart_arm64/versions/${RIVA_VERSION}/files/riva_quickstart_arm64_v${RIVA_VERSION}.tar.gz" -o "$TMP_TAR" 2>/dev/null || true
    
    if [ -s "$TMP_TAR" ] && file "$TMP_TAR" | grep -q "gzip compressed data"; then
      tar -xzf "$TMP_TAR" -C "$RIVA_DIR" --strip-components=1
      rm -f "$TMP_TAR"
      echo "✅ Extracted Riva Quickstart into $RIVA_DIR"
    else
      rm -f "$TMP_TAR"
      echo "══════════════════════════════════════════════════"
      echo "🔑 NGC Authentication / CLI Required for Riva"
      echo "══════════════════════════════════════════════════"
      echo "NVIDIA Riva Quickstart for ARM64 requires NGC authentication."
      echo "To download Riva Quickstart ARM64 via NGC CLI:"
      echo
      echo "1️⃣ Install NGC CLI (if not already installed):"
      echo "   wget -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip"
      echo "   unzip -o ngccli_linux.zip && chmod +x ngc-cli/ngc"
      echo
      echo "2️⃣ Generate your free NGC API Key:"
      echo "   👉 Visit: https://org.ngc.nvidia.com/setup/api-key"
      echo "   Run: ./ngc-cli/ngc config set"
      echo
      echo "3️⃣ Download Riva Quickstart ARM64:"
      echo "   ./ngc-cli/ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION} --dest $RIVA_DIR"
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
