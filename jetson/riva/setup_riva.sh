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
  echo "📥 Downloading Riva Quickstart ARM64 package..."
  TMP_TAR="$(mktemp)"
  
  # Download Riva quickstart tarball from NGC
  if curl -fsSL "https://ngc.nvidia.com/downloads/riva_quickstart_arm64_v${RIVA_VERSION}.tar.gz" -o "$TMP_TAR"; then
    tar -xzf "$TMP_TAR" -C "$RIVA_DIR" --strip-components=1
    rm -f "$TMP_TAR"
    echo "✅ Extracted Riva Quickstart into $RIVA_DIR"
  else
    echo "❌ Failed to download Riva quickstart package from NGC."
    echo "👉 Please manually download riva_quickstart_arm64_v${RIVA_VERSION}.tar.gz into $RIVA_DIR"
    exit 1
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
