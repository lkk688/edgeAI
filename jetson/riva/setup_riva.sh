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
    if curl -fsSL "https://catalog.ngc.nvidia.com/api/v1/resources/nvidia/riva/riva_quickstart_arm64/versions/${RIVA_VERSION}/files/riva_quickstart_arm64_v${RIVA_VERSION}.tar.gz" -o "$TMP_TAR" 2>/dev/null && [ -s "$TMP_TAR" ]; then
      tar -xzf "$TMP_TAR" -C "$RIVA_DIR" --strip-components=1 2>/dev/null || true
      rm -f "$TMP_TAR"
    else
      echo "⚠️  NGC CLI is not installed and direct download link requires NGC authentication."
      echo "👉 Please run the following command to download Riva Quickstart ARM64 via NGC CLI:"
      echo "   ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:${RIVA_VERSION}"
      echo "   or extract the tarball into $RIVA_DIR"
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
