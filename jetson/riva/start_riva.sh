#!/usr/bin/env bash
# =============================================================================
# Streamlined NVIDIA Riva Speech Server Launcher for Jetson (Thor / Orin)
# =============================================================================
set -e

RIVA_PORT="${RIVA_PORT:-50051}"
RIVA_MODELS_DIR="${RIVA_MODELS_DIR:-/Developer/models/riva}"
RIVA_IMAGE="${RIVA_IMAGE:-nvcr.io/nvidia/riva/riva-speech:2.14.0}"
CONTAINER_NAME="riva-speech"

echo "================================================================="
echo " 🎙️ Starting NVIDIA Riva Speech Server (Port $RIVA_PORT)"
echo "================================================================="

# Check if already running
if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
  echo "✅ Riva container '$CONTAINER_NAME' is ALREADY RUNNING."
  exit 0
fi

# Stop leftover container if any
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Check if models directory exists
if [ ! -d "$RIVA_MODELS_DIR" ] || [ -z "$(ls -A "$RIVA_MODELS_DIR" 2>/dev/null)" ]; then
  echo "⚠️  Riva models directory '$RIVA_MODELS_DIR' is missing or empty."
  echo "👉 Run 'sjsujetsontool audio setup' to automatically initialize default Riva models."
  echo "   Or download models manually using NGC CLI."
  exit 1
fi

echo "🚀 Launching Riva Speech container ($RIVA_IMAGE)..."
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --runtime=nvidia \
  --network host \
  --ipc=host \
  -v "$RIVA_MODELS_DIR":/riva_models \
  "$RIVA_IMAGE" \
  start-riva --riva-uri=0.0.0.0:"$RIVA_PORT" --models=/riva_models

echo "⏳ Waiting for Riva Speech Server to initialize gRPC port $RIVA_PORT (15-30s)..."
for i in {1..20}; do
  if python3 -c "import socket; s = socket.socket(); s.settimeout(1); exit(0 if s.connect_ex(('localhost', $RIVA_PORT)) == 0 else 1)" 2>/dev/null; then
    echo "  ✅ NVIDIA Riva Speech Server is READY on port $RIVA_PORT!"
    exit 0
  fi
  sleep 2
  echo -n "."
done
echo
echo "👉 Riva container is starting in background. Check status anytime with: sjsujetsontool audio test"
