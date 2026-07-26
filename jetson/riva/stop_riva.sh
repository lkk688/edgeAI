#!/usr/bin/env bash
# =============================================================================
# NVIDIA Riva Speech Server Stopper
# =============================================================================
CONTAINER_NAME="riva-speech"

echo "🛑 Stopping NVIDIA Riva Speech Server container..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Also stop Quickstart riva server if running via quickstart scripts
RIVA_DIR=$(find "$HOME" "/Developer" -maxdepth 3 -type d -name "riva_quickstart*" 2>/dev/null | head -n 1)
if [ -n "$RIVA_DIR" ] && [ -f "$RIVA_DIR/riva_stop.sh" ]; then
  (cd "$RIVA_DIR" && bash riva_stop.sh) 2>/dev/null || true
fi

echo "✅ NVIDIA Riva Speech Server stopped."
