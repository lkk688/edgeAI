#!/bin/bash
# setup_offload_server.sh — set up the HTTP object-detection offload server on a
# GPU workstation (e.g. lkk-alienware51, GTX 1080 Ti). Jetsons then offload over
# HTTP with `--offload <this-host>` — no SSH, no per-device keys.
#
# Run this ON THE SERVER. It creates a conda env, installs deps, and (optionally)
# starts the server. Files expected next to this script:
#   jetson_detection_server.py, jetson_object_detection_toolkit.py
#
#   ./setup_offload_server.sh            # set up env only
#   ./setup_offload_server.sh --run      # set up + launch the server now
#
# Env vars: OFFLOAD_CONDA_ENV (default objdet), DETECT_API_KEY (optional bearer
# token clients must send), DETECT_PORT (default 8000).

set -e
ENV_NAME="${OFFLOAD_CONDA_ENV:-objdet}"
PORT="${DETECT_PORT:-8000}"
PYVER="3.10"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1) Locate conda
if ! command -v conda >/dev/null 2>&1; then
  for p in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
  done
fi
command -v conda >/dev/null 2>&1 || { echo "❌ conda not found — install Miniconda first."; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2) Create / update the env. GTX 1080 Ti is Pascal (sm_61) → CUDA 11.8 wheels.
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "Creating conda env '${ENV_NAME}' (python ${PYVER}) ..."
  conda create -y -n "$ENV_NAME" python=$PYVER
fi
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda run -n "$ENV_NAME" pip install ultralytics transformers pillow numpy \
    opencv-python-headless fastapi "uvicorn[standard]"

# 3) Verify the GPU is visible
conda run -n "$ENV_NAME" python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"

echo "✅ Offload server env ready (conda env: ${ENV_NAME}, port ${PORT})."
echo "   Start it with:"
echo "   cd ${HERE} && DETECT_API_KEY=\"\${DETECT_API_KEY}\" conda run -n ${ENV_NAME} uvicorn jetson_detection_server:app --host 0.0.0.0 --port ${PORT}"

# 4) Optionally launch now (background)
if [ "$1" = "--run" ]; then
  cd "$HERE"
  echo "🚀 Launching server on 0.0.0.0:${PORT} ..."
  DETECT_API_KEY="${DETECT_API_KEY}" DETECT_PORT="${PORT}" \
    nohup conda run --no-capture-output -n "$ENV_NAME" \
    uvicorn jetson_detection_server:app --host 0.0.0.0 --port "${PORT}" \
    > "$HERE/detection_server.log" 2>&1 &
  sleep 1
  echo "   PID $! — logs: $HERE/detection_server.log"
fi
