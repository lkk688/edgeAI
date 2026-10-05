#!/bin/bash
# Run inside an image built with INSTALL_THOR_VLA=1: activate gr00t-venv, use the GPU,
# deactivate, then check the system Python still has its own torch/lerobot/transformers.
. /opt/gr00t-venv/bin/activate
python - <<PY
import torch, tensorrt
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("gr00t-venv ", torch.__version__, tensorrt.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), bool((x @ x).abs().mean() > 0))
PY
deactivate
python - <<PY
import torch, lerobot, transformers
print("system     ", torch.__version__, torch.cuda.is_available(), "lerobot", lerobot.__version__, "transformers", transformers.__version__)
PY
